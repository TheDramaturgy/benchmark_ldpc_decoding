/**
 * @file benchmark_ldpc_from_export_metadata.cc
 *
 * Read LDPC LLR datasets produced by the old header.bin exporter or by the new
 * metadata.json-only exporter, then benchmark bblib_ldpc_decoder_5gnr only.
 *
 * Supported dataset directory formats:
 *
 *   Old / size-sweep format:
 *     header.bin
 *     metadata.json              optional
 *     llrs_i8.bin
 *     ref_msg_u8.bin
 *
 *   New MCS-mother-rate format:
 *     metadata.json
 *     llrs_i8.bin
 *     ref_msg_u8.bin
 *     encoded_codewords_u8.bin   optional
 *     flexran_decoded_u8.bin     optional
 *
 * The benchmark excludes data generation, modulation, demodulation, file I/O,
 * and correctness checking from the timed section.
 */

#include <gflags/gflags.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gettime.h"
#include "memory_manage.h"
#include "phy_ldpc_decoder_5gnr.h"
#include "utils_ldpc.h"

DEFINE_string(input_root, "ldpc_llr_export",
              "Root export directory, or a single dataset directory containing "
              "metadata.json/header.bin plus llrs_i8.bin/ref_msg_u8.bin.");
DEFINE_string(output_csv, "flexran_ldpc_benchmark.csv",
              "CSV output path.");
DEFINE_uint64(warmup_repeats, 5,
              "Full-dataset warmup passes, not included in timing.");
DEFINE_uint64(repeats, 100,
              "Full-dataset measured passes.");
DEFINE_uint64(limit_codeblocks, 0,
              "If nonzero, benchmark only the first N codeblocks in each dataset.");
DEFINE_bool(verify, true,
            "After timing, compare the last decoded output with ref_msg_u8.bin.");
DEFINE_int32(max_iterations_override, -1,
             "If >= 0, override maxIterations from metadata/header.");
DEFINE_int32(early_termination_override, -1,
             "If 0 or 1, override earlyTermination from metadata/header.");
DEFINE_bool(prefer_metadata, true,
            "If true, use metadata.json even when header.bin is also present. "
            "This is safer for newly generated datasets.");

#pragma pack(push, 1)
struct DatasetHeader {
  char magic[8];
  uint32_t version;

  uint32_t mcs_index;
  uint32_t mcs_qm;
  uint32_t mcs_code_rate_x1024;

  uint32_t snr_index;
  double snr_db;
  double noise_sigma;

  uint64_t num_codeblocks;
  uint32_t llr_len;
  uint32_t msg_bits;
  uint32_t msg_bytes;
  uint32_t encoded_bytes;

  uint32_t base_graph;
  uint32_t zc;
  uint32_t n_rows;
  uint32_t num_filler_bits;
  uint32_t max_iterations;
  uint32_t early_termination;

  uint32_t mod_order_bits;
  uint32_t ofdm_data_num;
  double effective_cb_code_rate;

  uint32_t reserved[32];
};
#pragma pack(pop)

static_assert(sizeof(DatasetHeader) == 8 + 4 + 12 + 4 + 16 + 8 + 4 * 12 + 8 + 32 * 4,
              "Unexpected DatasetHeader size");

static void CheckFileOpen(const std::ifstream& file, const std::string& path) {
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file for reading: " + path);
  }
}

static void CheckFileOpen(const std::ofstream& file, const std::string& path) {
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file for writing: " + path);
  }
}

static std::string ReadTextFile(const std::filesystem::path& path) {
  std::ifstream f(path);
  CheckFileOpen(f, path.string());

  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

static std::string RegexEscape(const std::string& s) {
  std::string out;
  out.reserve(s.size() * 2);
  for (const char c : s) {
    switch (c) {
      case '\\': case '^': case '$': case '.': case '|':
      case '?': case '*': case '+': case '(': case ')':
      case '[': case ']': case '{': case '}':
        out.push_back('\\');
        [[fallthrough]];
      default:
        out.push_back(c);
    }
  }
  return out;
}

static bool JsonFindNumber(const std::string& json,
                           const std::string& key,
                           double* value) {
  const std::regex r(
      "\\\"" + RegexEscape(key) + "\\\"\\s*:\\s*"
      "([-+]?(?:[0-9]*\\.)?[0-9]+(?:[eE][-+]?[0-9]+)?)");

  std::smatch m;
  if (!std::regex_search(json, m, r)) {
    return false;
  }

  *value = std::stod(m[1].str());
  return true;
}

static bool JsonFindBool(const std::string& json,
                         const std::string& key,
                         bool* value) {
  const std::regex r(
      "\\\"" + RegexEscape(key) + "\\\"\\s*:\\s*"
      "(true|false|0|1)(?![0-9A-Za-z_])");

  std::smatch m;
  if (!std::regex_search(json, m, r)) {
    return false;
  }

  const std::string s = m[1].str();
  *value = (s == "true" || s == "1");
  return true;
}

static bool JsonFindString(const std::string& json,
                           const std::string& key,
                           std::string* value) {
  const std::regex r(
      "\\\"" + RegexEscape(key) + "\\\"\\s*:\\s*\\\"([^\\\"]*)\\\"");

  std::smatch m;
  if (!std::regex_search(json, m, r)) {
    return false;
  }

  *value = m[1].str();
  return true;
}

static double JsonGetDoubleAny(const std::string& json,
                               const std::vector<std::string>& keys,
                               double default_value) {
  for (const auto& key : keys) {
    double value = 0.0;
    if (JsonFindNumber(json, key, &value)) {
      return value;
    }
  }
  return default_value;
}

static uint64_t JsonGetU64Any(const std::string& json,
                              const std::vector<std::string>& keys,
                              uint64_t default_value) {
  for (const auto& key : keys) {
    double value = 0.0;
    if (JsonFindNumber(json, key, &value)) {
      return static_cast<uint64_t>(std::llround(value));
    }

    bool b = false;
    if (JsonFindBool(json, key, &b)) {
      return b ? 1u : 0u;
    }
  }
  return default_value;
}

static uint32_t JsonGetU32Any(const std::string& json,
                              const std::vector<std::string>& keys,
                              uint32_t default_value) {
  return static_cast<uint32_t>(JsonGetU64Any(json, keys, default_value));
}

static uint32_t ParseIndexFromPath(const std::filesystem::path& path,
                                   const std::string& prefix,
                                   uint32_t default_value) {
  const std::regex r("^" + prefix + "_([0-9]+)$");

  for (const auto& part : path) {
    const std::string s = part.string();
    std::smatch m;
    if (std::regex_match(s, m, r)) {
      return static_cast<uint32_t>(std::stoul(m[1].str()));
    }
  }

  return default_value;
}

static void ValidateHeaderFields(const DatasetHeader& h,
                                 const std::filesystem::path& dataset_dir) {
  std::ostringstream missing;

  if (h.num_codeblocks == 0) missing << " num_codeblocks";
  if (h.llr_len == 0) missing << " llr_len";
  if (h.msg_bits == 0) missing << " msg_bits/msg_bits_per_cb";
  if (h.msg_bytes == 0) missing << " msg_bytes/msg_bytes_per_cb";
  if (h.base_graph == 0) missing << " base_graph";
  if (h.zc == 0) missing << " zc";
  if (h.n_rows == 0) missing << " n_rows";

  const std::string miss = missing.str();
  if (!miss.empty()) {
    throw std::runtime_error("Dataset metadata/header is missing required field(s):" +
                             miss + " in " + dataset_dir.string());
  }
}

static DatasetHeader ReadHeaderBin(const std::filesystem::path& dataset_dir) {
  const std::string path = (dataset_dir / "header.bin").string();
  std::ifstream f(path, std::ios::binary);
  CheckFileOpen(f, path);

  DatasetHeader h {};
  f.read(reinterpret_cast<char*>(&h), sizeof(h));

  if (f.gcount() != static_cast<std::streamsize>(sizeof(h))) {
    throw std::runtime_error("Could not read complete header: " + path);
  }

  if (std::strncmp(h.magic, "LDPCDS1", 7) != 0) {
    throw std::runtime_error("Invalid dataset magic in: " + path);
  }

  ValidateHeaderFields(h, dataset_dir);
  return h;
}

static DatasetHeader ReadMetadataJson(const std::filesystem::path& dataset_dir) {
  const std::filesystem::path path = dataset_dir / "metadata.json";
  const std::string json = ReadTextFile(path);

  DatasetHeader h {};
  std::memset(&h, 0, sizeof(h));
  std::memcpy(h.magic, "LDPCDS1", 7);
  h.version = JsonGetU32Any(json, {"version", "dataset_version"}, 3);

  h.mcs_index = JsonGetU32Any(
      json, {"mcs_index"}, ParseIndexFromPath(dataset_dir, "mcs", 0));
  h.mcs_qm = JsonGetU32Any(
      json, {"mcs_qm", "mod_order_bits", "qm"}, 0);
  h.mcs_code_rate_x1024 = JsonGetU32Any(
      json, {"mcs_code_rate_x1024", "code_rate_x1024", "R_x1024"}, 0);

  if (h.mcs_code_rate_x1024 == 0) {
    const double r = JsonGetDoubleAny(
        json, {"mcs_code_rate", "mother_code_rate", "effective_cb_code_rate"}, 0.0);
    if (r > 0.0 && r <= 1.0) {
      h.mcs_code_rate_x1024 = static_cast<uint32_t>(std::llround(r * 1024.0));
    }
  }

  h.snr_index = JsonGetU32Any(
      json, {"snr_index"}, ParseIndexFromPath(dataset_dir, "snr", 0));
  h.snr_db = JsonGetDoubleAny(json, {"snr_db"}, 0.0);
  h.noise_sigma = JsonGetDoubleAny(json, {"noise_sigma", "sigma"}, 0.0);

  h.num_codeblocks = JsonGetU64Any(json, {"num_codeblocks", "num_samples"}, 0);
  h.llr_len = JsonGetU32Any(json, {"llr_len", "num_channel_llrs"}, 0);
  h.msg_bits = JsonGetU32Any(json, {"msg_bits_per_cb", "msg_bits", "num_msg_bits"}, 0);
  h.msg_bytes = JsonGetU32Any(
      json, {"msg_bytes_per_cb", "msg_bytes"},
      h.msg_bits == 0 ? 0 : static_cast<uint32_t>((h.msg_bits + 7) / 8));
  h.encoded_bytes = JsonGetU32Any(
      json, {"encoded_bytes_per_cb", "encoded_bytes"},
      h.llr_len == 0 ? 0 : static_cast<uint32_t>((h.llr_len + 7) / 8));

  h.base_graph = JsonGetU32Any(json, {"base_graph", "generated_base_graph"}, 0);
  h.zc = JsonGetU32Any(json, {"zc", "generated_zc"}, 0);
  h.n_rows = JsonGetU32Any(json, {"n_rows", "num_rows"}, 0);
  h.num_filler_bits = JsonGetU32Any(json, {"num_filler_bits", "filler_bits"}, 0);
  h.max_iterations = JsonGetU32Any(json, {"max_iterations", "max_iter"}, 5);
  h.early_termination = JsonGetU32Any(json, {"early_termination", "early_term"}, 1);

  h.mod_order_bits = JsonGetU32Any(
      json, {"mod_order_bits", "mcs_qm", "qm"}, h.mcs_qm);
  h.ofdm_data_num = JsonGetU32Any(json, {"ofdm_data_num"}, 0);

  h.effective_cb_code_rate = JsonGetDoubleAny(
      json,
      {"effective_cb_code_rate", "mother_code_rate", "mcs_code_rate"},
      h.llr_len == 0 ? 0.0 : static_cast<double>(h.msg_bits) /
                                static_cast<double>(h.llr_len));

  ValidateHeaderFields(h, dataset_dir);
  return h;
}

static DatasetHeader ReadDatasetDescription(
    const std::filesystem::path& dataset_dir) {
  const bool has_metadata = std::filesystem::exists(dataset_dir / "metadata.json");
  const bool has_header = std::filesystem::exists(dataset_dir / "header.bin");

  if (FLAGS_prefer_metadata && has_metadata) {
    try {
      return ReadMetadataJson(dataset_dir);
    } catch (const std::exception& e) {
      if (!has_header) {
        throw;
      }
      std::cerr << "WARNING: failed to parse metadata.json in " << dataset_dir
                << ": " << e.what() << "\n"
                << "         Falling back to header.bin\n";
    }
  }

  if (has_header) {
    return ReadHeaderBin(dataset_dir);
  }

  if (has_metadata) {
    return ReadMetadataJson(dataset_dir);
  }

  throw std::runtime_error("Dataset directory has neither metadata.json nor header.bin: " +
                           dataset_dir.string());
}

static bool IsDatasetDir(const std::filesystem::path& dir) {
  const bool has_description =
      std::filesystem::exists(dir / "metadata.json") ||
      std::filesystem::exists(dir / "header.bin");

  return has_description &&
         std::filesystem::exists(dir / "llrs_i8.bin") &&
         std::filesystem::exists(dir / "ref_msg_u8.bin");
}

static std::vector<std::filesystem::path> FindDatasets(
    const std::filesystem::path& root) {
  std::vector<std::filesystem::path> dirs;

  if (IsDatasetDir(root)) {
    dirs.push_back(root);
    return dirs;
  }

  if (!std::filesystem::exists(root)) {
    throw std::runtime_error("Input root does not exist: " + root.string());
  }

  for (const auto& entry : std::filesystem::recursive_directory_iterator(root)) {
    if (!entry.is_directory()) {
      continue;
    }

    if (IsDatasetDir(entry.path())) {
      dirs.push_back(entry.path());
    }
  }

  std::sort(dirs.begin(), dirs.end());
  return dirs;
}

static void LoadLlrs(const std::filesystem::path& dataset_dir,
                     const DatasetHeader& h,
                     size_t num_codeblocks,
                     Table<int8_t>& llrs) {
  const std::string path = (dataset_dir / "llrs_i8.bin").string();
  std::ifstream f(path, std::ios::binary);
  CheckFileOpen(f, path);

  llrs.Calloc(num_codeblocks, h.llr_len, Agora_memory::Alignment_t::kAlign64);

  for (size_t cb = 0; cb < num_codeblocks; cb++) {
    f.read(reinterpret_cast<char*>(llrs[cb]), h.llr_len * sizeof(int8_t));
    if (!f) {
      throw std::runtime_error("Failed while reading LLRs from: " + path);
    }
  }
}

static void DecodeAll(const DatasetHeader& h,
                      size_t num_codeblocks,
                      Table<int8_t>& llrs,
                      Table<uint8_t>& decoded_codewords,
                      bblib_ldpc_decoder_5gnr_request& request,
                      bblib_ldpc_decoder_5gnr_response& response) {
  for (size_t cb = 0; cb < num_codeblocks; cb++) {
    request.varNodes = llrs[cb];
    response.compactedMessageBytes = decoded_codewords[cb];

    bblib_ldpc_decoder_5gnr(&request, &response);
  }
}

static double Mean(const std::vector<double>& x) {
  if (x.empty()) {
    return 0.0;
  }
  return std::accumulate(x.begin(), x.end(), 0.0) / static_cast<double>(x.size());
}

static double StddevSample(const std::vector<double>& x) {
  if (x.size() <= 1) {
    return 0.0;
  }

  const double mean = Mean(x);
  double acc = 0.0;
  for (double v : x) {
    const double d = v - mean;
    acc += d * d;
  }

  return std::sqrt(acc / static_cast<double>(x.size() - 1));
}

struct VerifyResult {
  uint64_t bit_errors = 0;
  uint64_t total_bits = 0;
  uint64_t block_errors = 0;
  uint64_t total_blocks = 0;

  double ber() const {
    return total_bits == 0 ? 0.0 : static_cast<double>(bit_errors) / total_bits;
  }

  double bler() const {
    return total_blocks == 0 ? 0.0 : static_cast<double>(block_errors) / total_blocks;
  }
};

static VerifyResult VerifyDecoded(const std::filesystem::path& dataset_dir,
                                  const DatasetHeader& h,
                                  size_t num_codeblocks,
                                  Table<uint8_t>& decoded_codewords) {
  VerifyResult result;
  result.total_blocks = num_codeblocks;
  result.total_bits = static_cast<uint64_t>(num_codeblocks) * h.msg_bits;

  const std::string path = (dataset_dir / "ref_msg_u8.bin").string();
  std::ifstream f(path, std::ios::binary);
  CheckFileOpen(f, path);

  std::vector<uint8_t> ref(h.msg_bytes);

  for (size_t cb = 0; cb < num_codeblocks; cb++) {
    f.read(reinterpret_cast<char*>(ref.data()), h.msg_bytes);
    if (!f) {
      throw std::runtime_error("Failed while reading reference bytes from: " + path);
    }

    bool block_error = false;
    for (size_t j = 0; j < h.msg_bytes; j++) {
      const uint8_t input = ref[j];
      const uint8_t output = decoded_codewords[cb][j];

      if (input == output) {
        continue;
      }

      for (size_t k = 0; k < 8; k++) {
        const size_t bit_idx = j * 8 + k;
        if (bit_idx >= h.msg_bits) {
          break;
        }

        const uint8_t mask = static_cast<uint8_t>(1u << k);
        if ((input & mask) != (output & mask)) {
          result.bit_errors++;
          block_error = true;
        }
      }
    }

    if (block_error) {
      result.block_errors++;
    }
  }

  return result;
}

static void WriteCsvHeader(std::ofstream& csv) {
  csv
      << "dataset_dir,"
      << "mcs_index,mcs_qm,mcs_code_rate_x1024,"
      << "snr_index,snr_db,noise_sigma,"
      << "num_codeblocks,llr_len,msg_bits,msg_bytes,"
      << "base_graph,zc,n_rows,max_iterations,early_termination,"
      << "effective_cb_code_rate,"
      << "warmup_repeats,repeats,"
      << "mean_us_per_cb,stddev_us_per_cb,min_us_per_cb,max_us_per_cb,"
      << "ber,bler,bit_errors,total_bits,block_errors,total_blocks\n";
}

static size_t RoundUpDivSize(size_t a, size_t b) {
  return (a + b - 1) / b;
}

static size_t SelectZSimdAvx512(size_t z) {
  constexpr size_t kSimdLenInt16 = 32;   // Is16vec32
  constexpr size_t kCacheInt16Alignment = 64;

  size_t z_simd;

  if (z < kCacheInt16Alignment) {
    z_simd = RoundUpDivSize(z, kSimdLenInt16) * kSimdLenInt16;

    if ((z_simd - z) < kSimdLenInt16) {
      z_simd += kSimdLenInt16;
    }
  } else {
    z_simd = RoundUpDivSize(z, kCacheInt16Alignment) * kCacheInt16Alignment;

    if ((z_simd - z) < kSimdLenInt16) {
      z_simd += kCacheInt16Alignment;
    }
  }

  return z_simd;
}

static void WriteCsvRow(std::ofstream& csv,
                        const std::filesystem::path& dataset_dir,
                        const DatasetHeader& h,
                        size_t num_codeblocks,
                        uint32_t max_iterations,
                        uint32_t early_termination,
                        const std::vector<double>& us_per_cb,
                        const VerifyResult& verify) {
  const double mean = Mean(us_per_cb);
  const double stddev = StddevSample(us_per_cb);
  const auto [min_it, max_it] = std::minmax_element(us_per_cb.begin(), us_per_cb.end());

  csv
      << dataset_dir.string() << ","
      << h.mcs_index << ","
      << h.mcs_qm << ","
      << h.mcs_code_rate_x1024 << ","
      << h.snr_index << ","
      << std::setprecision(12) << h.snr_db << ","
      << std::setprecision(12) << h.noise_sigma << ","
      << num_codeblocks << ","
      << h.llr_len << ","
      << h.msg_bits << ","
      << h.msg_bytes << ","
      << h.base_graph << ","
      << h.zc << ","
      << h.n_rows << ","
      << max_iterations << ","
      << early_termination << ","
      << std::setprecision(12) << h.effective_cb_code_rate << ","
      << FLAGS_warmup_repeats << ","
      << FLAGS_repeats << ","
      << std::setprecision(12) << mean << ","
      << std::setprecision(12) << stddev << ","
      << std::setprecision(12) << (us_per_cb.empty() ? 0.0 : *min_it) << ","
      << std::setprecision(12) << (us_per_cb.empty() ? 0.0 : *max_it) << ","
      << std::setprecision(12) << verify.ber() << ","
      << std::setprecision(12) << verify.bler() << ","
      << verify.bit_errors << ","
      << verify.total_bits << ","
      << verify.block_errors << ","
      << verify.total_blocks
      << "\n";
}

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  const std::filesystem::path root(FLAGS_input_root);
  std::vector<std::filesystem::path> datasets = FindDatasets(root);

  if (datasets.empty()) {
    throw std::runtime_error("No datasets found under: " + FLAGS_input_root +
                             " . A valid dataset directory must contain "
                             "llrs_i8.bin, ref_msg_u8.bin, and metadata.json or header.bin.");
  }

  std::ofstream csv(FLAGS_output_csv);
  CheckFileOpen(csv, FLAGS_output_csv);
  WriteCsvHeader(csv);

  const double freq_ghz = GetTime::MeasureRdtscFreq();

  std::cout << "Found " << datasets.size() << " dataset(s)\n";

  for (const auto& dataset_dir : datasets) {
    DatasetHeader h = ReadDatasetDescription(dataset_dir);

    size_t num_codeblocks = static_cast<size_t>(h.num_codeblocks);
    if (FLAGS_limit_codeblocks > 0) {
      num_codeblocks =
          std::min<size_t>(num_codeblocks, static_cast<size_t>(FLAGS_limit_codeblocks));
    }

    uint32_t max_iterations = h.max_iterations;
    if (FLAGS_max_iterations_override >= 0) {
      max_iterations = static_cast<uint32_t>(FLAGS_max_iterations_override);
    }

    uint32_t early_termination = h.early_termination;
    if (FLAGS_early_termination_override >= 0) {
      early_termination = static_cast<uint32_t>(FLAGS_early_termination_override);
    }

    std::cout << "\n============================================================\n";
    std::cout << "Dataset: " << dataset_dir << "\n";
    std::cout << "  MCS=" << h.mcs_index
              << " Qm=" << h.mcs_qm
              << " R_x1024=" << h.mcs_code_rate_x1024
              << " SNR=" << h.snr_db << " dB"
              << " sigma=" << h.noise_sigma << "\n";
    std::cout << "  CBs=" << num_codeblocks
              << " llr_len=" << h.llr_len
              << " msg_bits=" << h.msg_bits
              << " BG=" << h.base_graph
              << " Zc=" << h.zc
              << " nRows=" << h.n_rows
              << " effectiveRate=" << h.effective_cb_code_rate
              << " maxIter=" << max_iterations
              << " earlyTerm=" << early_termination << "\n";

    Table<int8_t> llrs;
    Table<uint8_t> decoded_codewords;
    LoadLlrs(dataset_dir, h, num_codeblocks, llrs);
    decoded_codewords.Calloc(num_codeblocks, h.msg_bytes,
                             Agora_memory::Alignment_t::kAlign64);

    bblib_ldpc_decoder_5gnr_request request {};
    bblib_ldpc_decoder_5gnr_response response {};

    request.numChannelLlrs = h.llr_len;
    request.numFillerBits = h.num_filler_bits;
    request.maxIterations = max_iterations;
    request.enableEarlyTermination = early_termination;
    request.Zc = h.zc;
    request.baseGraph = h.base_graph;
    request.nRows = h.n_rows;

    response.numMsgBits = h.msg_bits;

    const size_t n_systematic_cols = (h.base_graph == 1) ? 22 : 10;
    const size_t n_cols = n_systematic_cols + h.n_rows;
    const size_t z_simd = SelectZSimdAvx512(h.zc);
    const size_t response_varnodes_len = (n_cols - 1) * h.zc + z_simd + 64;

    auto* resp_var_nodes = static_cast<int16_t*>(
        Agora_memory::PaddedAlignedAlloc(Agora_memory::Alignment_t::kAlign64,
                                         response_varnodes_len * sizeof(int16_t)));

    std::memset(resp_var_nodes, 0, response_varnodes_len * sizeof(int16_t));
    response.varNodes = resp_var_nodes;

    for (size_t r = 0; r < FLAGS_warmup_repeats; r++) {
      DecodeAll(h, num_codeblocks, llrs, decoded_codewords, request, response);
    }

    std::vector<double> us_per_cb;
    us_per_cb.reserve(static_cast<size_t>(FLAGS_repeats));

    for (size_t r = 0; r < FLAGS_repeats; r++) {
      const size_t start_tsc = GetTime::WorkerRdtsc();
      DecodeAll(h, num_codeblocks, llrs, decoded_codewords, request, response);
      const size_t duration = GetTime::WorkerRdtsc() - start_tsc;

      const double total_us = GetTime::CyclesToUs(duration, freq_ghz);
      us_per_cb.push_back(total_us / static_cast<double>(num_codeblocks));
    }

    VerifyResult verify;
    if (FLAGS_verify) {
      verify = VerifyDecoded(dataset_dir, h, num_codeblocks, decoded_codewords);
    }

    const double mean = Mean(us_per_cb);
    const double stddev = StddevSample(us_per_cb);

    std::cout << "  mean latency = " << mean << " us/CB"
              << "  stddev = " << stddev << " us/CB";
    if (FLAGS_verify) {
      std::cout << "  BER=" << verify.ber()
                << "  BLER=" << verify.bler()
                << "  bit_errors=" << verify.bit_errors << "/" << verify.total_bits;
    }
    std::cout << "\n";

    WriteCsvRow(csv, dataset_dir, h, num_codeblocks,
                max_iterations, early_termination, us_per_cb, verify);

    std::free(resp_var_nodes);
    llrs.Free();
    decoded_codewords.Free();
  }

  std::cout << "\nWrote CSV: " << FLAGS_output_csv << std::endl;
  return 0;
}
