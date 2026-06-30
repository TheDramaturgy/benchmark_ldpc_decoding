/**
 * @file export_llr_dataset_mcs_mother_rate.cc
 * @brief Export FlexRAN/Agora LDPC code-block LLR datasets using the
 *        selected MCS code rate as the LDPC mother-code rate.
 *
 * This is intentionally different from the NR-like exporter that generates a
 * full rate-1/3 mother code and then performs NR rate matching.  Here each MCS
 * code rate is written into the temporary Agora config, so Agora/FlexRAN chooses
 * an LDPC graph directly for that code rate.  Low-rate MCS values that would
 * require more LDPC rows than FlexRAN supports are skipped by default.
 *
 * Output layout:
 *   <export_root>/mcs_XX/snr_YY/
 *     metadata.json
 *     llrs_i8.bin
 *     ref_msg_u8.bin
 *     encoded_codewords_u8.bin
 *     flexran_decoded_u8.bin   optional
 *
 * The LLRs are exactly the int8 varNodes consumed by:
 *   bblib_ldpc_decoder_5gnr_request.varNodes
 * and have length:
 *   ldpc_config.NumCbCodewLen()
 */

#include <gflags/gflags.h>
#include <immintrin.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

#include "armadillo"
#include "comms-lib.h"
#include "config.h"
#include "data_generator.h"
#include "datatype_conversion.h"
#include "gettime.h"
#include "memory_manage.h"
#include "modulation.h"
#include "phy_ldpc_decoder_5gnr.h"
#include "utils_ldpc.h"

DEFINE_string(profile, "random",
              "Input user-byte profile: random or 123.");
DEFINE_string(conf_file,
              TOSTRING(PROJECT_DIRECTORY) "/files/config/ci/tddconfig-sim-ul.json",
              "Base Agora config file. A temporary copy is patched per MCS.");
DEFINE_string(export_root, "ldpc_llr_export_direct_mcs_rate",
              "Root directory for exported LDPC LLR datasets.");
DEFINE_string(mcs_list, "5-9,11-28",
              "Comma-separated MCS list/ranges, e.g. '5,6,7' or '5-9,11-28'. "
              "MCS 0-4 and 10 are skipped by default because direct BG1 LDPC "
              "would require more than 46 rows.");
DEFINE_int32(num_codeblocks, 1024,
             "Number of code blocks to export per MCS/SNR point.");
DEFINE_double(snr_min_db, 10.0, "Minimum SNR in dB.");
DEFINE_double(snr_max_db, 10.0, "Maximum SNR in dB.");
DEFINE_int32(num_snr, 1, "Number of SNR points.");
DEFINE_bool(skip_unsupported_mcs, true,
            "Skip MCS entries whose direct MCS code rate would require an "
            "unsupported LDPC graph. If false, throw.");
DEFINE_bool(export_flexran_decoded, true,
            "Also export FlexRAN decoded bytes for sanity checking.");
DEFINE_bool(patch_all_code_rate_fields, true,
            "Patch every JSON field named code_rate. This is usually wanted "
            "for the offline exporter because UL and DL config are both parsed.");
DEFINE_bool(patch_all_modulation_fields, true,
            "Patch every JSON field named modulation. This is usually wanted "
            "for the offline exporter because UL and DL config are both parsed.");
DEFINE_bool(allow_generated_rate_mismatch, true,
            "Continue if Agora rounds the generated LDPC effective code rate "
            "away from the exact MCS code rate. Metadata records both values.");

namespace {

struct McsEntry {
  int index;
  int qm;
  int r_x1024;
  const char* modulation;
};

static const std::map<int, McsEntry>& McsTableQam64() {
  static const std::map<int, McsEntry> table = {
      {0, {0, 2, 120, "QPSK"}},  {1, {1, 2, 157, "QPSK"}},
      {2, {2, 2, 193, "QPSK"}},  {3, {3, 2, 251, "QPSK"}},
      {4, {4, 2, 308, "QPSK"}},  {5, {5, 2, 379, "QPSK"}},
      {6, {6, 2, 449, "QPSK"}},  {7, {7, 2, 526, "QPSK"}},
      {8, {8, 2, 602, "QPSK"}},  {9, {9, 2, 679, "QPSK"}},
      {10, {10, 4, 340, "16QAM"}}, {11, {11, 4, 378, "16QAM"}},
      {12, {12, 4, 434, "16QAM"}}, {13, {13, 4, 490, "16QAM"}},
      {14, {14, 4, 553, "16QAM"}}, {15, {15, 4, 616, "16QAM"}},
      {16, {16, 4, 658, "16QAM"}}, {17, {17, 6, 438, "64QAM"}},
      {18, {18, 6, 466, "64QAM"}}, {19, {19, 6, 517, "64QAM"}},
      {20, {20, 6, 567, "64QAM"}}, {21, {21, 6, 616, "64QAM"}},
      {22, {22, 6, 666, "64QAM"}}, {23, {23, 6, 719, "64QAM"}},
      {24, {24, 6, 772, "64QAM"}}, {25, {25, 6, 822, "64QAM"}},
      {26, {26, 6, 873, "64QAM"}}, {27, {27, 6, 910, "64QAM"}},
      {28, {28, 6, 948, "64QAM"}},
  };
  return table;
}

static double McsRate(const McsEntry& mcs) {
  return static_cast<double>(mcs.r_x1024) / 1024.0;
}

static bool DirectBg1McsRateSupported(const McsEntry& mcs) {
  // BG1 with full available FlexRAN rows has minimum direct rate 22/(20+46)=1/3.
  // MCS 10 has 340/1024 = 0.33203125, so it is still below 1/3.
  return McsRate(mcs) >= (1.0 / 3.0);
}

static std::vector<int> ParseMcsList(const std::string& spec) {
  std::set<int> out;
  std::stringstream ss(spec);
  std::string token;
  while (std::getline(ss, token, ',')) {
    token.erase(remove_if(token.begin(), token.end(), ::isspace), token.end());
    if (token.empty()) continue;
    const size_t dash = token.find('-');
    if (dash == std::string::npos) {
      out.insert(std::stoi(token));
    } else {
      const int a = std::stoi(token.substr(0, dash));
      const int b = std::stoi(token.substr(dash + 1));
      if (a <= b) {
        for (int v = a; v <= b; v++) out.insert(v);
      } else {
        for (int v = a; v >= b; v--) out.insert(v);
      }
    }
  }
  return std::vector<int>(out.begin(), out.end());
}

static std::vector<double> BuildSnrList() {
  if (FLAGS_num_snr <= 0) {
    throw std::runtime_error("--num_snr must be positive");
  }
  std::vector<double> snrs;
  if (FLAGS_num_snr == 1) {
    snrs.push_back(FLAGS_snr_min_db);
    return snrs;
  }
  for (int i = 0; i < FLAGS_num_snr; i++) {
    const double t = static_cast<double>(i) / static_cast<double>(FLAGS_num_snr - 1);
    snrs.push_back(FLAGS_snr_min_db + t * (FLAGS_snr_max_db - FLAGS_snr_min_db));
  }
  return snrs;
}

static double SnrDbToSigma(double snr_db) {
  return std::pow(10.0, -snr_db / 20.0);
}

static std::string ReadTextFile(const std::string& path) {
  std::ifstream in(path);
  if (!in.is_open()) throw std::runtime_error("Could not open file: " + path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

static void WriteTextFile(const std::string& path, const std::string& text) {
  std::ofstream out(path);
  if (!out.is_open()) throw std::runtime_error("Could not write file: " + path);
  out << text;
}

static void MakeDirIfNeeded(const std::string& path) {
  if (path.empty()) return;
  if (mkdir(path.c_str(), 0755) != 0 && errno != EEXIST) {
    throw std::runtime_error("Failed to create directory: " + path);
  }
}

static void MakeDirRecursive(const std::string& path) {
  if (path.empty()) return;
  std::string cur;
  for (char ch : path) {
    cur.push_back(ch);
    if (ch == '/') {
      if (cur.size() > 1) MakeDirIfNeeded(cur.substr(0, cur.size() - 1));
    }
  }
  MakeDirIfNeeded(path);
}

static void CheckFileOpen(const std::ofstream& file, const std::string& path) {
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file for writing: " + path);
  }
}

static std::string MpcsDirName(const std::string& root, int mcs) {
  std::ostringstream oss;
  oss << root << "/mcs_" << std::setw(2) << std::setfill('0') << mcs;
  return oss.str();
}

static std::string RunDirName(const std::string& root, int mcs, size_t snr_idx) {
  std::ostringstream oss;
  oss << MpcsDirName(root, mcs) << "/snr_" << std::setw(2)
      << std::setfill('0') << snr_idx;
  return oss.str();
}

static std::string PatchJsonNumberFieldAll(std::string json,
                                           const std::string& field,
                                           const std::string& value) {
  const std::regex r("\\\"" + field + "\\\"\\s*:\\s*[-+]?[0-9]+(?:\\.[0-9]+)?(?:[eE][-+]?[0-9]+)?");
  const std::string repl = "\"" + field + "\": " + value;
  return std::regex_replace(json, r, repl);
}

static std::string PatchJsonStringFieldAll(std::string json,
                                           const std::string& field,
                                           const std::string& value) {
  const std::regex r("\\\"" + field + "\\\"\\s*:\\s*\\\"[^\\\"]*\\\"");
  const std::string repl = "\"" + field + "\": \"" + value + "\"";
  return std::regex_replace(json, r, repl);
}

static std::string BuildMcsConfig(const std::string& base_config,
                                  const McsEntry& mcs) {
  std::string patched = base_config;
  std::ostringstream rate;
  rate << std::setprecision(15) << McsRate(mcs);

  if (FLAGS_patch_all_code_rate_fields) {
    patched = PatchJsonNumberFieldAll(patched, "code_rate", rate.str());
  }
  if (FLAGS_patch_all_modulation_fields) {
    patched = PatchJsonStringFieldAll(patched, "modulation", mcs.modulation);
  }
  return patched;
}

static size_t RoundUpSize(size_t x, size_t align) {
  return ((x + align - 1) / align) * align;
}

static size_t BitsToBytesLocal(size_t bits) {
  return (bits + 7) / 8;
}

static double SafeDiv(double a, double b) {
  return b == 0.0 ? 0.0 : a / b;
}

static std::string AerialCodeRateForGraph(uint32_t base_graph,
                                          uint32_t zc,
                                          double fallback_rate) {
  // Aerial's Python API does not accept BG/Zc directly; it infers BG/Zc from
  // tb_size and code_rate.  For BG1 and small tb_size, a true low MCS rate can
  // make Aerial infer BG2, so metadata records a graph-forcing Aerial rate.
  if (base_graph == 1) return "0.68";
  std::ostringstream oss;
  oss << std::setprecision(8) << fallback_rate;
  return oss.str();
}

static uint32_t AerialTbSize(uint32_t base_graph, uint32_t zc,
                             uint32_t msg_bits) {
  if (base_graph == 1) return 22u * zc - 24u;
  if (base_graph == 2) return 10u * zc - 24u;
  return msg_bits >= 24 ? msg_bits - 24 : msg_bits;
}

static void WriteMetadata(const std::string& meta_path,
                          const McsEntry& mcs,
                          double snr_db,
                          double sigma,
                          Config& cfg,
                          Direction dir,
                          const LDPCconfig& ldpc_config,
                          size_t num_codeblocks,
                          size_t llr_len,
                          size_t msg_bits,
                          size_t msg_bytes,
                          size_t encoded_bytes,
                          size_t padded_mod_symbols,
                          size_t demod_valid_llrs) {
  std::ofstream meta(meta_path);
  CheckFileOpen(meta, meta_path);

  const uint32_t bg = static_cast<uint32_t>(ldpc_config.BaseGraph());
  const uint32_t zc = static_cast<uint32_t>(ldpc_config.ExpansionFactor());
  const uint32_t aerial_prepend = 2u * zc;
  const uint32_t aerial_input_len = static_cast<uint32_t>(llr_len + aerial_prepend);
  const double mcs_rate = McsRate(mcs);
  const double effective_rate = SafeDiv(static_cast<double>(msg_bits),
                                        static_cast<double>(llr_len));

  meta << "{\n";
  meta << "  \"format\": \"agora_flexran_ldpc_direct_mcs_rate_v1\",\n";
  meta << "  \"description\": \"FlexRAN LDPC code-block LLRs generated with MCS code rate used directly as the LDPC mother-code rate; no NR rate matching is applied.\",\n";
  meta << "  \"conf_file\": \"" << FLAGS_conf_file << "\",\n";
  meta << "  \"profile\": \"" << FLAGS_profile << "\",\n";

  meta << "  \"mcs_index\": " << mcs.index << ",\n";
  meta << "  \"mcs_qm\": " << mcs.qm << ",\n";
  meta << "  \"mcs_modulation\": \"" << mcs.modulation << "\",\n";
  meta << "  \"mcs_code_rate_x1024\": " << mcs.r_x1024 << ",\n";
  meta << "  \"mcs_code_rate\": " << std::setprecision(15) << mcs_rate << ",\n";
  meta << "  \"mother_code_rate_source\": \"mcs_code_rate\",\n";
  meta << "  \"mother_code_rate\": " << std::setprecision(15) << mcs_rate << ",\n";

  meta << "  \"snr_db\": " << std::setprecision(8) << snr_db << ",\n";
  meta << "  \"noise_sigma\": " << std::setprecision(15) << sigma << ",\n";

  meta << "  \"num_codeblocks\": " << num_codeblocks << ",\n";
  meta << "  \"llr_file\": \"llrs_i8.bin\",\n";
  meta << "  \"llr_dtype\": \"int8\",\n";
  meta << "  \"llr_shape\": [" << num_codeblocks << ", " << llr_len << "],\n";
  meta << "  \"llr_len\": " << llr_len << ",\n";
  meta << "  \"llr_layout\": \"flexran_varNodes_excluding_first_2Zc\",\n";
  meta << "  \"llr_sign\": \"same sign convention consumed by bblib_ldpc_decoder_5gnr_request.varNodes\",\n";

  meta << "  \"ref_file\": \"ref_msg_u8.bin\",\n";
  meta << "  \"ref_dtype\": \"uint8\",\n";
  meta << "  \"ref_shape\": [" << num_codeblocks << ", " << msg_bytes << "],\n";
  meta << "  \"msg_bits_per_cb\": " << msg_bits << ",\n";
  meta << "  \"msg_bytes_per_cb\": " << msg_bytes << ",\n";
  meta << "  \"bit_order_in_ref_bytes\": \"little\",\n";

  meta << "  \"encoded_file\": \"encoded_codewords_u8.bin\",\n";
  meta << "  \"encoded_dtype\": \"uint8\",\n";
  meta << "  \"encoded_shape\": [" << num_codeblocks << ", " << encoded_bytes << "],\n";
  meta << "  \"encoded_bytes_per_cb\": " << encoded_bytes << ",\n";

  meta << "  \"base_graph\": " << bg << ",\n";
  meta << "  \"zc\": " << zc << ",\n";
  meta << "  \"n_rows\": " << ldpc_config.NumRows() << ",\n";
  meta << "  \"num_filler_bits\": 0,\n";
  meta << "  \"max_iterations\": " << ldpc_config.MaxDecoderIter() << ",\n";
  meta << "  \"early_termination\": "
       << (ldpc_config.EarlyTermination() ? "true" : "false") << ",\n";

  meta << "  \"modulation\": \"" << cfg.Modulation(dir) << "\",\n";
  meta << "  \"mod_order_bits\": " << cfg.ModOrderBits(dir) << ",\n";
  meta << "  \"ofdm_data_num\": " << cfg.OfdmDataNum() << ",\n";
  meta << "  \"padded_mod_symbols\": " << padded_mod_symbols << ",\n";
  meta << "  \"demod_valid_llrs_per_cb\": " << demod_valid_llrs << ",\n";

  meta << "  \"rate_matching_enabled\": false,\n";
  meta << "  \"rate_match_len\": " << llr_len << ",\n";
  meta << "  \"rate_match_granularity_re\": 0,\n";
  meta << "  \"rv\": 0,\n";
  meta << "  \"export_stage_id\": 4,\n";
  meta << "  \"export_stage\": \"direct_mcs_rate_after_demod_before_ldpc_decode\",\n";
  meta << "  \"effective_cb_code_rate\": " << std::setprecision(15)
       << effective_rate << ",\n";
  meta << "  \"generated_rate_minus_mcs_rate\": " << std::setprecision(15)
       << (effective_rate - mcs_rate) << ",\n";

  meta << "  \"aerial_tb_size\": " << AerialTbSize(bg, zc, msg_bits) << ",\n";
  meta << "  \"aerial_code_rate\": " << AerialCodeRateForGraph(bg, zc, mcs_rate) << ",\n";
  meta << "  \"aerial_note\": \"aerial_code_rate is graph-selection metadata; mcs_code_rate/mother_code_rate is the true direct LDPC rate used in Agora.\",\n";
  meta << "  \"aerial_rate_match_len\": " << llr_len << ",\n";
  meta << "  \"aerial_input_len\": " << aerial_input_len << ",\n";
  meta << "  \"aerial_prepend_punctured_llrs\": " << aerial_prepend << "\n";

  meta << "}\n";
}

static std::vector<int8_t> MakeZeroPaddedInputBuffer(
    const std::vector<int8_t>& information,
    size_t msg_bytes,
    size_t input_size) {
  if (information.size() < msg_bytes) {
    throw std::runtime_error("information block is smaller than msg_bytes");
  }
  std::vector<int8_t> input(input_size, 0);
  std::memcpy(input.data(), information.data(), msg_bytes);
  return input;
}

static void ExportOneRun(const std::string& run_dir,
                         Config& cfg,
                         Direction dir,
                         const McsEntry& mcs,
                         double snr_db,
                         double sigma,
                         DataGenerator::Profile profile,
                         std::default_random_engine& generator,
                         std::normal_distribution<double>& distribution) {
  MakeDirRecursive(run_dir);

  const LDPCconfig& ldpc_config = cfg.LdpcConfig(dir);
  const size_t num_codeblocks = static_cast<size_t>(FLAGS_num_codeblocks);
  const size_t llr_len = ldpc_config.NumCbCodewLen();
  const size_t msg_bits = ldpc_config.NumCbLen();
  const size_t msg_bytes = BitsToBytesLocal(msg_bits);
  const size_t encoded_bytes = ldpc_config.NumEncodedBytes();
  const size_t qm = cfg.ModOrderBits(dir);
  const size_t num_mod_symbols = (llr_len + qm - 1) / qm;
  const size_t padded_mod_symbols = RoundUpSize(num_mod_symbols, 32);
  const size_t demod_valid_llrs = padded_mod_symbols * qm;

  if (qm != static_cast<size_t>(mcs.qm)) {
    std::ostringstream oss;
    oss << "Agora Config modulation order Qm=" << qm
        << " differs from MCS Qm=" << mcs.qm
        << ". Check JSON modulation patching.";
    throw std::runtime_error(oss.str());
  }

  const double effective_rate = SafeDiv(static_cast<double>(msg_bits),
                                        static_cast<double>(llr_len));
  const double mcs_rate = McsRate(mcs);
  if (std::fabs(effective_rate - mcs_rate) > 0.05) {
    std::ostringstream oss;
    oss << "Generated LDPC effective rate " << effective_rate
        << " differs substantially from requested MCS rate " << mcs_rate
        << " for MCS " << mcs.index
        << ". BG=" << ldpc_config.BaseGraph()
        << " Zc=" << ldpc_config.ExpansionFactor()
        << " nRows=" << ldpc_config.NumRows();
    if (!FLAGS_allow_generated_rate_mismatch) throw std::runtime_error(oss.str());
    std::cerr << "WARNING: " << oss.str() << "\n";
  }

  std::cout << "\n============================================================\n";
  std::cout << "MCS " << mcs.index
            << "  Qm=" << mcs.qm
            << "  target R=" << mcs_rate
            << "  Config modulation=" << cfg.Modulation(dir)
            << "  Config Qm=" << qm
            << "  BG=" << ldpc_config.BaseGraph()
            << "  Zc=" << ldpc_config.ExpansionFactor()
            << "  K=" << msg_bits
            << "  N=" << llr_len
            << "  nRows=" << ldpc_config.NumRows()
            << "  effective R=" << effective_rate << "\n";
  std::cout << "SNR " << snr_db << " dB  sigma=" << sigma << "\n";
  std::cout << "Export directory: " << run_dir << "\n";
  std::cout << "============================================================\n";

  DataGenerator data_generator(&cfg, 0 /* RNG seed */, profile);
  const size_t input_size = Roundup<64>(LdpcEncodingInputBufSize(
      ldpc_config.BaseGraph(), ldpc_config.ExpansionFactor()));

  std::vector<std::vector<int8_t>> information(num_codeblocks);
  std::vector<std::vector<int8_t>> encoded_codewords(num_codeblocks);

  for (size_t cb = 0; cb < num_codeblocks; cb++) {
    data_generator.GenRawData(dir, information.at(cb), cb % cfg.UeAntNum());
    auto input = MakeZeroPaddedInputBuffer(information.at(cb), msg_bytes, input_size);
    data_generator.GenCodeblock(dir, input.data(), encoded_codewords.at(cb));
  }

  Table<complex_float> modulated_codewords;
  modulated_codewords.Calloc(num_codeblocks, padded_mod_symbols,
                             Agora_memory::Alignment_t::kAlign64);
  Table<int8_t> demod_data_all_symbols;
  demod_data_all_symbols.Calloc(num_codeblocks, demod_valid_llrs,
                                Agora_memory::Alignment_t::kAlign64);

  std::vector<uint8_t> mod_input(padded_mod_symbols, 0);

  for (size_t cb = 0; cb < num_codeblocks; cb++) {
    std::fill(mod_input.begin(), mod_input.end(), 0);
    if (encoded_codewords.at(cb).size() < encoded_bytes) {
      throw std::runtime_error("encoded_codewords[cb] is smaller than encoded_bytes");
    }

    AdaptBitsForMod(reinterpret_cast<const uint8_t*>(encoded_codewords[cb].data()),
                    mod_input.data(), encoded_bytes, qm);

    for (size_t j = 0; j < padded_mod_symbols; j++) {
      modulated_codewords[cb][j] = ModSingleUint8(mod_input[j], cfg.ModTable(dir));
    }

    for (size_t j = 0; j < padded_mod_symbols; j++) {
      complex_float noise = {
          static_cast<float>(distribution(generator) * sigma),
          static_cast<float>(distribution(generator) * sigma)};
      modulated_codewords[cb][j].re += noise.re;
      modulated_codewords[cb][j].im += noise.im;
    }

    switch (qm) {
      case 2:
        DemodQpskSoftAvx2(reinterpret_cast<float*>(modulated_codewords[cb]),
                          demod_data_all_symbols[cb], padded_mod_symbols);
        break;
      case 4:
        Demod16qamSoftAvx2(reinterpret_cast<float*>(modulated_codewords[cb]),
                           demod_data_all_symbols[cb], padded_mod_symbols);
        break;
      case 6:
        Demod64qamSoftAvx2(reinterpret_cast<float*>(modulated_codewords[cb]),
                           demod_data_all_symbols[cb], padded_mod_symbols);
        break;
      case 8:
        Demod256qamSoftAvx2(reinterpret_cast<float*>(modulated_codewords[cb]),
                            demod_data_all_symbols[cb], padded_mod_symbols);
        break;
      default:
        throw std::runtime_error("Unsupported modulation order");
    }
  }

  struct bblib_ldpc_decoder_5gnr_request dec_req {};
  struct bblib_ldpc_decoder_5gnr_response dec_resp {};
  dec_req.numChannelLlrs = llr_len;
  dec_req.numFillerBits = 0;
  dec_req.maxIterations = ldpc_config.MaxDecoderIter();
  dec_req.enableEarlyTermination = ldpc_config.EarlyTermination();
  dec_req.Zc = ldpc_config.ExpansionFactor();
  dec_req.baseGraph = ldpc_config.BaseGraph();
  dec_req.nRows = ldpc_config.NumRows();
  dec_resp.numMsgBits = msg_bits;

  auto* resp_var_nodes = static_cast<int16_t*>(
      Agora_memory::PaddedAlignedAlloc(Agora_memory::Alignment_t::kAlign64,
                                       1024 * 1024 * sizeof(int16_t)));
  dec_resp.varNodes = resp_var_nodes;

  Table<uint8_t> decoded_codewords;
  decoded_codewords.Calloc(num_codeblocks, msg_bytes,
                           Agora_memory::Alignment_t::kAlign64);

  const std::string llr_path = run_dir + "/llrs_i8.bin";
  const std::string ref_path = run_dir + "/ref_msg_u8.bin";
  const std::string enc_path = run_dir + "/encoded_codewords_u8.bin";
  const std::string meta_path = run_dir + "/metadata.json";

  {
    std::ofstream f(llr_path, std::ios::binary);
    CheckFileOpen(f, llr_path);
    for (size_t cb = 0; cb < num_codeblocks; cb++) {
      f.write(reinterpret_cast<const char*>(demod_data_all_symbols[cb]),
              llr_len * sizeof(int8_t));
    }
  }

  {
    std::ofstream f(ref_path, std::ios::binary);
    CheckFileOpen(f, ref_path);
    for (size_t cb = 0; cb < num_codeblocks; cb++) {
      f.write(reinterpret_cast<const char*>(information[cb].data()),
              msg_bytes * sizeof(uint8_t));
    }
  }

  {
    std::ofstream f(enc_path, std::ios::binary);
    CheckFileOpen(f, enc_path);
    for (size_t cb = 0; cb < num_codeblocks; cb++) {
      f.write(reinterpret_cast<const char*>(encoded_codewords[cb].data()),
              encoded_bytes * sizeof(uint8_t));
    }
  }

  WriteMetadata(meta_path, mcs, snr_db, sigma, cfg, dir, ldpc_config,
                num_codeblocks, llr_len, msg_bits, msg_bytes, encoded_bytes,
                padded_mod_symbols, demod_valid_llrs);

  double freq_ghz = GetTime::MeasureRdtscFreq();
  size_t start_tsc = GetTime::WorkerRdtsc();
  for (size_t cb = 0; cb < num_codeblocks; cb++) {
    dec_req.varNodes = demod_data_all_symbols[cb];
    dec_resp.compactedMessageBytes = decoded_codewords[cb];
    bblib_ldpc_decoder_5gnr(&dec_req, &dec_resp);
  }
  size_t duration = GetTime::WorkerRdtsc() - start_tsc;
  std::printf("Decoding of %zu blocks takes %.2f us per block\n",
              num_codeblocks,
              GetTime::CyclesToUs(duration, freq_ghz) / num_codeblocks);

  if (FLAGS_export_flexran_decoded) {
    const std::string dec_path = run_dir + "/flexran_decoded_u8.bin";
    std::ofstream f(dec_path, std::ios::binary);
    CheckFileOpen(f, dec_path);
    for (size_t cb = 0; cb < num_codeblocks; cb++) {
      f.write(reinterpret_cast<const char*>(decoded_codewords[cb]),
              msg_bytes * sizeof(uint8_t));
    }
  }

  size_t bit_errors = 0;
  size_t block_errors = 0;
  for (size_t cb = 0; cb < num_codeblocks; cb++) {
    size_t block_bit_errors = 0;
    for (size_t byte = 0; byte < msg_bytes; byte++) {
      const uint8_t in = static_cast<uint8_t>(information[cb][byte]);
      const uint8_t out = decoded_codewords[cb][byte];
      uint8_t diff = in ^ out;
      if (byte + 1 == msg_bytes && (msg_bits % 8) != 0) {
        diff &= static_cast<uint8_t>((1u << (msg_bits % 8)) - 1u);
      }
      for (int k = 0; k < 8; k++) {
        if (diff & (1u << k)) {
          bit_errors++;
          block_bit_errors++;
        }
      }
    }
    if (block_bit_errors > 0) block_errors++;
  }

  const size_t total_bits = num_codeblocks * msg_bits;
  std::printf("BER: %zu/%zu = %.6e, BLER: %zu/%zu = %.6e\n",
              bit_errors, total_bits,
              static_cast<double>(bit_errors) / static_cast<double>(total_bits),
              block_errors, num_codeblocks,
              static_cast<double>(block_errors) / static_cast<double>(num_codeblocks));

  std::cout << "Exported direct-MCS-rate LDPC LLR dataset to: "
            << run_dir << std::endl;

  std::free(resp_var_nodes);
  modulated_codewords.Free();
  demod_data_all_symbols.Free();
  decoded_codewords.Free();
}

}  // namespace

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_num_codeblocks <= 0) {
    throw std::runtime_error("--num_codeblocks must be positive");
  }

  std::default_random_engine generator(
      static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count()));
  std::normal_distribution<double> distribution(0.0, 1.0);

  const DataGenerator::Profile profile =
      FLAGS_profile == "123" ? DataGenerator::Profile::kProfile123
                              : DataGenerator::Profile::kRandom;

  const auto& table = McsTableQam64();
  const std::vector<int> mcs_values = ParseMcsList(FLAGS_mcs_list);
  const std::vector<double> snrs = BuildSnrList();

  MakeDirRecursive(FLAGS_export_root);
  MakeDirRecursive(FLAGS_export_root + "/generated_configs");

  const std::string base_config = ReadTextFile(FLAGS_conf_file);

  for (int mcs_idx : mcs_values) {
    const auto it = table.find(mcs_idx);
    if (it == table.end()) {
      std::ostringstream oss;
      oss << "Unsupported/unknown MCS index " << mcs_idx
          << ". This exporter defines normal table entries 0..28.";
      if (FLAGS_skip_unsupported_mcs) {
        std::cerr << "WARNING: " << oss.str() << " Skipping.\n";
        continue;
      }
      throw std::runtime_error(oss.str());
    }

    const McsEntry& mcs = it->second;
    if (!DirectBg1McsRateSupported(mcs)) {
      std::ostringstream oss;
      oss << "MCS " << mcs.index << " has code rate " << McsRate(mcs)
          << " (" << mcs.r_x1024 << "/1024), below the direct BG1 LDPC "
          << "minimum of 1/3. It would require more than 46 BG1 rows. "
          << "Use the rate-matched exporter for this MCS instead.";
      if (FLAGS_skip_unsupported_mcs) {
        std::cerr << "WARNING: " << oss.str() << " Skipping.\n";
        continue;
      }
      throw std::runtime_error(oss.str());
    }

    const std::string patched = BuildMcsConfig(base_config, mcs);
    std::ostringstream cfg_path;
    cfg_path << FLAGS_export_root << "/generated_configs/mcs_"
             << std::setw(2) << std::setfill('0') << mcs.index
             << "_direct_rate.json";
    WriteTextFile(cfg_path.str(), patched);

    auto cfg = std::make_unique<Config>(cfg_path.str().c_str());
    Direction dir = cfg->Frame().NumULSyms() > 0 ? Direction::kUplink
                                                 : Direction::kDownlink;

    for (size_t snr_idx = 0; snr_idx < snrs.size(); snr_idx++) {
      const double snr_db = snrs[snr_idx];
      const double sigma = SnrDbToSigma(snr_db);
      const std::string run_dir = RunDirName(FLAGS_export_root, mcs.index, snr_idx);
      ExportOneRun(run_dir, *cfg, dir, mcs, snr_db, sigma,
                   profile, generator, distribution);
    }
  }

  return 0;
}
