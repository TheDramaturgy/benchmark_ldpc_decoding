#!/usr/bin/env bash
set -euo pipefail

# Benchmark every run directory in a generated LDPC dataset.
#
# Expected layouts supported:
#   ldpc_mcs_mother_rate/mcs_12/snr_00/metadata.json
#   ldpc_size_sweep/bg1_zc256/mcs_12/snr_00/metadata.json
#   any other tree where each run directory contains metadata.json
#
# Usage:
#   bash bench_final.sh [DATASET_ROOT] [OUTPUT_DIR]
#
# Example:
#   bash bench_final.sh ldpc_final_tesc cpu_results
#
# The script writes one CSV per run directory and also creates a combined CSV.

DATASET_ROOT="${1:-ldpc_final_tesc}"
OUT_DIR="${2:-cpu_results}"

BENCH_BIN="${BENCH_BIN:-./build/benchmark_ldpc}"
WARMUP_REPEATS="${WARMUP_REPEATS:-50}"
REPEATS="${REPEATS:-1000}"
VERIFY="${VERIFY:-true}"
EARLY_TERM_OVERRIDE="${EARLY_TERM_OVERRIDE:-0}"

# Optional. Leave empty to use the benchmark/default metadata value.
# Example:
#   MAX_ITER_OVERRIDE=5 bash bench_ldpc_cpu_dataset.sh ldpc_mcs_mother_rate cpu_results
MAX_ITER_OVERRIDE="${MAX_ITER_OVERRIDE:-}"

if [[ ! -x "$BENCH_BIN" ]]; then
  echo "ERROR: benchmark binary not found or not executable: $BENCH_BIN" >&2
  exit 1
fi

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "ERROR: dataset root does not exist: $DATASET_ROOT" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

mapfile -t RUN_DIRS < <(find "$DATASET_ROOT" -type f -name metadata.json -printf '%h\n' | sort)

if [[ "${#RUN_DIRS[@]}" -eq 0 ]]; then
  echo "ERROR: no metadata.json files found under: $DATASET_ROOT" >&2
  exit 1
fi

echo "Found ${#RUN_DIRS[@]} run directory/directories under $DATASET_ROOT"
echo "Writing per-run CSVs to: $OUT_DIR"
echo

COMBINED_CSV="$OUT_DIR/combined_cpu.csv"
rm -f "$COMBINED_CSV"

first_csv=1

for RUN_DIR in "${RUN_DIRS[@]}"; do
  REL="${RUN_DIR#"$DATASET_ROOT"/}"

  # Make a safe filename from the relative path.
  SAFE_NAME="$(echo "$REL" | sed 's#[/ ]#_#g')"
  OUT_CSV="$OUT_DIR/${SAFE_NAME}_cpu.csv"

  echo "================================================================================"
  echo "Run directory : $RUN_DIR"
  echo "Output CSV    : $OUT_CSV"
  echo "================================================================================"

  CMD=(
    "$BENCH_BIN"
    "--input_root=$RUN_DIR"
    "--output_csv=$OUT_CSV"
    "--warmup_repeats=$WARMUP_REPEATS"
    "--repeats=$REPEATS"
    "--verify=$VERIFY"
    "--early-termination-override=$EARLY_TERM_OVERRIDE"
  )

  if [[ -n "$MAX_ITER_OVERRIDE" ]]; then
    CMD+=("--max-iterations-override=$MAX_ITER_OVERRIDE")
  fi

  "${CMD[@]}"

  # Combine CSVs, preserving only the first header.
  if [[ -f "$OUT_CSV" ]]; then
    if [[ "$first_csv" -eq 1 ]]; then
      cat "$OUT_CSV" > "$COMBINED_CSV"
      first_csv=0
    else
      tail -n +2 "$OUT_CSV" >> "$COMBINED_CSV"
    fi
  else
    echo "WARNING: expected CSV was not created: $OUT_CSV" >&2
  fi

  echo
done

echo "Done."
echo "Combined CSV: $COMBINED_CSV"