#!/usr/bin/env bash
set -euo pipefail

RECOVERY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${RECOVERY_ROOT}/.." && pwd)"

# =========================
# Config
# =========================
THREADS=40
POOLS=(1)
COVERAGES=(13.8)

DATASET_DIR="ONT_FAST"
BASECALL_MODE="FAST"

# DATASET_DIR="ONT_HAC"
# BASECALL_MODE="HAC"

# DATASET_DIR="ONT_SUP"
# BASECALL_MODE="SUP"

# DATASET_DIR="NGS"
# BASECALL_MODE="NGS"

BASE_COV=29988

IN_ROOT="${PROJECT_ROOT}/sequencing_data/${DATASET_DIR}"
RES_ROOT_BASE="${PROJECT_ROOT}/results"
DEEPRESYNC_ROOT="${PROJECT_ROOT}"

MODEL_QUEUE=(
  "deepresync|training/runs/checkpoints"
)

DEEPRESYNC_BATCH_SIZE="${DEEPRESYNC_BATCH_SIZE:-512}"

if [[ -z "${INDEX_THRESHOLD:-}" ]]; then
  case "$BASECALL_MODE" in
    FAST) INDEX_THRESHOLD=2 ;;
    SUP | HAC | NGS) INDEX_THRESHOLD=3 ;;
  esac
fi

if [[ -z "${VALID_BIAS_SIZE:-}" ]]; then
  case "$BASECALL_MODE" in
    FAST) VALID_BIAS_SIZE=20 ;;
    SUP | HAC | NGS) VALID_BIAS_SIZE=8 ;;
  esac
fi

if [[ -z "${PRIMER_MIN_EDIT:-}" ]]; then
  case "$BASECALL_MODE" in
    FAST) PRIMER_MIN_EDIT=3 ;;
    SUP | HAC | NGS) PRIMER_MIN_EDIT=2 ;;
  esac
fi

mkdir -p "$RES_ROOT_BASE"

[[ -d "$IN_ROOT" ]] || { echo "[ERROR] missing input root: $IN_ROOT"; exit 1; }
[[ -d "$DEEPRESYNC_ROOT" ]] || { echo "[ERROR] missing DEEPRESYNC_ROOT: $DEEPRESYNC_ROOT"; exit 1; }
[[ -d "${DEEPRESYNC_ROOT}/training" ]] || { echo "[ERROR] missing training package: ${DEEPRESYNC_ROOT}/training"; exit 1; }
[[ -f "${RECOVERY_ROOT}/bin/.current_layout_built" ]] || {
  echo "[ERROR] recovery binaries do not match the current directory layout." >&2
  echo "        Run: bash recovery/build.sh" >&2
  exit 1
}

# =========================
# Helper
# =========================
log_step() {
  local step="$1"
  local msg="$2"
  echo 
  echo "++++++++++++++++++++++++++++++++++++++++++++++++"
  echo "[Step ${step}] ${msg}"
  echo "++++++++++++++++++++++++++++++++++++++++++++++++"
}

# =========================
# Main: model -> coverage -> pool
# =========================
for model_entry in "${MODEL_QUEUE[@]}"; do
  IFS="|" read -r MODEL_TAG DEEPRESYNC_RUN_REL <<< "$model_entry"
  DEEPRESYNC_RUN_DIR="${DEEPRESYNC_ROOT}/${DEEPRESYNC_RUN_REL}"
  DEEPRESYNC_CHECKPOINT="${DEEPRESYNC_RUN_DIR}/best.pt"
  RES_ROOT="${RES_ROOT_BASE}/${MODEL_TAG}"

  if [[ ! -f "$DEEPRESYNC_CHECKPOINT" ]]; then
    echo "[ERROR] missing checkpoint for ${MODEL_TAG}: ${DEEPRESYNC_CHECKPOINT}"
    exit 1
  fi

  export DEEPRESYNC_ROOT DEEPRESYNC_CHECKPOINT DEEPRESYNC_BATCH_SIZE
  mkdir -p "$RES_ROOT"


  for coverage in "${COVERAGES[@]}"; do
    num_reads=$(awk -v c="$coverage" -v b="$BASE_COV" 'BEGIN{print int(c*b)}')
    num_fastq_lines=$(( num_reads * 4 ))

    for pool in "${POOLS[@]}"; do

      source_fastq="${IN_ROOT}/UEP_Pool_${pool}_${BASECALL_MODE}.fastq"
      if [[ ! -f "$source_fastq" ]]; then
        echo "[WARN] missing input: $source_fastq"
        continue
      fi

      coverage_results_dir="${RES_ROOT}/pool${pool}/${coverage}x"
      mkdir -p "$coverage_results_dir"

      primer_results_dir="${coverage_results_dir}/primer_results"
      index_results_dir="${coverage_results_dir}/index_results"
      correction_results_dir="${coverage_results_dir}/correction_results"
      consensus_results_dir="${coverage_results_dir}/consensus_results"
      decoding_results_dir="${coverage_results_dir}/decoding_results"

      mkdir -p \
        "$primer_results_dir" \
        "$index_results_dir" \
        "$correction_results_dir" \
        "$consensus_results_dir" \
        "$decoding_results_dir"

      sampled_reads_fastq="${coverage_results_dir}/sampled_reads.fastq"
      head -n "$num_fastq_lines" "$source_fastq" > "$sampled_reads_fastq" || true
      lines_written=$(wc -l < "$sampled_reads_fastq" || echo 0)

      if (( lines_written < num_fastq_lines )); then
        rm -f "$sampled_reads_fastq"
        echo "[ERROR] not enough reads for model=${MODEL_TAG} pool${pool} coverage=${coverage}x"
        exit 1
      fi

      echo "[RUN] model=${MODEL_TAG} pool${pool} coverage=${coverage}x reads=${num_reads}"

      log_step 0 "Primer identification"
      primer_reference_file="${RECOVERY_ROOT}/reference/primer/primer_pair_${pool}.txt"
      [[ -f "$primer_reference_file" ]] || { echo "     [ERROR] missing $primer_reference_file"; exit 1; }

      "${RECOVERY_ROOT}/scripts/run_primer_identification.sh" \
        "$sampled_reads_fastq" "$primer_results_dir" "$THREADS" \
        "$primer_reference_file" "$VALID_BIAS_SIZE" "$PRIMER_MIN_EDIT"

      primer_trimmed_reads="${primer_results_dir}/valid_reads.fa"
      [[ -f "$primer_trimmed_reads" ]] || { echo "     [ERROR] missing $primer_trimmed_reads"; exit 1; }

      log_step 1 "read-wise index identification"
      "${RECOVERY_ROOT}/scripts/run_index_identification.sh" \
        "$primer_trimmed_reads" "$index_results_dir" "$THREADS" "${PROJECT_ROOT}/configureFiles" "$INDEX_THRESHOLD"

      index_results_file="${index_results_dir}/index_identification_results.txt"
      [[ -f "$index_results_file" ]] || { echo "     [ERROR] missing $index_results_file"; exit 1; }

      log_step 2 "individual-read correction"
      "${RECOVERY_ROOT}/scripts/run_indel_correction.sh" \
        "$primer_trimmed_reads" "$index_results_file" "$correction_results_dir" "$pool"

      read_posteriors_file="${correction_results_dir}/read_posteriors.bin"
      [[ -f "$read_posteriors_file" ]] || { echo "     [ERROR] missing $read_posteriors_file"; exit 1; }

      log_step 3 "multiple-copy consensus"
      bash "${RECOVERY_ROOT}/scripts/run_multiple_copy_consensus.sh" \
        "$read_posteriors_file" "$consensus_results_dir" "$pool" "$THREADS"

      decoding_llr_file="${consensus_results_dir}/llrs_for_decoding.bin"
      [[ -f "$decoding_llr_file" ]] || { echo "     [ERROR] missing $decoding_llr_file"; exit 1; }

      "${RECOVERY_ROOT}/bin/analysis" "${RECOVERY_ROOT}/reference" "$consensus_results_dir" "$pool"

      log_step 4 "global error correction"
      "${RECOVERY_ROOT}/scripts/run_decoding.sh" \
        "$decoding_llr_file" "$decoding_results_dir" "$pool" "$THREADS"

      decoded_bits_file="${decoding_results_dir}/src_information.txt"
      [[ -f "$decoded_bits_file" ]] || { echo "     [ERROR] missing $decoded_bits_file"; exit 1; }

      "${RECOVERY_ROOT}/scripts/run_recover.sh" \
        "$decoded_bits_file" "$decoding_results_dir" "$pool"

      echo "[DONE] model=${MODEL_TAG} pool${pool} coverage=${coverage}x"
    done
  done
done
