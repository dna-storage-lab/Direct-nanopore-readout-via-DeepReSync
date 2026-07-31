#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =========================
# Usage
# =========================
if [[ $# -ne 4 ]]; then
  echo "Usage: $0 <reads> <index_results> <results_path> <pool>"
  exit 1
fi

reads=$1
index_results=$2
results_path=$3
pool=$4

# reference="$SCRIPT_DIR/../reference"
config="$SCRIPT_DIR/../../configureFiles"
reference="$SCRIPT_DIR/../reference"

mkdir -p "$results_path"

POSTERIOR_BIN="$results_path/read_posteriors.bin"

DEEPRESYNC_PYTHON="${DEEPRESYNC_PYTHON:-python}"
DEEPRESYNC_ROOT="${DEEPRESYNC_ROOT:-$SCRIPT_DIR/../..}"
DEEPRESYNC_CHECKPOINT="${DEEPRESYNC_CHECKPOINT:-$DEEPRESYNC_ROOT/training/runs/checkpoints/best.pt}"
DEEPRESYNC_DEVICE="${DEEPRESYNC_DEVICE:-cuda}"
DEEPRESYNC_BATCH_SIZE="${DEEPRESYNC_BATCH_SIZE:-512}"
DEEPRESYNC_MAX_READ_LEN="${DEEPRESYNC_MAX_READ_LEN:-0}"
DEEPRESYNC_AMP="${DEEPRESYNC_AMP:-fp16}"
DEEPRESYNC_TEMPERATURE="${DEEPRESYNC_TEMPERATURE:-1.0}"
[[ -f "$DEEPRESYNC_CHECKPOINT" ]] || { echo "[ERROR] missing checkpoint: $DEEPRESYNC_CHECKPOINT" >&2; exit 1; }
DEEPRESYNC_EXTRA_ARGS=(--temperature "$DEEPRESYNC_TEMPERATURE")
if [[ -n "${DEEPRESYNC_SYMBOL_TEMPERATURE:-}" ]]; then
    DEEPRESYNC_EXTRA_ARGS+=(--symbol-temperature "$DEEPRESYNC_SYMBOL_TEMPERATURE")
fi
if [[ -n "${DEEPRESYNC_LOWER_TEMPERATURE:-}" ]]; then
    DEEPRESYNC_EXTRA_ARGS+=(--lower-temperature "$DEEPRESYNC_LOWER_TEMPERATURE")
fi

"$DEEPRESYNC_PYTHON" "$SCRIPT_DIR/deepresync_infer_posteriors.py" \
    --reads "$reads" \
    --index-results "$index_results" \
    --output "$POSTERIOR_BIN" \
    --checkpoint "$DEEPRESYNC_CHECKPOINT" \
    --watermark-path "$config/watermark_sequence_length_235" \
    --deepresync-root "$DEEPRESYNC_ROOT" \
    --device "$DEEPRESYNC_DEVICE" \
    --batch-size "$DEEPRESYNC_BATCH_SIZE" \
    --max-read-len "$DEEPRESYNC_MAX_READ_LEN" \
    --amp "$DEEPRESYNC_AMP" \
    --summary "$results_path/summary_deepresync.json" \
    "${DEEPRESYNC_EXTRA_ARGS[@]}"

[[ -f "$POSTERIOR_BIN" ]] || { echo "[ERROR] missing $POSTERIOR_BIN" >&2; exit 1; }

STATS_PYTHON="${STATS_PYTHON:-${DEEPRESYNC_PYTHON:-python}}"
STATS_BATCH_SIZE="${STATS_BATCH_SIZE:-20000}"
STATS_EXTRA_ARGS=(--stats-batch-size "$STATS_BATCH_SIZE")
if [[ "${STATS_SKIP_PER_READ:-0}" == "1" ]]; then
    STATS_EXTRA_ARGS+=(--skip-per-read-stats)
fi
if [[ "${STATS_SKIP_HARD_BASE:-0}" == "1" ]]; then
    STATS_EXTRA_ARGS+=(--skip-hard-base-fasta)
fi

"$STATS_PYTHON" "$SCRIPT_DIR/stat_read_posteriors.py" \
    --posterior-bin "$POSTERIOR_BIN" \
    --watermark-path "$config/watermark_sequence_length_235" \
    --reference-fasta "$reference/encoded_sequence.fasta" \
    --pool "$pool" \
    --payload-start 45 \
    --per-read-stats "$results_path/per_read_hamming.txt" \
    --hard-base-fasta "$results_path/per_read_hard_base.fasta" \
    --stats-json "$results_path/indel_hard_decision_stats.json" \
    "${STATS_EXTRA_ARGS[@]}"
