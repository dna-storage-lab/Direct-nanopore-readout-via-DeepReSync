#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECOVERY_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$RECOVERY_ROOT/.." && pwd)"

# =========================
# Config
# =========================
# Edit these two values when you want to evaluate another posterior file.
POOL=1
POSTERIOR_BIN="${PROJECT_ROOT}/results/deepresync/pool1/5.8x/correction_results/read_posteriors.bin"


# Usually no need to edit the values below.
STATS_PYTHON="python"
STATS_BATCH_SIZE=20000
STATS_SKIP_PER_READ=0
STATS_SKIP_HARD_BASE=0
PAYLOAD_START=45
CONFIG="${PROJECT_ROOT}/configureFiles"
REFERENCE="${RECOVERY_ROOT}/reference"

if [[ $# -gt 2 ]]; then
  echo "Usage: $0 [read_posteriors.bin] [pool]" >&2
  exit 1
fi

if [[ $# -ge 1 ]]; then
  POSTERIOR_BIN="$1"
fi

if [[ $# -ge 2 ]]; then
  POOL="$2"
fi

[[ -f "$POSTERIOR_BIN" ]] || {
  echo "[ERROR] missing posterior bin: $POSTERIOR_BIN" >&2
  echo "        Edit POSTERIOR_BIN near the top of $0, or pass it as:" >&2
  echo "        $0 /path/to/read_posteriors.bin $POOL" >&2
  exit 1
}

RESULTS_PATH="$(cd "$(dirname "$POSTERIOR_BIN")" && pwd)"
PER_READ_STATS="${RESULTS_PATH}/per_read_hamming.txt"
HARD_BASE_FASTA="${RESULTS_PATH}/per_read_hard_base.fasta"
STATS_JSON="${RESULTS_PATH}/indel_hard_decision_stats.json"

STATS_EXTRA_ARGS=(--stats-batch-size "$STATS_BATCH_SIZE")
if [[ "$STATS_SKIP_PER_READ" == "1" ]]; then
  STATS_EXTRA_ARGS+=(--skip-per-read-stats)
fi
if [[ "$STATS_SKIP_HARD_BASE" == "1" ]]; then
  STATS_EXTRA_ARGS+=(--skip-hard-base-fasta)
fi

"$STATS_PYTHON" "$SCRIPT_DIR/stat_read_posteriors.py" \
  --posterior-bin "$POSTERIOR_BIN" \
  --watermark-path "$CONFIG/watermark_sequence_length_235" \
  --reference-fasta "$REFERENCE/encoded_sequence.fasta" \
  --pool "$POOL" \
  --payload-start "$PAYLOAD_START" \
  --per-read-stats "$PER_READ_STATS" \
  --hard-base-fasta "$HARD_BASE_FASTA" \
  --stats-json "$STATS_JSON" \
  "${STATS_EXTRA_ARGS[@]}"

# echo "[DONE] posterior hard-decision statistics"
# echo "  posterior : $POSTERIOR_BIN"
# echo "  per-read  : $PER_READ_STATS"
# echo "  hard-base : $HARD_BASE_FASTA"
# echo "  summary   : $STATS_JSON"
