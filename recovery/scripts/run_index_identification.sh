#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =========================
# Usage
# =========================
if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "Usage: $0 <IN.fa> <OUT_dir> <OMP_NUM_THREADS> <configureFiles> [index_align_threshold]"
  echo "Example: $0 /path/to/pair1_seq.fa outputs/hac_res_pool1_downsample_3x 40 configureFiles 3"
  exit 1
fi

IN="$1"
OUT="$2"
OMP_T="$3"
configureFiles="$4"
INDEX_THRESHOLD="${5:-${INDEX_THRESHOLD:-3}}"

mkdir -p "$OUT"

export OMP_NUM_THREADS="$OMP_T"

"${SCRIPT_DIR}/../bin/index_identification" "$IN" "$OUT" "$configureFiles" "$INDEX_THRESHOLD"
