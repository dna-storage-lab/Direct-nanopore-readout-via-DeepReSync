#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =========================
# Usage
# =========================
if [[ $# -ne 4 ]]; then
  echo "Usage: $0 <read_posteriors.bin> <results_path> <pool> <threads>"
  exit 1
fi

read_posteriors=$1
results_path=$2
pool=$3
threads=$4

mkdir -p "$results_path"

# Consensus calibration parameters are intentionally not printed here.

"$SCRIPT_DIR/../bin/multiple_copy_consensus" \
    "$read_posteriors" \
    "$results_path" \
    "$pool" \
    "$threads"
