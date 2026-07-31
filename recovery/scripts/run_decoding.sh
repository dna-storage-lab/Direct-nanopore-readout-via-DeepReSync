#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =========================
# Usage & Argument Parsing
# =========================
if [[ $# -lt 4 ]]; then
    echo "Usage: $0 <inference_results> <decoding_save_path> <pool> <thread>"
    exit 1
fi

in="$1"
out="$2"
pool="$3"
thread="$4"

config="${SCRIPT_DIR}/../../configureFiles"

mkdir -p "$out"


# Run the decoder
"${SCRIPT_DIR}/../bin/ldpc_decoder" \
  "$in" \
  "$out" \
  "$pool" \
  "$thread" \
  "$config"
