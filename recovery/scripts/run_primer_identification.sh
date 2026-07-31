#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =========================
# Usage
# =========================
if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "Usage: $0 <input_subfastq> <result_output_prefix> <thread> <primer_list> <valid_bias_size> [primer_min_edit]"
  echo "Example: $0 in.fastq out_dir 20 ${SCRIPT_DIR}/primer_list_10nt.txt 20 3"
  exit 1
fi

input_subfastq="$1"
result_output_prefix="$2"
thread="$3"
primer_list="$4"
valid_bias_size="$5"
primer_min_edit="${6:-${PRIMER_MIN_EDIT:-2}}"

if ! [[ "$valid_bias_size" =~ ^[0-9]+$ ]]; then
  echo "Error: valid_bias_size must be a non-negative integer, got: $valid_bias_size" >&2
  exit 1
fi

if ! [[ "$primer_min_edit" =~ ^[0-9]+$ ]]; then
  echo "Error: primer_min_edit must be a non-negative integer, got: $primer_min_edit" >&2
  exit 1
fi

fastq_file="$input_subfastq"
primer_num=2
payload_size=260
result_output=${result_output_prefix}/valid_reads.fa

mkdir -p "$result_output_prefix"


"${SCRIPT_DIR}/../bin/primer_identification" \
                        "$fastq_file" \
                        "$primer_list" \
                        "$primer_num" \
                        "$payload_size" \
                        "$result_output" \
                        "$valid_bias_size" \
                        "$thread" \
                        "$primer_min_edit"
