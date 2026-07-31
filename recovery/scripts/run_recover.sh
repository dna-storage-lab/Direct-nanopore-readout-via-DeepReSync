#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <bit_file> <out_dir> <pool(1-6)>"
  exit 1
fi

bit_file="$1"
out_dir="$2"
pool="$3"

# padNum depends on pool
case "$pool" in
  1) padNum=868  ;;
  2) padNum=2956 ;;
  3) padNum=236  ;;
  4) padNum=6228 ;;
  5) padNum=44   ;;
  6) padNum=3484 ;;
  *) echo "[ERR] pool must be 1..6" >&2; exit 1 ;;
esac

[[ -f "$bit_file" ]] || { echo "[ERR] missing $bit_file" >&2; exit 1; }

mkdir -p "$out_dir"

tmp="${out_dir}/recover_BIT.tmp"
tr -d '\r\n' < "$bit_file" > "$tmp"

len=$(wc -c < "$tmp")
keep=$((len - padNum))
(( keep > 0 )) || { echo "[ERR] padNum too large" >&2; exit 1; }

out_bits="${out_dir}/recover_BIT.txt"
head -c "$keep" "$tmp" > "$out_bits"
rm -f "$tmp"


case "$pool" in
  1) archive_name="My_Uncle_Jules_Chinese_part1.7z" ;;
  2) archive_name="My_Uncle_Jules_Chinese_part2.7z" ;;
  3) archive_name="My_Uncle_Jules_English_part1.7z" ;;
  4) archive_name="My_Uncle_Jules_English_part2.7z" ;;
  5) archive_name="My_Uncle_Jules_French_part1.7z" ;;
  6) archive_name="My_Uncle_Jules_French_part2.7z" ;;
  *)
    echo "[ERROR] invalid pool: $pool"
    exit 1
    ;;
esac

archive_path="$out_dir/$archive_name"

# recovery_audio
"${SCRIPT_DIR}/../bin/recovery_audio" "$out_bits" "$archive_path" "$pool"

# unzip
7z x "$archive_path" -o"$out_dir" -y

