#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

mkdir -p bin

# Primer identification is distributed as a prebuilt binary.
[[ -f ./bin/primer_identification ]] || {
  echo "[ERROR] missing binary: ./bin/primer_identification" >&2
  exit 1
}
chmod +x ./bin/primer_identification


# index identification
gcc -O3 -march=native -DNDEBUG -std=gnu89 \
  -I./include \
  -c ./src/order5.c \
  -o ./bin/order5.o

g++ -O3 -march=native -DNDEBUG -fopenmp \
  -I./include \
  ./src/index_identification.cpp \
  ./src/fba_for_index.cpp \
  ./src/edlib.cpp \
  ./bin/order5.o \
  -lm \
  -o ./bin/index_identification


# multiple-copy consensus
g++ -O3 -w \
  -I./include \
  ./src/multiple_copy_consensus.cpp \
  ./src/utils.cpp \
  -o ./bin/multiple_copy_consensus


# LDPC decoder
g++ -O3 -std=c++17 \
  ./src/ldpc_decoder.cpp \
  -I./include \
  -L./lib \
  -ldecode \
  -Wl,-rpath,'$ORIGIN/../lib' \
  -pthread \
  -o ./bin/ldpc_decoder


# recovery audio
gcc ./src/recovery_audio.c \
  -lm \
  -o ./bin/recovery_audio


# analysis
g++ -O2 -std=c++11 \
  ./src/analysis.cpp \
  -o ./bin/analysis

touch ./bin/.current_layout_built
echo "[DONE] recovery binaries built for the current repository layout"
