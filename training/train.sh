#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

# SUP base-head training: cosine lr 2e-4 -> min_lr 3e-5.
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-512}"
EPOCHS="${EPOCHS:-500}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-100000}"
VAL_SAMPLES="${VAL_SAMPLES:-5000}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SAVE_DIR="training/runs/checkpoints"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

echo "[TRAIN] SUP base-head on GPU ${GPU_ID}"
echo "  batch_size: ${BATCH_SIZE}"
echo "  save_dir  : ${SAVE_DIR}"

python training/train_estimator.py \
  --train-samples "${TRAIN_SAMPLES}" \
  --val-samples "${VAL_SAMPLES}" \
  --seed 1 \
  --resample-each-epoch \
  --max-read-len 256 \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr 0.0002 \
  --adam-beta1 0.9 \
  --adam-beta2 0.999 \
  --lr-scheduler cosine \
  --min-lr 0.00003 \
  --weight-decay 0.0001 \
  --num-workers "${NUM_WORKERS}" \
  --prefetch-factor 4 \
  --tf32 \
  --amp fp16 \
  --log-detail compact \
  --d-model 128 \
  --nhead 8 \
  --encoder-layers 3 \
  --decoder-layers 3 \
  --dim-feedforward 512 \
  --dropout 0.1 \
  --base-loss-weight 1.0 \
  --aux-bit-loss-weight 0.10 \
  --lower-loss-weight 0.50 \
  --symbol-loss-weight 0.50 \
  --upper-info-loss-weight 0.20 \
  --symbol-hidden-size 0 \
  --symbol-layers 1 \
  --watermark-consistency-loss-weight 0.30 \
  --watermark-consistency-mode symbol_watermark \
  --save-dir "${SAVE_DIR}"
