#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# A5000 compute capability: 8.6
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"

# Required dataset environment variables:
#   PAPNET_COCO_TRAIN_JSON, PAPNET_COCO_TRAIN_ROOT
#   PAPNET_COCO_VAL_JSON, PAPNET_COCO_VAL_ROOT
# Optional:
#   PAPNET_COCO_CLASSES (comma-separated) or PAPNET_COCO_CLASSES_FILE
#   PAPNET_COCO_TRAIN_NAME / PAPNET_COCO_VAL_NAME (default: papnet_coco_train / papnet_coco_val)

CONFIG_PATH="${CONFIG_PATH:-configs/COCO-Custom-A5000-Base-RCNN-FPN-Fast-BCNet.yaml}"
NUM_GPUS="${NUM_GPUS:-1}"
EXTRA_OPTS=("$@")

python3 tools/train_net.py --num-gpus "${NUM_GPUS}" \
  --config-file "${CONFIG_PATH}" \
  "${EXTRA_OPTS[@]}"
