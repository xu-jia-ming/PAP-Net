#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# A5000 compute capability: 8.6
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"

# Required:
#   WEIGHT_PATH=/path/to/model.pth
#   PAPNET_COCO_TRAIN_JSON, PAPNET_COCO_TRAIN_ROOT
#   PAPNET_COCO_VAL_JSON, PAPNET_COCO_VAL_ROOT

if [[ -z "${WEIGHT_PATH:-}" ]]; then
  echo "ERROR: Please set WEIGHT_PATH before running eval." >&2
  exit 1
fi

CONFIG_PATH="${CONFIG_PATH:-configs/COCO-Custom-A5000-Base-RCNN-FPN-Fast-BCNet.yaml}"
NUM_GPUS="${NUM_GPUS:-1}"
EXTRA_OPTS=("$@")

python3 tools/train_net.py --num-gpus "${NUM_GPUS}" \
  --config-file "${CONFIG_PATH}" \
  --eval-only MODEL.WEIGHTS "${WEIGHT_PATH}" \
  "${EXTRA_OPTS[@]}"

cat <<'JSON_INFO'
Evaluation outputs are saved under OUTPUT_DIR/inference/ .
COCO segmentation JSON files include:
- coco_instances_amodal_results.json
- coco_instances_visible_results.json
JSON_INFO
