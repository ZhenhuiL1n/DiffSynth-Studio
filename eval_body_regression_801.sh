#!/usr/bin/env bash
set -euo pipefail

CKPT="/home/longnhat/Lin_workspace/8TB2/Lin/801_Project/805/DiffSynth-Studio/models/train/805_head_with_body_guard_768/epoch-8.safetensors"
INPUT_ROOT="/home/longnhat/Lin_workspace/8TB2/Lin/801_Project/805/DiffSynth-Studio/eval_samples_805/801"
OUTPUT_DIR="./out/body_regression_801"

python eval_head_plain_on_body_batch.py \
  --input_root "${INPUT_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --lora_path "${CKPT}" \
  --height 830 \
  --width 482 \
  --splits ood \
  --include ood_2 ood_3 ood_19 \
  --skip_existing \
  --seed 1 \
  --device cuda

echo "[INFO] Saved regression videos under ${OUTPUT_DIR}"
