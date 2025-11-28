#!/usr/bin/env bash
set -euo pipefail

CKPT="/home/longnhat/Lin_workspace/8TB2/Lin/801_Project/805/DiffSynth-Studio/models/train/805_head_with_body_guard_768/epoch-3.safetensors"
INPUT_ROOT="/home/longnhat/Lin_workspace/8TB2/Lin/801_Project/805/DiffSynth-Studio/eval_samples_805/heads"
OUTPUT_DIR="./out/heads_guard_eval_768"

python eval_head_only_batch.py \
  --input_root "${INPUT_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --lora_path "${CKPT}" \
  --height 768 \
  --width 768 \
  --num_frames 49 \
  --fps 24 \
  --seed 1

echo "[INFO] Videos saved to ${OUTPUT_DIR}"
