#!/usr/bin/env bash
set -euo pipefail

# Paths to checkpoints
HEAD_CKPT="/home/longnhat/Lin_workspace/8TB2/Lin/801_Project/805/DiffSynth-Studio/models/train/805_baseline_headonly/epoch-8.safetensors"
BODY_CKPT="/home/longnhat/Lin_workspace/8TB2/Lin/801_Project/805/DiffSynth-Studio/models/train/805_baseline_bodyonly/epoch-5.safetensors"

# Input roots
HEAD_INPUT="/home/longnhat/Lin_workspace/8TB2/Lin/801_Project/805/DiffSynth-Studio/eval_samples_805/heads"
BODY_INPUT="/home/longnhat/Lin_workspace/8TB2/Lin/801_Project/805/DiffSynth-Studio/eval_samples_805/bodys"

# Output roots
HEAD_OUTPUT="out/heads_baseline_eval"
BODY_OUTPUT="out/bodys_baseline_eval"

# Common settings
HEIGHT=512
WIDTH=512
SEED=1
SPLITS=("in_dist" "ood")

echo "[INFO] Evaluating head checkpoint: ${HEAD_CKPT}"
python eval_head_only_batch.py \
  --input_root "${HEAD_INPUT}" \
  --output_dir "${HEAD_OUTPUT}" \
  --lora_path "${HEAD_CKPT}" \
  --height ${HEIGHT} \
  --width ${WIDTH} \
  --splits "${SPLITS[@]}" \
  --seed ${SEED} \
  --device cuda

echo "[INFO] Evaluating body checkpoint: ${BODY_CKPT}"
python eval_body_only_batch.py \
  --input_root "${BODY_INPUT}" \
  --output_dir "${BODY_OUTPUT}" \
  --lora_path "${BODY_CKPT}" \
  --height ${HEIGHT} \
  --width ${WIDTH} \
  --splits "${SPLITS[@]}" \
  --seed ${SEED} \
  --device cuda

echo "[INFO] Head/body evaluations finished."
