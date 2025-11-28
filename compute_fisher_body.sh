#!/usr/bin/env bash
set -euo pipefail

# GPUs to use
CUDA_DEVICE_LIST=${CUDA_VISIBLE_DEVICES:-3,5,6}
export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE_LIST}

# Accelerate config (reuse training config if desired)
ACC_CONFIG="./acc_config/config.yaml"

# Input dataset (body rotation clips)
DATASET_BASE="/home/longnhat/Lin_workspace/8TB2/Lin/nas-lin-233/render_out"
METADATA_PATH="${DATASET_BASE}/metadata_body_aug.csv"

# Base WAN checkpoints
MODEL_ID_WITH_ORIGIN="Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,\
Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,\
Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth"

# Body LoRA checkpoint you already trained
BODY_LORA_CKPT="/home/longnhat/Lin_workspace/8TB2/Lin/801_Project/805/DiffSynth-Studio/checkpoints/body_best_ckpt/8.safetensors"

# Output tensors for Fisher & params
FISHER_DIR="./models/train/805_body_reference_best801"
FISHER_PATH="${FISHER_DIR}/body_fisher.pt"
PARAM_PATH="${FISHER_DIR}/body_params.pt"
mkdir -p "${FISHER_DIR}"

# Optional cap on the number of batches. Leave empty to sweep the whole dataset.
MAX_FISHER_STEPS=""

EXTRA_ARGS=()
if [[ -n "${MAX_FISHER_STEPS}" ]]; then
  EXTRA_ARGS+=("--max_fisher_steps" "${MAX_FISHER_STEPS}")
fi

accelerate launch --config_file "${ACC_CONFIG}" scripts/compute_fisher_wan.py \
  --dataset_base_path "${DATASET_BASE}" \
  --dataset_metadata_path "${METADATA_PATH}" \
  --height 512 \
  --width 512 \
  --num_frames 49 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "${MODEL_ID_WITH_ORIGIN}" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image" \
  --lora_checkpoint "${BODY_LORA_CKPT}" \
  --ewc_fisher "${FISHER_PATH}" \
  --ewc_prev_params "${PARAM_PATH}" \
  "${EXTRA_ARGS[@]}"

echo "[INFO] Fisher matrix saved to ${FISHER_PATH}"
echo "[INFO] Reference parameters saved to ${PARAM_PATH}"
