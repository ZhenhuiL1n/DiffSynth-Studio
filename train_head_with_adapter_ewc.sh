#!/usr/bin/env bash
set -euo pipefail

# Configurable paths ---------------------------------------------------------
CUDA_DEVICE_LIST=${CUDA_VISIBLE_DEVICES:3,4,5}

BODY_DATASET_BASE="/home/longnhat/Lin_workspace/8TB2/Lin/nas-lin-233/render_out"
BODY_METADATA_PATH="${BODY_DATASET_BASE}/metadata_body.csv"

HEAD_DATASET_BASE="/home/longnhat/Lin_workspace/8TB2/Lin/nas-lin-233/render_out"
HEAD_METADATA_PATH="${HEAD_DATASET_BASE}/metadata_head.csv"

MODEL_ID_WITH_ORIGIN="Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,\
Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,\
Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth"

BODY_LORA_CKPT="./models/train/805_baseline_bodyonly/epoch-5.safetensors"
REF_OUTPUT_DIR="./models/train/805_body_reference"
HEAD_OUTPUT_DIR="./models/train/805_head_with_body_guard"

FISHER_PATH="${REF_OUTPUT_DIR}/body_fisher.pt"
PARAM_PATH="${REF_OUTPUT_DIR}/body_params.pt"

mkdir -p "${REF_OUTPUT_DIR}" "${HEAD_OUTPUT_DIR}"

# Step 1: Accumulate Fisher information for the body adapter -----------------
echo "[INFO] Computing Fisher matrix for the body adapter..."
CUDA_VISIBLE_DEVICES=${CUDA_DEVICE_LIST} python examples/wanvideo/model_training/train.py \
  --dataset_base_path "${BODY_DATASET_BASE}" \
  --dataset_metadata_path "${BODY_METADATA_PATH}" \
  --height 512 \
  --width 512 \
  --num_frames 49 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "${MODEL_ID_WITH_ORIGIN}" \
  --learning_rate 1e-4 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${REF_OUTPUT_DIR}" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image" \
  --lora_checkpoint "${BODY_LORA_CKPT}" \
  --ewc_compute_fisher_only \
  --ewc_fisher "${FISHER_PATH}" \
  --ewc_prev_params "${PARAM_PATH}"

# Step 2: Train head adapter with EWC regularization -------------------------
echo "[INFO] Fine-tuning head adapter with EWC regularization..."
CUDA_VISIBLE_DEVICES=${CUDA_DEVICE_LIST} python examples/wanvideo/model_training/train.py \
  --dataset_base_path "${HEAD_DATASET_BASE}" \
  --dataset_metadata_path "${HEAD_METADATA_PATH}" \
  --height 512 \
  --width 512 \
  --num_frames 49 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "${MODEL_ID_WITH_ORIGIN}" \
  --learning_rate 1e-4 \
  --num_epochs 20 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${HEAD_OUTPUT_DIR}" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image" \
  --ewc_lambda 10.0 \
  --ewc_fisher "${FISHER_PATH}" \
  --ewc_prev_params "${PARAM_PATH}"

echo "[INFO] Training finished. Check ${HEAD_OUTPUT_DIR} for LoRA checkpoints."
