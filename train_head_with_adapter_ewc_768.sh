#!/usr/bin/env bash
set -euo pipefail

# GPU selection (comma-separated list). accelerate uses CUDA_VISIBLE_DEVICES.
# CUDA_DEVICE_LIST=${CUDA_VISIBLE_DEVICES:-3,5,6}
# export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE_LIST}

# Accelerate config to use
ACC_CONFIG="./acc_config/config.yaml"

# Head dataset
HEAD_DATASET_BASE="/home/longnhat/Lin_workspace/8TB2/Lin/nas-lin-233/render_out/805_final_set/"
HEAD_METADATA="/home/longnhat/Lin_workspace/8TB2/Lin/nas-lin-233/render_out/805_final_set/metadata_heads_all_805.csv"

# Model checkpoints
MODEL_ID_WITH_ORIGIN="Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,\
Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,\
Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth"
FISHER_PATH="./models/train/805_body_reference_best801/body_fisher.pt"
PARAM_PATH="./models/train/805_body_reference_best801/body_params.pt"

# Training output
HEAD_OUTPUT="./models/train/805_head_with_body_guard_768"

mkdir -p "${HEAD_OUTPUT}"

accelerate launch --config_file "${ACC_CONFIG}" examples/wanvideo/model_training/train.py \
  --dataset_base_path "${HEAD_DATASET_BASE}" \
  --dataset_metadata_path "${HEAD_METADATA}" \
  --height 768 \
  --width 768 \
  --num_frames 49 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "${MODEL_ID_WITH_ORIGIN}" \
  --learning_rate 1e-4 \
  --num_epochs 20 \
  --output_path "${HEAD_OUTPUT}" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image" \
  --ewc_lambda 10.0 \
  --ewc_fisher "${FISHER_PATH}" \
  --ewc_prev_params "${PARAM_PATH}"

echo "[INFO] Head training with EWC completed. Check ${HEAD_OUTPUT} for checkpoints."
