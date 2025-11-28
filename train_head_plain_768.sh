#!/usr/bin/env bash
set -euo pipefail

# GPU list for accelerate
CUDA_DEVICE_LIST=${CUDA_VISIBLE_DEVICES:-3,5,6}
export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE_LIST}
ACC_CONFIG="./acc_config/config.yaml"

# Data & checkpoints
HEAD_DATASET_BASE="/home/longnhat/Lin_workspace/8TB2/Lin/nas-lin-233/render_out/805_final_set/"
HEAD_METADATA="/home/longnhat/Lin_workspace/8TB2/Lin/nas-lin-233/render_out/805_final_set/metadata_heads_all_805.csv"
MODEL_ID_WITH_ORIGIN="Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,\
Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,\
Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth"
BODY_LORA_CKPT="/home/longnhat/Lin_workspace/8TB2/Lin/801_Project/805/DiffSynth-Studio/checkpoints/body_best_ckpt/8.safetensors"
OUTPUT_DIR="./models/train/805_head_plain_768"
mkdir -p "${OUTPUT_DIR}"

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
  --output_path "${OUTPUT_DIR}" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image" \
  --lora_checkpoint "${BODY_LORA_CKPT}"

echo "[INFO] Head fine-tune without EWC complete. Check ${OUTPUT_DIR}."
