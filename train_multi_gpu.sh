#!/usr/bin/env bash
set -euo pipefail

# Choose which 3 GPUs to use
export CUDA_VISIBLE_DEVICES=0,1,2

# Optional: tweak envs as you like
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

ACC_CONFIG="./acc_config/config.yaml"

accelerate launch --config_file "$ACC_CONFIG" examples/wanvideo/model_training/train.py \
  --dataset_base_path data_lora/ \
  --dataset_metadata_path data_lora/metadata.csv \
  --height 832 \
  --width 480 \
  --num_frames 49 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.2-TI2V-5B_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image"
