#!/usr/bin/env bash
set -euo pipefail

# Paths can be overridden when invoking the script, e.g. DATASET_BASE=/path ./finetune_body_lora_on_heads.sh
DATASET_BASE="${DATASET_BASE:-/home/longnhat/Lin_workspace/8TB2/Lin/nas-lin-233/render_out}"
HEAD_METADATA="${HEAD_METADATA:-$DATASET_BASE/metadata_head.csv}"
MODEL_ID_WITH_ORIGIN_PATHS="${MODEL_ID_WITH_ORIGIN_PATHS:-Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth}"
LORA_CHECKPOINT="${LORA_CHECKPOINT:-./models/train/805_baseline_bodyonly/epoch-5.safetensors}"
OUTPUT_PATH="${OUTPUT_PATH:-./models/train/vim }"
HEIGHT="${HEIGHT:-512}"
WIDTH="${WIDTH:-512}"
NUM_FRAMES="${NUM_FRAMES:-49}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
DATASET_REPEAT="${DATASET_REPEAT:-1}"
EXTRA_INPUTS="${EXTRA_INPUTS:-input_image}"
LORA_RANK="${LORA_RANK:-32}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"

export CUDA_VISIBLE_DEVICES

python examples/wanvideo/model_training/train.py \
  --dataset_base_path "${DATASET_BASE}" \
  --dataset_metadata_path "${HEAD_METADATA}" \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --num_frames "${NUM_FRAMES}" \
  --dataset_repeat "${DATASET_REPEAT}" \
  --model_id_with_origin_paths "${MODEL_ID_WITH_ORIGIN_PATHS}" \
  --learning_rate "${LEARNING_RATE}" \
  --num_epochs "${NUM_EPOCHS}" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_PATH}" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank "${LORA_RANK}" \
  --lora_checkpoint "${LORA_CHECKPOINT}" \
  --extra_inputs "${EXTRA_INPUTS}"
