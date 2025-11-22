#!/usr/bin/env bash
set -euo pipefail

# Override via environment variables when launching the script.
DATASET_BASE="${DATASET_BASE:-/home/longnhat/Lin_workspace/8TB2/Lin/nas-lin-233/render_out}"
BODY_METADATA="${BODY_METADATA:-$DATASET_BASE/metadata_body.csv}"
MODEL_ID_WITH_ORIGIN_PATHS="${MODEL_ID_WITH_ORIGIN_PATHS:-Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth}"
LORA_CHECKPOINT="${LORA_CHECKPOINT:-/path/to/headonly_checkpoint.safetensors}"
OUTPUT_PATH="${OUTPUT_PATH:-./models/train/805_head_to_body_finetune}"
HEIGHT="${HEIGHT:-512}"
WIDTH="${WIDTH:-512}"
NUM_FRAMES="${NUM_FRAMES:-49}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
DATASET_REPEAT="${DATASET_REPEAT:-1}"
EXTRA_INPUTS="${EXTRA_INPUTS:-input_image}"
LORA_RANK="${LORA_RANK:-32}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"

if [ ! -f "$LORA_CHECKPOINT" ]; then
  echo "[WARN] LORA_CHECKPOINT '$LORA_CHECKPOINT' does not exist. Set this env var once the head-only checkpoint is available." >&2
fi

export CUDA_VISIBLE_DEVICES

python examples/wanvideo/model_training/train.py \
  --dataset_base_path "${DATASET_BASE}" \
  --dataset_metadata_path "${BODY_METADATA}" \
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
