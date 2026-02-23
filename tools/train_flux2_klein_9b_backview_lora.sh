#!/usr/bin/env bash
set -euo pipefail

# Train FLUX.2-klein-9B LoRA on back-view-only images.
#
# It first generates metadata.csv from data/*/frame_*/rgb using target azimuth/elevation,
# then launches LoRA training.
#
# Example:
#   bash tools/train_flux2_klein_9b_backview_lora.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
DATASET_BASE_PATH="${DATASET_BASE_PATH:-${ROOT_DIR}/data}"
METADATA_PATH="${METADATA_PATH:-${ROOT_DIR}/data/back_view_dataset/metadata.csv}"
META_SCRIPT="${META_SCRIPT:-${ROOT_DIR}/tools/create_flux2_backview_metadata.py}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${ROOT_DIR}/examples/flux2/model_training/train.py}"

TARGET_AZIMUTH="${TARGET_AZIMUTH:-340.0}"
TARGET_ELEVATION="${TARGET_ELEVATION:-0.0}"
BACK_PROMPT="${BACK_PROMPT:-Back view of a woman in traditional Chinese hanfu dress, black background}"
SEQUENCE_START="${SEQUENCE_START:-}"
SEQUENCE_END="${SEQUENCE_END:-}"

MODEL_ID="${MODEL_ID:-black-forest-labs/FLUX.2-klein-9B}"
OUTPUT_PATH="${OUTPUT_PATH:-${ROOT_DIR}/models/train/FLUX.2-klein-9B-backview-lora}"

MAX_PIXELS="${MAX_PIXELS:-1048576}"
DATASET_REPEAT="${DATASET_REPEAT:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
LORA_RANK="${LORA_RANK:-32}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${ROOT_DIR}/models}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${ROOT_DIR}/models/modelscope_cache}"

if [[ ! -f "${META_SCRIPT}" ]]; then
  echo "Metadata script not found: ${META_SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "Training script not found: ${TRAIN_SCRIPT}" >&2
  exit 1
fi
if [[ ! -d "${DATASET_BASE_PATH}" ]]; then
  echo "Dataset base path not found: ${DATASET_BASE_PATH}" >&2
  exit 1
fi

mkdir -p "$(dirname "${METADATA_PATH}")"
mkdir -p "${OUTPUT_PATH}"

echo "Generating metadata..."
meta_args=(
  --data_root "${DATASET_BASE_PATH}"
  --output_csv "${METADATA_PATH}"
  --target_azimuth "${TARGET_AZIMUTH}"
  --target_elevation "${TARGET_ELEVATION}"
  --prompt "${BACK_PROMPT}"
)
if [[ -n "${SEQUENCE_START}" ]]; then
  meta_args+=(--sequence_start "${SEQUENCE_START}")
fi
if [[ -n "${SEQUENCE_END}" ]]; then
  meta_args+=(--sequence_end "${SEQUENCE_END}")
fi

"${PYTHON_BIN}" "${META_SCRIPT}" "${meta_args[@]}"

echo "Starting training..."
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "ACCELERATE_BIN=${ACCELERATE_BIN}"
echo "DATASET_BASE_PATH=${DATASET_BASE_PATH}"
echo "METADATA_PATH=${METADATA_PATH}"
echo "MODEL_ID=${MODEL_ID}"
echo "OUTPUT_PATH=${OUTPUT_PATH}"
echo "SEQUENCE_START=${SEQUENCE_START:-<none>} SEQUENCE_END=${SEQUENCE_END:-<none>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "DIFFSYNTH_SKIP_DOWNLOAD=${DIFFSYNTH_SKIP_DOWNLOAD}"

"${ACCELERATE_BIN}" launch "${TRAIN_SCRIPT}" \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --dataset_metadata_path "${METADATA_PATH}" \
  --data_file_keys "image" \
  --max_pixels "${MAX_PIXELS}" \
  --dataset_repeat "${DATASET_REPEAT}" \
  --model_id_with_origin_paths "${MODEL_ID}:text_encoder/*.safetensors,${MODEL_ID}:transformer/*.safetensors,${MODEL_ID}:vae/diffusion_pytorch_model.safetensors" \
  --tokenizer_path "${MODEL_ID}:tokenizer/" \
  --learning_rate "${LEARNING_RATE}" \
  --num_epochs "${NUM_EPOCHS}" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_PATH}" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,to_out.0,add_q_proj,add_k_proj,add_v_proj,to_add_out,linear_in,linear_out,to_qkv_mlp_proj,single_transformer_blocks.0.attn.to_out,single_transformer_blocks.1.attn.to_out,single_transformer_blocks.2.attn.to_out,single_transformer_blocks.3.attn.to_out,single_transformer_blocks.4.attn.to_out,single_transformer_blocks.5.attn.to_out,single_transformer_blocks.6.attn.to_out,single_transformer_blocks.7.attn.to_out,single_transformer_blocks.8.attn.to_out,single_transformer_blocks.9.attn.to_out,single_transformer_blocks.10.attn.to_out,single_transformer_blocks.11.attn.to_out,single_transformer_blocks.12.attn.to_out,single_transformer_blocks.13.attn.to_out,single_transformer_blocks.14.attn.to_out,single_transformer_blocks.15.attn.to_out,single_transformer_blocks.16.attn.to_out,single_transformer_blocks.17.attn.to_out,single_transformer_blocks.18.attn.to_out,single_transformer_blocks.19.attn.to_out,single_transformer_blocks.20.attn.to_out,single_transformer_blocks.21.attn.to_out,single_transformer_blocks.22.attn.to_out,single_transformer_blocks.23.attn.to_out" \
  --lora_rank "${LORA_RANK}" \
  --use_gradient_checkpointing

echo "Done. Checkpoints in: ${OUTPUT_PATH}"
