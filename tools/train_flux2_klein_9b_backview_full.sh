#!/usr/bin/env bash
set -euo pipefail

# Full finetune FLUX.2-klein on back-view-only images.
#
# It first generates metadata.csv from data/*/frame_*/rgb using target azimuth/elevation
# (optionally filtered by sequence range), then launches full DIT finetuning.
#
# Example:
#   bash tools/train_flux2_klein_9b_backview_full.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-}"
DATASET_BASE_PATH="${DATASET_BASE_PATH:-${ROOT_DIR}/data}"
METADATA_PATH="${METADATA_PATH:-${ROOT_DIR}/data/back_view_dataset/metadata_full.csv}"
META_SCRIPT="${META_SCRIPT:-${ROOT_DIR}/tools/create_flux2_backview_metadata.py}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${ROOT_DIR}/examples/flux2/model_training/train.py}"

TARGET_AZIMUTH="${TARGET_AZIMUTH:-340.0}"
TARGET_ELEVATION="${TARGET_ELEVATION:-0.0}"
BACK_PROMPT="${BACK_PROMPT:-Back view of a woman in traditional Chinese hanfu dress, black background}"
SEQUENCE_START="${SEQUENCE_START:-}"
SEQUENCE_END="${SEQUENCE_END:-}"

MODEL_ID="${MODEL_ID:-black-forest-labs/FLUX.2-klein-4B}"
OUTPUT_PATH="${OUTPUT_PATH:-${ROOT_DIR}/models/train/FLUX.2-klein-4B-backview-full}"

MAX_PIXELS="${MAX_PIXELS:-786432}"
DATASET_REPEAT="${DATASET_REPEAT:-1}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
TRAINABLE_MODELS="${TRAINABLE_MODELS:-dit}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
USE_GRADIENT_CHECKPOINTING_OFFLOAD="${USE_GRADIENT_CHECKPOINTING_OFFLOAD:-false}"

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

echo "Starting full finetune..."
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "ACCELERATE_BIN=${ACCELERATE_BIN}"
echo "ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-<none>}"
echo "DATASET_BASE_PATH=${DATASET_BASE_PATH}"
echo "METADATA_PATH=${METADATA_PATH}"
echo "MODEL_ID=${MODEL_ID}"
echo "OUTPUT_PATH=${OUTPUT_PATH}"
echo "SEQUENCE_START=${SEQUENCE_START:-<none>} SEQUENCE_END=${SEQUENCE_END:-<none>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "LR=${LEARNING_RATE} EPOCHS=${NUM_EPOCHS} DATASET_REPEAT=${DATASET_REPEAT} TRAINABLE_MODELS=${TRAINABLE_MODELS}"
echo "GRAD_ACC=${GRADIENT_ACCUMULATION_STEPS} GC_OFFLOAD=${USE_GRADIENT_CHECKPOINTING_OFFLOAD}"
echo "DIFFSYNTH_SKIP_DOWNLOAD=${DIFFSYNTH_SKIP_DOWNLOAD}"

launch_args=()
if [[ -n "${ACCELERATE_CONFIG}" ]]; then
  launch_args+=(--config_file "${ACCELERATE_CONFIG}")
fi

train_args=(
  --dataset_base_path "${DATASET_BASE_PATH}"
  --dataset_metadata_path "${METADATA_PATH}"
  --data_file_keys "image"
  --max_pixels "${MAX_PIXELS}"
  --dataset_repeat "${DATASET_REPEAT}"
  --model_id_with_origin_paths "${MODEL_ID}:text_encoder/*.safetensors,${MODEL_ID}:transformer/*.safetensors,${MODEL_ID}:vae/diffusion_pytorch_model.safetensors"
  --tokenizer_path "${MODEL_ID}:tokenizer/"
  --learning_rate "${LEARNING_RATE}"
  --num_epochs "${NUM_EPOCHS}"
  --remove_prefix_in_ckpt "pipe.dit."
  --output_path "${OUTPUT_PATH}"
  --trainable_models "${TRAINABLE_MODELS}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --use_gradient_checkpointing
)
if [[ "${USE_GRADIENT_CHECKPOINTING_OFFLOAD}" == "true" ]]; then
  train_args+=(--use_gradient_checkpointing_offload)
fi

"${ACCELERATE_BIN}" launch "${launch_args[@]}" "${TRAIN_SCRIPT}" "${train_args[@]}"

echo "Done. Full finetune checkpoints in: ${OUTPUT_PATH}"
