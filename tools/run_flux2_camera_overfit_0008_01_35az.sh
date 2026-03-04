#!/usr/bin/env bash
set -euo pipefail

# Overfit camera adapter on sequence 0008_01 using 35 azimuth angles (0..340, step 10), elevation fixed to 0.
# This is designed as a quick sanity test for camera-conditioning behavior.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${ROOT_DIR}/examples/flux2/model_training/train_camera.py}"
META_SCRIPT="${META_SCRIPT:-${ROOT_DIR}/tools/create_flux2_camera_metadata.py}"

DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/data}"
FOLDERS="${FOLDERS:-0008_01}"
FRAMES_PER_FOLDER="${FRAMES_PER_FOLDER:-1}"   # 1 frame => strongest overfit check
ANGLES_CSV="${ANGLES_CSV:-0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340}"
BASE_AZIMUTH="${BASE_AZIMUTH:-160.0}"
BASE_ELEVATION="${BASE_ELEVATION:-0.0}"
PROMPT="${PROMPT:-a woman in traditional Chinese hanfu dress, black background}"

MODEL_SIZE="${MODEL_SIZE:-4B}"                # 4B is lighter for quick overfit checks
MODEL_TYPE="${MODEL_TYPE:-distilled}"         # distilled|base
METADATA_PATH="${METADATA_PATH:-${ROOT_DIR}/data/camera_dataset/metadata_overfit_0008_01_35az.csv}"
OUTPUT_PATH="${OUTPUT_PATH:-${ROOT_DIR}/models/train/FLUX.2-klein-${MODEL_SIZE}-camera-adapter-overfit-0008_01-35az}"
CAMERA_ADAPTER_INIT_CHECKPOINT="${CAMERA_ADAPTER_INIT_CHECKPOINT:-}"

MAX_PIXELS="${MAX_PIXELS:-786432}"
DATASET_REPEAT="${DATASET_REPEAT:-200}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
SAVE_STEPS="${SAVE_STEPS:-50}"   # set to "none" (or empty) for epoch-only checkpoints
TRAINABLE_MODELS="${TRAINABLE_MODELS:-camera_adapter}"
USE_GRADIENT_CHECKPOINTING_OFFLOAD="${USE_GRADIENT_CHECKPOINTING_OFFLOAD:-false}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${ROOT_DIR}/models}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${ROOT_DIR}/models/modelscope_cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -f "${META_SCRIPT}" ]]; then
  echo "Metadata script not found: ${META_SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "Training script not found: ${TRAIN_SCRIPT}" >&2
  exit 1
fi

if [[ "${MODEL_TYPE}" == "base" ]]; then
  TRANSFORMER_ID="black-forest-labs/FLUX.2-klein-base-${MODEL_SIZE}"
else
  TRANSFORMER_ID="black-forest-labs/FLUX.2-klein-${MODEL_SIZE}"
fi

if [[ "${MODEL_SIZE}" == "9B" ]]; then
  TEXT_ENCODER_ID="black-forest-labs/FLUX.2-klein-9B"
  TOKENIZER_ID="black-forest-labs/FLUX.2-klein-9B"
else
  TEXT_ENCODER_ID="black-forest-labs/FLUX.2-klein-4B"
  TOKENIZER_ID="black-forest-labs/FLUX.2-klein-4B"
fi

if [[ -n "${CAMERA_ADAPTER_INIT_CHECKPOINT}" && ! -f "${CAMERA_ADAPTER_INIT_CHECKPOINT}" ]]; then
  echo "Init camera adapter checkpoint not found: ${CAMERA_ADAPTER_INIT_CHECKPOINT}" >&2
  exit 1
fi

mkdir -p "$(dirname "${METADATA_PATH}")" "${OUTPUT_PATH}"

echo "Generating overfit metadata..."
"${PYTHON_BIN}" "${META_SCRIPT}" \
  --data_root "${DATA_ROOT}" \
  --folders "${FOLDERS}" \
  --frames_per_folder "${FRAMES_PER_FOLDER}" \
  --angles "${ANGLES_CSV}" \
  --base_azimuth "${BASE_AZIMUTH}" \
  --base_elevation "${BASE_ELEVATION}" \
  --output_csv "${METADATA_PATH}" \
  --prompt "${PROMPT}"

echo "Starting camera adapter overfit training..."
echo "MODEL_SIZE=${MODEL_SIZE} MODEL_TYPE=${MODEL_TYPE}"
echo "TRANSFORMER_ID=${TRANSFORMER_ID}"
echo "TEXT_ENCODER_ID=${TEXT_ENCODER_ID}"
echo "METADATA_PATH=${METADATA_PATH}"
echo "OUTPUT_PATH=${OUTPUT_PATH}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NUM_PROCESSES=${NUM_PROCESSES} ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-<none>}"
echo "CAMERA_ADAPTER_INIT_CHECKPOINT=${CAMERA_ADAPTER_INIT_CHECKPOINT:-<none>}"
echo "TRAINABLE_MODELS=${TRAINABLE_MODELS}"
echo "USE_GRADIENT_CHECKPOINTING_OFFLOAD=${USE_GRADIENT_CHECKPOINTING_OFFLOAD}"
echo "GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "SAVE_STEPS=${SAVE_STEPS}"
echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"

launch_args=(--num_processes "${NUM_PROCESSES}")
if [[ -n "${ACCELERATE_CONFIG}" ]]; then
  launch_args+=(--config_file "${ACCELERATE_CONFIG}")
fi

model_id_with_origin_paths="${TEXT_ENCODER_ID}:text_encoder/*.safetensors,${TRANSFORMER_ID}:transformer/*.safetensors,${TEXT_ENCODER_ID}:vae/diffusion_pytorch_model.safetensors"
train_args=(
  --dataset_base_path "${DATA_ROOT}"
  --dataset_metadata_path "${METADATA_PATH}"
  --data_file_keys "image,edit_image"
  --extra_inputs "edit_image"
  --max_pixels "${MAX_PIXELS}"
  --dataset_repeat "${DATASET_REPEAT}"
  --batch_size "${BATCH_SIZE}"
  --model_id_with_origin_paths "${model_id_with_origin_paths}"
  --tokenizer_path "${TOKENIZER_ID}:tokenizer/"
  --learning_rate "${LEARNING_RATE}"
  --num_epochs "${NUM_EPOCHS}"
  --remove_prefix_in_ckpt "pipe.camera_adapter."
  --output_path "${OUTPUT_PATH}"
  --trainable_models "${TRAINABLE_MODELS}"
  --camera_key_mode "delta"
  --camera_scale 1.0
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --use_gradient_checkpointing
)

save_steps_lc="${SAVE_STEPS,,}"
if [[ -n "${SAVE_STEPS}" && "${save_steps_lc}" != "none" ]]; then
  train_args+=(--save_steps "${SAVE_STEPS}")
else
  echo "Checkpoint mode: epoch-only (no --save_steps)"
fi

if [[ -n "${CAMERA_ADAPTER_INIT_CHECKPOINT}" ]]; then
  # Local checkpoint paths must go through --model_paths, not --model_id_with_origin_paths.
  model_paths_json="$("${PYTHON_BIN}" - <<PY
import json
print(json.dumps(["${CAMERA_ADAPTER_INIT_CHECKPOINT}"]))
PY
)"
  train_args+=(--model_paths "${model_paths_json}")
fi
if [[ "${USE_GRADIENT_CHECKPOINTING_OFFLOAD}" == "true" ]]; then
  train_args+=(--use_gradient_checkpointing_offload)
fi

"${ACCELERATE_BIN}" launch "${launch_args[@]}" "${TRAIN_SCRIPT}" "${train_args[@]}"

echo "Done. Checkpoints in: ${OUTPUT_PATH}"
