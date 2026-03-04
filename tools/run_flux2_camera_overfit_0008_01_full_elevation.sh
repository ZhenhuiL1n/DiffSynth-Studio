#!/usr/bin/env bash
set -euo pipefail

# Two-stage camera overfit/finetune on 0008_01_full_elevation (camera head only by default):
# 1) Warm-up camera adapter
# 2) Small-LR finetune from stage1 checkpoint

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${ROOT_DIR}/examples/flux2/model_training/train_camera.py}"
META_SCRIPT="${META_SCRIPT:-${ROOT_DIR}/tools/create_flux2_camera_metadata.py}"

DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/data}"
FOLDERS="${FOLDERS:-0008_01_full_elevation}"
FRAMES_PER_FOLDER="${FRAMES_PER_FOLDER:-0}"  # 0 => use first 10 frames per folder
FRAME_SAMPLING="${FRAME_SAMPLING:-first}"    # first|evenly
ANGLES_CSV="${ANGLES_CSV:-0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350}"
ELEVATIONS_CSV="${ELEVATIONS_CSV:--40,-30,-20,-10,0,10,20,30,40}"
BASE_AZIMUTH="${BASE_AZIMUTH:-160.0}"
BASE_ELEVATION="${BASE_ELEVATION:-0.0}"
PROMPT="${PROMPT:-a woman in traditional Chinese hanfu dress, black background}"
METADATA_PATH="${METADATA_PATH:-${ROOT_DIR}/data/camera_dataset/metadata_0008_01_full_elevation.csv}"

MODEL_SIZE="${MODEL_SIZE:-4B}"        # 4B default for stability
MODEL_TYPE="${MODEL_TYPE:-distilled}" # distilled|base

RUN_TAG="${RUN_TAG:-0008_01_full_elevation}"
BASE_OUT_DIR="${BASE_OUT_DIR:-${ROOT_DIR}/models/train/FLUX.2-klein-${MODEL_SIZE}-camera-adapter-overfit-${RUN_TAG}}"
STAGE1_OUT="${STAGE1_OUT:-${BASE_OUT_DIR}/stage1_warmup}"
STAGE2_OUT="${STAGE2_OUT:-${BASE_OUT_DIR}/stage2_small_lr}"

LEARNING_RATE_STAGE1="${LEARNING_RATE_STAGE1:-1e-4}"
NUM_EPOCHS_STAGE1="${NUM_EPOCHS_STAGE1:-3}"
TRAINABLE_MODELS_STAGE1="${TRAINABLE_MODELS_STAGE1:-camera_adapter}"
DATASET_REPEAT_STAGE1="${DATASET_REPEAT_STAGE1:-1}"
MAX_PIXELS_STAGE1="${MAX_PIXELS_STAGE1:-786432}"
ACCELERATE_CONFIG_STAGE1="${ACCELERATE_CONFIG_STAGE1:-}"
USE_GRADIENT_CHECKPOINTING_OFFLOAD_STAGE1="${USE_GRADIENT_CHECKPOINTING_OFFLOAD_STAGE1:-false}"

LEARNING_RATE_STAGE2="${LEARNING_RATE_STAGE2:-5e-6}"
NUM_EPOCHS_STAGE2="${NUM_EPOCHS_STAGE2:-7}"
TRAINABLE_MODELS_STAGE2="${TRAINABLE_MODELS_STAGE2:-camera_adapter}"
DATASET_REPEAT_STAGE2="${DATASET_REPEAT_STAGE2:-1}"
MAX_PIXELS_STAGE2="${MAX_PIXELS_STAGE2:-524288}"
ACCELERATE_CONFIG_STAGE2="${ACCELERATE_CONFIG_STAGE2:-${ROOT_DIR}/tools/accelerate_config_flux2_camera_3gpu_zero2_cpuoffload.yaml}"
USE_GRADIENT_CHECKPOINTING_OFFLOAD_STAGE2="${USE_GRADIENT_CHECKPOINTING_OFFLOAD_STAGE2:-true}"

SKIP_WARMUP="${SKIP_WARMUP:-true}"
LEARNING_RATE_SINGLE="${LEARNING_RATE_SINGLE:-2e-5}"
NUM_EPOCHS_SINGLE="${NUM_EPOCHS_SINGLE:-10}"
TRAINABLE_MODELS_SINGLE="${TRAINABLE_MODELS_SINGLE:-camera_adapter}"
DATASET_REPEAT_SINGLE="${DATASET_REPEAT_SINGLE:-1}"
MAX_PIXELS_SINGLE="${MAX_PIXELS_SINGLE:-393216}"
ACCELERATE_CONFIG_SINGLE="${ACCELERATE_CONFIG_SINGLE:-${ACCELERATE_CONFIG_STAGE2}}"
USE_GRADIENT_CHECKPOINTING_OFFLOAD_SINGLE="${USE_GRADIENT_CHECKPOINTING_OFFLOAD_SINGLE:-true}"

SAVE_STEPS="${SAVE_STEPS:-200}"
CAMERA_SCALE="${CAMERA_SCALE:-1.0}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,6}"
export NUM_PROCESSES="${NUM_PROCESSES:-3}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${ROOT_DIR}/models}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${ROOT_DIR}/models/modelscope_cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"

# Keep world size aligned with selected visible GPUs unless user explicitly requests otherwise.
gpu_count="$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")"
if [[ -z "${NUM_PROCESSES}" || "${NUM_PROCESSES}" -le 0 ]]; then
  NUM_PROCESSES="${gpu_count}"
fi
if [[ "${NUM_PROCESSES}" -ne "${gpu_count}" ]]; then
  echo "Warning: NUM_PROCESSES=${NUM_PROCESSES} but CUDA_VISIBLE_DEVICES has ${gpu_count} GPUs (${CUDA_VISIBLE_DEVICES})." >&2
  echo "Overriding NUM_PROCESSES to ${gpu_count} to avoid single-GPU fallback." >&2
  NUM_PROCESSES="${gpu_count}"
fi

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "Training script not found: ${TRAIN_SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${META_SCRIPT}" ]]; then
  echo "Metadata script not found: ${META_SCRIPT}" >&2
  exit 1
fi

if [[ "${FRAMES_PER_FOLDER}" -le 0 ]]; then
  FRAMES_PER_FOLDER=10
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

mkdir -p "$(dirname "${METADATA_PATH}")" "${STAGE1_OUT}" "${STAGE2_OUT}"

echo "Generating full-elevation metadata..."
"${PYTHON_BIN}" "${META_SCRIPT}" \
  --data_root "${DATA_ROOT}" \
  --folders "${FOLDERS}" \
  --frames_per_folder "${FRAMES_PER_FOLDER}" \
  --frame_sampling "${FRAME_SAMPLING}" \
  --angles "${ANGLES_CSV}" \
  --elevations="${ELEVATIONS_CSV}" \
  --base_azimuth "${BASE_AZIMUTH}" \
  --base_elevation "${BASE_ELEVATION}" \
  --output_csv "${METADATA_PATH}" \
  --prompt "${PROMPT}"

model_id_with_origin_paths="${TEXT_ENCODER_ID}:text_encoder/*.safetensors,${TRANSFORMER_ID}:transformer/*.safetensors,${TEXT_ENCODER_ID}:vae/diffusion_pytorch_model.safetensors"

run_stage() {
  local stage_name="$1"
  local learning_rate="$2"
  local num_epochs="$3"
  local dataset_repeat="$4"
  local max_pixels="$5"
  local trainable_models="$6"
  local accelerate_config="$7"
  local use_offload="$8"
  local output_path="$9"
  local init_ckpt="${10}"

  local launch_args=(--num_processes "${NUM_PROCESSES}")
  if [[ -n "${accelerate_config}" ]]; then
    launch_args+=(--config_file "${accelerate_config}")
  elif [[ "${NUM_PROCESSES}" -gt 1 ]]; then
    launch_args+=(--multi_gpu)
  fi

  local train_args=(
    --dataset_base_path "${DATA_ROOT}"
    --dataset_metadata_path "${METADATA_PATH}"
    --data_file_keys "image,edit_image"
    --extra_inputs "edit_image"
    --batch_size "${BATCH_SIZE}"
    --max_pixels "${max_pixels}"
    --dataset_repeat "${dataset_repeat}"
    --model_id_with_origin_paths "${model_id_with_origin_paths}"
    --tokenizer_path "${TOKENIZER_ID}:tokenizer/"
    --learning_rate "${learning_rate}"
    --num_epochs "${num_epochs}"
    --save_steps "${SAVE_STEPS}"
    --remove_prefix_in_ckpt "pipe.camera_adapter."
    --output_path "${output_path}"
    --trainable_models "${trainable_models}"
    --camera_key_mode "delta"
    --camera_scale "${CAMERA_SCALE}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --use_gradient_checkpointing
  )

  if [[ -n "${init_ckpt}" ]]; then
    local model_paths_json
    model_paths_json="$("${PYTHON_BIN}" - <<PY
import json
print(json.dumps(["${init_ckpt}"]))
PY
)"cd /home/ren/lin/nas-lin-233/Code/DiffSynth-Studio

CUDA_VISIBLE_DEVICES=3 \
NUM_PROCESSES=1 \
BATCH_SIZE=3 \
FRAMES_PER_FOLDER=10 \
FRAME_SAMPLING=first \
METADATA_PATH=data/camera_dataset/metadata_0008_01_full_elevation_first10.csv \
ACCELERATE_CONFIG_SINGLE="" \
bash tools/run_flux2_camera_overfit_0008_01_full_elevation.sh
    train_args+=(--model_paths "${model_paths_json}")
  fi
  if [[ "${use_offload}" == "true" ]]; then
    train_args+=(--use_gradient_checkpointing_offload)
  fi

  echo "=== ${stage_name} ==="
  echo "output=${output_path}"
  echo "lr=${learning_rate} epochs=${num_epochs} trainable=${trainable_models}"
  echo "batch_size=${BATCH_SIZE} max_pixels=${max_pixels} dataset_repeat=${dataset_repeat} offload=${use_offload}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} NUM_PROCESSES=${NUM_PROCESSES}"
  echo "ACCELERATE_CONFIG=${accelerate_config:-<none>}"
  echo "INIT_CKPT=${init_ckpt:-<none>}"

  "${ACCELERATE_BIN}" launch "${launch_args[@]}" "${TRAIN_SCRIPT}" "${train_args[@]}"
}

if [[ "${SKIP_WARMUP}" == "true" ]]; then
  run_stage \
    "Single-stage training" \
    "${LEARNING_RATE_SINGLE}" \
    "${NUM_EPOCHS_SINGLE}" \
    "${DATASET_REPEAT_SINGLE}" \
    "${MAX_PIXELS_SINGLE}" \
    "${TRAINABLE_MODELS_SINGLE}" \
    "${ACCELERATE_CONFIG_SINGLE}" \
    "${USE_GRADIENT_CHECKPOINTING_OFFLOAD_SINGLE}" \
    "${STAGE2_OUT}" \
    ""
  echo "Done."
  echo "Metadata: ${METADATA_PATH}"
  echo "Single-stage output: ${STAGE2_OUT}"
  exit 0
fi

run_stage \
  "Stage1 warm-up" \
  "${LEARNING_RATE_STAGE1}" \
  "${NUM_EPOCHS_STAGE1}" \
  "${DATASET_REPEAT_STAGE1}" \
  "${MAX_PIXELS_STAGE1}" \
  "${TRAINABLE_MODELS_STAGE1}" \
  "${ACCELERATE_CONFIG_STAGE1}" \
  "${USE_GRADIENT_CHECKPOINTING_OFFLOAD_STAGE1}" \
  "${STAGE1_OUT}" \
  ""

resume_ckpt="$(find "${STAGE1_OUT}" -maxdepth 1 -type f -name 'epoch-*.safetensors' | sort -V | tail -n1 || true)"
if [[ -z "${resume_ckpt}" ]]; then
  resume_ckpt="$(find "${STAGE1_OUT}" -maxdepth 1 -type f -name 'step-*.safetensors' | sort -V | tail -n1 || true)"
fi
if [[ -z "${resume_ckpt}" ]]; then
  echo "No checkpoint found in stage1 output: ${STAGE1_OUT}" >&2
  exit 1
fi

run_stage \
  "Stage2 small-LR finetune" \
  "${LEARNING_RATE_STAGE2}" \
  "${NUM_EPOCHS_STAGE2}" \
  "${DATASET_REPEAT_STAGE2}" \
  "${MAX_PIXELS_STAGE2}" \
  "${TRAINABLE_MODELS_STAGE2}" \
  "${ACCELERATE_CONFIG_STAGE2}" \
  "${USE_GRADIENT_CHECKPOINTING_OFFLOAD_STAGE2}" \
  "${STAGE2_OUT}" \
  "${resume_ckpt}"

echo "Done."
echo "Metadata: ${METADATA_PATH}"
echo "Stage1 output: ${STAGE1_OUT}"
echo "Stage2 output: ${STAGE2_OUT}"
