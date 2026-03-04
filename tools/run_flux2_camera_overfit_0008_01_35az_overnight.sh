#!/usr/bin/env bash
set -euo pipefail

# Two-stage overnight overfit schedule:
# 1) Warm-up (small LR, short epochs)
# 2) Continue from latest stage-1 checkpoint (higher LR, longer epochs)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_SCRIPT="${ROOT_DIR}/tools/run_flux2_camera_overfit_0008_01_35az.sh"

if [[ ! -f "${BASE_SCRIPT}" ]]; then
  echo "Base script not found: ${BASE_SCRIPT}" >&2
  exit 1
fi

RUN_TAG="${RUN_TAG:-overnight_0008_01_35az}"
BASE_OUT_DIR="${BASE_OUT_DIR:-${ROOT_DIR}/models/train/FLUX.2-klein-4B-camera-adapter-overfit-${RUN_TAG}}"
STAGE1_OUT="${STAGE1_OUT:-${BASE_OUT_DIR}/stage1_lr1e4_e3}"
STAGE2_OUT="${STAGE2_OUT:-${BASE_OUT_DIR}/stage2_lr2e4_e7}"

LEARNING_RATE_STAGE1="${LEARNING_RATE_STAGE1:-1e-4}"
NUM_EPOCHS_STAGE1="${NUM_EPOCHS_STAGE1:-3}"
LEARNING_RATE_STAGE2="${LEARNING_RATE_STAGE2:-5e-6}"
NUM_EPOCHS_STAGE2="${NUM_EPOCHS_STAGE2:-7}"
TRAINABLE_MODELS_STAGE1="${TRAINABLE_MODELS_STAGE1:-camera_adapter}"
TRAINABLE_MODELS_STAGE2="${TRAINABLE_MODELS_STAGE2:-camera_adapter,dit}"
MAX_PIXELS_STAGE1="${MAX_PIXELS_STAGE1:-786432}"
MAX_PIXELS_STAGE2="${MAX_PIXELS_STAGE2:-524288}"
ACCELERATE_CONFIG_STAGE1="${ACCELERATE_CONFIG_STAGE1:-}"
ACCELERATE_CONFIG_STAGE2="${ACCELERATE_CONFIG_STAGE2:-${ROOT_DIR}/tools/accelerate_config_flux2_full_4gpu_zero2_cpuoffload.yaml}"
USE_GRADIENT_CHECKPOINTING_OFFLOAD_STAGE1="${USE_GRADIENT_CHECKPOINTING_OFFLOAD_STAGE1:-false}"
USE_GRADIENT_CHECKPOINTING_OFFLOAD_STAGE2="${USE_GRADIENT_CHECKPOINTING_OFFLOAD_STAGE2:-true}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5}"
export NUM_PROCESSES="${NUM_PROCESSES:-4}"

echo "=== Overnight camera-overfit schedule ==="
echo "Stage1: LR=${LEARNING_RATE_STAGE1}, epochs=${NUM_EPOCHS_STAGE1}, trainable=${TRAINABLE_MODELS_STAGE1}, max_pixels=${MAX_PIXELS_STAGE1}, offload=${USE_GRADIENT_CHECKPOINTING_OFFLOAD_STAGE1}, out=${STAGE1_OUT}"
echo "Stage2: LR=${LEARNING_RATE_STAGE2}, epochs=${NUM_EPOCHS_STAGE2}, trainable=${TRAINABLE_MODELS_STAGE2}, max_pixels=${MAX_PIXELS_STAGE2}, offload=${USE_GRADIENT_CHECKPOINTING_OFFLOAD_STAGE2}, out=${STAGE2_OUT}"
echo "Stage2 accelerate config: ${ACCELERATE_CONFIG_STAGE2:-<none>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, NUM_PROCESSES=${NUM_PROCESSES}"
echo "========================================="

echo "[1/2] Running warm-up stage..."
LEARNING_RATE="${LEARNING_RATE_STAGE1}" \
NUM_EPOCHS="${NUM_EPOCHS_STAGE1}" \
TRAINABLE_MODELS="${TRAINABLE_MODELS_STAGE1}" \
MAX_PIXELS="${MAX_PIXELS_STAGE1}" \
ACCELERATE_CONFIG="${ACCELERATE_CONFIG_STAGE1}" \
USE_GRADIENT_CHECKPOINTING_OFFLOAD="${USE_GRADIENT_CHECKPOINTING_OFFLOAD_STAGE1}" \
OUTPUT_PATH="${STAGE1_OUT}" \
bash "${BASE_SCRIPT}" "$@"

resume_ckpt="$(find "${STAGE1_OUT}" -maxdepth 1 -type f -name 'epoch-*.safetensors' | sort -V | tail -n1 || true)"
if [[ -z "${resume_ckpt}" ]]; then
  resume_ckpt="$(find "${STAGE1_OUT}" -maxdepth 1 -type f -name 'step-*.safetensors' | sort -V | tail -n1 || true)"
fi
if [[ -z "${resume_ckpt}" ]]; then
  echo "No checkpoint found in stage1 output: ${STAGE1_OUT}" >&2
  exit 1
fi
echo "Stage1 resume checkpoint: ${resume_ckpt}"

echo "[2/2] Running continued stage..."
LEARNING_RATE="${LEARNING_RATE_STAGE2}" \
NUM_EPOCHS="${NUM_EPOCHS_STAGE2}" \
TRAINABLE_MODELS="${TRAINABLE_MODELS_STAGE2}" \
MAX_PIXELS="${MAX_PIXELS_STAGE2}" \
ACCELERATE_CONFIG="${ACCELERATE_CONFIG_STAGE2}" \
USE_GRADIENT_CHECKPOINTING_OFFLOAD="${USE_GRADIENT_CHECKPOINTING_OFFLOAD_STAGE2}" \
OUTPUT_PATH="${STAGE2_OUT}" \
CAMERA_ADAPTER_INIT_CHECKPOINT="${resume_ckpt}" \
bash "${BASE_SCRIPT}" "$@"

echo "Overnight schedule complete."
echo "Stage1 output: ${STAGE1_OUT}"
echo "Stage2 output: ${STAGE2_OUT}"
