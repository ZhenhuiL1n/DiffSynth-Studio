#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3,4,5,6}"
export SEQUENCE_START="${SEQUENCE_START:-0008_01}"
export SEQUENCE_END="${SEQUENCE_END:-0133_07}"
export MODEL_ID="${MODEL_ID:-black-forest-labs/FLUX.2-klein-4B}"
export METADATA_PATH="${METADATA_PATH:-${ROOT_DIR}/data/back_view_dataset/metadata_full_0008_01_to_0133_07.csv}"
export OUTPUT_PATH="${OUTPUT_PATH:-${ROOT_DIR}/models/train/FLUX.2-klein-4B-backview-full-0008_01_to_0133_07}"
export LEARNING_RATE="${LEARNING_RATE:-5e-6}"
export MAX_PIXELS="${MAX_PIXELS:-786432}"
export ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${ROOT_DIR}/tools/accelerate_config_flux2_full_4gpu_zero3.yaml}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export USE_GRADIENT_CHECKPOINTING_OFFLOAD="${USE_GRADIENT_CHECKPOINTING_OFFLOAD:-false}"

bash "${ROOT_DIR}/tools/train_flux2_klein_9b_backview_full.sh" "$@"
