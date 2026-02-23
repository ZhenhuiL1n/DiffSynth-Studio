#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3,4,5,6}"
export SEQUENCE_START="${SEQUENCE_START:-0008_01}"
export SEQUENCE_END="${SEQUENCE_END:-0133_07}"
export METADATA_PATH="${METADATA_PATH:-${ROOT_DIR}/data/back_view_dataset/metadata_0008_01_to_0133_07.csv}"
export OUTPUT_PATH="${OUTPUT_PATH:-${ROOT_DIR}/models/train/FLUX.2-klein-9B-backview-lora-0008_01_to_0133_07}"

bash "${ROOT_DIR}/tools/train_flux2_klein_9b_backview_lora.sh" "$@"
