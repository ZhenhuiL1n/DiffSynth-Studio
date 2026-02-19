#!/usr/bin/env bash
set -euo pipefail

# Prompt-only vanilla FLUX.2-klein img2img sweep on all frames of one sequence.
#
# Defaults are set for your current experiment:
# - sequence: data/0008_01
# - reference view: view_0016_az160.0_el0.0.png
# - prompts: left90 / right90 / back
#
# Example:
#   bash tools/run_flux2_klein_vanilla_rotation_eval.sh
#
# Optional overrides:
#   SEQ_DIR=/abs/path/to/data/0008_01 \
#   OUT_ROOT=outputs/vanilla_prompt_0008_01 \
#   CUDA_VISIBLE_DEVICES=0 \
#   STEPS=4 SEED=0 \
#   MODEL_ID=black-forest-labs/FLUX.2-klein-9B \
#   bash tools/run_flux2_klein_vanilla_rotation_eval.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-diffsynth}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SEQ_DIR="${SEQ_DIR:-/home/ren/lin/nas-lin-233/Code/DiffSynth-Studio/data/0008_01}"
REF_VIEW_FILE="${REF_VIEW_FILE:-view_0016_az160.0_el0.0.png}"
INFER_SCRIPT="${INFER_SCRIPT:-${ROOT_DIR}/examples/flux2/model_inference/FLUX.2-klein-9B-img2img.py}"
MODEL_ID="${MODEL_ID:-black-forest-labs/FLUX.2-klein-9B}"
STEPS="${STEPS:-4}"
SEED="${SEED:-0}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs/vanilla_prompt_rotation_0008_01}"

if [[ ! -f "${INFER_SCRIPT}" ]]; then
  echo "Inference script not found: ${INFER_SCRIPT}" >&2
  exit 1
fi
if [[ ! -d "${SEQ_DIR}" ]]; then
  echo "Sequence directory not found: ${SEQ_DIR}" >&2
  exit 1
fi

mapfile -t FRAME_DIRS < <(find "${SEQ_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'frame_*' | sort -V)
if [[ "${#FRAME_DIRS[@]}" -eq 0 ]]; then
  echo "No frame_* directories found under ${SEQ_DIR}" >&2
  exit 1
fi

declare -a PROMPT_SPECS=(
  "left90| Left view of a woman in traditional Chinese hanfu dress, seen from left side, 90 degree apart, black background, same pose as reference view"
  "right90| Right view of a woman in traditional Chinese hanfu dress, seen from right side, 90 degree apart, black background, same pose as reference view"
  "back| Back view of a woman in traditional Chinese hanfu dress, seen from back side, 180 degree apart, black background, same pose as reference view"
)

mkdir -p "${OUT_ROOT}"

echo "Sequence dir: ${SEQ_DIR}"
echo "Frames: ${#FRAME_DIRS[@]}"
echo "Reference view: ${REF_VIEW_FILE}"
echo "Output root: ${OUT_ROOT}"
echo "Model: ${MODEL_ID} | steps=${STEPS} seed=${SEED} | env=${CONDA_ENV}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

total=$(( ${#FRAME_DIRS[@]} * ${#PROMPT_SPECS[@]} ))
done_count=0
missing_count=0

for frame_dir in "${FRAME_DIRS[@]}"; do
  frame_name="$(basename "${frame_dir}")"
  input_image="${frame_dir}/rgb/${REF_VIEW_FILE}"
  if [[ ! -f "${input_image}" ]]; then
    echo "[WARN] Missing input for ${frame_name}: ${input_image}" >&2
    missing_count=$((missing_count + 1))
    continue
  fi

  for spec in "${PROMPT_SPECS[@]}"; do
    label="${spec%%|*}"
    prompt="${spec#*|}"
    out_dir="${OUT_ROOT}/${label}"
    out_path="${out_dir}/${frame_name}.png"
    mkdir -p "${out_dir}"

    done_count=$((done_count + 1))
    echo "[${done_count}/${total}] ${frame_name} -> ${label}"

    conda run -n "${CONDA_ENV}" python "${INFER_SCRIPT}" \
      --input_image "${input_image}" \
      --prompt "${prompt}" \
      --output "${out_path}" \
      --seed "${SEED}" \
      --steps "${STEPS}" \
      --model_id "${MODEL_ID}"
  done
done

echo "Completed. Outputs saved to: ${OUT_ROOT}"
if [[ "${missing_count}" -gt 0 ]]; then
  echo "Missing reference images in ${missing_count} frame(s)." >&2
fi
