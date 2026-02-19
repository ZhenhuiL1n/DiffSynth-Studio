#!/usr/bin/env bash
set -euo pipefail

# Pose-stability stress test on a hard subset of frames.
#
# Default target sequence and subset:
#   data/0008_01, frames: 44,68,82,96,112,126
#
# Example:
#   bash tools/run_flux2_klein_pose_stability_subset.sh
#
# Optional overrides:
#   CUDA_VISIBLE_DEVICES=0 \
#   CONDA_ENV=diffsynth \
#   SEQ_DIR=/home/ren/lin/nas-lin-233/Code/DiffSynth-Studio/data/0008_01 \
#   FRAME_IDS=44,68,82,96,112,126 \
#   STEPS=4 SEED=0 \
#   OUT_ROOT=outputs/pose_stability_enhanced_prompt \
#   bash tools/run_flux2_klein_pose_stability_subset.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CONDA_ENV="${CONDA_ENV:-diffsynth}"
SEQ_DIR="${SEQ_DIR:-/home/ren/lin/nas-lin-233/Code/DiffSynth-Studio/data/0008_01}"
FRAME_IDS="${FRAME_IDS:-44,68,82,96,112,126}"
REF_VIEW_FILE="${REF_VIEW_FILE:-view_0016_az160.0_el0.0.png}"
INFER_SCRIPT="${INFER_SCRIPT:-${ROOT_DIR}/examples/flux2/model_inference/FLUX.2-klein-9B-img2img.py}"
MODEL_ID="${MODEL_ID:-black-forest-labs/FLUX.2-klein-9B}"
STEPS="${STEPS:-4}"
SEED="${SEED:-0}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs/pose_stability_enhanced_prompt}"

if [[ ! -f "${INFER_SCRIPT}" ]]; then
  echo "Inference script not found: ${INFER_SCRIPT}" >&2
  exit 1
fi
if [[ ! -d "${SEQ_DIR}" ]]; then
  echo "Sequence directory not found: ${SEQ_DIR}" >&2
  exit 1
fi

IFS=',' read -r -a FRAME_LIST <<< "${FRAME_IDS}"
if [[ "${#FRAME_LIST[@]}" -eq 0 ]]; then
  echo "No frame IDs provided in FRAME_IDS." >&2
  exit 1
fi

# Strong pose-lock template:
# - keep exact pose and silhouette
# - only change viewpoint
declare -a PROMPT_SPECS=(
  "left90|Same person and exact same pose as the reference image. Keep the full-body silhouette, head tilt, shoulder line, arm and hand positions, body proportions, hanfu structure, embroidery details, fabric folds, hairstyle, and colors unchanged. Keep black background. Only rotate camera viewpoint to left profile, 90 degrees."
  "right90|Same person and exact same pose as the reference image. Keep the full-body silhouette, head tilt, shoulder line, arm and hand positions, body proportions, hanfu structure, embroidery details, fabric folds, hairstyle, and colors unchanged. Keep black background. Only rotate camera viewpoint to right profile, 90 degrees."
  "back|Same person and exact same pose as the reference image. Keep the full-body silhouette, head tilt, shoulder line, arm and hand positions, body proportions, hanfu structure, embroidery details, fabric folds, hairstyle, and colors unchanged. Keep black background. Only rotate camera viewpoint to back view, 180 degrees."
)

mkdir -p "${OUT_ROOT}"
INPUT_DUMP_DIR="${OUT_ROOT}/input"
mkdir -p "${INPUT_DUMP_DIR}"

echo "Sequence dir: ${SEQ_DIR}"
echo "Frame IDs: ${FRAME_IDS}"
echo "Reference view: ${REF_VIEW_FILE}"
echo "Output root: ${OUT_ROOT}"
echo "Model: ${MODEL_ID} | steps=${STEPS} seed=${SEED} | env=${CONDA_ENV}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

total=$(( ${#FRAME_LIST[@]} * ${#PROMPT_SPECS[@]} ))
done_count=0
missing_count=0

for frame_id_raw in "${FRAME_LIST[@]}"; do
  frame_id="$(echo "${frame_id_raw}" | xargs)"
  frame_dir="${SEQ_DIR}/frame_${frame_id}"
  input_image="${frame_dir}/rgb/${REF_VIEW_FILE}"
  if [[ ! -f "${input_image}" ]]; then
    echo "[WARN] Missing reference image: ${input_image}" >&2
    missing_count=$((missing_count + 1))
    continue
  fi

  # Save input reference image alongside outputs for visual comparison.
  cp -f "${input_image}" "${INPUT_DUMP_DIR}/frame_${frame_id}.png"

  for spec in "${PROMPT_SPECS[@]}"; do
    label="${spec%%|*}"
    prompt="${spec#*|}"
    out_dir="${OUT_ROOT}/${label}"
    out_path="${out_dir}/frame_${frame_id}.png"
    mkdir -p "${out_dir}"

    done_count=$((done_count + 1))
    echo "[${done_count}/${total}] frame_${frame_id} -> ${label}"

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
  echo "Missing inputs in ${missing_count} frame(s)." >&2
fi
