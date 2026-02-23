#!/usr/bin/env bash
set -euo pipefail

# Pose-stability subset test with depth-aware noise injection.
#
# Defaults:
# - sequence: data/0008_01
# - frames: 44,68,82,96,112,126
# - reference RGB view: view_0016_az160.0_el0.0.png
# - depth view per target:
#   right90 -> az70.0, left90 -> az250.0, back -> az340.0
#
# Example:
#   bash tools/run_flux2_klein_pose_stability_subset_depth_noise.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"

CONDA_ENV="${CONDA_ENV:-diffsynth}"
SEQ_DIR="${SEQ_DIR:-/home/ren/lin/nas-lin-233/Code/DiffSynth-Studio/data/0008_01}"
FRAME_IDS="${FRAME_IDS:-44,68,82,96,112,126}"
REF_VIEW_FILE="${REF_VIEW_FILE:-view_0016_az160.0_el0.0.png}"
RIGHT90_DEPTH_AZ="${RIGHT90_DEPTH_AZ:-70.0}"
LEFT90_DEPTH_AZ="${LEFT90_DEPTH_AZ:-250.0}"
BACK_DEPTH_AZ="${BACK_DEPTH_AZ:-340.0}"
DEPTH_EL="${DEPTH_EL:-0.0}"
DEPTH_VIEW_FILE_FALLBACK="${DEPTH_VIEW_FILE_FALLBACK:-${REF_VIEW_FILE}}"
INFER_SCRIPT="${INFER_SCRIPT:-${ROOT_DIR}/examples/flux2/model_inference/FLUX.2-klein-9B-img2img-depth-noise.py}"
MODEL_ID="${MODEL_ID:-black-forest-labs/FLUX.2-klein-9B}"
STEPS="${STEPS:-4}"
SEED="${SEED:-0}"
DEPTH_NOISE_BLEND="${DEPTH_NOISE_BLEND:-0.25}"
DEPTH_NOISE_MODULATION="${DEPTH_NOISE_MODULATION:-1.0}"
DEPTH_NOISE_INVERT="${DEPTH_NOISE_INVERT:-false}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs/pose_stability_depth_noise_subset_0008_01}"

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

# declare -a PROMPT_SPECS=(
#   "left90|Same person and exact same pose as the reference image. Keep the full-body silhouette, head tilt, shoulder line, arm and hand positions, body proportions, hanfu structure, embroidery details, fabric folds, hairstyle, and colors unchanged. Keep black background. Only rotate camera viewpoint to left profile, 90 degrees."
#   "right90|Same person and exact same pose as the reference image. Keep the full-body silhouette, head tilt, shoulder line, arm and hand positions, body proportions, hanfu structure, embroidery details, fabric folds, hairstyle, and colors unchanged. Keep black background. Only rotate camera viewpoint to right profile, 90 degrees."
#   "back|Same person and exact same pose as the reference image. Keep the full-body silhouette, head tilt, shoulder line, arm and hand positions, body proportions, hanfu structure, embroidery details, fabric folds, hairstyle, and colors unchanged. Keep black background. Only rotate camera viewpoint to back view, 180 degrees."
# )

declare -a PROMPT_SPECS=(
  "left90| Left view of a woman in traditional Chinese hanfu dress, seen from left side, 90 degree apart, black background, same pose as reference view"
  "right90| Right view of a woman in traditional Chinese hanfu dress, seen from right side, 90 degree apart, black background, same pose as reference view"
  "back| Back view of a woman in traditional Chinese hanfu dress, seen from back side, 180 degree apart, black background, same pose as reference view"
)

mkdir -p "${OUT_ROOT}"
INPUT_DUMP_DIR="${OUT_ROOT}/input"
DEPTH_DUMP_DIR="${OUT_ROOT}/depth"
mkdir -p "${INPUT_DUMP_DIR}" "${DEPTH_DUMP_DIR}"

echo "Sequence dir: ${SEQ_DIR}"
echo "Frame IDs: ${FRAME_IDS}"
echo "RGB view file: ${REF_VIEW_FILE}"
echo "Depth azimuth map: right90=${RIGHT90_DEPTH_AZ}, left90=${LEFT90_DEPTH_AZ}, back=${BACK_DEPTH_AZ}, el=${DEPTH_EL}"
echo "Output root: ${OUT_ROOT}"
echo "Model: ${MODEL_ID} | steps=${STEPS} seed=${SEED} | env=${CONDA_ENV}"
echo "Depth noise: blend=${DEPTH_NOISE_BLEND} modulation=${DEPTH_NOISE_MODULATION} invert=${DEPTH_NOISE_INVERT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "DIFFSYNTH_SKIP_DOWNLOAD=${DIFFSYNTH_SKIP_DOWNLOAD}"

total=$(( ${#FRAME_LIST[@]} * ${#PROMPT_SPECS[@]} ))
done_count=0
missing_count=0
shopt -s nullglob

for frame_id_raw in "${FRAME_LIST[@]}"; do
  frame_id="$(echo "${frame_id_raw}" | xargs)"
  frame_dir="${SEQ_DIR}/frame_${frame_id}"
  input_image="${frame_dir}/rgb/${REF_VIEW_FILE}"

  if [[ ! -f "${input_image}" ]]; then
    echo "[WARN] Missing RGB image: ${input_image}" >&2
    missing_count=$((missing_count + 1))
    continue
  fi

  cp -f "${input_image}" "${INPUT_DUMP_DIR}/frame_${frame_id}.png"

  for spec in "${PROMPT_SPECS[@]}"; do
    label="${spec%%|*}"
    prompt="${spec#*|}"

    case "${label}" in
      right90) target_depth_az="${RIGHT90_DEPTH_AZ}" ;;
      left90) target_depth_az="${LEFT90_DEPTH_AZ}" ;;
      back) target_depth_az="${BACK_DEPTH_AZ}" ;;
      *) target_depth_az="" ;;
    esac

    if [[ -n "${target_depth_az}" ]]; then
      depth_candidates=( "${frame_dir}/depth/"*"_az${target_depth_az}_el${DEPTH_EL}.png" )
      if [[ "${#depth_candidates[@]}" -gt 0 ]]; then
        depth_image="${depth_candidates[0]}"
      else
        depth_image="${frame_dir}/depth/${DEPTH_VIEW_FILE_FALLBACK}"
      fi
    else
      depth_image="${frame_dir}/depth/${DEPTH_VIEW_FILE_FALLBACK}"
    fi

    if [[ ! -f "${depth_image}" ]]; then
      echo "[WARN] Missing depth image for ${label}: ${depth_image}" >&2
      missing_count=$((missing_count + 1))
      continue
    fi

    out_dir="${OUT_ROOT}/${label}"
    out_path="${out_dir}/frame_${frame_id}.png"
    mkdir -p "${out_dir}"
    cp -f "${depth_image}" "${DEPTH_DUMP_DIR}/frame_${frame_id}_${label}.png"

    done_count=$((done_count + 1))
    echo "[${done_count}/${total}] frame_${frame_id} -> ${label} (depth: $(basename "${depth_image}"))"

    cmd=(
      conda run -n "${CONDA_ENV}" python "${INFER_SCRIPT}"
      --input_image "${input_image}"
      --depth_image "${depth_image}"
      --prompt "${prompt}"
      --output "${out_path}"
      --seed "${SEED}"
      --steps "${STEPS}"
      --model_id "${MODEL_ID}"
      --depth_noise_blend "${DEPTH_NOISE_BLEND}"
      --depth_noise_modulation "${DEPTH_NOISE_MODULATION}"
    )

    if [[ "${DEPTH_NOISE_INVERT}" == "true" ]]; then
      cmd+=(--depth_noise_invert)
    fi

    "${cmd[@]}"
  done
done

echo "Completed. Outputs saved to: ${OUT_ROOT}"
if [[ "${missing_count}" -gt 0 ]]; then
  echo "Missing inputs in ${missing_count} frame(s)." >&2
fi
