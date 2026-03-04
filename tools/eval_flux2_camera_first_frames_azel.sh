#!/usr/bin/env bash
set -euo pipefail

# Evaluate camera-adapter checkpoints over first N frames with azimuth + elevation sweeps.
# Per frame, generates:
# - input image
# - pred images for each (delta_azimuth, target_elevation) pair
# - GT copies
# - compare panels: Input | Pred | GT
# - compare video and contact sheet

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
INFER_SCRIPT="${INFER_SCRIPT:-${ROOT_DIR}/examples/flux2/model_inference/FLUX.2-klein-camera-adapter-inference.py}"

export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${ROOT_DIR}/models}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${ROOT_DIR}/models/modelscope_cache}"

MODEL_SIZE="${MODEL_SIZE:-4B}"        # 4B|9B
MODEL_TYPE="${MODEL_TYPE:-distilled}" # distilled|base
DEVICE="${DEVICE:-cuda}"
STEPS="${STEPS:-4}"
SEED="${SEED:-0}"
CAMERA_SCALE="${CAMERA_SCALE:-1.0}"
CFG_SCALE="${CFG_SCALE:-1.0}"
EMBEDDED_GUIDANCE="${EMBEDDED_GUIDANCE:-1.0}"
PROMPT="${PROMPT:-a woman in traditional Chinese hanfu dress, black background}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"
FINETUNE_CHECKPOINT="${FINETUNE_CHECKPOINT:-}"
OVERRIDE_CAMERA_FROM_FINETUNE="${OVERRIDE_CAMERA_FROM_FINETUNE:-1}"
OVERRIDE_DIT_FROM_FINETUNE="${OVERRIDE_DIT_FROM_FINETUNE:-0}"

DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/data}"
SEQ_ID="${SEQ_ID:-0008_01_full_elevation}"
START_FRAME="${START_FRAME:-0}"
FIRST_N_FRAMES="${FIRST_N_FRAMES:-10}"
BASE_AZ_INT="${BASE_AZ_INT:-160}" # source azimuth (absolute)
BASE_EL_INT="${BASE_EL_INT:-0}"   # source elevation (absolute)
AZ_ANGLES_CSV="${AZ_ANGLES_CSV:-0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350}"
ELEVATIONS_CSV="${ELEVATIONS_CSV:--40,-30,-20,-10,0,10,20,30,40}" # target absolute elevations

TRAIN_OUTPUT_PATH="${TRAIN_OUTPUT_PATH:-${ROOT_DIR}/models/train/FLUX.2-klein-${MODEL_SIZE}-camera-adapter-overfit-0008_01-35az}"
CAMERA_ADAPTER_PATH="${CAMERA_ADAPTER_PATH:-}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs/eval_camera_first_frames_azel}"
CONTACT_SHEET_COLS="${CONTACT_SHEET_COLS:-8}"

if [[ ! -f "${INFER_SCRIPT}" ]]; then
  echo "Inference script not found: ${INFER_SCRIPT}" >&2
  exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found in PATH." >&2
  exit 1
fi

if [[ -z "${CAMERA_ADAPTER_PATH}" ]]; then
  latest="$(find "${TRAIN_OUTPUT_PATH}" -maxdepth 1 -type f -name 'epoch-*.safetensors' | sort -V | tail -n1 || true)"
  if [[ -z "${latest}" ]]; then
    latest="$(find "${TRAIN_OUTPUT_PATH}" -maxdepth 1 -type f -name 'step-*.safetensors' | sort -V | tail -n1 || true)"
  fi
  if [[ -z "${latest}" ]]; then
    echo "No checkpoint found under ${TRAIN_OUTPUT_PATH}. Set CAMERA_ADAPTER_PATH explicitly." >&2
    exit 1
  fi
  CAMERA_ADAPTER_PATH="${latest}"
fi

if [[ ! -f "${CAMERA_ADAPTER_PATH}" ]]; then
  echo "Camera adapter checkpoint not found: ${CAMERA_ADAPTER_PATH}" >&2
  exit 1
fi

find_view_image() {
  local rgb_dir="$1"
  local view_id="$2"
  local az="$3"
  local el="$4"
  local az_fmt el_fmt p ext
  az_fmt="$(printf '%.1f' "${az}")"
  el_fmt="$(printf '%.1f' "${el}")"
  for ext in png webp jpg jpeg; do
    printf -v p "%s/view_%04d_az%s_el%s.%s" "${rgb_dir}" "${view_id}" "${az_fmt}" "${el_fmt}" "${ext}"
    if [[ -f "${p}" ]]; then
      echo "${p}"
      return 0
    fi
  done
  return 1
}

signed_tag() {
  local x="$1"
  if (( x < 0 )); then
    printf "m%03d" "$(( -x ))"
  else
    printf "p%03d" "${x}"
  fi
}

echo "Camera adapter eval (first frames, az+el)"
echo "SEQ_ID=${SEQ_ID} START_FRAME=${START_FRAME} FIRST_N_FRAMES=${FIRST_N_FRAMES}"
echo "BASE_AZ_INT=${BASE_AZ_INT} BASE_EL_INT=${BASE_EL_INT}"
echo "AZ_ANGLES_CSV=${AZ_ANGLES_CSV}"
echo "ELEVATIONS_CSV=${ELEVATIONS_CSV}"
echo "CAMERA_ADAPTER_PATH=${CAMERA_ADAPTER_PATH}"
echo "OUT_ROOT=${OUT_ROOT}"

mkdir -p "${OUT_ROOT}"
printf "%s\n" "${CAMERA_ADAPTER_PATH}" > "${OUT_ROOT}/checkpoint_used.txt"

IFS=',' read -r -a AZ_ANGLES <<< "${AZ_ANGLES_CSV}"
IFS=',' read -r -a TARGET_ELEVATIONS <<< "${ELEVATIONS_CSV}"

for ((frame_id = START_FRAME; frame_id < START_FRAME + FIRST_N_FRAMES; frame_id++)); do
  frame_root="${OUT_ROOT}/frame_$(printf '%04d' "${frame_id}")"
  pred_dir="${frame_root}/pred"
  gt_dir="${frame_root}/gt"
  cmp_dir="${frame_root}/compare"
  mkdir -p "${pred_dir}" "${gt_dir}" "${cmp_dir}"

  rgb_dir="${DATA_ROOT}/${SEQ_ID}/frame_${frame_id}/rgb"
  if [[ ! -d "${rgb_dir}" ]]; then
    echo "[WARN] Missing rgb dir, skipping frame_${frame_id}: ${rgb_dir}" >&2
    continue
  fi

  source_view_id=$(( BASE_AZ_INT / 10 ))
  SOURCE_IMAGE="$(find_view_image "${rgb_dir}" "${source_view_id}" "${BASE_AZ_INT}" "${BASE_EL_INT}" || true)"
  if [[ -z "${SOURCE_IMAGE}" ]]; then
    echo "[WARN] Source image missing for frame_${frame_id}, skip." >&2
    continue
  fi
  cp -f "${SOURCE_IMAGE}" "${frame_root}/input_original.${SOURCE_IMAGE##*.}"
  ffmpeg -y -i "${SOURCE_IMAGE}" -frames:v 1 "${frame_root}/input.png" >/dev/null 2>&1

  done_count=0
  for az in "${AZ_ANGLES[@]}"; do
    az="$(echo "${az}" | xargs)"
    [[ -z "${az}" ]] && continue
    az_int="$(printf '%.0f' "${az}")"

    target_az=$(( (BASE_AZ_INT + az_int + 3600) % 360 ))
    target_view_id=$(( target_az / 10 ))
    az_tag="$(signed_tag "${az_int}")"

    for target_el in "${TARGET_ELEVATIONS[@]}"; do
      target_el="$(echo "${target_el}" | xargs)"
      [[ -z "${target_el}" ]] && continue

      target_el_int="$(printf '%.0f' "${target_el}")"
      delta_el=$(( target_el_int - BASE_EL_INT ))
      el_tag="$(signed_tag "${target_el_int}")"

      pred_out="${pred_dir}/az${az_tag}_el${el_tag}.png"
      gt_out="${gt_dir}/az${az_tag}_el${el_tag}.png"
      cmp_out="${cmp_dir}/az${az_tag}_el${el_tag}.png"

      gt_src="$(find_view_image "${rgb_dir}" "${target_view_id}" "${target_az}" "${target_el}" || true)"
      if [[ -z "${gt_src}" ]]; then
        echo "[WARN] Missing GT frame=${frame_id} az=${az} target_el=${target_el}" >&2
        continue
      fi

      done_count=$((done_count + 1))
      echo "[frame ${frame_id}] ${done_count}: az=${az_int} target_az=${target_az} target_el=${target_el_int} delta_el=${delta_el}"

      infer_args=(
        --input_image "${SOURCE_IMAGE}"
        --prompt "${PROMPT}"
        --output "${pred_out}"
        --model_size "${MODEL_SIZE}"
        --model_type "${MODEL_TYPE}"
        --camera_adapter_path "${CAMERA_ADAPTER_PATH}"
        --delta_azimuth "${az_int}"
        --delta_elevation "${delta_el}"
        --camera_scale "${CAMERA_SCALE}"
        --cfg_scale "${CFG_SCALE}"
        --embedded_guidance "${EMBEDDED_GUIDANCE}"
        --override_camera_from_finetune "${OVERRIDE_CAMERA_FROM_FINETUNE}"
        --override_dit_from_finetune "${OVERRIDE_DIT_FROM_FINETUNE}"
        --seed "${SEED}"
        --steps "${STEPS}"
        --device "${DEVICE}"
      )
      if [[ -n "${NEGATIVE_PROMPT}" ]]; then
        infer_args+=(--negative_prompt "${NEGATIVE_PROMPT}")
      fi
      if [[ -n "${FINETUNE_CHECKPOINT}" ]]; then
        infer_args+=(--finetune_checkpoint "${FINETUNE_CHECKPOINT}")
      fi
      "${PYTHON_BIN}" "${INFER_SCRIPT}" "${infer_args[@]}"

      cp -f "${gt_src}" "${gt_out}"

      ffmpeg -y \
        -i "${frame_root}/input.png" \
        -i "${pred_out}" \
        -i "${gt_out}" \
        -filter_complex "\
[0:v]drawtext=text='Input':x=20:y=20:fontsize=34:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=8[v0];\
[1:v]drawtext=text='Pred dAZ=${az_int} el=${target_el_int}':x=20:y=20:fontsize=34:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=8[v1];\
[2:v]drawtext=text='GT dAZ=${az_int} el=${target_el_int}':x=20:y=20:fontsize=34:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=8[v2];\
[v0][v1][v2]hstack=inputs=3[v]" \
        -map "[v]" "${cmp_out}" >/dev/null 2>&1
    done
  done

  ffmpeg -y -framerate 6 -pattern_type glob -i "${cmp_dir}/az*_el*.png" \
    -c:v libx264 -pix_fmt yuv420p "${frame_root}/compare_azel.mp4" >/dev/null 2>&1 || true

  "${PYTHON_BIN}" - "${frame_root}" "${CONTACT_SHEET_COLS}" <<'PY'
import math
import sys
from pathlib import Path
from PIL import Image

frame_root = Path(sys.argv[1])
cols = max(1, int(sys.argv[2]))
paths = sorted((frame_root / "compare").glob("az*_el*.png"))
if not paths:
    print(f"No compare images for {frame_root}, skip contact sheet.")
    raise SystemExit(0)

first = Image.open(paths[0]).convert("RGB")
w, h = first.size
rows = math.ceil(len(paths) / cols)
canvas = Image.new("RGB", (cols * w, rows * h), (0, 0, 0))
for i, p in enumerate(paths):
    img = Image.open(p).convert("RGB")
    if img.size != (w, h):
        img = img.resize((w, h), Image.BICUBIC)
    x = (i % cols) * w
    y = (i // cols) * h
    canvas.paste(img, (x, y))
out = frame_root / "compare_azel_grid.png"
canvas.save(out)
print(f"Saved: {out}")
PY
done

echo "Done."
echo "Output root: ${OUT_ROOT}"
