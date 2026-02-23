#!/usr/bin/env bash
set -euo pipefail

# Sequence-level evaluation grid:
# left=input view, middle=LoRA output, right=GT back view.
#
# Outputs:
#   ${OUT_ROOT}/input/frame_x.png
#   ${OUT_ROOT}/pred/frame_x.png
#   ${OUT_ROOT}/gt/frame_x.png
#   ${OUT_ROOT}/grid/frame_x.png
#   ${OUT_ROOT}/grid_compare.mp4 (optional)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
INFER_SCRIPT="${INFER_SCRIPT:-${ROOT_DIR}/examples/flux2/model_inference/FLUX.2-klein-9B-img2img.py}"

SEQ_DIR="${SEQ_DIR:-${ROOT_DIR}/data/0008_01}"
REF_VIEW_FILE="${REF_VIEW_FILE:-view_0016_az160.0_el0.0.png}"
GT_VIEW_FILE="${GT_VIEW_FILE:-view_0034_az340.0_el0.0.png}"
PROMPT="${PROMPT:-Back view of a woman in traditional Chinese hanfu dress, black background}"

MODEL_ID="${MODEL_ID:-black-forest-labs/FLUX.2-klein-9B}"
LORA_PATH="${LORA_PATH:-${ROOT_DIR}/models/train/FLUX.2-klein-9B-backview-lora/epoch-4.safetensors}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs/eval_backview_grid_0008_01}"

DEVICE="${DEVICE:-cuda}"
MODE="${MODE:-input}"
DENOISING_STRENGTH="${DENOISING_STRENGTH:-0.15}"
CFG_SCALE="${CFG_SCALE:-2.0}"
STEPS="${STEPS:-4}"
SEED="${SEED:-42}"

DRAW_LABELS="${DRAW_LABELS:-true}"
MAKE_VIDEO="${MAKE_VIDEO:-true}"
FPS="${FPS:-12}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"
MAX_FRAMES="${MAX_FRAMES:-0}" # 0 means all

if [[ ! -f "${INFER_SCRIPT}" ]]; then
  echo "Inference script not found: ${INFER_SCRIPT}" >&2
  exit 1
fi
if [[ ! -d "${SEQ_DIR}" ]]; then
  echo "Sequence directory not found: ${SEQ_DIR}" >&2
  exit 1
fi
if [[ ! -f "${LORA_PATH}" ]]; then
  echo "LoRA checkpoint not found: ${LORA_PATH}" >&2
  exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found in PATH." >&2
  exit 1
fi

mapfile -t FRAME_DIRS < <(find "${SEQ_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'frame_*' | sort -V)
if [[ "${#FRAME_DIRS[@]}" -eq 0 ]]; then
  echo "No frame_* directories found under ${SEQ_DIR}" >&2
  exit 1
fi
if [[ "${MAX_FRAMES}" -gt 0 && "${MAX_FRAMES}" -lt "${#FRAME_DIRS[@]}" ]]; then
  FRAME_DIRS=("${FRAME_DIRS[@]:0:${MAX_FRAMES}}")
fi

mkdir -p "${OUT_ROOT}/input" "${OUT_ROOT}/pred" "${OUT_ROOT}/gt" "${OUT_ROOT}/grid"

echo "Running sequence grid evaluation"
echo "SEQ_DIR=${SEQ_DIR}"
echo "REF_VIEW_FILE=${REF_VIEW_FILE}"
echo "GT_VIEW_FILE=${GT_VIEW_FILE}"
echo "MODEL_ID=${MODEL_ID}"
echo "LORA_PATH=${LORA_PATH}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "DEVICE=${DEVICE} MODE=${MODE} DENOISING_STRENGTH=${DENOISING_STRENGTH} CFG_SCALE=${CFG_SCALE} STEPS=${STEPS} SEED=${SEED}"
echo "frames=${#FRAME_DIRS[@]} DRAW_LABELS=${DRAW_LABELS} MAKE_VIDEO=${MAKE_VIDEO} FPS=${FPS}"

done_count=0
missing_count=0

for frame_dir in "${FRAME_DIRS[@]}"; do
  frame_name="$(basename "${frame_dir}")"
  input_image="${frame_dir}/rgb/${REF_VIEW_FILE}"
  gt_image="${frame_dir}/rgb/${GT_VIEW_FILE}"

  if [[ ! -f "${input_image}" ]]; then
    echo "[WARN] Missing input image for ${frame_name}: ${input_image}" >&2
    missing_count=$((missing_count + 1))
    continue
  fi
  if [[ ! -f "${gt_image}" ]]; then
    echo "[WARN] Missing GT image for ${frame_name}: ${gt_image}" >&2
    missing_count=$((missing_count + 1))
    continue
  fi

  out_input="${OUT_ROOT}/input/${frame_name}.png"
  out_pred="${OUT_ROOT}/pred/${frame_name}.png"
  out_gt="${OUT_ROOT}/gt/${frame_name}.png"
  out_grid="${OUT_ROOT}/grid/${frame_name}.png"

  cp -f "${input_image}" "${out_input}"
  cp -f "${gt_image}" "${out_gt}"

  done_count=$((done_count + 1))
  echo "[${done_count}/${#FRAME_DIRS[@]}] ${frame_name}"

  if [[ "${SKIP_EXISTING}" != "true" || ! -f "${out_pred}" ]]; then
    "${PYTHON_BIN}" "${INFER_SCRIPT}" \
      --input_image "${input_image}" \
      --prompt "${PROMPT}" \
      --output "${out_pred}" \
      --model_id "${MODEL_ID}" \
      --lora_path "${LORA_PATH}" \
      --device "${DEVICE}" \
      --mode "${MODE}" \
      --denoising_strength "${DENOISING_STRENGTH}" \
      --cfg_scale "${CFG_SCALE}" \
      --steps "${STEPS}" \
      --seed "${SEED}"
  fi

  if [[ "${DRAW_LABELS}" == "true" ]]; then
    ffmpeg -y \
      -i "${out_input}" \
      -i "${out_pred}" \
      -i "${out_gt}" \
      -filter_complex "\
[0:v]drawtext=text='Input':x=20:y=20:fontsize=42:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=10[v0];\
[1:v]drawtext=text='LoRA':x=20:y=20:fontsize=42:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=10[v1];\
[2:v]drawtext=text='GT':x=20:y=20:fontsize=42:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=10[v2];\
[v0][v1][v2]hstack=inputs=3[v]" \
      -map "[v]" "${out_grid}" >/dev/null 2>&1
  else
    ffmpeg -y \
      -i "${out_input}" \
      -i "${out_pred}" \
      -i "${out_gt}" \
      -filter_complex "[0:v][1:v][2:v]hstack=inputs=3[v]" \
      -map "[v]" "${out_grid}" >/dev/null 2>&1
  fi
done

if [[ "${MAKE_VIDEO}" == "true" ]]; then
  ffmpeg -y -framerate "${FPS}" -i "${OUT_ROOT}/grid/frame_%d.png" \
    -c:v libx264 -pix_fmt yuv420p "${OUT_ROOT}/grid_compare.mp4" >/dev/null 2>&1 || true
fi

echo "Done. Grid frames: ${OUT_ROOT}/grid"
if [[ "${MAKE_VIDEO}" == "true" ]]; then
  echo "Video (if frame index is contiguous): ${OUT_ROOT}/grid_compare.mp4"
fi
if [[ "${missing_count}" -gt 0 ]]; then
  echo "Missing inputs in ${missing_count} frame(s)." >&2
fi
