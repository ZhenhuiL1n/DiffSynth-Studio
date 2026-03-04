#!/usr/bin/env bash
set -euo pipefail

# Evaluate a camera-adapter overfit run on sequence 0008_01 for 35 azimuth deltas.
# Generates:
# - pred images for each angle
# - copied GT images
# - per-angle compare panels: Input | Pred | GT
# - compare video + contact sheet

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
INFER_SCRIPT="${INFER_SCRIPT:-${ROOT_DIR}/examples/flux2/model_inference/FLUX.2-klein-camera-adapter-inference.py}"

# Prefer offline/local model loading during long eval loops.
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${ROOT_DIR}/models}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${ROOT_DIR}/models/modelscope_cache}"

MODEL_SIZE="${MODEL_SIZE:-4B}"         # 4B|9B
MODEL_TYPE="${MODEL_TYPE:-distilled}"  # distilled|base
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
OVERRIDE_DIT_FROM_FINETUNE="${OVERRIDE_DIT_FROM_FINETUNE:-1}"

SEQ_ID="${SEQ_ID:-0008_01}"
FRAME_ID="${FRAME_ID:-0}"              # frame index used for overfit check
BASE_AZ_INT="${BASE_AZ_INT:-160}"      # source view azimuth
BASE_EL_INT="${BASE_EL_INT:-0}"        # source view elevation
ANGLES_CSV="${ANGLES_CSV:-0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340}"

DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/data}"
TRAIN_OUTPUT_PATH="${TRAIN_OUTPUT_PATH:-${ROOT_DIR}/models/train/FLUX.2-klein-${MODEL_SIZE}-camera-adapter-overfit-0008_01-35az}"
CAMERA_ADAPTER_PATH="${CAMERA_ADAPTER_PATH:-}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs/eval_camera_overfit_0008_01_35az}"

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

source_view_id=$(( BASE_AZ_INT / 10 ))
printf -v source_view_file "view_%04d_az%d.0_el%d.0.png" "${source_view_id}" "${BASE_AZ_INT}" "${BASE_EL_INT}"
SOURCE_IMAGE="${DATA_ROOT}/${SEQ_ID}/frame_${FRAME_ID}/rgb/${source_view_file}"
if [[ ! -f "${SOURCE_IMAGE}" ]]; then
  echo "Source image not found: ${SOURCE_IMAGE}" >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}/pred" "${OUT_ROOT}/gt" "${OUT_ROOT}/compare"
cp -f "${SOURCE_IMAGE}" "${OUT_ROOT}/input.png"

echo "Camera adapter eval"
echo "SEQ_ID=${SEQ_ID} FRAME_ID=${FRAME_ID}"
echo "SOURCE_IMAGE=${SOURCE_IMAGE}"
echo "CAMERA_ADAPTER_PATH=${CAMERA_ADAPTER_PATH}"
echo "OUT_ROOT=${OUT_ROOT}"

IFS=',' read -r -a ANGLES <<< "${ANGLES_CSV}"
done_count=0
for angle in "${ANGLES[@]}"; do
  angle="$(echo "${angle}" | xargs)"
  [[ -z "${angle}" ]] && continue

  target_az=$(( (BASE_AZ_INT + angle + 3600) % 360 ))
  target_view_id=$(( target_az / 10 ))
  printf -v target_view_file "view_%04d_az%d.0_el%d.0.png" "${target_view_id}" "${target_az}" "${BASE_EL_INT}"

  pred_out="${OUT_ROOT}/pred/az$(printf '%03d' "${angle}").png"
  gt_src="${DATA_ROOT}/${SEQ_ID}/frame_${FRAME_ID}/rgb/${target_view_file}"
  gt_out="${OUT_ROOT}/gt/az$(printf '%03d' "${angle}").png"
  cmp_out="${OUT_ROOT}/compare/az$(printf '%03d' "${angle}").png"

  if [[ ! -f "${gt_src}" ]]; then
    echo "[WARN] Missing GT for az=${angle}: ${gt_src}" >&2
    continue
  fi

  done_count=$((done_count + 1))
  echo "[${done_count}] az=${angle} target=${target_view_file}"

  infer_args=(
    --input_image "${SOURCE_IMAGE}"
    --prompt "${PROMPT}"
    --output "${pred_out}"
    --model_size "${MODEL_SIZE}"
    --model_type "${MODEL_TYPE}"
    --camera_adapter_path "${CAMERA_ADAPTER_PATH}"
    --delta_azimuth "${angle}"
    --delta_elevation 0.0
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
    -i "${OUT_ROOT}/input.png" \
    -i "${pred_out}" \
    -i "${gt_out}" \
    -filter_complex "\
[0:v]drawtext=text='Input':x=20:y=20:fontsize=38:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=10[v0];\
[1:v]drawtext=text='Pred az=${angle}':x=20:y=20:fontsize=38:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=10[v1];\
[2:v]drawtext=text='GT az=${angle}':x=20:y=20:fontsize=38:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=10[v2];\
[v0][v1][v2]hstack=inputs=3[v]" \
    -map "[v]" "${cmp_out}" >/dev/null 2>&1
done

# Compare video over angles.
# Use glob instead of %03d sequence because frames are named az000, az010, ..., az340.
ffmpeg -y -framerate 4 -pattern_type glob -i "${OUT_ROOT}/compare/az*.png" \
  -c:v libx264 -pix_fmt yuv420p "${OUT_ROOT}/compare_35angles.mp4" >/dev/null 2>&1 || true

# Contact sheet (7 x 5 = 35)
"${PYTHON_BIN}" - "${OUT_ROOT}" <<'PY'
import math
import sys
from pathlib import Path
from PIL import Image

root = Path(sys.argv[1])
compare_dir = root / "compare"
paths = sorted(compare_dir.glob("az*.png"))
if not paths:
    print("No compare images found; skip contact sheet.")
    raise SystemExit(0)

cols = 7
rows = math.ceil(len(paths) / cols)
first = Image.open(paths[0]).convert("RGB")
w, h = first.size
canvas = Image.new("RGB", (cols * w, rows * h), (0, 0, 0))

for i, p in enumerate(paths):
    img = Image.open(p).convert("RGB")
    if img.size != (w, h):
        img = img.resize((w, h), Image.BICUBIC)
    x = (i % cols) * w
    y = (i // cols) * h
    canvas.paste(img, (x, y))

out = root / "compare_35angles_grid_7x5.png"
canvas.save(out)
print(f"Saved contact sheet: {out}")
PY

echo "Done."
echo "Input:   ${OUT_ROOT}/input.png"
echo "Pred:    ${OUT_ROOT}/pred"
echo "GT:      ${OUT_ROOT}/gt"
echo "Compare: ${OUT_ROOT}/compare"
echo "Video:   ${OUT_ROOT}/compare_35angles.mp4"
echo "Grid:    ${OUT_ROOT}/compare_35angles_grid_7x5.png"
