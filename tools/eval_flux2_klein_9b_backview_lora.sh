#!/usr/bin/env bash
set -euo pipefail

# Evaluate FLUX.2-klein-9B back-view LoRA using image+prompt pairs from metadata.csv.
#
# Default behavior:
# - runs LoRA inference on all rows in metadata
# - saves outputs under OUT_ROOT/lora/<relative_image_path>
#
# Optional:
# - set COMPARE_BASE=true to also run base model inference under OUT_ROOT/base/...
# - set MAX_ITEMS>0 to only evaluate the first N rows
#
# Example:
#   CUDA_VISIBLE_DEVICES=3 \
#   bash tools/eval_flux2_klein_9b_backview_lora.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
INFER_SCRIPT="${INFER_SCRIPT:-${ROOT_DIR}/examples/flux2/model_inference/FLUX.2-klein-9B-img2img.py}"
DATASET_BASE_PATH="${DATASET_BASE_PATH:-${ROOT_DIR}/data}"
METADATA_PATH="${METADATA_PATH:-${ROOT_DIR}/data/back_view_dataset/metadata.csv}"
MODEL_ID="${MODEL_ID:-black-forest-labs/FLUX.2-klein-9B}"
LORA_PATH="${LORA_PATH:-${ROOT_DIR}/models/train/FLUX.2-klein-9B-backview-lora/epoch-4.safetensors}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs/eval_backview_lora_epoch4}"

DEVICE="${DEVICE:-cuda}"
MODE="${MODE:-input}"
DENOISING_STRENGTH="${DENOISING_STRENGTH:-0.15}"
CFG_SCALE="${CFG_SCALE:-2.0}"
STEPS="${STEPS:-4}"
SEED="${SEED:-42}"

MAX_ITEMS="${MAX_ITEMS:-0}"           # 0 means all rows
COMPARE_BASE="${COMPARE_BASE:-false}" # true/false
SKIP_EXISTING="${SKIP_EXISTING:-true}"

if [[ ! -f "${INFER_SCRIPT}" ]]; then
  echo "Inference script not found: ${INFER_SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${METADATA_PATH}" ]]; then
  echo "Metadata not found: ${METADATA_PATH}" >&2
  exit 1
fi
if [[ ! -d "${DATASET_BASE_PATH}" ]]; then
  echo "Dataset base path not found: ${DATASET_BASE_PATH}" >&2
  exit 1
fi
if [[ ! -f "${LORA_PATH}" ]]; then
  echo "LoRA checkpoint not found: ${LORA_PATH}" >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}/lora"
if [[ "${COMPARE_BASE}" == "true" ]]; then
  mkdir -p "${OUT_ROOT}/base"
fi
cp -f "${METADATA_PATH}" "${OUT_ROOT}/metadata_snapshot.csv"

rows_tsv="$(mktemp /tmp/backview_eval_rows.XXXXXX.tsv)"
trap 'rm -f "${rows_tsv}"' EXIT

"${PYTHON_BIN}" - "${METADATA_PATH}" "${MAX_ITEMS}" > "${rows_tsv}" <<'PY'
import csv
import sys

metadata = sys.argv[1]
max_items = int(sys.argv[2])

count = 0
with open(metadata, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    if "image" not in reader.fieldnames or "prompt" not in reader.fieldnames:
        raise SystemExit("metadata.csv must contain 'image' and 'prompt' columns")
    for row in reader:
        image = row["image"].strip()
        prompt = row["prompt"]
        if not image:
            continue
        print(f"{image}\t{prompt}")
        count += 1
        if max_items > 0 and count >= max_items:
            break
PY

total="$(wc -l < "${rows_tsv}")"
if [[ "${total}" -eq 0 ]]; then
  echo "No evaluation rows found in metadata: ${METADATA_PATH}" >&2
  exit 1
fi

echo "Running evaluation"
echo "INFER_SCRIPT=${INFER_SCRIPT}"
echo "DATASET_BASE_PATH=${DATASET_BASE_PATH}"
echo "METADATA_PATH=${METADATA_PATH}"
echo "LORA_PATH=${LORA_PATH}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "DEVICE=${DEVICE} MODE=${MODE} DENOISING_STRENGTH=${DENOISING_STRENGTH} CFG_SCALE=${CFG_SCALE} STEPS=${STEPS} SEED=${SEED}"
echo "COMPARE_BASE=${COMPARE_BASE} SKIP_EXISTING=${SKIP_EXISTING} MAX_ITEMS=${MAX_ITEMS} TOTAL=${total}"

done_count=0
while IFS=$'\t' read -r rel_image prompt; do
  done_count=$((done_count + 1))
  input_image="${DATASET_BASE_PATH}/${rel_image}"
  out_lora="${OUT_ROOT}/lora/${rel_image}"

  if [[ ! -f "${input_image}" ]]; then
    echo "[WARN ${done_count}/${total}] Missing input image: ${input_image}" >&2
    continue
  fi

  mkdir -p "$(dirname "${out_lora}")"
  if [[ "${SKIP_EXISTING}" != "true" || ! -f "${out_lora}" ]]; then
    echo "[${done_count}/${total}] LoRA -> ${rel_image}"
    "${PYTHON_BIN}" "${INFER_SCRIPT}" \
      --input_image "${input_image}" \
      --prompt "${prompt}" \
      --output "${out_lora}" \
      --model_id "${MODEL_ID}" \
      --lora_path "${LORA_PATH}" \
      --device "${DEVICE}" \
      --mode "${MODE}" \
      --denoising_strength "${DENOISING_STRENGTH}" \
      --cfg_scale "${CFG_SCALE}" \
      --steps "${STEPS}" \
      --seed "${SEED}"
  fi

  if [[ "${COMPARE_BASE}" == "true" ]]; then
    out_base="${OUT_ROOT}/base/${rel_image}"
    mkdir -p "$(dirname "${out_base}")"
    if [[ "${SKIP_EXISTING}" != "true" || ! -f "${out_base}" ]]; then
      echo "[${done_count}/${total}] Base -> ${rel_image}"
      "${PYTHON_BIN}" "${INFER_SCRIPT}" \
        --input_image "${input_image}" \
        --prompt "${prompt}" \
        --output "${out_base}" \
        --model_id "${MODEL_ID}" \
        --device "${DEVICE}" \
        --mode "${MODE}" \
        --denoising_strength "${DENOISING_STRENGTH}" \
        --cfg_scale "${CFG_SCALE}" \
        --steps "${STEPS}" \
        --seed "${SEED}"
    fi
  fi
done < "${rows_tsv}"

echo "Evaluation complete. Outputs saved to: ${OUT_ROOT}"
