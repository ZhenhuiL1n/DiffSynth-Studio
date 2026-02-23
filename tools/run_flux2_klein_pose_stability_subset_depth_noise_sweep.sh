#!/usr/bin/env bash
set -euo pipefail

# Hyperparameter sweep for depth influence strength.
# This script runs the existing subset depth-noise pipeline at 5 blend levels
# and stores each run under a dedicated subfolder.
#
# Default output layout:
#   outputs/pose_stability_depth_noise_subset_0008_01/depth_strength_sweep_5levels/
#     blend_0p10/
#     blend_0p30/
#     blend_0p50/
#     blend_0p70/
#     blend_0p90/
#
# Example:
#   bash tools/run_flux2_klein_pose_stability_subset_depth_noise_sweep.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_SCRIPT="${BASE_SCRIPT:-${ROOT_DIR}/tools/run_flux2_klein_pose_stability_subset_depth_noise.sh}"

if [[ ! -f "${BASE_SCRIPT}" ]]; then
  echo "Base script not found: ${BASE_SCRIPT}" >&2
  exit 1
fi

# Sweep target: depth-noise blend strength.
STRENGTH_LEVELS="${STRENGTH_LEVELS:-0.10,0.30,0.50,0.70,0.90}"

# Keep modulation fixed by default so blend is the only changed variable.
DEPTH_NOISE_MODULATION="${DEPTH_NOISE_MODULATION:-1.0}"
DEPTH_NOISE_INVERT="${DEPTH_NOISE_INVERT:-false}"

# Parent output and new sweep subfolder.
PARENT_OUT="${PARENT_OUT:-${ROOT_DIR}/outputs/pose_stability_depth_noise_subset_0008_01}"
SWEEP_SUBDIR="${SWEEP_SUBDIR:-depth_strength_sweep_5levels}"
SWEEP_ROOT="${PARENT_OUT}/${SWEEP_SUBDIR}"
mkdir -p "${SWEEP_ROOT}"

# Pass-through defaults (can be overridden by environment variables).
CONDA_ENV="${CONDA_ENV:-diffsynth}"
SEQ_DIR="${SEQ_DIR:-/home/ren/lin/nas-lin-233/Code/DiffSynth-Studio/data/0008_01}"
FRAME_IDS="${FRAME_IDS:-44,68,82,96,112,126}"
REF_VIEW_FILE="${REF_VIEW_FILE:-view_0016_az160.0_el0.0.png}"
RIGHT90_DEPTH_AZ="${RIGHT90_DEPTH_AZ:-70.0}"
LEFT90_DEPTH_AZ="${LEFT90_DEPTH_AZ:-250.0}"
BACK_DEPTH_AZ="${BACK_DEPTH_AZ:-340.0}"
DEPTH_EL="${DEPTH_EL:-0.0}"
MODEL_ID="${MODEL_ID:-black-forest-labs/FLUX.2-klein-9B}"
STEPS="${STEPS:-4}"
SEED="${SEED:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"

echo "Base script: ${BASE_SCRIPT}"
echo "Sweep levels (depth_noise_blend): ${STRENGTH_LEVELS}"
echo "Sweep output root: ${SWEEP_ROOT}"
echo "Fixed modulation: ${DEPTH_NOISE_MODULATION} | invert=${DEPTH_NOISE_INVERT}"

IFS=',' read -r -a LEVELS <<< "${STRENGTH_LEVELS}"
if [[ "${#LEVELS[@]}" -ne 5 ]]; then
  echo "Expected exactly 5 levels in STRENGTH_LEVELS, got ${#LEVELS[@]}: ${STRENGTH_LEVELS}" >&2
  exit 1
fi

for level_raw in "${LEVELS[@]}"; do
  level="$(echo "${level_raw}" | xargs)"
  level_tag="${level//./p}"
  out_root="${SWEEP_ROOT}/blend_${level_tag}"

  echo ""
  echo "=============================="
  echo "Running depth blend = ${level}"
  echo "Output: ${out_root}"
  echo "=============================="

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD}" \
  CONDA_ENV="${CONDA_ENV}" \
  SEQ_DIR="${SEQ_DIR}" \
  FRAME_IDS="${FRAME_IDS}" \
  REF_VIEW_FILE="${REF_VIEW_FILE}" \
  RIGHT90_DEPTH_AZ="${RIGHT90_DEPTH_AZ}" \
  LEFT90_DEPTH_AZ="${LEFT90_DEPTH_AZ}" \
  BACK_DEPTH_AZ="${BACK_DEPTH_AZ}" \
  DEPTH_EL="${DEPTH_EL}" \
  MODEL_ID="${MODEL_ID}" \
  STEPS="${STEPS}" \
  SEED="${SEED}" \
  DEPTH_NOISE_BLEND="${level}" \
  DEPTH_NOISE_MODULATION="${DEPTH_NOISE_MODULATION}" \
  DEPTH_NOISE_INVERT="${DEPTH_NOISE_INVERT}" \
  OUT_ROOT="${out_root}" \
  bash "${BASE_SCRIPT}"
done

echo ""
echo "Sweep completed."
echo "All outputs are under: ${SWEEP_ROOT}"
