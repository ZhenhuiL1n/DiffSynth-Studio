#!/usr/bin/env bash
set -euo pipefail

# Paths ----------------------------------------------------------------------
BODY_CKPT="/home/longnhat/Lin_workspace/8TB2/Lin/801_Project/805/DiffSynth-Studio/models/train/head_to_body_run1/epoch-3.safetensors"
HEAD_CKPT="/home/longnhat/Lin_workspace/8TB2/Lin/801_Project/805/DiffSynth-Studio/models/train/805_body_to_head/epoch-3.safetensors"

BODY_INPUT="/home/longnhat/Lin_workspace/8TB2/Lin/801_Project/805/DiffSynth-Studio/eval_samples_805/bodys"
HEAD_INPUT="/home/longnhat/Lin_workspace/8TB2/Lin/801_Project/805/DiffSynth-Studio/eval_samples_805/heads"

BODY_OUTPUT="out/body_run1_eval"
HEAD_OUTPUT="out/head_run_body_to_head_eval"

HEIGHT=512
WIDTH=512
SEED=1

# Split stems ----------------------------------------------------------------
make_groups() {
  local dir=$1
  local group_size=$2
  python3 - <<'PY' "$dir" "$group_size"
import sys
from pathlib import Path
dir_path = Path(sys.argv[1])
group_size = int(sys.argv[2])
stems = sorted(p.stem for p in dir_path.glob("*.png"))
groups = [stems[i:i+group_size] for i in range(0, len(stems), group_size)]
for idx, group in enumerate(groups):
    print(f"GROUP {idx}:{' '.join(group)}")
PY
}

mapfile -t BODY_GROUPS_OOD < <(make_groups "${BODY_INPUT}/ood" 5 | grep GROUP)
mapfile -t HEAD_GROUPS_OOD < <(make_groups "${HEAD_INPUT}/ood" 5 | grep GROUP)

launch_eval() {
  local gpu=$1
  local script=$2
  local input_root=$3
  local output_root=$4
  local ckpt=$5
  local includes=$6
  local split=$7
  CUDA_VISIBLE_DEVICES=${gpu} python "${script}" \
    --input_root "${input_root}" \
    --output_dir "${output_root}" \
    --lora_path "${ckpt}" \
    --height ${HEIGHT} \
    --width ${WIDTH} \
    --splits "${split}" \
    --include ${includes} \
    --skip_existing \
    --seed ${SEED} \
    --device cuda &
}

echo "[INFO] Launching body OOD shards…"
gpu_id=0
for entry in "${BODY_GROUPS_OOD[@]}"; do
  names=${entry#*:}
  launch_eval ${gpu_id} eval_body_only_batch.py "${BODY_INPUT}" "${BODY_OUTPUT}" "${BODY_CKPT}" "${names}" "ood"
  gpu_id=$((gpu_id + 1))
  if [ ${gpu_id} -ge 7 ]; then gpu_id=0; fi
done

echo "[INFO] Launching head OOD shards…"
for entry in "${HEAD_GROUPS_OOD[@]}"; do
  names=${entry#*:}
  launch_eval ${gpu_id} eval_head_only_batch.py "${HEAD_INPUT}" "${HEAD_OUTPUT}" "${HEAD_CKPT}" "${names}" "ood"
  gpu_id=$((gpu_id + 1))
  if [ ${gpu_id} -ge 7 ]; then gpu_id=0; fi
done

wait
echo "[INFO] Parallel evaluations completed."
