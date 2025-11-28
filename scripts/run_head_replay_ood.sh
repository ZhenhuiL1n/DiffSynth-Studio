#!/usr/bin/env bash
set -euo pipefail

CKPT="/home/longnhat/Lin_workspace/8TB2/Lin/801_Project/805/DiffSynth-Studio/models/train/805_body_to_head/epoch-3.safetensors"
INPUT_ROOT="/home/longnhat/Lin_workspace/8TB2/Lin/801_Project/805/DiffSynth-Studio/eval_samples_805/heads"
OUTPUT_ROOT="out/head_body_to_head_eval"
HEIGHT=768
WIDTH=768
SEED=1
GROUP_SIZE=5

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

mapfile -t HEAD_GROUPS_OOD < <(make_groups "${INPUT_ROOT}/ood" ${GROUP_SIZE} | grep GROUP)

launch_shard() {
  local gpu=$1
  local includes=$2
  CUDA_VISIBLE_DEVICES=${gpu} python eval_head_only_batch.py \
    --input_root "${INPUT_ROOT}" \
    --output_dir "${OUTPUT_ROOT}" \
    --lora_path "${CKPT}" \
    --height ${HEIGHT} \
    --width ${WIDTH} \
    --splits ood \
    --include ${includes} \
    --skip_existing \
    --seed ${SEED} \
    --device cuda &
}

echo "[INFO] Launching head replay OOD shards…"
gpu_id=0
for entry in "${HEAD_GROUPS_OOD[@]}"; do
  names=${entry#*:}
  launch_shard ${gpu_id} "${names}"
  gpu_id=$((gpu_id + 1))
  if [ ${gpu_id} -ge 7 ]; then gpu_id=0; fi
done

wait
echo "[INFO] Replay OOD evaluation completed."
