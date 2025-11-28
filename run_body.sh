#!/usr/bin/env bash
set -euo pipefail

CKPT=/home/longnhat/Lin_workspace/8TB2/Lin/801_Project/805/DiffSynth-Studio/models/train/805_head_with_body_guard_768/epoch-8.safetensors
INPUT_ROOT=/home/longnhat/Lin_workspace/8TB2/Lin/801_Project/805/DiffSynth-Studio/eval_samples_805/801
OUTPUT_DIR=out/body_regression_801
HEIGHT=830
WIDTH=482

CUDA_VISIBLE_DEVICES=0 python eval_head_plain_on_body_batch.py \
	  --input_root "${INPUT_ROOT}" --output_dir "${OUTPUT_DIR}" \
	    --lora_path "${CKPT}" --height ${HEIGHT} --width ${WIDTH} \
	      --splits ood --include ood_0 ood_1 ood_10 ood_11 ood_12 \
	        --skip_existing --seed 1 --device cuda &

CUDA_VISIBLE_DEVICES=1 python eval_head_plain_on_body_batch.py \
	  --input_root "${INPUT_ROOT}" --output_dir "${OUTPUT_DIR}" \
	    --lora_path "${CKPT}" --height ${HEIGHT} --width ${WIDTH} \
	      --splits ood --include ood_13 ood_14 ood_15 ood_16 ood_17 \
	        --skip_existing --seed 1 --device cuda &

CUDA_VISIBLE_DEVICES=2 python eval_head_plain_on_body_batch.py \
	  --input_root "${INPUT_ROOT}" --output_dir "${OUTPUT_DIR}" \
	    --lora_path "${CKPT}" --height ${HEIGHT} --width ${WIDTH} \
	      --splits ood --include ood_18 ood_19 ood_2 ood_20 ood_21 \
	        --skip_existing --seed 1 --device cuda &

CUDA_VISIBLE_DEVICES=3 python eval_head_plain_on_body_batch.py \
	  --input_root "${INPUT_ROOT}" --output_dir "${OUTPUT_DIR}" \
	    --lora_path "${CKPT}" --height ${HEIGHT} --width ${WIDTH} \
	      --splits ood --include ood_22 ood_23 ood_24 ood_25 ood_26 \
	        --skip_existing --seed 1 --device cuda &

CUDA_VISIBLE_DEVICES=4 python eval_head_plain_on_body_batch.py \
	  --input_root "${INPUT_ROOT}" --output_dir "${OUTPUT_DIR}" \
	    --lora_path "${CKPT}" --height ${HEIGHT} --width ${WIDTH} \
	      --splits ood --include ood_27 ood_28 ood_29 ood_3 ood_30 \
	        --skip_existing --seed 1 --device cuda &

CUDA_VISIBLE_DEVICES=5 python eval_head_plain_on_body_batch.py \
	  --input_root "${INPUT_ROOT}" --output_dir "${OUTPUT_DIR}" \
	    --lora_path "${CKPT}" --height ${HEIGHT} --width ${WIDTH} \
	      --splits ood --include ood_31 ood_32 ood_33 ood_34 ood_4 \
	        --skip_existing --seed 1 --device cuda &

CUDA_VISIBLE_DEVICES=6 python eval_head_plain_on_body_batch.py \
	  --input_root "${INPUT_ROOT}" --output_dir "${OUTPUT_DIR}" \
	    --lora_path "${CKPT}" --height ${HEIGHT} --width ${WIDTH} \
	      --splits ood --include ood_5 ood_6 ood_7 ood_8 ood_9 \
	        --skip_existing --seed 1 --device cuda &

wait

