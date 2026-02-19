#!/bin/bash
# Camera adapter training for FLUX.2-klein
# Supports both 4B and 9B models, distilled and base variants.

# ===== Configuration =====
MODEL_SIZE="9B"        # "4B" or "9B"
MODEL_TYPE="distilled" # "distilled" or "base"
# ==========================

# Resolve model IDs
if [ "$MODEL_TYPE" = "base" ]; then
    TRANSFORMER_ID="black-forest-labs/FLUX.2-klein-base-${MODEL_SIZE}"
else
    TRANSFORMER_ID="black-forest-labs/FLUX.2-klein-${MODEL_SIZE}"
fi

# Text encoder and VAE always come from the distilled model
if [ "$MODEL_SIZE" = "9B" ]; then
    TEXT_ENCODER_ID="black-forest-labs/FLUX.2-klein-9B"
    TOKENIZER_ID="black-forest-labs/FLUX.2-klein-9B"
else
    TEXT_ENCODER_ID="black-forest-labs/FLUX.2-klein-4B"
    TOKENIZER_ID="black-forest-labs/FLUX.2-klein-4B"
fi

echo "=== Camera Adapter Training ==="
echo "Model Size:  ${MODEL_SIZE}"
echo "Model Type:  ${MODEL_TYPE}"
echo "Transformer: ${TRANSFORMER_ID}"
echo "Text Encoder: ${TEXT_ENCODER_ID}"
echo "==============================="

accelerate launch examples/flux2/model_training/train_camera.py \
  --dataset_base_path data/camera_dataset \
  --dataset_metadata_path data/camera_dataset/metadata.csv \
  --data_file_keys "image,edit_image" \
  --extra_inputs "edit_image" \
  --max_pixels 1048576 \
  --dataset_repeat 10 \
  --model_id_with_origin_paths "${TEXT_ENCODER_ID}:text_encoder/*.safetensors,${TRANSFORMER_ID}:transformer/*.safetensors,${TEXT_ENCODER_ID}:vae/diffusion_pytorch_model.safetensors" \
  --tokenizer_path "${TOKENIZER_ID}:tokenizer/" \
  --learning_rate 1e-4 \
  --num_epochs 50 \
  --remove_prefix_in_ckpt "pipe.camera_adapter." \
  --output_path "./models/train/FLUX.2-klein-${MODEL_SIZE}-camera-adapter" \
  --trainable_models "camera_adapter" \
  --use_gradient_checkpointing
