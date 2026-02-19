export CUDA_VISIBLE_DEVICES="3,4,5,6"
taskset -c 0-63 accelerate launch examples/flux2/model_training/train_camera.py \
  --dataset_base_path /home/ren/lin/nas-lin-233/Code/DiffSynth-Studio/data \
  --dataset_metadata_path /home/ren/lin/nas-lin-233/Code/DiffSynth-Studio/data/camera_dataset/metadata.csv \
  --data_file_keys "image,edit_image" \
  --extra_inputs "edit_image" \
  --model_id_with_origin_paths "black-forest-labs/FLUX.2-klein-9B:text_encoder/*.safetensors,black-forest-labs/FLUX.2-klein-9B:transformer/*.safetensors,black-forest-labs/FLUX.2-klein-9B:vae/diffusion_pytorch_model.safetensors" \
  --model_paths '["/home/ren/lin/nas-lin-233/Code/DiffSynth-Studio/models/train/FLUX.2-klein-9B-camera-adapter/epoch-10.safetensors"]' \
  --tokenizer_path "black-forest-labs/FLUX.2-klein-9B:tokenizer/" \
  --trainable_models "camera_adapter" \
  --camera_key_mode "delta" \
  --camera_scale 1.0 \
  --learning_rate 1e-4 \
  --num_epochs 50 \
  --remove_prefix_in_ckpt "pipe.camera_adapter." \
  --output_path /home/ren/lin/nas-lin-233/Code/DiffSynth-Studio/models/train/FLUX.2-klein-9B-camera-adapter \
  --use_gradient_checkpointing


