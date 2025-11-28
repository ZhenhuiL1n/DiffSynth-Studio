export CUDA_VISIBLE_DEVICES=0,1

accelerate launch --config_file accelerate_config.yaml \
  examples/wanvideo/model_training/train.kd.py \
  --dataset_base_path data_lora/ \
  --dataset_metadata_path data_lora/metadata.csv \
  --height 832 \
  --width 480 \
  --num_frames 49 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
  --student_model_id_or_paths '["/ckpts/student/dit.safetensors","/ckpts/student/text_encoder.pth","/ckpts/student/vae.pth"]' \
  --teacher_model_id_or_paths "./models/train/Wan2.2-TI2V-5B_lora+face_bigger_ipadapter" \
  --teacher_device cuda:0 \
  --student_device cuda:1 \
  --dtype bf16 \
  --learning_rate 1e-5 \
  --batch_size 1 \
  --epochs 10 \
  --lambda_kd 0.5 \
  --use_gradient_checkpointing_offload \
  --output_path ./models/train/WAN-KD
