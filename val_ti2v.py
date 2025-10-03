import torch
from PIL import Image
from diffsynth import save_video, VideoData, load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth", offload_device="cpu"),
    ],
)
pipe.load_lora(pipe.dit, "/home/longnhat/Lin_workspace/8TB2/Lin/801_Project/DiffSynth-Studio/models/train/Wan2.2-TI2V-5B_lora/epoch-1.safetensors", alpha=1)
pipe.enable_vram_management()

input_image = VideoData("/home/longnhat/Lin_workspace/8TB2/Lin/801_Project/DiffSynth-Studio/data_lora/videos_padded/0000.mp4", height=832, width=480)[0]

video = pipe(
    prompt="a rotation video of the human",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    input_image=input_image,
    height=832,
    width=480,
    num_frames=17,
    seed=1, tiled=True,
)
save_video(video, "test_lora_video_Wan2.2-TI2V-5B.mp4", fps=17, quality=5)
