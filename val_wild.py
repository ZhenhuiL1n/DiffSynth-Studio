import io
import os
import torch
import requests
from PIL import Image, ImageOps
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig

# ---------- helpers ----------
def load_image_fit(source: str, target_w: int, target_h: int, keep_aspect: bool = True, pad_color=(0, 0, 0)) -> Image.Image:
    """
    Load an image from local path or URL, apply EXIF orientation, and
    resize to (target_w, target_h). If keep_aspect=True, letterbox-pad.
    """
    if source.startswith("http://") or source.startswith("https://"):
        resp = requests.get(source, timeout=20)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    else:
        img = Image.open(source).convert("RGB")

    # Fix orientation from EXIF
    img = ImageOps.exif_transpose(img)

    if keep_aspect:
        # Letterbox to avoid distortion
        img = ImageOps.pad(
            img, (target_w, target_h),
            method=Image.Resampling.LANCZOS,
            color=pad_color, centering=(0.5, 0.5)
        )
    else:
        # Direct stretch to target size
        img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)

    return img

# ---------- pipeline ----------
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth", offload_device="cpu"),
    ],
)

# Load your LoRA (adjust path as needed)
pipe.load_lora(
    pipe.dit,
    "/home/longnhat/Lin_workspace/8TB2/Lin/801_Project/DiffSynth-Studio/models/train/Wan2.2-TI2V-5B_lora/epoch-3.safetensors",
    alpha=1
)
pipe.enable_vram_management()

# ---------- user inputs ----------
# Use a local file OR a URL
image_source = "/path/to/any_wild_image.jpg"
# image_source = "https://example.com/some_image.jpg"

H, W = 832, 480  # keep multiples of 16; this matches your training/eval
input_image = load_image_fit(image_source, target_w=W, target_h=H, keep_aspect=True)

# ---------- inference ----------
video = pipe(
    prompt="a rotation video of the human",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    input_image=input_image,   # <- pass the PIL image directly
    height=H,
    width=W,
    num_frames=49,
    seed=1,
    tiled=True,
)

out_path = "wild_image_49F_Wan2.2-TI2V-5B_lora.mp4"
save_video(video, out_path, fps=24, quality=5)
print(f"Saved to {out_path}")
