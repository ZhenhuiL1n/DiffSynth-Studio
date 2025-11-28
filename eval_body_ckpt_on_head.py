"""
Run the baseline body checkpoint (full model) on the head evaluation set without LoRA finetuning.

Loads the checkpoint directly into the DiT and renders 768x768 head videos (in_dist/ood by default).
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import List

import requests
import torch
from PIL import Image, ImageOps

from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        default="/home/longnhat/Lin_workspace/8TB2/Lin/801_Project/805/DiffSynth-Studio/checkpoints/body_best_ckpt/8.safetensors",
        help="Path to the baseline body checkpoint (full model weights).",
    )
    ap.add_argument(
        "--input_root",
        default="/home/longnhat/Lin_workspace/8TB2/Lin/801_Project/805/DiffSynth-Studio/eval_samples_805/heads",
        help="Head evaluation root containing split subfolders.",
    )
    ap.add_argument(
        "--output_dir",
        default="out/body_ckpt_on_head_eval",
        help="Directory to save rendered videos (split subfolders created automatically).",
    )
    ap.add_argument("--height", type=int, default=768, help="Output height.")
    ap.add_argument("--width", type=int, default=768, help="Output width.")
    ap.add_argument("--num_frames", type=int, default=49, help="Frames per video.")
    ap.add_argument("--fps", type=int, default=24, help="FPS for saved videos.")
    ap.add_argument("--quality", type=int, default=5, help="FFMPEG quality (1-31).")
    ap.add_argument("--seed", type=int, default=1, help="Base random seed.")
    ap.add_argument("--device", default="cuda", help="Device for inference.")
    ap.add_argument(
        "--prompt",
        default=(
            "a smooth 360-degree rotation video of a human head, centered, neutral pose, "
            "consistent lighting, clean background, steady camera orbit"
        ),
        help="Positive prompt.",
    )
    ap.add_argument(
        "--negative_prompt",
        default=(
            "vivid neon tone, overexposed, static, blurry details, subtitles, stylized, painting, still frame, gray overall, "
            "worst quality, low quality, JPEG artifacts, ugly, deformed, extra fingers, bad hands, bad face, disfigured, "
            "mutated limbs, fused fingers, unmoving frame, cluttered background, three legs, crowded background, walking backwards"
        ),
        help="Negative prompt.",
    )
    ap.add_argument(
        "--splits",
        nargs="+",
        default=["in_dist", "ood"],
        help="Split subfolders to evaluate.",
    )
    ap.add_argument(
        "--tiled",
        action="store_true",
        help="Enable tiled generation (helps with VRAM).",
    )
    return ap.parse_args()


def list_images(root: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    return sorted(p for p in root.iterdir() if p.suffix.lower() in exts)


def load_image_fit(source: Path, target_w: int, target_h: int, pad_color=(0, 0, 0)) -> Image.Image:
    if str(source).startswith(("http://", "https://")):
        resp = requests.get(str(source), timeout=20)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    else:
        img = Image.open(source).convert("RGB")

    img = ImageOps.exif_transpose(img)
    img = ImageOps.pad(
        img,
        (target_w, target_h),
        method=Image.Resampling.LANCZOS,
        color=pad_color,
        centering=(0.5, 0.5),
    )
    return img


def build_pipeline(device: str, checkpoint: str) -> WanVideoPipeline:
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(
                model_id="Wan-AI/Wan2.2-TI2V-5B",
                origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                offload_device="cpu",
            ),
            ModelConfig(
                model_id="Wan-AI/Wan2.2-TI2V-5B",
                origin_file_pattern="diffusion_pytorch_model*.safetensors",
                offload_device="cpu",
            ),
            ModelConfig(
                model_id="Wan-AI/Wan2.2-TI2V-5B",
                origin_file_pattern="Wan2.2_VAE.pth",
                offload_device="cpu",
            ),
        ],
    )
    # Load full checkpoint directly (no LoRA)
    if checkpoint.endswith(".safetensors"):
        from safetensors.torch import load_file

        state = load_file(checkpoint, device=device)
    else:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
    missing, unexpected = pipe.dit.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[WARN] Missing keys: {missing} | Unexpected keys: {unexpected}")
    pipe.enable_vram_management()
    return pipe


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    pipe = build_pipeline(args.device, args.checkpoint)
    print(f"[INFO] Loaded baseline checkpoint: {args.checkpoint}")

    for split in args.splits:
        split_dir = input_root / split
        images = list_images(split_dir)
        if not images:
            print(f"[skip] No images found in {split_dir}")
            continue
        split_out = output_root / split
        split_out.mkdir(parents=True, exist_ok=True)
        print(f"[{split}] Evaluating {len(images)} prompts from {split_dir}")

        for idx, img_path in enumerate(images):
            input_image = load_image_fit(img_path, target_w=args.width, target_h=args.height)
            video = pipe(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                input_image=input_image,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                seed=args.seed + idx,
                tiled=args.tiled,
            )
            out_path = split_out / f"{img_path.stem}.mp4"
            save_video(video, str(out_path), fps=args.fps, quality=args.quality)
            print(f"[{split}] [{idx + 1}/{len(images)}] Saved {out_path}")


if __name__ == "__main__":
    main()
