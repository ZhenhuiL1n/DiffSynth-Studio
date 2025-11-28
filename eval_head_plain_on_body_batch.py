"""
Evaluate the plain head fine-tune checkpoint on the 801 body evaluation set.

Useful for checking cross-domain performance (head LoRA → body prompts/images)
using ``eval_samples_805/801`` as the default source.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import List
import torch

import requests
from PIL import Image, ImageOps

from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_root",
        default="/home/longnhat/Lin_workspace/8TB2/Lin/801_Project/805/DiffSynth-Studio/eval_samples_805/801",
        help="Root directory containing 801 body splits (e.g., in_dist, ood).",
    )
    parser.add_argument(
        "--output_dir",
        default="out/head_plain_on_801_eval",
        help="Where to save rendered videos (split subfolders created automatically).",
    )
    parser.add_argument(
        "--lora_path",
        default="/home/longnhat/Lin_workspace/8TB2/Lin/801_Project/805/DiffSynth-Studio/models/train/805_head_plain_768/epoch-5.safetensors",
        help="Path to the head plain LoRA checkpoint.",
    )
    parser.add_argument("--lora_alpha", type=float, default=1.0, help="LoRA alpha.")
    parser.add_argument("--height", type=int, default=830, help="Output height.")
    parser.add_argument("--width", type=int, default=482, help="Output width.")
    parser.add_argument("--num_frames", type=int, default=49, help="Frames per video.")
    parser.add_argument("--fps", type=int, default=24, help="FPS for saved videos.")
    parser.add_argument("--quality", type=int, default=5, help="FFMPEG quality (1-31).")
    parser.add_argument("--seed", type=int, default=1, help="Base seed.")
    parser.add_argument("--device", default="cuda", help="Device for inference.")
    parser.add_argument(
        "--prompt",
        default=(
            "a smooth 360-degree rotation video of a full human body, centered, neutral pose, "
            "consistent lighting, clean background, steady camera orbit"
        ),
        help="Positive prompt used for body renders.",
    )
    parser.add_argument(
        "--negative_prompt",
        default=(
            "vivid neon tone, overexposed, static, blurry details, subtitles, stylized, painting, still frame, gray overall, "
            "worst quality, low quality, JPEG artifacts, ugly, deformed, extra fingers, bad hands, bad face, disfigured, "
            "mutated limbs, fused fingers, unmoving frame, cluttered background, three legs, crowded background, walking backwards"
        ),
        help="Negative prompt.",
    )
    parser.add_argument(
        "--tiled",
        action="store_true",
        help="Enable tiled generation (helps with VRAM).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["in_dist", "ood"],
        help="Which subfolders under input_root to evaluate.",
    )
    parser.add_argument(
        "--include",
        nargs="+",
        default=None,
        help="Optional list of filename stems (without extension) to process.",
    )
    parser.add_argument(
        "--include_file",
        type=str,
        default=None,
        help="Optional text file (one stem per line) specifying which inputs to process.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip rendering if the output mp4 already exists.",
    )
    return parser.parse_args()


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


def build_pipeline(device: str, lora_path: str, lora_alpha: float) -> WanVideoPipeline:
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
    pipe.load_lora(pipe.dit, lora_path, alpha=lora_alpha)
    pipe.enable_vram_management()
    return pipe


def load_include_set(args: argparse.Namespace) -> set[str]:
    include: set[str] = set()
    if args.include:
        include.update(args.include)
    if args.include_file:
        file_path = Path(args.include_file)
        if not file_path.is_file():
            raise SystemExit(f"include_file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    include.add(line)
    return include


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    pipe = build_pipeline(args.device, args.lora_path, args.lora_alpha)
    print(f"Loaded head plain LoRA from: {args.lora_path}")
    include_set = load_include_set(args)

    for split in args.splits:
        split_dir = input_root / split
        images = list_images(split_dir)
        if include_set:
            images = [img for img in images if img.stem in include_set]
        if not images:
            print(f"[skip] No images found in {split_dir}")
            continue

        split_out = output_root / split
        split_out.mkdir(parents=True, exist_ok=True)
        print(f"Evaluating {len(images)} body images under {split_dir}")

        for idx, img_path in enumerate(images):
            input_image = load_image_fit(img_path, target_w=args.width, target_h=args.height)
            out_path = split_out / f"{img_path.stem}.mp4"
            if args.skip_existing and out_path.exists():
                print(f"[{split}] [skip-existing] {out_path}")
                continue
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

            save_video(video, str(out_path), fps=args.fps, quality=args.quality)
            print(f"[{split}] [{idx + 1}/{len(images)}] Saved {out_path}")


if __name__ == "__main__":
    main()
