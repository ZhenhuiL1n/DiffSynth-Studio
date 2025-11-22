"""
Evaluate a mixed-training LoRA on both head and body datasets for baseline comparison.

Runs two evaluations using the same mixed checkpoint:
1) Head datasets (in_dist/ood) with a head prompt.
2) Body datasets (in_dist/ood) with a body prompt.
"""

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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lora_path",
        default="/home/longnhat/Lin_workspace/8TB2/Lin/801_Project/805/DiffSynth-Studio/models/train/805_mixed/epoch-5.safetensors",
        help="Path to the mixed-training LoRA checkpoint.",
    )
    parser.add_argument("--lora_alpha", type=float, default=1.0, help="LoRA alpha.")
    parser.add_argument("--height", type=int, default=512, help="Output height.")
    parser.add_argument("--width", type=int, default=512, help="Output width.")
    parser.add_argument("--num_frames", type=int, default=49, help="Frames per video.")
    parser.add_argument("--fps", type=int, default=24, help="FPS for saved videos.")
    parser.add_argument("--quality", type=int, default=5, help="FFMPEG quality (1-31).")
    parser.add_argument("--seed", type=int, default=1, help="Base seed.")
    parser.add_argument("--device", default="cuda", help="Device for inference.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["in_dist", "ood"],
        help="Split subfolders to evaluate under each input root.",
    )
    parser.add_argument(
        "--head_input_root",
        default="/home/longnhat/Lin_workspace/8TB2/Lin/801_Project/805/DiffSynth-Studio/eval_samples_805/heads",
        help="Root containing head evaluation images.",
    )
    parser.add_argument(
        "--body_input_root",
        default="/home/longnhat/Lin_workspace/8TB2/Lin/801_Project/805/DiffSynth-Studio/eval_samples_805/bodys",
        help="Root containing body evaluation images.",
    )
    parser.add_argument(
        "--head_output_root",
        default="out/mixed/head_eval",
        help="Where to save head evaluation videos (split subfolders will be created).",
    )
    parser.add_argument(
        "--body_output_root",
        default="out/mixed/body_eval",
        help="Where to save body evaluation videos (split subfolders will be created).",
    )
    parser.add_argument(
        "--head_prompt",
        default=(
            "a smooth 360-degree rotation video of a human head, centered, neutral pose, "
            "consistent lighting, clean background, steady camera orbit"
        ),
        help="Positive prompt for head evaluation.",
    )
    parser.add_argument(
        "--body_prompt",
        default=(
            "a smooth 360-degree rotation video of a full human body, centered, neutral pose, "
            "consistent lighting, clean background, steady camera orbit"
        ),
        help="Positive prompt for body evaluation.",
    )
    parser.add_argument(
        "--negative_prompt",
        default=(
            "vivid neon tone, overexposed, static, blurry details, subtitles, stylized, painting, still frame, gray overall, "
            "worst quality, low quality, JPEG artifacts, ugly, deformed, extra fingers, bad hands, bad face, disfigured, "
            "mutated limbs, fused fingers, unmoving frame, cluttered background, three legs, crowded background, walking backwards"
        ),
        help="Shared negative prompt.",
    )
    parser.add_argument(
        "--tiled",
        action="store_true",
        help="Enable tiled generation (helps with VRAM).",
    )
    return parser.parse_args()


def list_images(root: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    return sorted(p for p in root.iterdir() if p.suffix.lower() in exts)


def load_image_fit(source: Path, target_w: int, target_h: int, pad_color=(0, 0, 0)) -> Image.Image:
    """Load an image (local or URL), respect EXIF orientation, and pad/resize to target size."""
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


def run_eval(
    name: str,
    input_root: Path,
    output_root: Path,
    prompt: str,
    negative_prompt: str,
    pipe: WanVideoPipeline,
    args: argparse.Namespace,
) -> None:
    for split in args.splits:
        split_dir = input_root / split
        images = list_images(split_dir)
        if not images:
            print(f"[{name}] [skip] No images found in {split_dir}")
            continue

        split_out = output_root / split
        split_out.mkdir(parents=True, exist_ok=True)
        print(f"[{name}] Evaluating {len(images)} images under {split_dir}")

        for idx, img_path in enumerate(images):
            input_image = load_image_fit(img_path, target_w=args.width, target_h=args.height)
            video = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                input_image=input_image,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                seed=args.seed + idx,
                tiled=args.tiled,
            )

            out_path = split_out / f"{img_path.stem}.mp4"
            save_video(video, str(out_path), fps=args.fps, quality=args.quality)
            print(f"[{name}] [{split}] [{idx + 1}/{len(images)}] Saved {out_path}")


def main() -> None:
    args = parse_args()
    pipe = build_pipeline(args.device, args.lora_path, args.lora_alpha)
    print(f"Loaded mixed LoRA: {args.lora_path}")

    # Head evaluation
    run_eval(
        name="head_eval_mixed",
        input_root=Path(args.head_input_root),
        output_root=Path(args.head_output_root),
        prompt=args.head_prompt,
        negative_prompt=args.negative_prompt,
        pipe=pipe,
        args=args,
    )

    # Body evaluation
    run_eval(
        name="body_eval_mixed",
        input_root=Path(args.body_input_root),
        output_root=Path(args.body_output_root),
        prompt=args.body_prompt,
        negative_prompt=args.negative_prompt,
        pipe=pipe,
        args=args,
    )


if __name__ == "__main__":
    main()
