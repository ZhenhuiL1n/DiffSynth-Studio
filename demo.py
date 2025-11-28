#!/usr/bin/env python3
"""
Simple demo script to render a few sample inputs with pre-trained checkpoints.

Usage:
    python demo.py --mode head --ckpt checkpoints/ewc_best/ewc.safetensors
    python demo.py --mode body --ckpt checkpoints/body_best_ckpt/8.safetensors
"""

import argparse
from pathlib import Path
from typing import List
import torch
from PIL import Image, ImageOps
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline


HEAD_PROMPT = (
    "a smooth 360-degree rotation video of a human head, centered, neutral pose, "
    "consistent lighting, clean background, steady camera orbit"
)
BODY_PROMPT = (
    "a smooth 360-degree rotation video of a full human body, centered, neutral pose, "
    "consistent lighting, clean background, steady camera orbit"
)
NEGATIVE_PROMPT = (
    "vivid neon tone, overexposed, static, blurry details, subtitles, stylized, painting, still frame, gray overall, "
    "worst quality, low quality, JPEG artifacts, ugly, deformed, extra fingers, bad hands, bad face, disfigured, "
    "mutated limbs, fused fingers, unmoving frame, cluttered background, three legs, crowded background, walking backwards"
)


TEST_EXAMPLES = {
    "head": [
        Path("test_examples/others/ood_head_7.png"),
        Path("test_examples/others/ood_head_12.png"),
        Path("test_examples/others/ood_head_13.png"),
        Path("test_examples/others/ood_head_14.png"),
    ],
    "body": [
        Path("test_examples/ewc_body/ood_10.png"),
        Path("test_examples/ewc_body/ood_27.png"),
        Path("test_examples/others/ood_32.png"),
        Path("test_examples/others/ood_24.png"),
    ],
}


def build_pipeline(device: str, ckpt: Path) -> WanVideoPipeline:
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
    pipe.load_lora(pipe.dit, str(ckpt), alpha=1.0)
    pipe.enable_vram_management()
    return pipe


def load_image_fit(path: Path, width: int, height: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img = ImageOps.exif_transpose(img)
    img = ImageOps.pad(
        img,
        (width, height),
        method=Image.Resampling.LANCZOS,
        color=(0, 0, 0),
        centering=(0.5, 0.5),
    )
    return img


def render_samples(mode: str, ckpt: Path, output_dir: Path, device: str, examples: List[Path]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = build_pipeline(device, ckpt)
    prompt = HEAD_PROMPT if mode == "head" else BODY_PROMPT

    for idx, example in enumerate(examples, 1):
        if not example.is_file():
            print(f"[skip] Example not found: {example}")
            continue
        img = load_image_fit(example, width=768, height=768)
        print(f"[{mode}] Rendering {example.name} …")
        video = pipeline(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            input_image=img,
            height=768,
            width=768,
            num_frames=49,
            seed=idx,
        )
        out_path = output_dir / f"{example.stem}.mp4"
        save_video(video, str(out_path), fps=24, quality=5)
        print(f"Saved {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple demo for head/body checkpoints.")
    parser.add_argument("--mode", choices=["head", "body"], default="head", help="Which preset to run.")
    parser.add_argument("--ckpt", required=True, help="Path to the .safetensors checkpoint.")
    parser.add_argument("--device", default="cuda", help="Device for inference.")
    parser.add_argument("--output_dir", default="demo_output", help="Directory to store demo mp4s.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    examples = TEST_EXAMPLES[args.mode]
    render_samples(
        mode=args.mode,
        ckpt=ckpt_path,
        output_dir=Path(args.output_dir) / args.mode,
        device=args.device,
        examples=examples,
    )


if __name__ == "__main__":
    main()
