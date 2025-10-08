import argparse
from pathlib import Path
import torch
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Validate TI2V LoRA with an input media (video/image).")
    parser.add_argument("--media_path", required=True, help="Path to input media (.mp4 video or image)")
    parser.add_argument("--model_name", default="Wan2.2-TI2V-5B_lora", help="LoRA model directory under models/train")
    parser.add_argument("--n_epoch", type=int, default=2, help="Epoch number of the LoRA checkpoint")
    parser.add_argument("--fps", type=int, default=15, help="FPS for the output video (default: 15)")
    parser.add_argument("--num_frames", type=int, required=True, help="Temporal resolution of the model")
    parser.add_argument("--out_path", type=str, required=False, default=None)
    args = parser.parse_args()

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth", offload_device="cpu"),
        ],
    )

    repo_root = Path(__file__).resolve().parent
    lora_path = repo_root / "models" / "train" / args.model_name / f"epoch-{args.n_epoch}.safetensors"
    pipe.load_lora(pipe.dit, str(lora_path), alpha=1)
    pipe.enable_vram_management()

    media_arg_path = Path(args.media_path)

    if args.out_path is None:
        outputs_dir = repo_root / "outputs"
    else:
        outputs_dir = Path(args.out_path)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Build list of media paths: either a single media or many from a text file
    media_paths = []
    if media_arg_path.suffix.lower() == ".txt":
        for line in media_arg_path.read_text(encoding="utf-8").splitlines():
            candidate = line.strip()
            if not candidate or candidate.startswith("#"):
                continue
            candidate_path = Path(candidate)
            if not candidate_path.is_absolute():
                candidate_path = media_arg_path.parent / candidate
            media_paths.append(candidate_path)
    else:
        media_paths.append(media_arg_path)

    for media_path in media_paths:
        ext = media_path.suffix.lower()
        if ext == ".mp4":
            input_image = VideoData(str(media_path), height=832, width=480)[0]
        else:
            input_image = Image.open(media_path).convert("RGB")

        video = pipe(
            prompt="a rotation video of the human",
            negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            input_image=input_image,
            height=832,
            width=480,
            num_frames=args.num_frames,
            seed=1, tiled=True,
        )

        media_base_name = media_path.stem
        output_filename = f"{args.model_name}_epoch-{args.n_epoch}_{media_base_name}.mp4"
        save_video(video, str(outputs_dir / output_filename), fps=args.fps, quality=5)


if __name__ == "__main__":
    main()
