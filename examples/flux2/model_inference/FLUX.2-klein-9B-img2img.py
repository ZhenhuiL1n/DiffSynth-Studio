from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig
from diffsynth.core import load_state_dict
from PIL import Image
import torch
import argparse


def main():
    parser = argparse.ArgumentParser(description="FLUX.2-klein-9B Image-to-Image Translation")
    parser.add_argument("--input_image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--prompt", type=str, default="", help="Text prompt describing the desired output")
    parser.add_argument("--output", type=str, default="output_img2img.jpg", help="Output image path")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--steps", type=int, default=4, help="Number of inference steps (klein models use ~4)")
    parser.add_argument("--cfg_scale", type=float, default=4.0, help="Classifier-free guidance scale")
    parser.add_argument("--mode", type=str, default="edit", choices=["edit", "input"], help="edit: reference-edit mode, input: true img2img mode")
    parser.add_argument("--denoising_strength", type=float, default=0.35, help="Only used in mode=input; lower keeps identity more")
    parser.add_argument("--model_id", type=str, default="black-forest-labs/FLUX.2-klein-9B", help="Model ID")
    parser.add_argument("--lora_path", type=str, default=None, help="Optional LoRA checkpoint path")
    parser.add_argument("--dit_checkpoint", type=str, default=None, help="Optional full-finetune DIT checkpoint path (epoch-*.safetensors)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Inference device")
    args = parser.parse_args()

    # Load pipeline
    pipe = Flux2ImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=args.device,
        model_configs=[
            ModelConfig(model_id=args.model_id, origin_file_pattern="text_encoder/*.safetensors"),
            ModelConfig(model_id=args.model_id, origin_file_pattern="transformer/*.safetensors"),
            ModelConfig(model_id=args.model_id, origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ],
        tokenizer_config=ModelConfig(model_id=args.model_id, origin_file_pattern="tokenizer/"),
    )
    if args.lora_path:
        pipe.load_lora(pipe.dit, args.lora_path)
        print(f"Loaded LoRA: {args.lora_path}")
    if args.dit_checkpoint:
        state_dict = load_state_dict(args.dit_checkpoint, torch_dtype=torch.bfloat16)
        pipe.dit.load_state_dict(state_dict, strict=True)
        print(f"Loaded DIT checkpoint: {args.dit_checkpoint}")

    # Load input image
    input_img = Image.open(args.input_image).convert("RGB")
    print(f"Input image size: {input_img.size}")

    if args.mode == "input":
        image = pipe(
            args.prompt,
            input_image=input_img,
            denoising_strength=args.denoising_strength,
            cfg_scale=args.cfg_scale,
            seed=args.seed,
            rand_device=args.device,
            num_inference_steps=args.steps,
        )
    else:
        image = pipe(
            args.prompt,
            edit_image=[input_img],
            cfg_scale=args.cfg_scale,
            seed=args.seed,
            rand_device=args.device,
            num_inference_steps=args.steps,
        )
    image.save(args.output)
    print(f"Saved output to: {args.output}")


if __name__ == "__main__":
    main()
