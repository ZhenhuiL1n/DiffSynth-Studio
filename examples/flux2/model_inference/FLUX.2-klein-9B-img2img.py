from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig
from PIL import Image
import torch
import argparse


def main():
    parser = argparse.ArgumentParser(description="FLUX.2-klein-9B Image-to-Image Translation")
    parser.add_argument("--input_image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt describing the desired output")
    parser.add_argument("--output", type=str, default="output_img2img.jpg", help="Output image path")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--steps", type=int, default=4, help="Number of inference steps (klein models use ~4)")
    parser.add_argument("--model_id", type=str, default="black-forest-labs/FLUX.2-klein-9B", help="Model ID")
    args = parser.parse_args()

    # Load pipeline
    pipe = Flux2ImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id=args.model_id, origin_file_pattern="text_encoder/*.safetensors"),
            ModelConfig(model_id=args.model_id, origin_file_pattern="transformer/*.safetensors"),
            ModelConfig(model_id=args.model_id, origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ],
        tokenizer_config=ModelConfig(model_id=args.model_id, origin_file_pattern="tokenizer/"),
    )

    # Load input image
    input_img = Image.open(args.input_image).convert("RGB")
    print(f"Input image size: {input_img.size}")

    # Run img2img via edit_image (FLUX.2-klein's native editing mode)
    # The model treats edit_image as a reference and generates based on the prompt
    image = pipe(
        args.prompt,
        edit_image=[input_img],
        seed=args.seed,
        rand_device="cuda",
        num_inference_steps=args.steps,
    )
    image.save(args.output)
    print(f"Saved output to: {args.output}")


if __name__ == "__main__":
    main()
