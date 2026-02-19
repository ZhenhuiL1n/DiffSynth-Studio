import argparse
import torch
from diffsynth import ModelConfig
from diffsynth.pipelines.flux2_image import Flux2ImagePipeline
from PIL import Image


def resolve_camera_condition(args):
    # Priority: explicit delta > target-source > raw azimuth/elevation args.
    if args.delta_azimuth is not None or args.delta_elevation is not None:
        az = 0.0 if args.delta_azimuth is None else args.delta_azimuth
        el = 0.0 if args.delta_elevation is None else args.delta_elevation
        source = "delta"
        return az, el, source

    if (
        args.source_azimuth is not None
        and args.source_elevation is not None
        and args.target_azimuth is not None
        and args.target_elevation is not None
    ):
        az = args.target_azimuth - args.source_azimuth
        el = args.target_elevation - args.source_elevation
        source = "target_minus_source"
        return az, el, source

    return args.azimuth, args.elevation, "absolute"


def main():
    parser = argparse.ArgumentParser(description="Camera adapter inference for FLUX.2-klein")
    parser.add_argument("--input_image", type=str, required=True, help="Path to source view image")
    parser.add_argument("--prompt", type=str, default="", help="Text prompt (avoid putting rotation words here)")
    parser.add_argument("--output", type=str, default="output.png", help="Output path")

    # Backward-compatible absolute args.
    parser.add_argument("--azimuth", type=float, default=0.0, help="Camera conditioning azimuth in degrees")
    parser.add_argument("--elevation", type=float, default=0.0, help="Camera conditioning elevation in degrees")

    # Preferred rotation-control args.
    parser.add_argument("--delta_azimuth", type=float, default=None, help="Desired azimuth rotation delta in degrees")
    parser.add_argument("--delta_elevation", type=float, default=None, help="Desired elevation rotation delta in degrees")
    parser.add_argument("--source_azimuth", type=float, default=None, help="Source azimuth in degrees")
    parser.add_argument("--source_elevation", type=float, default=None, help="Source elevation in degrees")
    parser.add_argument("--target_azimuth", type=float, default=None, help="Target azimuth in degrees")
    parser.add_argument("--target_elevation", type=float, default=None, help="Target elevation in degrees")

    parser.add_argument("--camera_scale", type=float, default=1.0, help="Camera adapter strength")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--steps", type=int, default=4, help="Number of inference steps (4 for distilled, 50 for base)")
    parser.add_argument("--model_size", type=str, default="9B", choices=["4B", "9B"], help="Model size")
    parser.add_argument("--model_type", type=str, default="distilled", choices=["distilled", "base"], help="Distilled or base model")
    parser.add_argument("--camera_adapter_path", type=str, required=True, help="Path to trained camera adapter checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Inference device")
    args = parser.parse_args()

    camera_azimuth, camera_elevation, camera_source = resolve_camera_condition(args)

    if args.model_type == "base":
        transformer_id = f"black-forest-labs/FLUX.2-klein-base-{args.model_size}"
    else:
        transformer_id = f"black-forest-labs/FLUX.2-klein-{args.model_size}"

    if args.model_size == "9B":
        text_encoder_id = "black-forest-labs/FLUX.2-klein-9B"
    else:
        text_encoder_id = "black-forest-labs/FLUX.2-klein-4B"

    pipe = Flux2ImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=args.device,
        model_configs=[
            ModelConfig(model_id=text_encoder_id, origin_file_pattern="text_encoder/*.safetensors"),
            ModelConfig(model_id=transformer_id, origin_file_pattern="transformer/*.safetensors"),
            ModelConfig(model_id=text_encoder_id, origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
            ModelConfig(path=args.camera_adapter_path),
        ],
        tokenizer_config=ModelConfig(model_id=text_encoder_id, origin_file_pattern="tokenizer/"),
    )

    input_image = Image.open(args.input_image).convert("RGB")
    print(
        f"Camera conditioning source={camera_source}, "
        f"azimuth={camera_azimuth:.3f}, elevation={camera_elevation:.3f}"
    )

    image = pipe(
        prompt=args.prompt,
        edit_image=input_image,
        camera_azimuth=camera_azimuth,
        camera_elevation=camera_elevation,
        camera_scale=args.camera_scale,
        seed=args.seed,
        rand_device=args.device,
        num_inference_steps=args.steps,
    )
    image.save(args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
