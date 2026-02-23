import argparse
import torch
from diffsynth import ModelConfig
import numpy as np
import torch.nn.functional as F
from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, Flux2Unit_NoiseInitializer
from diffsynth.utils.controlnet.annotator import Annotator
from PIL import Image


class DepthAwareNoiseInitializer(Flux2Unit_NoiseInitializer):
    def __init__(
        self,
        depth_source_image: Image.Image,
        depth_noise_blend: float = 0.0,
        depth_noise_modulation: float = 1.0,
        depth_noise_invert: bool = False,
        depth_model_path: str = "models/Annotators",
        annotator_device: str = "cuda",
    ):
        super().__init__()
        self.depth_source_image = depth_source_image
        self.depth_noise_blend = float(depth_noise_blend)
        self.depth_noise_modulation = float(depth_noise_modulation)
        self.depth_noise_invert = bool(depth_noise_invert)
        self.depth_model_path = depth_model_path
        self.annotator_device = annotator_device
        self._warned_fallback = False

    @staticmethod
    def _zscore(x: torch.Tensor, eps: float = 1e-6):
        return (x - x.mean()) / (x.std(unbiased=False) + eps)

    def _make_depth_feature(self, height: int, width: int, device: str, dtype: torch.dtype) -> torch.Tensor:
        annotator = Annotator("depth", model_path=self.depth_model_path, device=self.annotator_device)
        depth_image = annotator(self.depth_source_image)
        depth_np = np.array(depth_image.convert("L"), dtype=np.float32)
        if self.depth_noise_invert:
            depth_np = 255.0 - depth_np
        depth = torch.from_numpy(depth_np).to(device=device, dtype=torch.float32) / 255.0
        depth = depth.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        depth = F.interpolate(depth, size=(height // 16, width // 16), mode="bilinear", align_corners=False)
        depth = depth.reshape(1, -1, 1).to(dtype=dtype)  # [1, HW, 1]
        return self._zscore(depth)

    def process(self, pipe: Flux2ImagePipeline, height, width, seed, rand_device):
        base = super().process(pipe, height, width, seed, rand_device)
        noise = base["noise"]

        if self.depth_noise_blend <= 0.0:
            return base

        try:
            depth_feature = self._make_depth_feature(height, width, pipe.device, pipe.torch_dtype)
        except Exception as e:
            if not self._warned_fallback:
                print(f"[WARN] Depth-aware noise fallback to vanilla noise: {e}")
                self._warned_fallback = True
            return base

        mod = 1.0 + self.depth_noise_modulation * depth_feature
        structured = self._zscore(noise * mod)

        alpha = max(0.0, min(1.0, self.depth_noise_blend))
        mixed_noise = self._zscore((1.0 - alpha) * noise + alpha * structured)
        return {"noise": mixed_noise}


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
    parser.add_argument("--depth_noise_blend", type=float, default=0.0, help="0~1. Blend ratio for depth-aware structured noise (inference only)")
    parser.add_argument("--depth_noise_modulation", type=float, default=1.0, help="Depth modulation strength before blend")
    parser.add_argument("--depth_noise_invert", action="store_true", help="Invert depth map before creating structured noise")
    parser.add_argument("--depth_model_path", type=str, default="models/Annotators", help="Local controlnet_aux annotator model path")
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

    if args.depth_noise_blend > 0.0:
        replaced = False
        for i, unit in enumerate(pipe.units):
            if isinstance(unit, Flux2Unit_NoiseInitializer):
                pipe.units[i] = DepthAwareNoiseInitializer(
                    depth_source_image=input_image,
                    depth_noise_blend=args.depth_noise_blend,
                    depth_noise_modulation=args.depth_noise_modulation,
                    depth_noise_invert=args.depth_noise_invert,
                    depth_model_path=args.depth_model_path,
                    annotator_device=args.device,
                )
                replaced = True
                break
        if not replaced:
            raise RuntimeError("Flux2Unit_NoiseInitializer not found in pipeline units.")

        print(
            "Depth-aware noise enabled: "
            f"blend={args.depth_noise_blend:.3f}, modulation={args.depth_noise_modulation:.3f}, "
            f"invert={args.depth_noise_invert}, depth_model_path={args.depth_model_path}"
        )

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
