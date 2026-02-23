from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig, Flux2Unit_NoiseInitializer
from diffsynth.utils.controlnet.annotator import Annotator
from PIL import Image
import torch
import argparse
import numpy as np
import torch.nn.functional as F


class DepthAwareNoiseInitializer(Flux2Unit_NoiseInitializer):
    def __init__(
        self,
        depth_source_image: Image.Image = None,
        depth_image: Image.Image = None,
        depth_noise_blend: float = 0.0,
        depth_noise_modulation: float = 1.0,
        depth_noise_invert: bool = False,
        depth_model_path: str = "models/Annotators",
        annotator_device: str = "cuda",
    ):
        super().__init__()
        self.depth_source_image = depth_source_image
        self.depth_image = depth_image
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
        if self.depth_image is not None:
            depth_image = self.depth_image
        else:
            if self.depth_source_image is None:
                raise ValueError("Either depth_source_image or depth_image must be provided.")
            annotator = Annotator("depth", model_path=self.depth_model_path, device=self.annotator_device)
            depth_image = annotator(self.depth_source_image)
        depth_np = np.array(depth_image.convert("L"), dtype=np.float32)
        if self.depth_noise_invert:
            depth_np = 255.0 - depth_np
        depth = torch.from_numpy(depth_np).to(device=device, dtype=torch.float32) / 255.0
        depth = depth.unsqueeze(0).unsqueeze(0)
        depth = F.interpolate(depth, size=(height // 16, width // 16), mode="bilinear", align_corners=False)
        depth = depth.reshape(1, -1, 1).to(dtype=dtype)
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


def main():
    parser = argparse.ArgumentParser(description="FLUX.2-klein-9B Image-to-Image with depth-aware noise")
    parser.add_argument("--input_image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt describing the desired output")
    parser.add_argument("--output", type=str, default="output_img2img_depth_noise.jpg", help="Output image path")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--steps", type=int, default=4, help="Number of inference steps (klein models use ~4)")
    parser.add_argument("--model_id", type=str, default="black-forest-labs/FLUX.2-klein-9B", help="Model ID")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Inference device")
    parser.add_argument("--depth_image", type=str, default=None, help="Path to precomputed depth map image (optional)")
    parser.add_argument("--depth_noise_blend", type=float, default=0.25, help="0~1. Blend ratio for depth-aware structured noise")
    parser.add_argument("--depth_noise_modulation", type=float, default=1.0, help="Depth modulation strength before blend")
    parser.add_argument("--depth_noise_invert", action="store_true", help="Invert depth map before creating structured noise")
    parser.add_argument("--depth_model_path", type=str, default="models/Annotators", help="Local controlnet_aux annotator model path")
    args = parser.parse_args()

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

    input_img = Image.open(args.input_image).convert("RGB")
    print(f"Input image size: {input_img.size}")
    depth_img = Image.open(args.depth_image).convert("L") if args.depth_image is not None else None
    if args.depth_image is not None:
        print(f"Using precomputed depth image: {args.depth_image}")

    replaced = False
    for i, unit in enumerate(pipe.units):
        if isinstance(unit, Flux2Unit_NoiseInitializer):
            pipe.units[i] = DepthAwareNoiseInitializer(
                depth_source_image=input_img,
                depth_image=depth_img,
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

    image = pipe(
        args.prompt,
        edit_image=[input_img],
        seed=args.seed,
        rand_device=args.device,
        num_inference_steps=args.steps,
    )
    image.save(args.output)
    print(f"Saved output to: {args.output}")


if __name__ == "__main__":
    main()
