import torch
import torch.nn as nn
from typing import List, Union
from PIL import Image

from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from ip_adapter.ip_adapter import ImageProjModel

try:
    from safetensors.torch import load_file as safe_load_file
except Exception:
    safe_load_file = None
import os

class WanVideoIPAdapter(nn.Module):
    """
    Turns reference images into 1280-d tokens compatible with WAN `clip_feature`.
    """
    def __init__(
        self,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        clip_vision_model: str = "openai/clip-vit-large-patch14",
        num_tokens: int = 16,
        cross_attention_dim: int = 1280,  # WAN clip_feature dim
        ipadapter_weight_path: Union[str, None] = None,
        freeze_vision: bool = True,
    ):
        super().__init__()
        self.device, self.dtype = device, torch_dtype

        # CLIP-Vision encoder + processor
        self.vision = CLIPVisionModelWithProjection.from_pretrained(
            clip_vision_model
        ).to(device=self.device, dtype=torch.float16)  # keep in fp16
        self.processor = CLIPImageProcessor.from_pretrained(clip_vision_model)

        if freeze_vision:
            for p in self.vision.parameters():
                p.requires_grad = False

        # IP-Adapter projector: (clip_emb_dim -> tokens[num_tokens, cross_attention_dim])
        clip_emb_dim = self.vision.config.projection_dim  # e.g., 768 for ViT-L/14
        self.image_proj = ImageProjModel(
            clip_embeddings_dim=clip_emb_dim,
            cross_attention_dim=cross_attention_dim,
            # num_tokens=num_tokens,
        ).to(device=self.device, dtype=self.dtype)

        # Load IP-Adapter projector weights if provided (e.g., SDXL projector)
        if ipadapter_weight_path is not None and os.path.exists(ipadapter_weight_path):
            sd = None
            if safe_load_file is not None and ipadapter_weight_path.endswith(".safetensors"):
                sd = safe_load_file(ipadapter_weight_path, device="cpu")
            else:
                sd = torch.load(ipadapter_weight_path, map_location="cpu")
            # Weight files from IP-Adapter commonly store the projector under keys like:
            #  - "image_proj.*"  (diffusers-integrations often)
            #  - direct ".*"     (bare)
            # We do a forgiving load.
            matched = {}
            for k, v in sd.items():
                if k.startswith("image_proj."):
                    matched[k.replace("image_proj.", "")] = v
                else:
                    matched[k] = v
            missing, unexpected = self.image_proj.load_state_dict(matched, strict=False)
            if len(missing) > 0:
                print(f"[WanVideoIPAdapter] Missing keys in projector: {missing}")
            if len(unexpected) > 0:
                print(f"[WanVideoIPAdapter] Unexpected keys in projector: {unexpected}")

        self._scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)

    @torch.no_grad()
    def encode(self, pil_images: List[Image.Image]) -> torch.Tensor:
        """
        Returns IP-Adapter tokens to be appended to WAN `clip_feature`.
        Shape: [B, num_tokens, 1280]
        """
        breakpoint()
        pixel = self.processor(images=pil_images, return_tensors="pt")["pixel_values"]
        pixel = pixel.to(self.vision.device, dtype=self.vision.dtype)
        vision_out = self.vision(pixel)
        # .image_embeds: [B, projection_dim], already projected by CLIP head
        image_embeds = vision_out.image_embeds.to(dtype=self.dtype)
        tokens = self.image_proj(image_embeds)  # [B, num_tokens, cross_attention_dim]
        return tokens * self._scale

    def set_scale(self, value: float):
        self._scale.data = torch.tensor(float(value), device=self._scale.device)
        
        
# diffsynth/models/wan_video_ipadapter_ctx.py
import os
from typing import List, Union
from PIL import Image
import torch
import torch.nn as nn

from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from ip_adapter.ip_adapter import ImageProjModel  # pip install git+https://github.com/tencent-ailab/IP-Adapter.git

try:
    from safetensors.torch import load_file as safe_load_file
except Exception:
    safe_load_file = None


class IPAdapterContextBridge(nn.Module):
    """
    Encodes reference images with CLIP-Vision + IP-Adapter projector and maps the
    resulting tokens into WAN's *text* token space (text_dim, e.g., 4096).
    """
    def __init__(
        self,
        text_dim: int,                            # e.g., 4096 (queried from WAN at runtime)
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        clip_vision_model: str = "openai/clip-vit-large-patch14",
        num_tokens: int = 16,
        cross_attention_dim: int = 1280,          # IP-Adapter token dim
        ipadapter_weight_path: Union[str, None] = None,
        freeze_vision: bool = True,
        freeze_projector: bool = True,
    ):
        super().__init__()
        self.device, self.dtype = device, torch_dtype

        # 1) CLIP-Vision
        self.vision = CLIPVisionModelWithProjection.from_pretrained(
            clip_vision_model
        ).to(device=self.device, dtype=torch.float16)
        self.processor = CLIPImageProcessor.from_pretrained(clip_vision_model)
        if freeze_vision:
            for p in self.vision.parameters():
                p.requires_grad = False

        # 2) IP-Adapter projector -> [B, num_tokens, 1280]
        clip_emb_dim = self.vision.config.projection_dim
        self.image_proj = ImageProjModel(
            clip_embeddings_dim=clip_emb_dim,
            cross_attention_dim=cross_attention_dim,
            # num_tokens=num_tokens,
        ).to(device=self.device, dtype=self.dtype)
        if freeze_projector:
            for p in self.image_proj.parameters():
                p.requires_grad = False

        # 3) Small bridge into WAN text token space (e.g., 1280 -> 4096)
        self.to_text = nn.Sequential(
            nn.LayerNorm(cross_attention_dim),
            nn.Linear(cross_attention_dim, text_dim, bias=True),
        ).to(device=self.device, dtype=self.dtype)

        # optional projector weights
        if ipadapter_weight_path and os.path.exists(ipadapter_weight_path):
            sd = safe_load_file(ipadapter_weight_path, device="cpu") if (
                safe_load_file is not None and ipadapter_weight_path.endswith(".safetensors")
            ) else torch.load(ipadapter_weight_path, map_location="cpu")
            # try friendly key loading
            cleaned = {}
            for k, v in sd.items():
                cleaned[k.replace("image_proj.", "") if k.startswith("image_proj.") else k] = v
            self.image_proj.load_state_dict(cleaned, strict=False)

        self._scale = nn.Parameter(torch.tensor(1.0, device=self.device), requires_grad=False)

    @torch.no_grad()
    def encode_to_text(self, images: List[Image.Image]) -> torch.Tensor:
        """
        -> Tokens in *text* space: [B, N, text_dim]
        """
        if len(images) == 0:
            raise ValueError("IPAdapterContextBridge.encode_to_text: empty image list")

        px = self.processor(images=images, return_tensors="pt")["pixel_values"]
        px = px.to(self.vision.device, dtype=self.vision.dtype)
        img_emb = self.vision(px).image_embeds.to(dtype=self.dtype)           # [B, clip_emb_dim]
        tok1280 = self.image_proj(img_emb)                                    # [B, N, 1280]
        tok4096 = self.to_text(tok1280) * self._scale                         # [B, N, text_dim]
        return tok4096

    def set_scale(self, value: float):
        self._scale.data = torch.tensor(float(value), device=self._scale.device)

