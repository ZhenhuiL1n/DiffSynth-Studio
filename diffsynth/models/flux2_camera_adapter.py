import torch
import torch.nn as nn
from .general_modules import RMSNorm


class CameraEncoder(nn.Module):
    """Encodes camera parameters (azimuth, elevation) into token embeddings.

    Uses sinusoidal Fourier features for smooth interpolation between angles.
    Input: (batch, 4) = [sin(az), cos(az), sin(el), cos(el)]
    Output: (batch, num_tokens, output_dim)
    """

    def __init__(self, output_dim=4096, num_tokens=4, num_fourier_features=64):
        super().__init__()
        self.num_tokens = num_tokens
        self.output_dim = output_dim
        # 4 raw inputs × num_fourier_features × 2 (sin + cos) + 4 raw
        input_dim = 4 * num_fourier_features * 2 + 4
        # Keep Fourier frequencies fixed during training and deterministic across init/load.
        self.register_buffer("fourier_freqs", self.build_default_fourier_freqs(num_fourier_features))
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim * num_tokens),
        )
        self.norm = nn.LayerNorm(output_dim)

    @staticmethod
    def build_default_fourier_freqs(num_fourier_features: int, dtype=torch.float32, device="cpu"):
        # Deterministic frequency basis so checkpoints without this buffer remain usable.
        base = torch.arange(1, num_fourier_features + 1, dtype=dtype, device=device)
        base = base / float(max(num_fourier_features, 1)) * 2.0
        return base.unsqueeze(0).repeat(4, 1)

    def forward(self, camera_params):
        """camera_params: (batch, 4) = [sin(az), cos(az), sin(el), cos(el)]"""
        # Fourier features: project each input through learned frequencies
        # camera_params: (B, 4), fourier_freqs: (4, F)
        x = camera_params.unsqueeze(-1) * self.fourier_freqs.unsqueeze(0)  # (B, 4, F)
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)  # (B, 4, 2F)
        x = x.reshape(x.shape[0], -1)  # (B, 4*2F)
        x = torch.cat([x, camera_params], dim=-1)  # (B, 4*2F + 4)
        x = self.proj(x)  # (B, output_dim * num_tokens)
        x = x.reshape(-1, self.num_tokens, self.output_dim)  # (B, T, D)
        x = self.norm(x)
        return x


class CameraAdapterModule(nn.Module):
    """Per-block decoupled K/V cross-attention for camera conditioning.

    Same architecture as IpAdapterModule.
    """

    def __init__(self, num_attention_heads, attention_head_dim, input_dim):
        super().__init__()
        self.num_heads = num_attention_heads
        self.head_dim = attention_head_dim
        output_dim = num_attention_heads * attention_head_dim
        self.to_k_cam = nn.Linear(input_dim, output_dim, bias=False)
        self.to_v_cam = nn.Linear(input_dim, output_dim, bias=False)
        self.norm_added_k = RMSNorm(attention_head_dim, eps=1e-5, elementwise_affine=False)

    def forward(self, hidden_states):
        batch_size = hidden_states.shape[0]
        cam_k = self.to_k_cam(hidden_states)
        cam_k = cam_k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        cam_k = self.norm_added_k(cam_k)
        cam_v = self.to_v_cam(hidden_states)
        cam_v = cam_v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        return cam_k, cam_v


class Flux2CameraAdapter(nn.Module):
    """Camera adapter for FLUX.2-klein.

    Can match FLUX.2-klein variants via `from_dit_config`.
    Works for both 4B (24 heads, 25 blocks) and 9B (32 heads, 32 blocks).
    """

    def __init__(
        self,
        num_attention_heads=32,
        attention_head_dim=128,
        cross_attention_dim=4096,
        num_tokens=4,
        num_blocks=32,
        num_fourier_features=64,
    ):
        super().__init__()
        self.camera_encoder = CameraEncoder(
            output_dim=cross_attention_dim,
            num_tokens=num_tokens,
            num_fourier_features=num_fourier_features,
        )
        self.adapter_modules = nn.ModuleList(
            [
                CameraAdapterModule(num_attention_heads, attention_head_dim, cross_attention_dim)
                for _ in range(num_blocks)
            ]
        )
        self.call_block_id = {i: i for i in range(num_blocks)}

    def forward(self, camera_params, scale=1.0):
        """
        Args:
            camera_params: (batch, 4) = [sin(az), cos(az), sin(el), cos(el)]
            scale: float, controls strength of camera conditioning
        Returns:
            dict mapping block_id -> {"cam_k": ..., "cam_v": ..., "scale": ...}
        """
        camera_params = camera_params.to(dtype=self.camera_encoder.fourier_freqs.dtype, device=self.camera_encoder.fourier_freqs.device)
        camera_embeds = self.camera_encoder(camera_params)  # (B, num_tokens, D)
        cam_kv_dict = {}
        for block_id in self.call_block_id:
            adapter_id = self.call_block_id[block_id]
            cam_k, cam_v = self.adapter_modules[adapter_id](camera_embeds)
            cam_kv_dict[block_id] = {
                "cam_k": cam_k,
                "cam_v": cam_v,
                "scale": scale,
            }
        return cam_kv_dict

    @staticmethod
    def from_dit_config(dit):
        """Create a CameraAdapter matching a Flux2DiT's dimensions."""
        if hasattr(dit, "transformer_blocks") and len(dit.transformer_blocks) > 0:
            attn = dit.transformer_blocks[0].attn
            num_heads = getattr(attn, "heads", getattr(attn, "num_heads", 32))
            head_dim = getattr(attn, "head_dim", 128)
            cross_attention_dim = getattr(attn, "inner_dim", num_heads * head_dim)
            num_double = len(dit.transformer_blocks)
            num_single = len(dit.single_transformer_blocks)
        else:
            num_heads = 32
            head_dim = 128
            cross_attention_dim = num_heads * head_dim
            num_double = 8
            num_single = 24

        return Flux2CameraAdapter(
            num_attention_heads=num_heads,
            attention_head_dim=head_dim,
            cross_attention_dim=cross_attention_dim,
            num_blocks=num_double + num_single,
        )

    @staticmethod
    def infer_architecture_from_keys_dict(keys_dict):
        """Infer adapter architecture from a checkpoint key/shape dict."""
        if not isinstance(keys_dict, dict):
            return None

        required = (
            "camera_encoder.proj.0.weight",
            "camera_encoder.proj.2.weight",
        )
        if any(key not in keys_dict for key in required):
            return None

        proj0_shape = list(keys_dict["camera_encoder.proj.0.weight"])
        proj2_shape = list(keys_dict["camera_encoder.proj.2.weight"])
        if len(proj0_shape) != 2 or len(proj2_shape) != 2:
            return None

        cross_attention_dim = int(proj0_shape[0])
        if proj2_shape[1] != cross_attention_dim or cross_attention_dim <= 0:
            return None
        if proj2_shape[0] % cross_attention_dim != 0:
            return None
        num_tokens = int(proj2_shape[0] // cross_attention_dim)
        fourier_key = "camera_encoder.fourier_freqs"
        if fourier_key in keys_dict and len(list(keys_dict[fourier_key])) == 2:
            num_fourier_features = int(list(keys_dict[fourier_key])[1])
        else:
            # input_dim = 4 * num_fourier_features * 2 + 4
            input_dim = int(proj0_shape[1])
            if (input_dim - 4) % 8 != 0:
                return None
            num_fourier_features = int((input_dim - 4) // 8)

        block_ids = []
        for key in keys_dict:
            if key.startswith("adapter_modules.") and key.endswith(".to_k_cam.weight"):
                split_key = key.split(".")
                if len(split_key) >= 4 and split_key[1].isdigit():
                    block_ids.append(int(split_key[1]))
        if len(block_ids) == 0:
            return None
        num_blocks = max(block_ids) + 1

        k0_key = f"adapter_modules.{min(block_ids)}.to_k_cam.weight"
        k0_shape = list(keys_dict.get(k0_key, []))
        if len(k0_shape) != 2 or k0_shape[0] <= 0:
            return None

        attention_head_dim = 128
        if k0_shape[0] % attention_head_dim != 0:
            return None
        num_attention_heads = int(k0_shape[0] // attention_head_dim)

        return {
            "num_attention_heads": num_attention_heads,
            "attention_head_dim": attention_head_dim,
            "cross_attention_dim": cross_attention_dim,
            "num_tokens": num_tokens,
            "num_blocks": num_blocks,
            "num_fourier_features": num_fourier_features,
        }

    @staticmethod
    def state_dict_converter():
        return Flux2CameraAdapterStateDictConverter()


class Flux2CameraAdapterStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        # Direct mapping - no conversion needed for our own format
        return state_dict

    def from_civitai(self, state_dict):
        return self.from_diffusers(state_dict)
