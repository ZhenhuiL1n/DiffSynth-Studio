import torch
import torch.nn as nn
from typing import Optional
from .general_modules import RMSNorm


class CameraMLPResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.ff(self.norm(x))


class LegacyCameraEncoder(nn.Module):
    """Original camera encoder kept for backward compatibility with old checkpoints."""

    def __init__(self, output_dim=4096, num_tokens=4, num_fourier_features=64):
        super().__init__()
        self.num_tokens = num_tokens
        self.output_dim = output_dim
        input_dim = 4 * num_fourier_features * 2 + 4
        self.register_buffer("fourier_freqs", self.build_default_fourier_freqs(num_fourier_features))
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim * num_tokens),
        )
        self.norm = nn.LayerNorm(output_dim)

    @staticmethod
    def build_default_fourier_freqs(num_fourier_features: int, dtype=torch.float32, device="cpu"):
        base = torch.arange(1, num_fourier_features + 1, dtype=dtype, device=device)
        base = base / float(max(num_fourier_features, 1)) * 2.0
        return base.unsqueeze(0).repeat(4, 1)

    def forward(self, camera_params):
        x = camera_params.unsqueeze(-1) * self.fourier_freqs.unsqueeze(0)
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        x = x.reshape(x.shape[0], -1)
        x = torch.cat([x, camera_params], dim=-1)
        x = self.proj(x)
        x = x.reshape(-1, self.num_tokens, self.output_dim)
        x = self.norm(x)
        return x


class CameraEncoder(nn.Module):
    """Control-MLP camera encoder.

    Input: (batch, 4) = [sin(az), cos(az), sin(el), cos(el)]
    Output: (batch, num_tokens, output_dim)
    """

    def __init__(
        self,
        output_dim=4096,
        num_tokens=4,
        num_fourier_features=64,
        hidden_dim=1024,
        num_mlp_blocks=2,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        input_dim = 4 * num_fourier_features * 2 + 4
        self.register_buffer("fourier_freqs", LegacyCameraEncoder.build_default_fourier_freqs(num_fourier_features))
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([CameraMLPResidualBlock(hidden_dim) for _ in range(num_mlp_blocks)])
        self.output_proj = nn.Linear(hidden_dim, output_dim * num_tokens)
        self.token_bias = nn.Parameter(torch.zeros(num_tokens, output_dim))
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, camera_params):
        x = camera_params.unsqueeze(-1) * self.fourier_freqs.unsqueeze(0)
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        x = x.reshape(x.shape[0], -1)
        x = torch.cat([x, camera_params], dim=-1)
        x = self.input_proj(x)
        x = torch.nn.functional.silu(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_proj(x).reshape(-1, self.num_tokens, self.output_dim)
        x = x + self.token_bias.unsqueeze(0)
        x = self.norm(x)
        return x


class CameraAdapterModule(nn.Module):
    """Per-block decoupled K/V camera cross-attention.

    Supports both full projection and low-rank factorized projection.
    """

    def __init__(self, num_attention_heads, attention_head_dim, input_dim, projection_rank: Optional[int] = 256):
        super().__init__()
        self.num_heads = num_attention_heads
        self.head_dim = attention_head_dim
        output_dim = num_attention_heads * attention_head_dim
        self.projection_rank = projection_rank

        self.use_factorized_projection = (
            projection_rank is not None and 0 < int(projection_rank) < min(input_dim, output_dim)
        )
        if self.use_factorized_projection:
            rank = int(projection_rank)
            self.to_k_cam_in = nn.Linear(input_dim, rank, bias=False)
            self.to_k_cam_out = nn.Linear(rank, output_dim, bias=False)
            self.to_v_cam_in = nn.Linear(input_dim, rank, bias=False)
            self.to_v_cam_out = nn.Linear(rank, output_dim, bias=False)
        else:
            self.to_k_cam = nn.Linear(input_dim, output_dim, bias=False)
            self.to_v_cam = nn.Linear(input_dim, output_dim, bias=False)
        self.norm_added_k = RMSNorm(attention_head_dim, eps=1e-5, elementwise_affine=False)

    def forward(self, hidden_states):
        batch_size = hidden_states.shape[0]
        if self.use_factorized_projection:
            cam_k = self.to_k_cam_out(self.to_k_cam_in(hidden_states))
            cam_v = self.to_v_cam_out(self.to_v_cam_in(hidden_states))
        else:
            cam_k = self.to_k_cam(hidden_states)
            cam_v = self.to_v_cam(hidden_states)
        cam_k = cam_k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        cam_k = self.norm_added_k(cam_k)
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
        encoder_type="control_mlp",
        camera_hidden_dim=1024,
        camera_num_mlp_blocks=2,
        projection_rank: Optional[int] = 256,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        if encoder_type == "legacy":
            self.camera_encoder = LegacyCameraEncoder(
                output_dim=cross_attention_dim,
                num_tokens=num_tokens,
                num_fourier_features=num_fourier_features,
            )
        else:
            self.camera_encoder = CameraEncoder(
                output_dim=cross_attention_dim,
                num_tokens=num_tokens,
                num_fourier_features=num_fourier_features,
                hidden_dim=camera_hidden_dim,
                num_mlp_blocks=camera_num_mlp_blocks,
            )
        self.adapter_modules = nn.ModuleList(
            [
                CameraAdapterModule(
                    num_attention_heads,
                    attention_head_dim,
                    cross_attention_dim,
                    projection_rank=projection_rank,
                )
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
        """Create a CameraAdapter matching a Flux2DiT's dimensions.

        Defaults to Control-MLP encoder + low-rank per-block projection.
        """
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
            encoder_type="control_mlp",
            camera_hidden_dim=1024,
            camera_num_mlp_blocks=2,
            projection_rank=256,
        )

    @staticmethod
    def infer_architecture_from_keys_dict(keys_dict):
        """Infer adapter architecture from a checkpoint key/shape dict."""
        if not isinstance(keys_dict, dict):
            return None

        encoder_type = None
        camera_hidden_dim = 1024
        camera_num_mlp_blocks = 2
        cross_attention_dim = None
        num_tokens = None

        # New control-MLP encoder format.
        if "camera_encoder.input_proj.weight" in keys_dict and "camera_encoder.output_proj.weight" in keys_dict:
            encoder_type = "control_mlp"
            in_shape = list(keys_dict["camera_encoder.input_proj.weight"])
            out_shape = list(keys_dict["camera_encoder.output_proj.weight"])
            if len(in_shape) != 2 or len(out_shape) != 2:
                return None
            camera_hidden_dim = int(in_shape[0])
            output_rows = int(out_shape[0])
            if camera_hidden_dim <= 0 or out_shape[1] != camera_hidden_dim:
                return None

            if "camera_encoder.token_bias" in keys_dict and len(list(keys_dict["camera_encoder.token_bias"])) == 2:
                token_bias_shape = list(keys_dict["camera_encoder.token_bias"])
                num_tokens = int(token_bias_shape[0])
                cross_attention_dim = int(token_bias_shape[1])
                if output_rows != num_tokens * cross_attention_dim:
                    return None
            else:
                # Fallback: assume default 4 tokens if token bias is missing.
                num_tokens = 4
                if output_rows % num_tokens != 0:
                    return None
                cross_attention_dim = int(output_rows // num_tokens)

            block_ids = []
            for key in keys_dict:
                if key.startswith("camera_encoder.blocks.") and key.endswith(".ff.0.weight"):
                    parts = key.split(".")
                    if len(parts) >= 4 and parts[2].isdigit():
                        block_ids.append(int(parts[2]))
            if len(block_ids) > 0:
                camera_num_mlp_blocks = max(block_ids) + 1

        # Legacy encoder format.
        elif "camera_encoder.proj.0.weight" in keys_dict and "camera_encoder.proj.2.weight" in keys_dict:
            encoder_type = "legacy"
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
        else:
            return None

        fourier_key = "camera_encoder.fourier_freqs"
        if fourier_key in keys_dict and len(list(keys_dict[fourier_key])) == 2:
            num_fourier_features = int(list(keys_dict[fourier_key])[1])
        else:
            if encoder_type == "legacy":
                input_dim = int(list(keys_dict["camera_encoder.proj.0.weight"])[1])
            else:
                input_dim = int(list(keys_dict["camera_encoder.input_proj.weight"])[1])
            if (input_dim - 4) % 8 != 0:
                return None
            num_fourier_features = int((input_dim - 4) // 8)

        projection_rank = None
        block_ids = []
        for key in keys_dict:
            if key.startswith("adapter_modules.") and (
                key.endswith(".to_k_cam.weight") or key.endswith(".to_k_cam_out.weight")
            ):
                split_key = key.split(".")
                if len(split_key) >= 4 and split_key[1].isdigit():
                    block_ids.append(int(split_key[1]))
        if len(block_ids) == 0:
            return None
        num_blocks = max(block_ids) + 1

        block0 = min(block_ids)
        k0_full_key = f"adapter_modules.{block0}.to_k_cam.weight"
        k0_factorized_out_key = f"adapter_modules.{block0}.to_k_cam_out.weight"
        k0_factorized_in_key = f"adapter_modules.{block0}.to_k_cam_in.weight"
        if k0_full_key in keys_dict:
            k0_shape = list(keys_dict.get(k0_full_key, []))
            if len(k0_shape) != 2 or k0_shape[0] <= 0:
                return None
        elif k0_factorized_out_key in keys_dict and k0_factorized_in_key in keys_dict:
            k0_shape = list(keys_dict.get(k0_factorized_out_key, []))
            k0_in_shape = list(keys_dict.get(k0_factorized_in_key, []))
            if len(k0_shape) != 2 or len(k0_in_shape) != 2 or k0_shape[0] <= 0:
                return None
            projection_rank = int(k0_in_shape[0])
        else:
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
            "encoder_type": encoder_type,
            "camera_hidden_dim": camera_hidden_dim,
            "camera_num_mlp_blocks": camera_num_mlp_blocks,
            "projection_rank": projection_rank,
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
