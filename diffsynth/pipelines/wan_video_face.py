import torch, warnings, glob, os, types, random
import torchvision.ops as tvops
import numpy as np
from PIL import Image
from einops import repeat, reduce
from typing import Optional, Union
from dataclasses import dataclass
from modelscope import snapshot_download
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional, List, Union
from typing_extensions import Literal

from ..utils import BasePipeline, ModelConfig, PipelineUnit, PipelineUnitRunner
from ..models import ModelManager, load_state_dict
from ..models.wan_video_dit import WanModel, RMSNorm, sinusoidal_embedding_1d
from ..models.wan_video_dit_s2v import rope_precompute
from ..models.wan_video_text_encoder import WanTextEncoder, T5RelativeEmbedding, T5LayerNorm
from ..models.wan_video_vae import WanVideoVAE, RMS_norm, CausalConv3d, Upsample
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.wan_video_vace import VaceWanModel
from ..models.wan_video_motion_controller import WanMotionControllerModel
from ..models.wan_video_animate_adapter import WanAnimateAdapter
from ..schedulers.flow_match import FlowMatchScheduler
from ..prompters import WanPrompter
from ..vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear, WanAutoCastLayerNorm
from ..lora import GeneralLoRALoader
from .wan_video_ipadapter import WanVideoIPAdapter, IPAdapterContextBridge


class WanVideoPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16, tokenizer_path=None):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16, time_division_factor=4, time_division_remainder=1
        )
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.dit2: WanModel = None
        self.vae: WanVideoVAE = None
        self.motion_controller: WanMotionControllerModel = None
        self.vace: VaceWanModel = None
        self.vace2: VaceWanModel = None
        self.animate_adapter: WanAnimateAdapter = None
        self.in_iteration_models = ("dit", "motion_controller", "vace", "animate_adapter")
        self.in_iteration_models_2 = ("dit2", "motion_controller", "vace2", "animate_adapter")
        self.unit_runner = PipelineUnitRunner()
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_FaceBBox(
                detector=self._build_face_detector((832, 480)),
                det_size=(832, 480),
                prob=1.0,
                run_in_train_only=True,
            ),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_S2V(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_ImageEmbedderVAE(),
            WanVideoUnit_ImageEmbedderCLIP(),
            WanVideoUnit_ImageEmbedderFused(),
            WanVideoUnit_FunControl(),
            WanVideoUnit_FunReference(),
            WanVideoUnit_FunCameraControl(),
            WanVideoUnit_SpeedControl(),
            WanVideoUnit_VACE(),
            WanVideoPostUnit_AnimateVideoSplit(),
            WanVideoPostUnit_AnimatePoseLatents(),
            WanVideoPostUnit_AnimateFacePixelValues(),
            WanVideoPostUnit_AnimateInpaint(),
            WanVideoUnit_UnifiedSequenceParallel(),
            WanVideoUnit_TeaCache(),
            WanVideoUnit_CfgMerger(),
        ]
        self.post_units = [
            WanVideoPostUnit_S2V(),
        ]
        self.model_fn = model_fn_wan_video
        self.lpips_loss_fn = LPIPS(net='vgg')
        self.lpips_loss_fn.to(self.device)
        self.lpips_loss_fn.eval()
        
        self.ipadapter = None
        self.ipadapter_scale = 0.0
        self.ipadapter_num_tokens = 16
        
    def _build_face_detector(self, det_size=(1024,1024)):
        try:
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        except:
            local_rank = 0
        app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection"],
            providers=[('CUDAExecutionProvider', {"device_id": local_rank})],
        )
        app.prepare(ctx_id=local_rank, det_size=det_size)
        return app
    
    def enable_ipadapter(
        self,
        clip_vision_model: str = "openai/clip-vit-large-patch14",
        ipadapter_weight_path: str = "models/IpAdapter/stable_diffusion_xl/ip-adapter_sdxl.safetensors",
        num_tokens: int = 16,
        scale: float = 0.6,
    ):
        # WAN uses 1280-d `clip_feature` tokens (see wan_video.py).
        self.ipadapter = WanVideoIPAdapter(
            device=self.device,
            torch_dtype=self.torch_dtype,
            clip_vision_model=clip_vision_model,
            num_tokens=num_tokens,
            cross_attention_dim=1280,
            ipadapter_weight_path=ipadapter_weight_path,
        )
        self.ipadapter.set_scale(scale)
        self.ipadapter_scale = scale
        self.ipadapter_num_tokens = num_tokens
        self.ipadapter.eval()
        
    def enable_ipadapter_context(
        self,
        clip_vision_model: str = "openai/clip-vit-large-patch14",
        ipadapter_weight_path: str = "models/IpAdapter/sdxl/ip-adapter_sdxl.safetensors",
        num_tokens: int = 16,
        scale: float = 0.6,
        freeze_vision: bool = True,
        freeze_projector: bool = True,
    ):
        # WAN text token dim (e.g., 4096). Safer to read from the model:
        text_dim = int(self.dit.text_embedding[0].in_features)  # Linear(text_dim -> dim)
        self.ip_ctx = IPAdapterContextBridge(
            text_dim=text_dim,
            device=self.device,
            torch_dtype=self.torch_dtype,
            clip_vision_model=clip_vision_model,
            num_tokens=num_tokens,
            cross_attention_dim=1280,
            ipadapter_weight_path=ipadapter_weight_path,
            freeze_vision=freeze_vision,
            freeze_projector=freeze_projector,
        )
        self.ip_ctx.set_scale(scale)
        self.ip_ctx_scale = scale
        self.ip_ctx_num_tokens = num_tokens
        # self.ip_ctx.eval()
        
    @torch.no_grad()
    def _append_ipadapter_tokens(
        self,
        clip_feature: torch.Tensor,
        ip_images: List[Image.Image],
        t_sigma: torch.Tensor = None,
        t_gate_range=(0.0, 0.3),
    ) -> torch.Tensor:
        """
        Appends IP-Adapter tokens to existing WAN `clip_feature`.
        - clip_feature: [B, Lc, 1280]
        - returns:      [B, Lc + N, 1280]
        Optional `t_sigma`: per-step noise (sigma) to gate IP-Adapter strength at low noise.
        """
        if self.ipadapter is None or self.ipadapter_scale <= 0.0 or not ip_images:
            return clip_feature

        ipa = self.ipadapter.encode(ip_images).to(dtype=clip_feature.dtype, device=clip_feature.device)  # [B, N, 1280]

        # Optional: gate by sigma so IP-Adapter is strongest late in denoising
        if t_sigma is not None:
            # linear gate: 1.0 at sigma<=t_gate_range[0], 0.0 at sigma>=t_gate_range[1]
            lo, hi = t_gate_range
            g = (hi - t_sigma.clamp(min=lo, max=hi)) / max(hi - lo, 1e-6)
            g = g.clamp(0, 1).view(-1, 1, 1)  # [B,1,1]
            ipa = ipa * g

        return torch.cat([clip_feature, ipa], dim=1)
    
    @torch.no_grad()
    def _append_ip_to_context(
        self,
        context: torch.Tensor,               # [B_ctx, L, text_dim]
        ip_images: list,                     # list[PIL.Image]
        t_sigma: torch.Tensor = None,        # [B] (from scheduler sigmas)
        t_gate_range=(0.0, 0.30),            # active near low noise
    ) -> torch.Tensor:
        if (self.ip_ctx is None) or (self.ip_ctx_scale <= 0.0) or (not ip_images):
            return context

        ipa = self.ip_ctx.encode_to_text(ip_images)  # [B_ipa, N, text_dim]
        ipa = ipa.to(device=context.device, dtype=context.dtype)

        # If batch sizes differ (CFG or batching), repeat IPA across batch.
        B_ctx, _, _ = context.shape
        B_ipa = ipa.shape[0]
        if B_ipa == 1 and B_ctx > 1:
            ipa = ipa.repeat(B_ctx, 1, 1)
        elif B_ipa != B_ctx:
            ipa = ipa[:B_ctx]

        # Optional: gate by noise sigma so guidance is strongest at late steps.
        if t_sigma is not None:
            lo, hi = t_gate_range
            g = (hi - t_sigma.clamp(min=lo, max=hi)) / max(hi - lo, 1e-6)
            g = g.clamp(0, 1).view(-1, 1, 1)         # [B,1,1]
            ipa = ipa * g

        # Prepend IPA tokens so they get high attention priority.
        return torch.cat([ipa, context], dim=1)

    
    def load_lora(
        self,
        module: torch.nn.Module,
        lora_config: Union[ModelConfig, str] = None,
        alpha=1,
        hotload=False,
        state_dict=None,
    ):
        if state_dict is None:
            if isinstance(lora_config, str):
                lora = load_state_dict(lora_config, torch_dtype=self.torch_dtype, device=self.device)
            else:
                lora_config.download_if_necessary()
                lora = load_state_dict(lora_config.path, torch_dtype=self.torch_dtype, device=self.device)
        else:
            lora = state_dict
        if hotload:
            for name, module in module.named_modules():
                if isinstance(module, AutoWrappedLinear):
                    lora_a_name = f'{name}.lora_A.default.weight'
                    lora_b_name = f'{name}.lora_B.default.weight'
                    if lora_a_name in lora and lora_b_name in lora:
                        module.lora_A_weights.append(lora[lora_a_name] * alpha)
                        module.lora_B_weights.append(lora[lora_b_name])
        else:
            loader = GeneralLoRALoader(torch_dtype=self.torch_dtype, device=self.device)
            loader.load(module, lora, alpha=alpha)
            
    @torch.no_grad()
    def _sigma_from_timestep_id(self, timestep_id: torch.Tensor) -> torch.Tensor:
         # scheduler.sigmas: shape [num_inference_steps]
         sigma = self.scheduler.sigmas[timestep_id.cpu()]
         return sigma.to(dtype=self.torch_dtype, device=self.device)
 
    def _predict_x0_from_model_output(self, noisy_sample: torch.Tensor, model_output: torch.Tensor, sigma: torch.Tensor):
         """
         In Flow-Match training used here:
             x_sigma = (1 - sigma) * x0   sigma * eps
             d x_sigma / d sigma = (eps - x0)  (this is the network's target/output)
         =>  x0 = x_sigma - sigma * (eps - x0) = x_sigma - sigma * model_output
         """
         while sigma.ndim < noisy_sample.ndim:
             sigma = sigma.view(-1, *([1] * (noisy_sample.ndim - 1)))  # broadcast
         return noisy_sample - sigma * model_output
     
    # def _build_face_mask_lat(
    #     self,
    #     face_boxes: list,                # list of [x1,y1,x2,y2] per *input* frame (pixel coords)
    #     face_src_size: tuple,            # (H_src, W_src) of the frames that boxes refer to
    #     T_lat: int, H_lat: int, W_lat: int,
    #     device,
    #     dilate_time: bool = True,
    # ):
    #     """
    #     Map frame-level pixel bboxes -> binary mask on the latent grid [T_lat, H_lat, W_lat].
    #     Respects the pipeline's temporal packing: remainder=1 (first frame), stride=4 afterwards.
    #     """
    #     import numpy as np
    #     mask = torch.zeros((T_lat, H_lat, W_lat), device=device, dtype=torch.float32)
    #     if not face_boxes or face_src_size is None:
    #         return mask

    #     H_src, W_src = int(face_src_size[0]), int(face_src_size[1])
    #     sx = W_lat / max(1, W_src)
    #     sy = H_lat / max(1, H_src)

    #     stride_t = getattr(self, "time_division_factor", 4)
    #     rem_t    = getattr(self, "time_division_remainder", 1)

    #     for t_in, (x1, y1, x2, y2) in enumerate(face_boxes):
    #         if x2 <= x1 or y2 <= y1:
    #             continue
    #         # temporal downsample: 0..(rem_t-1) -> 0, then buckets of stride_t
    #         t_lat = 0 if t_in < rem_t else 1 + (t_in - rem_t) // stride_t
    #         if t_lat < 0 or t_lat >= T_lat:
    #             continue

    #         x1l = max(0, int(np.floor(x1 * sx))); x2l = min(W_lat, int(np.ceil(x2 * sx)))
    #         y1l = max(0, int(np.floor(y1 * sy))); y2l = min(H_lat, int(np.ceil(y2 * sy)))
    #         if x2l > x1l and y2l > y1l:
    #             mask[t_lat, y1l:y2l, x1l:x2l] = 1.0

    #     if dilate_time:
    #         # cover temporal receptive field boundaries
    #         mask = torch.nn.functional.max_pool3d(mask[None,None], kernel_size=(3,1,1), stride=1, padding=(1,0,0)).squeeze(0).squeeze(0)
    #     return mask
    
    def vae_decode_time_chunked(
        self,
        latents: torch.Tensor,
        chunk_latents: int = 5,
        overlap_latents: int = 1,
        *,
        tiled: bool = True,
        tile_size: tuple | None = (30, 52),
        tile_stride: tuple | None = (15, 26),
        use_gradient_checkpointing: bool = False,
    ):
        assert latents.ndim == 5, f"expected [B,Cz,Tz,Hl,Wl], got {latents.shape}"
        B, Cz, Tz, Hl, Wl = latents.shape
        assert chunk_latents >= 2, "chunk_latents must be >= 2"
        assert 1 <= overlap_latents < chunk_latents, "need 1 <= overlap < chunk"

        starts = [0]
        stride = chunk_latents - overlap_latents
        # breakpoint()
        while True:
            nxt = starts[-1] + stride
            if nxt + chunk_latents >= Tz:
                if starts[-1] + chunk_latents != Tz:
                    starts.append(max(Tz - chunk_latents, 0))
                break
            starts.append(nxt)

        outs = []
        for i, s in enumerate(starts):
            e = min(s + chunk_latents, Tz)
            z_slice = latents[:, :, s:e]

            if use_gradient_checkpointing:
                def _fn(x):
                    return self.vae.decode(
                        x, device=self.device,
                        tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
                    )
                out_chunk = torch.utils.checkpoint.checkpoint(_fn, z_slice, use_reentrant=False)
            else:
                out_chunk = self.vae.decode(
                    z_slice, device=self.device,
                    tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
                )

            if i == 0:
                outs.append(out_chunk)
            else:
                drop_pix = 1 + 4 * (overlap_latents - 1)   # H=1 => 1; H=2 => 5; ...
                outs.append(out_chunk[:, :, drop_pix:, :, :])

            del out_chunk

        video = torch.cat(outs, dim=2)  # [B,3,T,H,W]
        return video
    
    def training_loss(self, **inputs):
        use_decoded = inputs.get("use_decoded_loss", False)
        if use_decoded:
            return self.training_loss_on_decoded(**inputs)
        else:
            return self.training_loss_on_latents(**inputs)
    
    def training_loss_on_decoded(self, **inputs):
        import os, numpy as np, torch
        import torch.nn.functional as F
        import torchvision.ops as tvops
        from PIL import Image, ImageDraw
        # self.vae.eval()

        max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * self.scheduler.num_train_timesteps)
        min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * self.scheduler.num_train_timesteps)
        timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
        timestep = self.scheduler.timesteps[timestep_id].to(dtype=self.torch_dtype, device=self.device)

        inputs["latents"] = self.scheduler.add_noise(inputs["input_latents"], inputs["noise"], timestep)
        training_target = self.scheduler.training_target(inputs["input_latents"], inputs["noise"], timestep)

        noise_pred = self.model_fn(**inputs, timestep=timestep)

        loss_latent = F.mse_loss(noise_pred.float(), training_target.float())
        loss_latent = loss_latent * self.scheduler.training_weight(timestep)
        # loss = loss_latent
        # breakpoint()
        tiled = bool(inputs.get("tiled", True))
        tile_size = inputs.get("tile_size", (30, 52))
        tile_stride = inputs.get("tile_stride", (15, 26))

        chunk_latents = int(inputs.get("chunk_latents", 5))
        overlap_latents = int(inputs.get("overlap_latents", 1))
        vae_ckpt = bool(inputs.get("vae_ckpt", True))

        sigma = self._sigma_from_timestep_id(timestep_id)
        x0_pred_lat = self._predict_x0_from_model_output(inputs["latents"], noise_pred, sigma)

        pred_vid = self.vae_decode_time_chunked(
            x0_pred_lat,
            chunk_latents=chunk_latents,
            overlap_latents=overlap_latents,
            tiled=tiled, tile_size=tile_size, tile_stride=tile_stride,
            use_gradient_checkpointing=vae_ckpt,
        )
        pred_imgs = ((pred_vid.clamp(-1, 1) + 1) / 2).permute(0, 2, 1, 3, 4).contiguous()
        B, T, C, H, W = pred_imgs.shape

        gt_input = inputs.get("input_video")
        if isinstance(gt_input, (list, tuple)):
            gt_vid_tensor = self.preprocess_video(gt_input)  # [-1,1], [1,3,T,H,W]
        elif isinstance(gt_input, torch.Tensor):
            gt_vid_tensor = gt_input
        else:
            gt_vid_tensor = self.vae_decode_time_chunked(
                inputs["input_latents"],
                chunk_latents=chunk_latents,
                overlap_latents=overlap_latents,
                tiled=tiled, tile_size=tile_size, tile_stride=tile_stride,
                use_gradient_checkpointing=vae_ckpt,
            )
            
        gt_imgs = ((gt_vid_tensor.clamp(-1, 1) + 1) / 2).permute(0, 2, 1, 3, 4).contiguous()  # [B,T,C,H,W]

        face_rois = inputs.get("face_rois", None)
        if (face_rois is None) or (isinstance(face_rois, torch.Tensor) and face_rois.numel() == 0):
            boxes = inputs.get("face_boxes", None)
            if boxes is not None and len(boxes) > 0:
                rois = []
                for t, (x1, y1, x2, y2) in enumerate(boxes):
                    if x2 > x1 and y2 > y1:
                        rois.append([float(t), float(x1), float(y1), float(x2), float(y2)])
                face_rois = torch.tensor(rois, dtype=torch.float16, device=self.device) if len(rois) else None

        face_loss_weight = float(inputs.get("face_loss_weight", 0.2))
        face_crop_size = inputs.get("face_crop_size", (128, 128))

        loss_face = torch.tensor(0.0, dtype=pred_imgs.dtype, device=self.device)
        if (face_rois is not None) and (face_rois.numel() > 0):
            pred_bt = pred_imgs.view(B * T, C, H, W)
            # breakpoint()
            gt_bt   = gt_imgs.view(B * T, C, H, W)
            # breakpoint()
            crops_pred = tvops.roi_align(pred_bt.to(torch.float16), face_rois, output_size=face_crop_size, aligned=True)
            crops_gt   = tvops.roi_align(gt_bt.to(torch.float16),   face_rois, output_size=face_crop_size, aligned=True)
            loss_face = F.mse_loss(crops_pred, crops_gt) + self.lpips_loss_fn(crops_pred * 2 - 1, crops_gt * 2 - 1).mean()

        loss = loss_latent + face_loss_weight * loss_face

        if bool(inputs.get("save_debug", False)):
            os.makedirs("./debug", exist_ok=True)
            if isinstance(gt_input, (list, tuple)) and len(gt_input) == T:
                gt_frames_pil = list(gt_input)
            else:
                gt_u8 = (gt_imgs[0].detach().cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)  # [T,H,W,3], RGB
                gt_frames_pil = [Image.fromarray(gt_u8[t]) for t in range(T)]
            # breakpoint()
            pred_u8 = (pred_imgs[0].to(torch.float16).detach().cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
            pred_frames_pil = [Image.fromarray(pred_u8[t]) for t in range(T)]

            boxes = inputs.get("face_boxes", None) or []
            for t in range(T):
                if t < len(boxes):
                    x1, y1, x2, y2 = boxes[t]
                    if x2 > x1 and y2 > y1:
                        dgt = ImageDraw.Draw(gt_frames_pil[t])
                        dpd = ImageDraw.Draw(pred_frames_pil[t])
                        color = (0, 255, 0)
                        for k in range(2):
                            dgt.rectangle([x1 - k, y1 - k, x2 + k, y2 + k], outline=color)
                            dpd.rectangle([x1 - k, y1 - k, x2 + k, y2 + k], outline=color)
                gt_im   = gt_frames_pil[t]
                pred_im = pred_frames_pil[t]

                if pred_im.size != gt_im.size:
                    pred_im = pred_im.resize(gt_im.size, Image.BILINEAR)
                if pred_im.mode != gt_im.mode:
                    pred_im = pred_im.convert(gt_im.mode)

                w, h = gt_im.size
                frame_to_save = Image.new(gt_im.mode, (w * 2, h))
                frame_to_save.paste(gt_im, (0, 0))
                frame_to_save.paste(pred_im, (w, 0))

                frame_to_save.save(f"./debug/concat_{t:03d}.png")
        # breakpoint()
        return loss
        
    def training_loss_on_latents(self, **inputs):
        max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * self.scheduler.num_train_timesteps)
        min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * self.scheduler.num_train_timesteps)
        timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
        timestep = self.scheduler.timesteps[timestep_id].to(dtype=self.torch_dtype, device=self.device)
        # breakpoint()
        inputs["latents"] = self.scheduler.add_noise(inputs["input_latents"], inputs["noise"], timestep)
        # training_target = self.scheduler.training_target(inputs["input_latents"], inputs["noise"], timestep)
        tgt_full = self.scheduler.training_target(inputs["input_latents"], inputs["noise"], timestep)

        # breakpoint()
        # breakpoint()
        # --- IP-Adapter → context injection (works even when clip_feature is absent) ---
        if "context" in inputs and self.ip_ctx is not None and self.ip_ctx_scale > 0.0:
            # Use explicit ipadapter_images if provided; otherwise reuse detected face crops.
            ip_imgs = inputs.get("ipadapter_images") or inputs.get("face_masked_images") or []
            if isinstance(ip_imgs, Image.Image):
                ip_imgs = [ip_imgs]

            # per-step sigma for gating
            t_sigma = self._sigma_from_timestep_id(timestep_id).to(inputs["context"].device)
            inputs["context"] = self._append_ip_to_context(inputs["context"], ip_imgs, t_sigma=t_sigma)
# -------------------------------------------------------------------------------

        # cond_text = self.encode_prompt(prompt, positive=True)        # {"context": ...}
        # cond_img  = self.encode_image(image, end_image, T, H, W, ...)# {"clip_feature": ..., "y": ...}
        # clip_feature, y = cond_img["clip_feature"], cond_img["y"]
        # # Add IP-Adapter tokens into clip_feature (decode-free, tiny VRAM)
        # if ipadapter_images is not None and len(ipadapter_images) > 0:
        #     # ipadapter_images: List[PIL.Image.Image], typically your face crops or a single identity image
        #     # Optional: t_sigma from your scheduler/noise for gating (if you have it at this point)
        #     t_sigma = None
        #     if isinstance(t, torch.Tensor) and hasattr(self.scheduler, "sigma"):
        #         t_sigma = self.scheduler.sigma(t).to(clip_feature.device).view(clip_feature.shape[0])

        #     clip_feature = self._append_ipadapter_tokens(
        #         clip_feature, ipadapter_images, t_sigma=t_sigma
        #     )
        # face_target = self.scheduler.training_target(inputs["face_latents"], inputs["noise"], timestep) 
        noise_pred = self.model_fn(**inputs, timestep=timestep)
        
        # noise_pred = self.model_fn(
        #     **inputs, timestep=timestep,
        #     context=cond_text["context"],    # text tokens
        #     clip_feature=clip_feature,       # NOW includes IP-Adapter tokens
        #     y=y,                             # WAN latent video cond
        #     # timestep=t,
        # )
        
        loss = torch.nn.functional.mse_loss(noise_pred.float(), tgt_full.float())
        loss = loss * self.scheduler.training_weight(timestep)            
            
        if "face_latents" in inputs and inputs["face_latents"] is not None:
            base_err = F.mse_loss(noise_pred.float(), tgt_full.float(), reduction='none')  # [B,C,T,Hl,Wl]
            per_loc  = base_err.mean(dim=1, keepdim=True)
            # delta is large where background was zeroed, small on the face
            delta = (inputs["input_latents"] - inputs["face_latents"]).float()             # [B,Cz,T,Hl,Wl]
            d2 = delta.pow(2).mean(dim=1, keepdim=True)                                 # [B,1,T,Hl,Wl]
            d2max = d2.amax(dim=(2,3,4), keepdim=True).clamp_min(1e-6)
            bg = d2 / d2max                                                              # 0≈face, 1≈background
            W_face = 1.0 - bg                                                               # 1≈face, 0≈background
            # smooth a bit to avoid speckles
            W_face = torch.nn.functional.avg_pool3d(W_face, kernel_size=(1,3,3), stride=1, padding=(0,1,1))

            # optional: later timesteps carry sharper face signal
            sigma = self._sigma_from_timestep_id(timestep_id)
            gate  = (1.0 - sigma.clamp(0,1)).view(1,1,1,1,1)                                # scale ∈[0,1]

            alpha = float(inputs.get("face_weight", 1))
            face_loss = (per_loc * W_face).sum() / (W_face.sum()).clamp_min(1.0) # area-normalized     * noise_pred.shape[1]
            loss = loss + alpha * gate * self.scheduler.training_weight(timestep) * face_loss
        
        # loss_face = torch.nn.functional.mse_loss(noise_pred.float(), face_target.float())
        
        # loss += 0.2 * loss_face * self.scheduler.training_weight(timestep) 
        # breakpoint()
        
        return loss
        
        # face_stream_weight = float(inputs.get("face_stream_weight", 0.3))     # weight for face stream loss
        # face_bbox_margin   = int(inputs.get("face_bbox_margin", 6))           # px to expand bbox before masking
        # mask_threshold     = float(inputs.get("face_mask_thresh", 1.0/255.0)) # threshold to detect non-black
        # use_external_face_only = isinstance(inputs.get("face_only_video", None), (list, tuple))

        # B, T, C, H, W = pred_imgs.shape

        # # 1) Build a binary face mask [B,T,1,H,W]. Prefer an explicit "face_only_video" if provided;
        # #    otherwise, synthesize face-only frames from GT frames + bboxes to derive the mask.
        # if use_external_face_only:
        #     # Provided by dataloader or upstream unit; must match T, H, W
        #     fo_tensor = self.preprocess_video(inputs["face_only_video"])  # [-1,1] -> [1,3,T,H,W]
        #     face_only_imgs = ((fo_tensor.clamp(-1, 1) + 1) / 2).permute(0, 2, 1, 3, 4).contiguous()
        #     face_mask = (face_only_imgs.sum(dim=2, keepdim=True) > mask_threshold).to(dtype=pred_imgs.dtype, device=self.device)
        # else:
        #     # Synthesize face-only frames to build mask (stays on CPU then to GPU as a mask only)
        #     gt_input = inputs.get("input_video")
        #     boxes = inputs.get("face_boxes", None) or []
        #     if isinstance(gt_input, (list, tuple)) and len(gt_input) == T:
        #         face_only_frames = self._make_face_only_video(gt_input, boxes, expand=face_bbox_margin, fill=(0, 0, 0))
        #         fo_tensor = self.preprocess_video(face_only_frames)  # [-1,1] -> [1,3,T,H,W]
        #         face_only_imgs = ((fo_tensor.clamp(-1, 1) + 1) / 2).permute(0, 2, 1, 3, 4).contiguous()
        #         face_mask = (face_only_imgs.sum(dim=2, keepdim=True) > mask_threshold).to(dtype=pred_imgs.dtype, device=self.device)
        #     else:
        #         # Fallback: build mask directly from boxes (no extra video tensor)
        #         face_mask = torch.zeros((B, T, 1, H, W), dtype=pred_imgs.dtype, device=self.device)
        #         for t in range(T):
        #             if t < len(boxes) and boxes[t] is not None:
        #                 bs = boxes[t] if isinstance(boxes[t][0], (list, tuple)) else [boxes[t]]
        #                 for (x1, y1, x2, y2) in bs:
        #                     x1 = max(0, int(x1 - face_bbox_margin)); y1 = max(0, int(y1 - face_bbox_margin))
        #                     x2 = min(W, int(x2 + face_bbox_margin)); y2 = min(H, int(y2 + face_bbox_margin))
        #                     if x2 > x1 and y2 > y1:
        #                         face_mask[:, t, :, y1:y2, x1:x2] = 1.0

        # # 2) Compute masked pixel loss on the face stream only (no ROI crops, no LPIPS by default)
        # if face_mask.sum() > 0:
        #     pred_face = pred_imgs * face_mask
        #     gt_face   = gt_imgs  * face_mask
        #     denom = (face_mask.sum() * C).clamp_min_(1.0)
        #     loss_face_stream = ((pred_face - gt_face) ** 2).sum() / denom
        # else:
        #     loss_face_stream = pred_imgs.new_zeros(())

        # # 3) Combine with the default (latent) loss
        # loss = loss_latent + face_stream_weight * loss_face_stream
        
        
        # base_err = (noise_pred.float() - training_target.float()) ** 2
        
        # face_boxes = inputs.get("face_boxes", None)            # list[[x1,y1,x2,y2], ...] per *input* frame
        # face_src   = inputs.get("face_source_size", None)      # (H_src, W_src)
        # lambda_face = float(inputs.get("face_weight", 3.0))    # weight multiplier in face regions

        # if face_boxes is not None and face_src is not None and len(face_boxes) > 0:
        #     B, C, T, H_lat, W_lat = noise_pred.shape
        #     face_mask_lat = self._build_face_mask_lat(
        #         face_boxes=face_boxes, face_src_size=face_src,
        #         T_lat=T, H_lat=H_lat, W_lat=W_lat, device=noise_pred.device, dilate_time=True
        #     )                                 # [T,H,W]
        #     W_face = face_mask_lat[None, None].to(dtype=noise_pred.dtype)   # [1,1,T,H,W]
        #     # region‑weighted diffusion loss
        #     loss = ((1.0 + lambda_face * W_face) * base_err).mean()
        # else:
        #     # fallback: vanilla MSE
        #     loss = base_err.mean()

        # # scheduler weight (same as before)
        # loss = loss * self.scheduler.training_weight(timestep)

        # # ===== Optional: latent x0 face MSE (still no decode) =====
        # beta_face_x0 = float(inputs.get("face_x0_weight", 0.1))
        # if beta_face_x0 > 0.0 and face_boxes is not None and face_src is not None and len(face_boxes) > 0:
        #     sigma = self._sigma_from_timestep_id(timestep_id)  # already implemented in file
        #     x0_hat = self._predict_x0_from_model_output(inputs["latents"], noise_pred, sigma)  # [B,Cz,T,H,W]
        #     x0_gt  = inputs["input_latents"]
        #     # reuse W_face; broadcast across channels
        #     denom = (W_face.sum() * x0_hat.shape[1]).clamp_min_(1e-6)
        #     L_face = ((x0_hat.float() - x0_gt.float())**2 * W_face).sum() / denom
        #     loss = loss + beta_face_x0 * L_face
        
        # pred = self._predict_x0_from_model_output(inputs["latents"], noise_pred, self._sigma_from_timestep_id(timestep_id))
        # breakpoint()
        # gt = inputs["input_latents"]
        # breakpoint()
        # pred_vid = self.vae.decode(pred, device=self.device)   # [B,3,T,H,W] in [-1,1]
        # gt_vid   = inputs["input_video"]
        
        # face_boxes = inputs.get("face_boxes", None)
        # face_src = inputs.get("face_source_size", None)
        # # temporal downsample mapping: t_src -> s_lat
        # factor = getattr(self, "time_division_factor", 4)
        # if face_boxes and face_src:
        #     H_src, W_src = face_src
        #     # Build per-frame mask in latent grid
        #     B, C, T, H_lat, W_lat = noise_pred.shape
        #     mask = torch.zeros((B, 1, T, H_lat, W_lat), device=noise_pred.device, dtype=noise_pred.dtype)
        #     sx = W_lat / W_src
        #     sy = H_lat / H_src
        #     for t, (x1,y1,x2,y2) in enumerate(face_boxes):
        #         s_lat = 0 if t == 0 else 1 + (t - 1) // factor
        #         if s_lat < 0 or s_lat >= T:
        #             continue  # frame maps outside (can happen if inputs were trimmed)
        #         if x2 > x1 and y2 > y1:
        #             x1l = max(0, int(x1 * sx)); x2l = min(W_lat, int(np.ceil(x2 * sx)))
        #             y1l = max(0, int(y1 * sy)); y2l = min(H_lat, int(np.ceil(y2 * sy)))
        #             # breakpoint()
        #             try:
        #                 mask[:, :, t, y1l:y2l, x1l:x2l] = 1.0
        #             except:
        #                 # breakpoint()
        #                 continue
        #     # Normalize so weight average is ~1
        #     base_err = (noise_pred.float() - training_target.float())**2
        #     denom = mask.sum() * C + 1e-6
        #     if denom.item() > 0:
        #         face_err = (base_err * mask).sum() / denom
        #         loss = loss + 0.2 * face_err  # <-- tune weight
        
        
        # face_rois = inputs.get("face_rois", None)
        # face_src  = inputs.get("face_source_size", None)  # (H_src, W_src)
        # if face_rois is not None and face_rois.numel() > 0 and face_src is not None:
        #     # Gate to later timesteps (less noise), and to subset of steps for memory
        #     timestep_id = timestep_id if 'timestep_id' in locals() else torch.tensor([0])
        #     sigma = _sigma_from_timestep_id(self, timestep_id)
        #     if sigma.mean().item() < 0.5:  # tune threshold if needed
        #         # 1) reconstruct x0 and decode
        #         x0_pred = _predict_x0_from_model_output(inputs["latents"], noise_pred, sigma)
        #         x0_gt   = inputs["input_latents"]
        #         pred_vid = self.vae.decode(x0_pred, device=self.device)   # [B,3,T,H,W] in [-1,1]
        #         gt_vid   = self.vae.decode(x0_gt,   device=self.device)
        #         pred_imgs = ((pred_vid.clamp(-1,1) + 1)/2).permute(0,2,1,3,4).contiguous()  # [B,T,3,H,W]
        #         gt_imgs   = ((gt_vid.clamp(-1,1) + 1)/2).permute(0,2,1,3,4).contiguous()

        #         # 2) Collapse [B,T] as batch for roi_align
        #         B,T = pred_imgs.shape[0], pred_imgs.shape[1]
        #         pred_bt = pred_imgs.reshape(B*T, 3, pred_imgs.shape[-2], pred_imgs.shape[-1])
        #         gt_bt   = gt_imgs.reshape(B*T, 3, gt_imgs.shape[-2], gt_imgs.shape[-1])
        #         # rois built with batch_idx=t (see unit); that's consistent with [B*T] layout when B=1
        #         crops_pred = tvops.roi_align(pred_bt, face_rois, output_size=(256,256), aligned=True)  # [N,3,256,256]
        #         crops_gt   = tvops.roi_align(gt_bt,   face_rois, output_size=(256,256), aligned=True)

        #         # 3) Face loss (use your class if you prefer LPIPS): here simple MSE to keep memory small
        #         face_mse = F.mse_loss(crops_pred.float(), crops_gt.float())
        #         loss = loss + 0.2 * face_mse  # 0.2 = weight, tune it
                
                
        #  # ---------------- Face Reconstruction Supervision (optional) ----------------
        # if self._face_sup_enabled and self._face_sup_weight > 0.0:
        #     # Optional sampling to limit overhead
        #     if random.random() < self._face_sup_prob:
        #         # 1) Recover sigma and predict clean latents x0
        #         sigma = self._sigma_from_timestep_id(timestep_id)
        #         x0_pred_latents = self._predict_x0_from_model_output(inputs["latents"], noise_pred, sigma)
        #         x0_gt_latents = inputs["input_latents"]

        #         # 2) Decode to pixel space [B, C=3, T, H, W], then to [B, T, C, H, W] in [0,1]
        #         #    Note: WAN-VAE outputs roughly in [-1, 1]; map to [0,1] for LPIPS / detectors.
        #         pred_vid = self.vae.decode(x0_pred_latents, device=self.device)
        #         gt_vid   = self.vae.decode(x0_gt_latents, device=self.device)

        #         pred_imgs = ((pred_vid.clamp(-1, 1) + 1) / 2).permute(0, 2, 1, 3, 4).contiguous()
        #         gt_imgs   = ((gt_vid.clamp(-1, 1) + 1) / 2).permute(0, 2, 1, 3, 4).contiguous()
        #         # breakpoint()
        #         import cv2
        #         for i in range(pred_imgs.shape[0]):
        #             for j in range(pred_imgs.shape[1]):
        #                 pred_img = (pred_imgs[i, j].to(torch.float16).detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        #                 gt_img = (gt_imgs[i, j].to(torch.float16).detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        #                 cv2.imwrite(f"debug/pred_{i}_{j}.png", cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR))
        #                 cv2.imwrite(f"debug/gt_{i}_{j}.png", cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR))
        #         # 3) Compute face loss (will be zero if no face is found)
        #         face_loss = self._face_loss(gt_imgs, pred_imgs)
        #         # breakpoint()
        #         # 4) Blend into the objective
        #         loss = loss + self._face_sup_weight * face_loss
        #         del pred_vid, gt_vid, pred_imgs, gt_imgs
        #         gc.collect()
        # return loss

    def enable_vram_management(self, num_persistent_param_in_dit=None, vram_limit=None, vram_buffer=0.5):
        self.vram_management_enabled = True
        if num_persistent_param_in_dit is not None:
            vram_limit = None
        else:
            if vram_limit is None:
                vram_limit = self.get_vram()
            vram_limit = vram_limit - vram_buffer
        if self.text_encoder is not None:
            dtype = next(iter(self.text_encoder.parameters())).dtype
            enable_vram_management(
                self.text_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Embedding: AutoWrappedModule,
                    T5RelativeEmbedding: AutoWrappedModule,
                    T5LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.dit is not None:
            dtype = next(iter(self.dit.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.dit,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,
                    RMSNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.Conv1d: AutoWrappedModule,
                    torch.nn.Embedding: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                max_num_param=num_persistent_param_in_dit,
                overflow_module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.dit2 is not None:
            dtype = next(iter(self.dit2.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.dit2,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,
                    RMSNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                max_num_param=num_persistent_param_in_dit,
                overflow_module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.vae is not None:
            dtype = next(iter(self.vae.parameters())).dtype
            enable_vram_management(
                self.vae,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    RMS_norm: AutoWrappedModule,
                    CausalConv3d: AutoWrappedModule,
                    Upsample: AutoWrappedModule,
                    torch.nn.SiLU: AutoWrappedModule,
                    torch.nn.Dropout: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=self.device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )
        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            enable_vram_management(
                self.image_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        if self.motion_controller is not None:
            dtype = next(iter(self.motion_controller.parameters())).dtype
            enable_vram_management(
                self.motion_controller,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        if self.vace is not None:
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.vace,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                    RMSNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.audio_encoder is not None:
            # TODO: need check
            dtype = next(iter(self.audio_encoder.parameters())).dtype
            enable_vram_management(
                self.audio_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.LayerNorm: AutoWrappedModule,
                    torch.nn.Conv1d: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )
            
            
    def initialize_usp(self):
        import torch.distributed as dist
        from xfuser.core.distributed import initialize_model_parallel, init_distributed_environment
        dist.init_process_group(backend="nccl", init_method="env://")
        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=1,
            ulysses_degree=dist.get_world_size(),
        )
        torch.cuda.set_device(dist.get_rank())
            
            
    def enable_usp(self):
        from xfuser.core.distributed import get_sequence_parallel_world_size
        from ..distributed.xdit_context_parallel import usp_attn_forward, usp_dit_forward

        for block in self.dit.blocks:
            block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
        self.dit.forward = types.MethodType(usp_dit_forward, self.dit)
        if self.dit2 is not None:
            for block in self.dit2.blocks:
                block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
            self.dit2.forward = types.MethodType(usp_dit_forward, self.dit2)
        self.sp_size = get_sequence_parallel_world_size()
        self.use_unified_sequence_parallel = True


    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/*"),
        audio_processor_config: ModelConfig = None,
        redirect_common_files: bool = True,
        use_usp=False,
    ):
        # Redirect model path
        if redirect_common_files:
            redirect_dict = {
                "models_t5_umt5-xxl-enc-bf16.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "Wan2.1_VAE.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": "Wan-AI/Wan2.1-I2V-14B-480P",
            }
            for model_config in model_configs:
                if model_config.origin_file_pattern is None or model_config.model_id is None:
                    continue
                if model_config.origin_file_pattern in redirect_dict and model_config.model_id != redirect_dict[model_config.origin_file_pattern]:
                    print(f"To avoid repeatedly downloading model files, ({model_config.model_id}, {model_config.origin_file_pattern}) is redirected to ({redirect_dict[model_config.origin_file_pattern]}, {model_config.origin_file_pattern}). You can use `redirect_common_files=False` to disable file redirection.")
                    model_config.model_id = redirect_dict[model_config.origin_file_pattern]
        
        # Initialize pipeline
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)
        if use_usp: pipe.initialize_usp()
        
        # Download and load models
        model_manager = ModelManager()
        for model_config in model_configs:
            model_config.download_if_necessary(use_usp=use_usp)
            model_manager.load_model(
                model_config.path,
                device=model_config.offload_device or device,
                torch_dtype=model_config.offload_dtype or torch_dtype
            )
        
        # Load models
        pipe.text_encoder = model_manager.fetch_model("wan_video_text_encoder")
        dit = model_manager.fetch_model("wan_video_dit", index=2)
        if isinstance(dit, list):
            pipe.dit, pipe.dit2 = dit
        else:
            pipe.dit = dit
        pipe.vae = model_manager.fetch_model("wan_video_vae")
        pipe.image_encoder = model_manager.fetch_model("wan_video_image_encoder")
        pipe.motion_controller = model_manager.fetch_model("wan_video_motion_controller")
        vace = model_manager.fetch_model("wan_video_vace", index=2)
        if isinstance(vace, list):
            pipe.vace, pipe.vace2 = vace
        else:
            pipe.vace = vace
        pipe.audio_encoder = model_manager.fetch_model("wans2v_audio_encoder")
        pipe.animate_adapter = model_manager.fetch_model("wan_video_animate_adapter")

        # Size division factor
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        # Initialize tokenizer
        tokenizer_config.download_if_necessary(use_usp=use_usp)
        pipe.prompter.fetch_models(pipe.text_encoder)
        pipe.prompter.fetch_tokenizer(tokenizer_config.path)

        if audio_processor_config is not None:
            audio_processor_config.download_if_necessary(use_usp=use_usp)
            from transformers import Wav2Vec2Processor
            pipe.audio_processor = Wav2Vec2Processor.from_pretrained(audio_processor_config.path)
        # Unified Sequence Parallel
        if use_usp: pipe.enable_usp()
        return pipe


    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: Optional[str] = "",
        # Image-to-video
        input_image: Optional[Image.Image] = None,
        # First-last-frame-to-video
        end_image: Optional[Image.Image] = None,
        # Video-to-video
        input_video: Optional[list[Image.Image]] = None,
        denoising_strength: Optional[float] = 1.0,
        # Speech-to-video
        input_audio: Optional[np.array] = None,
        audio_embeds: Optional[torch.Tensor] = None,
        audio_sample_rate: Optional[int] = 16000,
        s2v_pose_video: Optional[list[Image.Image]] = None,
        s2v_pose_latents: Optional[torch.Tensor] = None,
        motion_video: Optional[list[Image.Image]] = None,
        # ControlNet
        control_video: Optional[list[Image.Image]] = None,
        reference_image: Optional[Image.Image] = None,
        # Camera control
        camera_control_direction: Optional[Literal["Left", "Right", "Up", "Down", "LeftUp", "LeftDown", "RightUp", "RightDown"]] = None,
        camera_control_speed: Optional[float] = 1/54,
        camera_control_origin: Optional[tuple] = (0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
        # VACE
        vace_video: Optional[list[Image.Image]] = None,
        vace_video_mask: Optional[Image.Image] = None,
        vace_reference_image: Optional[Image.Image] = None,
        vace_scale: Optional[float] = 1.0,
        # Animate
        animate_pose_video: Optional[list[Image.Image]] = None,
        animate_face_video: Optional[list[Image.Image]] = None,
        animate_inpaint_video: Optional[list[Image.Image]] = None,
        animate_mask_video: Optional[list[Image.Image]] = None,
        # Randomness
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        # Shape
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames=81,
        # Classifier-free guidance
        cfg_scale: Optional[float] = 5.0,
        cfg_merge: Optional[bool] = False,
        # Boundary
        switch_DiT_boundary: Optional[float] = 0.875,
        # Scheduler
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        # Speed control
        motion_bucket_id: Optional[int] = None,
        # VAE tiling
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
        # Sliding window
        sliding_window_size: Optional[int] = None,
        sliding_window_stride: Optional[int] = None,
        # Teacache
        tea_cache_l1_thresh: Optional[float] = None,
        tea_cache_model_id: Optional[str] = "",
        # progress_bar
        progress_bar_cmd=tqdm,
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)
        
        # Inputs
        inputs_posi = {
            "prompt": prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }
        inputs_shared = {
            "input_image": input_image,
            "end_image": end_image,
            "input_video": input_video, "denoising_strength": denoising_strength,
            "control_video": control_video, "reference_image": reference_image,
            "camera_control_direction": camera_control_direction, "camera_control_speed": camera_control_speed, "camera_control_origin": camera_control_origin,
            "vace_video": vace_video, "vace_video_mask": vace_video_mask, "vace_reference_image": vace_reference_image, "vace_scale": vace_scale,
            "seed": seed, "rand_device": rand_device,
            "height": height, "width": width, "num_frames": num_frames,
            "cfg_scale": cfg_scale, "cfg_merge": cfg_merge,
            "sigma_shift": sigma_shift,
            "motion_bucket_id": motion_bucket_id,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "sliding_window_size": sliding_window_size, "sliding_window_stride": sliding_window_stride,
            "input_audio": input_audio, "audio_sample_rate": audio_sample_rate, "s2v_pose_video": s2v_pose_video, "audio_embeds": audio_embeds, "s2v_pose_latents": s2v_pose_latents, "motion_video": motion_video,
            "animate_pose_video": animate_pose_video, "animate_face_video": animate_face_video, "animate_inpaint_video": animate_inpaint_video, "animate_mask_video": animate_mask_video,
        }
        for unit in self.units:
            # breakpoint()
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            # Switch DiT if necessary
            if timestep.item() < switch_DiT_boundary * self.scheduler.num_train_timesteps and self.dit2 is not None and not models["dit"] is self.dit2:
                self.load_models_to_device(self.in_iteration_models_2)
                models["dit"] = self.dit2
                models["vace"] = self.vace2
                
            # Timestep
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            
            # Inference
            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
            if cfg_scale != 1.0:
                if cfg_merge:
                    noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)
                else:
                    noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])
            if "first_frame_latents" in inputs_shared:
                inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]
        
        # VACE (TODO: remove it)
        if vace_reference_image is not None or (animate_pose_video is not None and animate_face_video is not None):
            if vace_reference_image is not None and isinstance(vace_reference_image, list):
                f = len(vace_reference_image)
            else:
                f = 1
            inputs_shared["latents"] = inputs_shared["latents"][:, :, f:]
        # post-denoising, pre-decoding processing logic
        for unit in self.post_units:
            inputs_shared, _, _ = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)
        # Decode
        self.load_models_to_device(['vae'])
        video = self.vae.decode(inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video = self.vae_output_to_video(video)
        self.load_models_to_device([])

        return video
    
# THIS CLASS IS NEW   
class WanVideoUnit_FaceBBox(PipelineUnit):
    """
    Detect faces on GT frames and stash boxes for later losses.
    Inputs it tries to read (from inputs_shared):
      - input_video: list[PIL.Image] length T  (preferred at training time)
      - video: torch.Tensor [B=1, C, T, H, W] in [-1,1] (fallback)
      - height, width, num_frames: for metadata (optional)

    Outputs it writes into inputs_shared:
      - face_boxes: List[List[int]] length T  -> [[x1,y1,x2,y2], ...]
      - face_rois:  torch.FloatTensor [N,5] with batch indices (B*T layout)
      - face_source_size: (H, W)
    """
    def __init__(self, detector=None, det_size=(1024,1024), prob=1.0, run_in_train_only=True):
        super().__init__(seperate_cfg=False, input_params=("input_video", "video", "height", "width", "num_frames"))
        self.detector = detector      # e.g., insightface FaceAnalysis instance (prepared)
        self.det_size = det_size
        self.prob = float(prob)
        self.run_in_train_only = run_in_train_only

    @staticmethod
    def _pil_to_bgr_uint8(img_pil):
        arr = np.array(img_pil, dtype=np.uint8) # RGB HxWx3
        return arr[..., ::-1].copy()            # BGR
    
    def _make_face_masked_video(self, frames, face_boxes, expand: int = 0, fill=(0, 0, 0)):
        """Return a list of PIL RGB frames where only the detected face region is kept,
        the rest is black. Uses per-frame bboxes; if multiple boxes, unions them.
        """
        from PIL import Image, ImageDraw
        out = []
        last_box = None
        for t, img in enumerate(frames):
            w, h = img.size
            box = None
            if face_boxes is not None and t < len(face_boxes):
                b = face_boxes[t]
                if b is not None:
                    if isinstance(b[0], (list, tuple)):
                        xs = [bb[0] for bb in b]; ys = [bb[1] for bb in b]
                        xe = [bb[2] for bb in b]; ye = [bb[3] for bb in b]
                        x1, y1, x2, y2 = min(xs), min(ys), max(xe), max(ye)
                    else:
                        x1, y1, x2, y2 = b
                    x1 = max(0, int(x1 - expand)); y1 = max(0, int(y1 - expand))
                    x2 = min(w, int(x2 + expand)); y2 = min(h, int(y2 + expand))
                    if x2 > x1 and y2 > y1:
                        box = (x1, y1, x2, y2)
            if box is None:
                box = last_box
            if box is None:
                out.append(Image.new("RGB", (w, h), fill))
                continue
            last_box = box
            mask = Image.new("L", (w, h), 0)
            draw = ImageDraw.Draw(mask)
            draw.rectangle(box, fill=255)
            black = Image.new("RGB", (w, h), fill)
            out.append(Image.composite(img, black, mask))
        return out
    
    def _make_face_only_video(self, frames, face_boxes, expand: int = 0, out_size: int = 256):
        """
        Return a list of 256x256 RGBA frames with only the face region kept.
        - Unions multiple boxes per frame.
        - Crops to the (expanded) bbox, preserves aspect, pads with transparency.
        - If no detection on a frame, reuses the last valid box; otherwise outputs empty.
        """
        from PIL import Image

        out = []
        last_box = None

        for t, img in enumerate(frames):
            w, h = img.size
            box = None

            # Resolve bbox (union if multiple)
            if face_boxes is not None and t < len(face_boxes):
                b = face_boxes[t]
                if b is not None:
                    if isinstance(b[0], (list, tuple)):
                        xs = [bb[0] for bb in b]; ys = [bb[1] for bb in b]
                        xe = [bb[2] for bb in b]; ye = [bb[3] for bb in b]
                        x1, y1, x2, y2 = min(xs), min(ys), max(xe), max(ye)
                    else:
                        x1, y1, x2, y2 = b

                    # Expand + clamp
                    x1 = max(0, int(x1 - expand)); y1 = max(0, int(y1 - expand))
                    x2 = min(w, int(x2 + expand)); y2 = min(h, int(y2 + expand))
                    if x2 > x1 and y2 > y1:
                        box = (x1, y1, x2, y2)

            if box is None:
                box = last_box

            if box is None:
                # No detection yet → transparent 256x256
                out.append(Image.new("RGBA", (out_size, out_size), (0, 0, 0, 0)))
                continue

            last_box = box

            # Crop to face bbox
            crop = img.crop(box).convert("RGBA")
            cw, ch = crop.size

            # Fit inside 256x256 with padding (no distortion)
            scale = min(out_size / cw, out_size / ch)
            nw, nh = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
            crop_resized = crop.resize((nw, nh), Image.BICUBIC)

            canvas = Image.new("RGBA", (out_size, out_size), (0, 0, 0, 0))
            x_off = (out_size - nw) // 2
            y_off = (out_size - nh) // 2
            canvas.paste(crop_resized, (x_off, y_off), crop_resized)

            out.append(canvas)

        return out


    @torch.no_grad()
    def process(self, pipe, input_video=None, video=None, height=None, width=None, num_frames=None):
        # Optional gate: run only during training
        if self.run_in_train_only and not pipe.training:
            return {}

        # Optional probability gate (to save time)
        if self.prob < 1.0 and torch.rand(()) > self.prob:
            return {}

        # Pick source frames
        frames_bgr = []
        if input_video is not None and isinstance(input_video, (list, tuple)) and len(input_video) > 0:
            # list of PIL.Image
            H, W = input_video[0].size[1], input_video[0].size[0]
            for im in input_video:
                frames_bgr.append(self._pil_to_bgr_uint8(im))
        elif isinstance(video, torch.Tensor):
            # video: [B, C, T, H, W] in [-1,1]; assume B==1 here (training example uses per-sample)
            v01 = ((video.clamp(-1,1) + 1) * 127.5).round().to(torch.uint8)  # [1,3,T,H,W]
            v01 = v01[0].permute(1,3,4,0).contiguous()                        # [T,H,W,3] RGB
            H, W = int(v01.shape[1]), int(v01.shape[2])
            for t in range(v01.shape[0]):
                frames_bgr.append(v01[t].cpu().numpy()[..., ::-1].copy())
        else:
            return {}  # nothing to do

        # Run detection on each frame (on CPU; detector may still use CUDA internally)
        face_boxes = []
        for bgr in frames_bgr:
            dets = self.detector.get(bgr) if self.detector is not None else []
            if len(dets) > 0:
                best = max(dets, key=lambda d: d.det_score)
                x1, y1, x2, y2 = best.bbox.astype(int)[:4]
                # clamp box to image
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W-1, x2), min(H-1, y2)
                if x2 - x1 >= 10 and y2 - y1 >= 10:
                    face_boxes.append([int(x1), int(y1), int(x2), int(y2)])
                else:
                    face_boxes.append([0,0,0,0])
            else:
                face_boxes.append([0,0,0,0])

        # Build ROI tensor compatible with torchvision.ops.roi_align
        # We assume B=1 sample; otherwise, batch_idx mapping would be (b*T + t)
        rois = []
        for t, (x1,y1,x2,y2) in enumerate(face_boxes):
            if x2 > x1 and y2 > y1:
                rois.append([float(t), float(x1), float(y1), float(x2), float(y2)])  # use t as batch_idx over (T) slices

        face_rois = torch.tensor(rois, dtype=torch.float16, device=pipe.device) if len(rois) else torch.empty((0,5), dtype=torch.float16, device=pipe.device)
        
        if input_video is not None and isinstance(input_video, (list, tuple)) and len(input_video) > 0:
            frames_pil = list(input_video)
        else:
            from PIL import Image
            frames_pil = [Image.fromarray(bgr[..., ::-1].copy()) for bgr in frames_bgr]  # BGR->RGB

        # Use the pipeline helper to keep only the face region per frame
        # (falls back to last valid box per frame if a frame has no detection)
        face_masked_images_to_encode = self._make_face_masked_video(
            frames=frames_pil,
            face_boxes=face_boxes,
            expand=0,           # tweak if you want padding around the face
            fill=(0, 0, 0)      # black background
        )  #
        
        face_masked_images = self._make_face_only_video(
            frames=frames_pil,
            face_boxes=face_boxes,
            expand=0,           # tweak if you want padding around the face
            out_size=256      # output size
        )
        
        # for i, image in enumerate(face_masked_images):
        #     image.save(f'debug/debug_faces_{i}.png')
        return dict(
            face_boxes=face_boxes,
            face_rois=face_rois,
            face_source_size=(H, W),
            face_masked_images=face_masked_images,
            face_masked_images_to_encode=face_masked_images_to_encode,
        )



class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames"))

    def process(self, pipe: WanVideoPipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}



class WanVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames", "seed", "rand_device", "vace_reference_image"))

    def process(self, pipe: WanVideoPipeline, height, width, num_frames, seed, rand_device, vace_reference_image):
        length = (num_frames - 1) // 4 + 1
        if vace_reference_image is not None:
            f = len(vace_reference_image) if isinstance(vace_reference_image, list) else 1
            length += f
        shape = (1, pipe.vae.model.z_dim, length, height // pipe.vae.upsampling_factor, width // pipe.vae.upsampling_factor)
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        if vace_reference_image is not None:
            noise = torch.concat((noise[:, :, -f:], noise[:, :, :-f]), dim=2)
        return {"noise": noise}
    


class WanVideoUnit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "face_masked_images_to_encode", "noise", "tiled", "tile_size", "tile_stride", "vace_reference_image"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_video, face_masked_images_to_encode, noise, tiled, tile_size, tile_stride, vace_reference_image):
        if input_video is None:
            return {"latents": noise}
        pipe.load_models_to_device(["vae"])
        input_video = pipe.preprocess_video(input_video)
        face_video = pipe.preprocess_video(face_masked_images_to_encode)
        input_latents = pipe.vae.encode(input_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        face_latents = pipe.vae.encode(face_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device) 
        # face_latents = None
        if vace_reference_image is not None:
            if not isinstance(vace_reference_image, list):
                vace_reference_image = [vace_reference_image]
            vace_reference_image = pipe.preprocess_video(vace_reference_image)
            vace_reference_latents = pipe.vae.encode(vace_reference_image, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
            input_latents = torch.concat([vace_reference_latents, input_latents], dim=2)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents, "face_latents": face_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents}



class WanVideoUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive"},
            onload_model_names=("text_encoder",)
        )

    def process(self, pipe: WanVideoPipeline, prompt, positive) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        prompt_emb = pipe.prompter.encode_prompt(prompt, positive=positive, device=pipe.device)
        return {"context": prompt_emb}



class WanVideoUnit_ImageEmbedder(PipelineUnit):
    """
    Deprecated
    """
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("image_encoder", "vae")
        )

    def process(self, pipe: WanVideoPipeline, input_image, end_image, num_frames, height, width, tiled, tile_size, tile_stride):
        if input_image is None or pipe.image_encoder is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        clip_context = pipe.image_encoder.encode_image([image])
        msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
        msk[:, 1:] = 0
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            vae_input = torch.concat([image.transpose(0,1), torch.zeros(3, num_frames-2, height, width).to(image.device), end_image.transpose(0,1)],dim=1)
            if pipe.dit.has_image_pos_emb:
                clip_context = torch.concat([clip_context, pipe.image_encoder.encode_image([end_image])], dim=1)
            msk[:, -1:] = 1
        else:
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)

        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]
        
        y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"clip_feature": clip_context, "y": y}



class WanVideoUnit_ImageEmbedderCLIP(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "height", "width"),
            onload_model_names=("image_encoder",)
        )

    def process(self, pipe: WanVideoPipeline, input_image, end_image, height, width):
        if input_image is None or pipe.image_encoder is None or not pipe.dit.require_clip_embedding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        clip_context = pipe.image_encoder.encode_image([image])
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            if pipe.dit.has_image_pos_emb:
                clip_context = torch.concat([clip_context, pipe.image_encoder.encode_image([end_image])], dim=1)
        clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"clip_feature": clip_context}
    


class WanVideoUnit_ImageEmbedderVAE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_image, end_image, num_frames, height, width, tiled, tile_size, tile_stride):
        if input_image is None or not pipe.dit.require_vae_embedding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
        msk[:, 1:] = 0
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            vae_input = torch.concat([image.transpose(0,1), torch.zeros(3, num_frames-2, height, width).to(image.device), end_image.transpose(0,1)],dim=1)
            msk[:, -1:] = 1
        else:
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)

        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]
        
        y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"y": y}



class WanVideoUnit_ImageEmbedderFused(PipelineUnit):
    """
    Encode input image to latents using VAE. This unit is for Wan-AI/Wan2.2-TI2V-5B.
    """
    def __init__(self):
        super().__init__(
            input_params=("input_image", "latents", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_image, latents, height, width, tiled, tile_size, tile_stride):
        if input_image is None or not pipe.dit.fuse_vae_embedding_in_latents:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).transpose(0, 1)
        z = pipe.vae.encode([image], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        latents[:, :, 0: 1] = z
        return {"latents": latents, "fuse_vae_embedding_in_latents": True, "first_frame_latents": z}



class WanVideoUnit_FunControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("control_video", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride", "clip_feature", "y", "latents"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, control_video, num_frames, height, width, tiled, tile_size, tile_stride, clip_feature, y, latents):
        if control_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        control_video = pipe.preprocess_video(control_video)
        control_latents = pipe.vae.encode(control_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        control_latents = control_latents.to(dtype=pipe.torch_dtype, device=pipe.device)
        y_dim = pipe.dit.in_dim-control_latents.shape[1]-latents.shape[1]
        if clip_feature is None or y is None:
            clip_feature = torch.zeros((1, 257, 1280), dtype=pipe.torch_dtype, device=pipe.device)
            y = torch.zeros((1, y_dim, (num_frames - 1) // 4 + 1, height//8, width//8), dtype=pipe.torch_dtype, device=pipe.device)
        else:
            y = y[:, -y_dim:]
        y = torch.concat([control_latents, y], dim=1)
        return {"clip_feature": clip_feature, "y": y}
    


class WanVideoUnit_FunReference(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("reference_image", "height", "width", "reference_image"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, reference_image, height, width):
        if reference_image is None:
            return {}
        pipe.load_models_to_device(["vae"])
        reference_image = reference_image.resize((width, height))
        reference_latents = pipe.preprocess_video([reference_image])
        reference_latents = pipe.vae.encode(reference_latents, device=pipe.device)
        if pipe.image_encoder is None:
            return {"reference_latents": reference_latents}
        clip_feature = pipe.preprocess_image(reference_image)
        clip_feature = pipe.image_encoder.encode_image([clip_feature])
        return {"reference_latents": reference_latents, "clip_feature": clip_feature}



class WanVideoUnit_FunCameraControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "camera_control_direction", "camera_control_speed", "camera_control_origin", "latents", "input_image", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, height, width, num_frames, camera_control_direction, camera_control_speed, camera_control_origin, latents, input_image, tiled, tile_size, tile_stride):
        if camera_control_direction is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        camera_control_plucker_embedding = pipe.dit.control_adapter.process_camera_coordinates(
            camera_control_direction, num_frames, height, width, camera_control_speed, camera_control_origin)
        
        control_camera_video = camera_control_plucker_embedding[:num_frames].permute([3, 0, 1, 2]).unsqueeze(0)
        control_camera_latents = torch.concat(
            [
                torch.repeat_interleave(control_camera_video[:, :, 0:1], repeats=4, dim=2),
                control_camera_video[:, :, 1:]
            ], dim=2
        ).transpose(1, 2)
        b, f, c, h, w = control_camera_latents.shape
        control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, 4, c, h, w).transpose(2, 3)
        control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, c * 4, h, w).transpose(1, 2)
        control_camera_latents_input = control_camera_latents.to(device=pipe.device, dtype=pipe.torch_dtype)
        
        input_image = input_image.resize((width, height))
        input_latents = pipe.preprocess_video([input_image])
        input_latents = pipe.vae.encode(input_latents, device=pipe.device)
        y = torch.zeros_like(latents).to(pipe.device)
        y[:, :, :1] = input_latents
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)

        if y.shape[1] != pipe.dit.in_dim - latents.shape[1]:
            image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)
            y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
            y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
            msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
            msk[:, 1:] = 0
            msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
            msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
            msk = msk.transpose(1, 2)[0]
            y = torch.cat([msk,y])
            y = y.unsqueeze(0)
            y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"control_camera_latents_input": control_camera_latents_input, "y": y}



class WanVideoUnit_SpeedControl(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("motion_bucket_id",))

    def process(self, pipe: WanVideoPipeline, motion_bucket_id):
        if motion_bucket_id is None:
            return {}
        motion_bucket_id = torch.Tensor((motion_bucket_id,)).to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"motion_bucket_id": motion_bucket_id}



class WanVideoUnit_VACE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("vace_video", "vace_video_mask", "vace_reference_image", "vace_scale", "height", "width", "num_frames", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        vace_video, vace_video_mask, vace_reference_image, vace_scale,
        height, width, num_frames,
        tiled, tile_size, tile_stride
    ):
        if vace_video is not None or vace_video_mask is not None or vace_reference_image is not None:
            pipe.load_models_to_device(["vae"])
            if vace_video is None:
                vace_video = torch.zeros((1, 3, num_frames, height, width), dtype=pipe.torch_dtype, device=pipe.device)
            else:
                vace_video = pipe.preprocess_video(vace_video)
            
            if vace_video_mask is None:
                vace_video_mask = torch.ones_like(vace_video)
            else:
                vace_video_mask = pipe.preprocess_video(vace_video_mask, min_value=0, max_value=1)
            
            inactive = vace_video * (1 - vace_video_mask) + 0 * vace_video_mask
            reactive = vace_video * vace_video_mask + 0 * (1 - vace_video_mask)
            inactive = pipe.vae.encode(inactive, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            reactive = pipe.vae.encode(reactive, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            vace_video_latents = torch.concat((inactive, reactive), dim=1)
            
            vace_mask_latents = rearrange(vace_video_mask[0,0], "T (H P) (W Q) -> 1 (P Q) T H W", P=8, Q=8)
            vace_mask_latents = torch.nn.functional.interpolate(vace_mask_latents, size=((vace_mask_latents.shape[2] + 3) // 4, vace_mask_latents.shape[3], vace_mask_latents.shape[4]), mode='nearest-exact')
            
            if vace_reference_image is None:
                pass
            else:
                if not isinstance(vace_reference_image,list):
                    vace_reference_image = [vace_reference_image]

                vace_reference_image = pipe.preprocess_video(vace_reference_image)

                bs, c, f, h, w = vace_reference_image.shape
                new_vace_ref_images = []
                for j in range(f):
                    new_vace_ref_images.append(vace_reference_image[0, :, j:j+1])
                vace_reference_image = new_vace_ref_images
                
                vace_reference_latents = pipe.vae.encode(vace_reference_image, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
                vace_reference_latents = torch.concat((vace_reference_latents, torch.zeros_like(vace_reference_latents)), dim=1)
                vace_reference_latents = [u.unsqueeze(0) for u in vace_reference_latents]

                vace_video_latents = torch.concat((*vace_reference_latents, vace_video_latents), dim=2)
                vace_mask_latents = torch.concat((torch.zeros_like(vace_mask_latents[:, :, :f]), vace_mask_latents), dim=2)
            
            vace_context = torch.concat((vace_video_latents, vace_mask_latents), dim=1)
            return {"vace_context": vace_context, "vace_scale": vace_scale}
        else:
            return {"vace_context": None, "vace_scale": vace_scale}



class WanVideoUnit_UnifiedSequenceParallel(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=())

    def process(self, pipe: WanVideoPipeline):
        if hasattr(pipe, "use_unified_sequence_parallel"):
            if pipe.use_unified_sequence_parallel:
                return {"use_unified_sequence_parallel": True}
        return {}



class WanVideoUnit_TeaCache(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
            input_params_nega={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
        )

    def process(self, pipe: WanVideoPipeline, num_inference_steps, tea_cache_l1_thresh, tea_cache_model_id):
        if tea_cache_l1_thresh is None:
            return {}
        return {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id)}



class WanVideoUnit_CfgMerger(PipelineUnit):
    def __init__(self):
        super().__init__(take_over=True)
        self.concat_tensor_names = ["context", "clip_feature", "y", "reference_latents"]

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if not inputs_shared["cfg_merge"]:
            return inputs_shared, inputs_posi, inputs_nega
        for name in self.concat_tensor_names:
            tensor_posi = inputs_posi.get(name)
            tensor_nega = inputs_nega.get(name)
            tensor_shared = inputs_shared.get(name)
            if tensor_posi is not None and tensor_nega is not None:
                inputs_shared[name] = torch.concat((tensor_posi, tensor_nega), dim=0)
            elif tensor_shared is not None:
                inputs_shared[name] = torch.concat((tensor_shared, tensor_shared), dim=0)
        inputs_posi.clear()
        inputs_nega.clear()
        return inputs_shared, inputs_posi, inputs_nega


class WanVideoUnit_S2V(PipelineUnit):
    def __init__(self):
        super().__init__(
            take_over=True,
            onload_model_names=("audio_encoder", "vae",)
        )

    def process_audio(self, pipe: WanVideoPipeline, input_audio, audio_sample_rate, num_frames, fps=16, audio_embeds=None, return_all=False):
        if audio_embeds is not None:
            return {"audio_embeds": audio_embeds}
        pipe.load_models_to_device(["audio_encoder"])
        audio_embeds = pipe.audio_encoder.get_audio_feats_per_inference(input_audio, audio_sample_rate, pipe.audio_processor, fps=fps, batch_frames=num_frames-1, dtype=pipe.torch_dtype, device=pipe.device)
        if return_all:
            return audio_embeds
        else:
            return {"audio_embeds": audio_embeds[0]}

    def process_motion_latents(self, pipe: WanVideoPipeline, height, width, tiled, tile_size, tile_stride, motion_video=None):
        pipe.load_models_to_device(["vae"])
        motion_frames = 73
        kwargs = {}
        if motion_video is not None and len(motion_video) > 0:
            assert len(motion_video) == motion_frames, f"motion video must have {motion_frames} frames, but got {len(motion_video)}"
            motion_latents = pipe.preprocess_video(motion_video)
            kwargs["drop_motion_frames"] = False
        else:
            motion_latents = torch.zeros([1, 3, motion_frames, height, width], dtype=pipe.torch_dtype, device=pipe.device)
            kwargs["drop_motion_frames"] = True
        motion_latents = pipe.vae.encode(motion_latents, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        kwargs.update({"motion_latents": motion_latents})
        return kwargs

    def process_pose_cond(self, pipe: WanVideoPipeline, s2v_pose_video, num_frames, height, width, tiled, tile_size, tile_stride, s2v_pose_latents=None, num_repeats=1, return_all=False):
        if s2v_pose_latents is not None:
            return {"s2v_pose_latents": s2v_pose_latents}
        if s2v_pose_video is None:
            return {"s2v_pose_latents": None}
        pipe.load_models_to_device(["vae"])
        infer_frames = num_frames - 1
        input_video = pipe.preprocess_video(s2v_pose_video)[:, :, :infer_frames * num_repeats]
        # pad if not enough frames
        padding_frames = infer_frames * num_repeats - input_video.shape[2]
        input_video = torch.cat([input_video, -torch.ones(1, 3, padding_frames, height, width, device=input_video.device, dtype=input_video.dtype)], dim=2)
        input_videos = input_video.chunk(num_repeats, dim=2)
        pose_conds = []
        for r in range(num_repeats):
            cond = input_videos[r]
            cond = torch.cat([cond[:, :, 0:1].repeat(1, 1, 1, 1, 1), cond], dim=2)
            cond_latents = pipe.vae.encode(cond, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            pose_conds.append(cond_latents[:,:,1:])
        if return_all:
            return pose_conds
        else:
            return {"s2v_pose_latents": pose_conds[0]}

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if (inputs_shared.get("input_audio") is None and inputs_shared.get("audio_embeds") is None) or pipe.audio_encoder is None or pipe.audio_processor is None:
            return inputs_shared, inputs_posi, inputs_nega
        num_frames, height, width, tiled, tile_size, tile_stride = inputs_shared.get("num_frames"), inputs_shared.get("height"), inputs_shared.get("width"), inputs_shared.get("tiled"), inputs_shared.get("tile_size"), inputs_shared.get("tile_stride")
        input_audio, audio_embeds, audio_sample_rate = inputs_shared.pop("input_audio"), inputs_shared.pop("audio_embeds"), inputs_shared.get("audio_sample_rate")
        s2v_pose_video, s2v_pose_latents, motion_video = inputs_shared.pop("s2v_pose_video"), inputs_shared.pop("s2v_pose_latents"), inputs_shared.pop("motion_video")

        audio_input_positive = self.process_audio(pipe, input_audio, audio_sample_rate, num_frames, audio_embeds=audio_embeds)
        inputs_posi.update(audio_input_positive)
        inputs_nega.update({"audio_embeds": 0.0 * audio_input_positive["audio_embeds"]})

        inputs_shared.update(self.process_motion_latents(pipe, height, width, tiled, tile_size, tile_stride, motion_video))
        inputs_shared.update(self.process_pose_cond(pipe, s2v_pose_video, num_frames, height, width, tiled, tile_size, tile_stride, s2v_pose_latents=s2v_pose_latents))
        return inputs_shared, inputs_posi, inputs_nega

    @staticmethod
    def pre_calculate_audio_pose(pipe: WanVideoPipeline, input_audio=None, audio_sample_rate=16000, s2v_pose_video=None, num_frames=81, height=448, width=832, fps=16, tiled=True, tile_size=(30, 52), tile_stride=(15, 26)):
        assert pipe.audio_encoder is not None and pipe.audio_processor is not None, "Please load audio encoder and audio processor first."
        shapes = WanVideoUnit_ShapeChecker().process(pipe, height, width, num_frames)
        height, width, num_frames = shapes["height"], shapes["width"], shapes["num_frames"]
        unit = WanVideoUnit_S2V()
        audio_embeds = unit.process_audio(pipe, input_audio, audio_sample_rate, num_frames, fps, return_all=True)
        pose_latents = unit.process_pose_cond(pipe, s2v_pose_video, num_frames, height, width, num_repeats=len(audio_embeds), return_all=True, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        pose_latents = None if s2v_pose_video is None else pose_latents
        return audio_embeds, pose_latents, len(audio_embeds)


class WanVideoPostUnit_S2V(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("latents", "motion_latents", "drop_motion_frames"))

    def process(self, pipe: WanVideoPipeline, latents, motion_latents, drop_motion_frames):
        if pipe.audio_encoder is None or motion_latents is None or drop_motion_frames:
            return {}
        latents = torch.cat([motion_latents, latents[:,:,1:]], dim=2)
        return {"latents": latents}


class WanVideoPostUnit_AnimateVideoSplit(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("input_video", "animate_pose_video", "animate_face_video", "animate_inpaint_video", "animate_mask_video"))

    def process(self, pipe: WanVideoPipeline, input_video, animate_pose_video, animate_face_video, animate_inpaint_video, animate_mask_video):
        if input_video is None:
            return {}
        if animate_pose_video is not None:
            animate_pose_video = animate_pose_video[:len(input_video) - 4]
        if animate_face_video is not None:
            animate_face_video = animate_face_video[:len(input_video) - 4]
        if animate_inpaint_video is not None:
            animate_inpaint_video = animate_inpaint_video[:len(input_video) - 4]
        if animate_mask_video is not None:
            animate_mask_video = animate_mask_video[:len(input_video) - 4]
        return {"animate_pose_video": animate_pose_video, "animate_face_video": animate_face_video, "animate_inpaint_video": animate_inpaint_video, "animate_mask_video": animate_mask_video}


class WanVideoPostUnit_AnimatePoseLatents(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("animate_pose_video", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, animate_pose_video, tiled, tile_size, tile_stride):
        if animate_pose_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        animate_pose_video = pipe.preprocess_video(animate_pose_video)
        breakpoint()
        pose_latents = pipe.vae.encode(animate_pose_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"pose_latents": pose_latents}


# class WanVideoPostUnit_AnimateFacePixelValues(PipelineUnit):
#     def __init__(self):
#         super().__init__(take_over=True)

#     def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
#         if inputs_shared.get("animate_face_video", None) is None:
#             return {}
#         inputs_posi["face_pixel_values"] = pipe.preprocess_video(inputs_shared["animate_face_video"])
#         inputs_nega["face_pixel_values"] = torch.zeros_like(inputs_posi["face_pixel_values"]) - 1
#         return inputs_shared, inputs_posi, inputs_nega


class WanVideoPostUnit_AnimateFacePixelValues(PipelineUnit):
    def __init__(self): 
        super().__init__(take_over=True)

    # keyword-only args to match the runner's call style
    def process(self, pipe: WanVideoPipeline, *, inputs_shared, inputs_posi, inputs_nega):
        video = inputs_shared.get("animate_face_video", None)
        if video is None:
            # no-op, but STILL return the triple
            return inputs_shared, inputs_posi, inputs_nega

        face_px = pipe.preprocess_video(video)  # expect a tensor (B,T,C,H,W) or similar

        # Fill pos/neg; keep shapes/dtypes consistent
        inputs_posi["face_pixel_values"] = face_px
        inputs_nega["face_pixel_values"] = torch.full_like(face_px, -1)

        return inputs_shared, inputs_posi, inputs_nega


class WanVideoPostUnit_AnimateInpaint(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("animate_inpaint_video", "animate_mask_video", "input_image", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )
        
    def get_i2v_mask(self, lat_t, lat_h, lat_w, mask_len=1, mask_pixel_values=None, device="cuda"):
        if mask_pixel_values is None:
            msk = torch.zeros(1, (lat_t-1) * 4 + 1, lat_h, lat_w, device=device)
        else:
            msk = mask_pixel_values.clone()
        msk[:, :mask_len] = 1
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]
        return msk

    def process(self, pipe: WanVideoPipeline, animate_inpaint_video, animate_mask_video, input_image, tiled, tile_size, tile_stride):
        if animate_inpaint_video is None or animate_mask_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)

        bg_pixel_values = pipe.preprocess_video(animate_inpaint_video)
        y_reft = pipe.vae.encode(bg_pixel_values, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0].to(dtype=pipe.torch_dtype, device=pipe.device)
        _, lat_t, lat_h, lat_w = y_reft.shape
        
        ref_pixel_values = pipe.preprocess_video([input_image])
        ref_latents = pipe.vae.encode(ref_pixel_values, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        mask_ref = self.get_i2v_mask(1, lat_h, lat_w, 1, device=pipe.device)
        y_ref = torch.concat([mask_ref, ref_latents[0]]).to(dtype=torch.bfloat16, device=pipe.device)
        
        mask_pixel_values = 1 - pipe.preprocess_video(animate_mask_video, max_value=1, min_value=0)
        mask_pixel_values = rearrange(mask_pixel_values, "b c t h w -> (b t) c h w")
        mask_pixel_values = torch.nn.functional.interpolate(mask_pixel_values, size=(lat_h, lat_w), mode='nearest')
        mask_pixel_values = rearrange(mask_pixel_values, "(b t) c h w -> b t c h w", b=1)[:,:,0]
        msk_reft = self.get_i2v_mask(lat_t, lat_h, lat_w, 0, mask_pixel_values=mask_pixel_values, device=pipe.device)
        
        y_reft = torch.concat([msk_reft, y_reft]).to(dtype=torch.bfloat16, device=pipe.device)
        y = torch.concat([y_ref, y_reft], dim=1).unsqueeze(0)
        return {"y": y}


class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None
        
        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],
            "Wan2.1-T2V-14B": [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
            "Wan2.1-I2V-14B-480P": [2.57151496e+05, -3.54229917e+04,  1.40286849e+03, -1.35890334e+01, 1.32517977e-01],
            "Wan2.1-I2V-14B-720P": [ 8.10705460e+03,  2.13393892e+03, -3.72934672e+02,  1.66203073e+01, -4.17769401e-02],
        }
        if model_id not in self.coefficients_dict:
            supported_model_ids = ", ".join([i for i in self.coefficients_dict])
            raise ValueError(f"{model_id} is not a supported TeaCache model id. Please choose a valid model id in ({supported_model_ids}).")
        self.coefficients = self.coefficients_dict[model_id]

    def check(self, dit: WanModel, x, t_mod):
        modulated_inp = t_mod.clone()
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = self.coefficients
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = x.clone()
        return not should_calc

    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states



class TemporalTiler_BCTHW:
    def __init__(self):
        pass

    def build_1d_mask(self, length, left_bound, right_bound, border_width):
        x = torch.ones((length,))
        if border_width == 0:
            return x
        
        shift = 0.5
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + shift) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip((torch.arange(border_width) + shift) / border_width, dims=(0,))
        return x

    def build_mask(self, data, is_bound, border_width):
        _, _, T, _, _ = data.shape
        t = self.build_1d_mask(T, is_bound[0], is_bound[1], border_width[0])
        mask = repeat(t, "T -> 1 1 T 1 1")
        return mask
    
    def run(self, model_fn, sliding_window_size, sliding_window_stride, computation_device, computation_dtype, model_kwargs, tensor_names, batch_size=None):
        tensor_names = [tensor_name for tensor_name in tensor_names if model_kwargs.get(tensor_name) is not None]
        tensor_dict = {tensor_name: model_kwargs[tensor_name] for tensor_name in tensor_names}
        B, C, T, H, W = tensor_dict[tensor_names[0]].shape
        if batch_size is not None:
            B *= batch_size
        data_device, data_dtype = tensor_dict[tensor_names[0]].device, tensor_dict[tensor_names[0]].dtype
        value = torch.zeros((B, C, T, H, W), device=data_device, dtype=data_dtype)
        weight = torch.zeros((1, 1, T, 1, 1), device=data_device, dtype=data_dtype)
        for t in range(0, T, sliding_window_stride):
            if t - sliding_window_stride >= 0 and t - sliding_window_stride + sliding_window_size >= T:
                continue
            t_ = min(t + sliding_window_size, T)
            model_kwargs.update({
                tensor_name: tensor_dict[tensor_name][:, :, t: t_:, :].to(device=computation_device, dtype=computation_dtype) \
                    for tensor_name in tensor_names
            })
            model_output = model_fn(**model_kwargs).to(device=data_device, dtype=data_dtype)
            mask = self.build_mask(
                model_output,
                is_bound=(t == 0, t_ == T),
                border_width=(sliding_window_size - sliding_window_stride,)
            ).to(device=data_device, dtype=data_dtype)
            value[:, :, t: t_, :, :] += model_output * mask
            weight[:, :, t: t_, :, :] += mask
        value /= weight
        model_kwargs.update(tensor_dict)
        return value



def model_fn_wan_video(
    dit: WanModel,
    motion_controller: WanMotionControllerModel = None,
    vace: VaceWanModel = None,
    animate_adapter: WanAnimateAdapter = None,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    reference_latents = None,
    vace_context = None,
    vace_scale = 1.0,
    audio_embeds: Optional[torch.Tensor] = None,
    motion_latents: Optional[torch.Tensor] = None,
    s2v_pose_latents: Optional[torch.Tensor] = None,
    drop_motion_frames: bool = True,
    tea_cache: TeaCache = None,
    use_unified_sequence_parallel: bool = False,
    motion_bucket_id: Optional[torch.Tensor] = None,
    pose_latents=None,
    face_pixel_values=None,
    sliding_window_size: Optional[int] = None,
    sliding_window_stride: Optional[int] = None,
    cfg_merge: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    control_camera_latents_input = None,
    fuse_vae_embedding_in_latents: bool = False,
    **kwargs,
):
    if sliding_window_size is not None and sliding_window_stride is not None:
        model_kwargs = dict(
            dit=dit,
            motion_controller=motion_controller,
            vace=vace,
            latents=latents,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            y=y,
            reference_latents=reference_latents,
            vace_context=vace_context,
            vace_scale=vace_scale,
            tea_cache=tea_cache,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
            motion_bucket_id=motion_bucket_id,
        )
        return TemporalTiler_BCTHW().run(
            model_fn_wan_video,
            sliding_window_size, sliding_window_stride,
            latents.device, latents.dtype,
            model_kwargs=model_kwargs,
            tensor_names=["latents", "y"],
            batch_size=2 if cfg_merge else 1
        )
    # wan2.2 s2v
    if audio_embeds is not None:
        return model_fn_wans2v(
            dit=dit,
            latents=latents,
            timestep=timestep,
            context=context,
            audio_embeds=audio_embeds,
            motion_latents=motion_latents,
            s2v_pose_latents=s2v_pose_latents,
            drop_motion_frames=drop_motion_frames,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
        )

    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                            get_sequence_parallel_world_size,
                                            get_sp_group)

    # Timestep
    if dit.seperated_timestep and fuse_vae_embedding_in_latents:
        timestep = torch.concat([
            torch.zeros((1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device),
            torch.ones((latents.shape[2] - 1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device) * timestep
        ]).flatten()
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0))
        if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
            t_chunks = torch.chunk(t, get_sequence_parallel_world_size(), dim=1)
            t_chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, t_chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in t_chunks]
            t = t_chunks[get_sequence_parallel_rank()]
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    
    # Motion Controller
    if motion_bucket_id is not None and motion_controller is not None:
        t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)

    x = latents
    # Merged cfg
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)

    # Image Embedding
    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)
    if clip_feature is not None and dit.require_clip_embedding:
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)
    
    # Camera control
    x = dit.patchify(x, control_camera_latents_input)
    
    # Animate
    if animate_adapter != None:
        x, motion_vec = animate_adapter.after_patch_embedding(x, pose_latents, face_pixel_values)
    
    # Patchify
    f, h, w = x.shape[2:]
    x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
    
    # Reference image
    if reference_latents is not None:
        if len(reference_latents.shape) == 5:
            reference_latents = reference_latents[:, :, 0]
        reference_latents = dit.ref_conv(reference_latents).flatten(2).transpose(1, 2)
        x = torch.concat([reference_latents, x], dim=1)
        f += 1
    
    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
    
    # TeaCache
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False
        
    if vace_context is not None:
        vace_hints = vace(
            x, vace_context, context, t_mod, freqs,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload
        )
    
    # blocks
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            chunks = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)
            pad_shape = chunks[0].shape[1] - chunks[-1].shape[1]
            chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in chunks]
            x = chunks[get_sequence_parallel_rank()]
    if tea_cache_update:
        x = tea_cache.update(x)
    else:
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        for block_id, block in enumerate(dit.blocks):
            # Block
            if use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs,
                        use_reentrant=False,
                    )
            elif use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, context, t_mod, freqs,
                    use_reentrant=False,
                )
            else:
                x = block(x, context, t_mod, freqs)
            
            # VACE
            if vace_context is not None and block_id in vace.vace_layers_mapping:
                current_vace_hint = vace_hints[vace.vace_layers_mapping[block_id]]
                if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
                    current_vace_hint = torch.chunk(current_vace_hint, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
                    current_vace_hint = torch.nn.functional.pad(current_vace_hint, (0, 0, 0, chunks[0].shape[1] - current_vace_hint.shape[1]), value=0)
                x = x + current_vace_hint * vace_scale
            
            # Animate
            if pose_latents is not None and face_pixel_values is not None:
                x = animate_adapter.after_transformer_block(block_id, x, motion_vec)
        if tea_cache is not None:
            tea_cache.store(x)
            
    x = dit.head(x, t)
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = get_sp_group().all_gather(x, dim=1)
            x = x[:, :-pad_shape] if pad_shape > 0 else x
    # Remove reference latents
    if reference_latents is not None:
        x = x[:, reference_latents.shape[1]:]
        f -= 1
    x = dit.unpatchify(x, (f, h, w))
    return x


def model_fn_wans2v(
    dit,
    latents,
    timestep,
    context,
    audio_embeds,
    motion_latents,
    s2v_pose_latents,
    drop_motion_frames=True,
    use_gradient_checkpointing_offload=False,
    use_gradient_checkpointing=False,
    use_unified_sequence_parallel=False,
):
    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                            get_sequence_parallel_world_size,
                                            get_sp_group)
    origin_ref_latents = latents[:, :, 0:1]
    x = latents[:, :, 1:]

    # context embedding
    context = dit.text_embedding(context)

    # audio encode
    audio_emb_global, merged_audio_emb = dit.cal_audio_emb(audio_embeds)

    # x and s2v_pose_latents
    s2v_pose_latents = torch.zeros_like(x) if s2v_pose_latents is None else s2v_pose_latents
    x, (f, h, w) = dit.patchify(dit.patch_embedding(x) + dit.cond_encoder(s2v_pose_latents))
    seq_len_x = seq_len_x_global = x.shape[1] # global used for unified sequence parallel

    # reference image
    ref_latents, (rf, rh, rw) = dit.patchify(dit.patch_embedding(origin_ref_latents))
    grid_sizes = dit.get_grid_sizes((f, h, w), (rf, rh, rw))
    x = torch.cat([x, ref_latents], dim=1)
    # mask
    mask = torch.cat([torch.zeros([1, seq_len_x]), torch.ones([1, ref_latents.shape[1]])], dim=1).to(torch.long).to(x.device)
    # freqs
    pre_compute_freqs = rope_precompute(x.detach().view(1, x.size(1), dit.num_heads, dit.dim // dit.num_heads), grid_sizes, dit.freqs, start=None)
    # motion
    x, pre_compute_freqs, mask = dit.inject_motion(x, pre_compute_freqs, mask, motion_latents, drop_motion_frames=drop_motion_frames, add_last_motion=2)

    x = x + dit.trainable_cond_mask(mask).to(x.dtype)

    # tmod
    timestep = torch.cat([timestep, torch.zeros([1], dtype=timestep.dtype, device=timestep.device)])
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim)).unsqueeze(2).transpose(0, 2)

    if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
        world_size, sp_rank = get_sequence_parallel_world_size(), get_sequence_parallel_rank()
        assert x.shape[1] % world_size == 0, f"the dimension after chunk must be divisible by world size, but got {x.shape[1]} and {get_sequence_parallel_world_size()}"
        x = torch.chunk(x, world_size, dim=1)[sp_rank]
        seg_idxs = [0] + list(torch.cumsum(torch.tensor([x.shape[1]] * world_size), dim=0).cpu().numpy())
        seq_len_x_list = [min(max(0, seq_len_x - seg_idxs[i]), x.shape[1]) for i in range(len(seg_idxs)-1)]
        seq_len_x = seq_len_x_list[sp_rank]

    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)
        return custom_forward

    for block_id, block in enumerate(dit.blocks):
        if use_gradient_checkpointing_offload:
            with torch.autograd.graph.save_on_cpu():
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, context, t_mod, seq_len_x, pre_compute_freqs[0],
                    use_reentrant=False,
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(lambda x: dit.after_transformer_block(block_id, x, audio_emb_global, merged_audio_emb, seq_len_x)),
                    x,
                    use_reentrant=False,
                )
        elif use_gradient_checkpointing:
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                x, context, t_mod, seq_len_x, pre_compute_freqs[0],
                use_reentrant=False,
            )
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(lambda x: dit.after_transformer_block(block_id, x, audio_emb_global, merged_audio_emb, seq_len_x)),
                x,
                use_reentrant=False,
            )
        else:
            x = block(x, context, t_mod, seq_len_x, pre_compute_freqs[0])
            x = dit.after_transformer_block(block_id, x, audio_emb_global, merged_audio_emb, seq_len_x_global, use_unified_sequence_parallel)

    if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
        x = get_sp_group().all_gather(x, dim=1)

    x = x[:, :seq_len_x_global]
    x = dit.head(x, t[:-1])
    x = dit.unpatchify(x, (f, h, w))
    # make compatible with wan video
    x = torch.cat([origin_ref_latents, x], dim=2)
    return x



import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from insightface.app import FaceAnalysis
# from facenet_pytorch import MTCNN #TODO try this model 
from kiui.lpips import LPIPS
import gc
# import onnxruntime

# onnxruntime.preload_dlls(directory="")


class FaceSupervisionLoss(nn.Module):
    def __init__(self, det_size=(1024, 1024),  output_size=(256, 256), lpips_engine=None, mse_weight=1.0, lpips_weight=1.0, device="cuda:0", local_process_index=0):
        super(FaceSupervisionLoss, self).__init__()
        self.output_size = output_size
        self.mse_weight = mse_weight
        self.lpips_weight = lpips_weight
        self.device = device

        # breakpoint()
        # print(f"{device=}, {local_process_index=}")
        
        self.face_detector = FaceAnalysis(
            name="buffalo_l", 
            allowed_modules=["detection"], 
            providers=[('CUDAExecutionProvider', {"device_id": torch.cuda.current_device(), "user_compute_stream": str(torch.cuda.current_stream().cuda_stream)})],
        ) 

        ctx_id = local_process_index
        
        self.face_detector.prepare(ctx_id=ctx_id, det_size=det_size)

        if not lpips_engine:
            self.lpips_loss_fn = LPIPS(net='vgg')
            self.lpips_loss_fn.to(self.device)
        else:
            self.lpips_loss_fn = lpips_engine
    
    def _tensor_to_numpy(self, img_tensor):
        img_np = img_tensor.to(torch.float16).detach().cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        return img_np

    def _detect_face(self, img_tensor):
        img_np = self._tensor_to_numpy(img_tensor)
        
        with torch.no_grad():
            detections = self.face_detector.get(img_np)
            
        if len(detections) == 0:
            return None
        
        best_det = max(detections, key=lambda d: d.det_score)
        return best_det

    def _crop_and_resize_face(self, img_tensor, det):
        _, H, W = img_tensor.shape
        bbox = det.bbox.astype(int)  # [x1, y1, x2, y2, ...]
        x1, y1, x2, y2 = bbox[:4]
        
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(W, x2)
        y2 = min(H, y2)
        
        if x2 - x1 < 10 or y2 - y1 < 10:
            return None

        face_patch = torch.zeros((1, H, W), dtype=img_tensor.dtype, device=img_tensor.device)
        face_patch[:, y1:y2, x1:x2] = 1.0
        face_img = img_tensor * face_patch
        
        if self.lpips_weight > 0:
            face_img = F.interpolate(face_img.unsqueeze(0), size=self.output_size, mode='bilinear', align_corners=False).squeeze(0)
        
        return face_img
    
    def get_gt_face(self, image, output_size):
        det = self._detect_face(image)
        _, H, W = image.shape
        bbox = det.bbox.astype(int)  # [x1, y1, x2, y2, ...]
        x1, y1, x2, y2 = bbox[:4]
        
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(W, x2)
        y2 = min(H, y2)
        
        if x2 - x1 < 10 or y2 - y1 < 10:
            return None
        
        face_img = image[:, y1:y2, x1:x2]
        
        face_img = F.interpolate(face_img.unsqueeze(0), size=output_size, mode='bilinear', align_corners=False).squeeze(0)
        
        return face_img
    
    def get_new_intrinsics(self, image, intrinsics, output_size):
        out_w, out_h = output_size
        
        det = self._detect_face(image)
        bbox = det.bbox.astype(int)  # [x1, y1, x2, y2, ...]
        x1, y1, x2, y2 = bbox[:4]
        
        fx, fy, cx, cy = intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[4]

        crop_w = x2 - x1
        crop_h = y2 - y1

        sx = out_w  / crop_w
        sy = out_h  / crop_h

        fx2 = fx * sx
        fy2 = fy * sy

        cx2 = (cx - x1) * sx
        cy2 = (cy - y1) * sy
        
        return fx2, fy2, cx2, cy2

    def forward(self, gt_images, pred_images, detect=True):
        B, V, C, H, W = gt_images.shape
        gt_faces = []
        pred_faces = []
        # breakpoint()
        if detect:
            for b in range(B):
                for v in range(V):
                    gt_img = gt_images[b, v]  # [3, H, W]
                    pred_img = pred_images[b, v]  # [3, H, W]

                    gt_det = self._detect_face(gt_img)
                    pred_det = self._detect_face(pred_img)
                    
                    if gt_det is None or pred_det is None:
                        continue

                    gt_face_patch = self._crop_and_resize_face(gt_img, gt_det)
                    pred_face_patch = self._crop_and_resize_face(pred_img, pred_det)
                    
                    if gt_face_patch is None or pred_face_patch is None:
                        continue

                    # NOTE DEBUG ONLY
                    cv2.imwrite(f"./debug/{b}-{v}gt_test.png", gt_face_patch.to(torch.float16).detach().cpu().permute(1, 2, 0).numpy()*255)
                    cv2.imwrite(f"./debug/{b}-{v}pred_test.png", pred_face_patch.to(torch.float16).detach().cpu().permute(1, 2, 0).numpy()*255)
                    
                    # print("gt face device:", gt_face_patch.device)
                    # print("pred face device:", pred_face_patch.device)
                    
                    gt_faces.append(gt_face_patch)
                    pred_faces.append(pred_face_patch)
            
            if len(gt_faces) == 0 or len(pred_faces) == 0:
                return torch.tensor(0.0, device=self.device, requires_grad=True)
        
            gt_faces = torch.stack(gt_faces, dim=0).to(self.device) # [N, 3, output_H, output_W]
            pred_faces = torch.stack(pred_faces, dim=0).to(self.device)
        
        else:
            gt_faces = gt_images
            pred_faces = pred_images

        mse_loss = F.mse_loss(pred_faces, gt_faces)

        if self.lpips_weight > 0:
            # breakpoint()
            lpips_loss = self.lpips_loss_fn(gt_faces * 2 - 1, pred_faces * 2 - 1).mean()
        
        del gt_faces, pred_faces
        gc.collect()

        total_loss = self.mse_weight * mse_loss + self.lpips_weight * lpips_loss
        return total_loss