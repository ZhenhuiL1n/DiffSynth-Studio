import torch, os, json
from diffsynth import load_state_dict
from diffsynth.pipelines.wan_video_face import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, launch_training_task, wan_parser
from diffsynth.trainers.unified_dataset import UnifiedDataset, LoadVideo, ImageCropAndResize, ToAbsolutePath
from pathlib import Path
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from diffsynth.pipelines.wan_kd_mixin import WanKDMixin

class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32, lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
    ):
        super().__init__()
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False)
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs)
        
        kd = True
        if kd:
            # Load teacher model (assumed to be WAN Video Face)
            from copy import deepcopy
            self.teacher_model = deepcopy(self.pipe).eval()
            for param in self.teacher_model.dit.parameters():
                param.requires_grad = False
            # teacher_pipe = WanKDMixin()
            # teacher_pipe.attach_teacher(teacher_model)
            self.pipe.enable_kd_training(None)
            student_weights_path = "/home/longnhat/Lin_workspace/8TB2/Lin/801_Project/805/DiffSynth-Studio/models/train/805_baseline_bodyonly/epoch-5.safetensors"
            teacher_weights_path = "/home/longnhat/Lin_workspace/8TB2/Lin/801_Project/DiffSynth-Studio/models/train/805_baseline_teacher_weights/epoch-5.safetensors" # the same, just copy 
            self.teacher_model.load_lora(self.teacher_model.dit, str(teacher_weights_path), alpha=1)
            print(f"Loaded teacher weights from {teacher_weights_path}")
            self.pipe.load_lora(self.pipe.dit, str(student_weights_path), alpha=1)
            print(f"Initialized student weights from teacher weights {teacher_weights_path}")
            
            self.teacher_device = torch.device("cuda:1")
            # self.teacher_model.to(self.teacher_device)
        # # enable IP-Adapter (use SDXL projector weights -> 1280 dims)
        # self.pipe.enable_ipadapter(
        #     clip_vision_model="openai/clip-vit-large-patch14",
        #     ipadapter_weight_path="models/IpAdapter/stable_diffusion_xl/ip-adapter_sdxl.safetensors",
        #     num_tokens=16,
        #     scale=0.6,   # 0.3–0.8 typical; tune per dataset
        # )
        # self.pipe.enable_ipadapter_context(
        #     clip_vision_model="openai/clip-vit-large-patch14",
        #     ipadapter_weight_path="models/IpAdapter/sdxl/ip-adapter_sdxl.safetensors",
        #     num_tokens=16,
        #     scale=0.6,
        # )
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
        )
            
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        
        
    def forward_preprocess(self, data):
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}

        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        
        # Extra inputs
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        
        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}
    
    @torch.no_grad()
    def _to_teachers_device(self, inputs):
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.teacher_device, non_blocking=True)
            if isinstance(v, dict):
                inputs[k] = self._to_teachers_device(v)
            if isinstance(v, list):
                inputs[k] = [item.to(self.teacher_device, non_blocking=True) if isinstance(item, torch.Tensor) else item for item in v]
        return inputs
    
    @torch.no_grad()
    def kd_forward(self, inputs, timestep, kd_gate=None):
        # breakpoint()
        if self.teacher_model.device != self.teacher_device:
            self.teacher_model.to(self.teacher_device)
            torch.cuda.empty_cache()
            
        orig_device = inputs["latents"].device
        inputs_teacher = self._to_teachers_device(inputs)
        timestep = timestep.to(self.teacher_device)
        # max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * self.teacher_model.scheduler.num_train_timesteps)
        # min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * self.teacher_model.scheduler.num_train_timesteps)
        # timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
        # timestep = self.teacher_model.scheduler.timesteps[timestep_id].to(dtype=self.torch_dtype, device=self.device)
        # breakpoint()
        inputs["latents"] = self.teacher_model.scheduler.add_noise(inputs["input_latents"], inputs["noise"], timestep)
        # training_target = self.scheduler.training_target(inputs["input_latents"], inputs["noise"], timestep)
        tgt_full = self.teacher_model.scheduler.training_target(inputs["input_latents"], inputs["noise"], timestep)
        pred_teacher = self.teacher_model.model_fn(dit = self.teacher_model.dit, **inputs, timestep=timestep,)

        pred_teacher = pred_teacher.to(orig_device, non_blocking=True)

        if kd_gate is not None:
            pred_teacher = pred_teacher * kd_gate.view(-1, 1, 1, 1, 1)
        return pred_teacher
    
    def forward(self, data, inputs=None):
        # breakpoint()
        if inputs is None: inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        if hasattr(self, "teacher_model"):
            # self.pipe.teacher = self.teacher_model
            loss, student_pred, timestep_t = self.pipe.training_loss(**models, **inputs)
            pred_teacher = self.kd_forward(inputs, timestep_t, kd_gate=None)
            # kd_loss = torch.nn.functional.mse_loss(student_pred.float(), pred_teacher.float())
            # kd_loss = torch.nn.functional.mse_loss(student_pred.float(), pred_teacher.float())
            kd_loss = torch.nn.functional.kl_div(torch.log_softmax(student_pred, dim=1), torch.log_softmax(pred_teacher, dim=1), reduction='batchmean',log_target=True,) # this is superbig idk why
            if kd_loss.isnan().any() or kd_loss.isinf().any() or kd_loss.item() == 0:
                print(f"Warning: invalid KD loss encountered, {kd_loss.item()=}; skipping KD loss for this step.")
            loss = loss + 0.02 * kd_loss
            
        else:    
            loss = self.pipe.training_loss(**models, **inputs)
        
        return loss


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        ),
        special_operator_map={
            "animate_face_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16))
        }
    )
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
    )

    if args.eval_data:
        eval_data_kwargs = {
            "val_script_path": str(os.path.abspath(__file__).replace("examples/wanvideo/model_training/train.py", "val_ti2v.py")),
            "media_path": args.eval_media_path,
            "fps": args.eval_fps,
            "num_frames": args.num_frames,
        }
    else:
        eval_data_kwargs = None

    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        eval_data=args.eval_data,
        eval_data_kwargs=eval_data_kwargs,
    )
    # breakpoint()
    launch_training_task(dataset, model, model_logger, args=args)
