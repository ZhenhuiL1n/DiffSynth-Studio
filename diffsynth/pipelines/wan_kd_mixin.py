# diffsynth/pipelines/wan_kd_mixin.py
import copy
import torch
import torch.nn.functional as F

class WanKDMixin:
    def attach_teacher(self, teacher_pipe, teacher_device="cuda:0"):
        """
        teacher_pipe: a WanVideoPipeline (same arch) already loaded, will be frozen.
        """
        self.teacher = teacher_pipe
        self.teacher_device = torch.device(teacher_device)
        # Freeze teacher DiT and set eval
        self.teacher.dit.requires_grad_(False)
        self.teacher.dit.eval()
        # Make sure teacher scheduler matches (same FM formulation & boundaries)
        # If you set min/max timestep boundaries on student, mirror them here.
        # self.teacher.scheduler = copy.deepcopy(self.scheduler)

    # @torch.no_grad()
    # def _move_inputs_to(self, inputs, device):
    #     moved = {}
    #     for k, v in inputs.items():
    #         if isinstance(v, torch.Tensor):
    #             moved[k] = v.to(device, non_blocking=True)
    #         else:
    #             moved[k] = v
    #     return moved
    
    @torch.no_grad()
    def _to_teachers_device(self, inputs):
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.teacher_device, non_blocking=True)
            if isinstance(v, dict):
                inputs[k] = self._to_teacher_device(v)
            if isinstance(v, list):
                inputs[k] = [item.to(self.teacher_device, non_blocking=True) if isinstance(item, torch.Tensor) else item for item in v]
        return inputs

    @torch.no_grad()
    def kd_forward(self, inputs, timestep, kd_gate=None):
        """
        Compute teacher's prediction on the SAME (noisy_latents, timestep, inputs),
        return it on the student's device for loss computation.
        - noisy_latents: [B, C, T, H_l, W_l] on student device
        - timestep: torch.LongTensor or float tensor on student device
        - inputs: dict used by self.model_fn(**inputs, timestep=...)
        - kd_gate: optional [B] gate (0..1), typically a function of sigma(t)
        """
        assert hasattr(self, "teacher"), "Call attach_teacher(...) first."
        # Move data to teacher device
        # breakpoint()
        orig_device = inputs["latents"].device
        inputs_teacher = self._to_teachers_device(inputs)
        # noisy_latents_t = noisy_latents.to(self.teacher_device, non_blocking=True)
        timestep_t = timestep.to(self.teacher_device)
        # inputs_t = self._move_inputs_to(inputs, self.teacher_device)

        # Forward teacher
        # Your pipelines call either self.model_fn or self.dit directly;
        # model_fn usually wraps DiT with proper signature
        if hasattr(self.teacher, "model_fn"):
            pred_teacher = self.teacher.model_fn(**inputs, timestep=timestep_t,)
        else:
            # Fallback: direct DiT call (common signature in DiffSynth WAN)
            raise NotImplementedError("Teacher pipeline has no model_fn; implement direct DiT call if needed.")

        # Bring to student's device
        pred_teacher = pred_teacher.to(orig_device, non_blocking=True)

        if kd_gate is not None:
            pred_teacher = pred_teacher * kd_gate.view(-1, 1, 1, 1, 1)
        return pred_teacher
