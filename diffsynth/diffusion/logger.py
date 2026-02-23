import os, torch
from accelerate import Accelerator


class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0
        self.loss_log_path = os.path.join(self.output_path, "train_loss.txt")
        self._loss_log_initialized = False


    def on_step_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None, **kwargs):
        self.num_steps += 1
        if "loss" in kwargs:
            loss = kwargs["loss"]
            if isinstance(loss, torch.Tensor):
                loss_tensor = loss.detach().float()
                if loss_tensor.ndim == 0:
                    loss_tensor = loss_tensor.unsqueeze(0)
                gathered = accelerator.gather(loss_tensor)
                loss_value = gathered.mean().item()
            else:
                loss_value = float(loss)

            lr_value = kwargs.get("lr", None)
            epoch_id = kwargs.get("epoch_id", None)
            msg_fields = [f"step={self.num_steps}", f"loss={loss_value:.6f}"]
            if lr_value is not None:
                msg_fields.append(f"lr={float(lr_value):.8g}")
            if epoch_id is not None:
                msg_fields.append(f"epoch={int(epoch_id)}")
            msg = " ".join(msg_fields)
            accelerator.print(msg)

            if accelerator.is_main_process:
                os.makedirs(self.output_path, exist_ok=True)
                if not self._loss_log_initialized:
                    with open(self.loss_log_path, "a", encoding="utf-8") as f:
                        if f.tell() == 0:
                            f.write("step\tloss\tlr\tepoch\n")
                    self._loss_log_initialized = True
                with open(self.loss_log_path, "a", encoding="utf-8") as f:
                    lr_txt = "" if lr_value is None else f"{float(lr_value):.10g}"
                    epoch_txt = "" if epoch_id is None else str(int(epoch_id))
                    f.write(f"{self.num_steps}\t{loss_value:.10g}\t{lr_txt}\t{epoch_txt}\n")

        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id):
        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)


    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def save_model(self, accelerator: Accelerator, model: torch.nn.Module, file_name):
        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)
