#!/usr/bin/env python3
"""Compute diagonal Fisher information for WanVideo LoRA adapters."""
import argparse
import os
import sys
from pathlib import Path

# Ensure we import the local diffsynth package when executed as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffsynth.trainers.utils import wan_parser
from diffsynth.trainers.unified_dataset import (
    UnifiedDataset,
    LoadVideo,
    ImageCropAndResize,
    ToAbsolutePath,
)
from examples.wanvideo.model_training.train import WanTrainingModule
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch


def build_parser():
    parser = wan_parser()
    parser.description = "Compute Fisher matrix for WanVideo LoRA adapters"
    parser.add_argument("--max_fisher_steps", type=int, default=None, help="Optional cap on the number of batches used for Fisher accumulation.")
    return parser


def create_dataset(args):
    return UnifiedDataset(
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
            "animate_face_video": ToAbsolutePath(args.dataset_base_path)
            >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16))
        },
    )


def main():
    parser = build_parser()
    args = parser.parse_args()

    args.ewc_compute_fisher_only = True
    if args.ewc_fisher is None or args.ewc_prev_params is None:
        raise ValueError("--ewc_fisher and --ewc_prev_params must point to output files when computing Fisher.")

    dataset = create_dataset(args)
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
        ewc_lambda=0.0,
        ewc_fisher=args.ewc_fisher,
        ewc_prev_params=args.ewc_prev_params,
        ewc_compute_fisher_only=True,
    )

    dataloader = DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=args.dataset_num_workers)
    accelerator = Accelerator()
    model, dataloader = accelerator.prepare(model, dataloader)
    unwrapped = accelerator.unwrap_model(model)

    step_limit = args.max_fisher_steps
    progress = tqdm(dataloader, desc="Fisher", dynamic_ncols=True)
    for step, data in enumerate(progress):
        if step_limit is not None and step >= step_limit:
            break
        with accelerator.accumulate(model):
            loss = model(data)
            accelerator.backward(loss)
            unwrapped.accumulate_fisher()

    accelerator.wait_for_everyone()
    unwrapped.save_ewc_state()


if __name__ == "__main__":
    main()
