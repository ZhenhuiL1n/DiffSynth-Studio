#!/usr/bin/env python3
"""Preview the WanVideo training preprocessing pipeline on a single clip."""
from __future__ import annotations

import argparse
from pathlib import Path

from diffsynth.trainers.unified_dataset import UnifiedDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="Path to the input video (relative to base path or absolute).")
    parser.add_argument(
        "--base-path",
        default="",
        help="Dataset base path passed to the training script (default: current working directory).",
    )
    parser.add_argument("--height", type=int, default=832, help="Target height used during training.")
    parser.add_argument("--width", type=int, default=480, help="Target width used during training.")
    parser.add_argument("--max-pixels", type=int, default=1280 * 720, help="Dynamic-resolution cap from training.")
    parser.add_argument("--num-frames", type=int, default=49, help="Number of frames requested by the loader.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("preprocess_preview"),
        help="Where to dump the processed frames for inspection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    operator = UnifiedDataset.default_video_operator(
        base_path=args.base_path,
        max_pixels=args.max_pixels,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        time_division_factor=4,
        time_division_remainder=1,
    )

    frames = operator(args.video)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames):
        frame.save(args.output_dir / f"frame_{idx:04d}.png")

    print(f"Wrote {len(frames)} processed frames to {args.output_dir.resolve()}")
    if frames:
        print(f"Frame size after preprocessing: {frames[0].size[1]}x{frames[0].size[0]} (HxW)")


if __name__ == "__main__":
    main()
