#!/usr/bin/env python3
"""
Precompute Fréchet Video Distance statistics (mu, sigma) for a video dataset.
This lets you reuse the same real-data baseline across multiple evaluations.

Example:
    python results/precompute_fvd_stats.py \
        --videos /path/to/training/videos \
        --output results/stats/train_fvd_stats.npz \
        --num_frames 64 --frame_size 224 --device cpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from FVD_helper import compute_fvd_stats, save_fvd_stats


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Precompute FVD statistics for a folder of reference videos.")
    ap.add_argument("--videos", required=True, help="Directory containing reference/real videos.")
    ap.add_argument("--output", required=True, help="Path to save the resulting .npz file (mu, sigma, metadata).")
    ap.add_argument("--num_frames", type=int, default=64, help="Frames to sample uniformly from each video before feature extraction.")
    ap.add_argument("--frame_height", type=int, default=224, help="Resize height before feature extraction.")
    ap.add_argument("--frame_width", type=int, default=224, help="Resize width before feature extraction.")
    ap.add_argument("--device", type=str, default=None, help="torch device for feature extraction (e.g., cpu or cuda:0).")
    ap.add_argument("--encoder_frames", type=int, default=32, help="Temporal length expected by the video encoder (keep 32 for R(2+1)D).")
    ap.add_argument("--encoder_frame_size", type=int, default=112, help="Spatial resolution expected by the encoder (keep 112 for R(2+1)D).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    videos_path = Path(args.videos)
    if not videos_path.is_dir():
        raise SystemExit(f"Video folder does not exist: {videos_path}")

    mu, sigma, meta = compute_fvd_stats(
        videos_path,
        num_frames=args.num_frames,
        frame_height=args.frame_height,
        frame_width=args.frame_width,
        device=args.device,
        encoder_frames=args.encoder_frames,
        encoder_frame_size=args.encoder_frame_size,
    )
    meta_json = dict(meta)
    meta_json["videos_path"] = str(videos_path)
    meta_json["output"] = str(Path(args.output))
    save_fvd_stats(args.output, mu, sigma, meta_json)
    print(json.dumps({"saved_to": args.output, "meta": meta_json}, indent=2))


if __name__ == "__main__":
    main()
