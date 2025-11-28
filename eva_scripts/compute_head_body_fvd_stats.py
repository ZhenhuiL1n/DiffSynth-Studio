#!/usr/bin/env python3
"""
Compute and persist FVD statistics for two datasets (head/body) in a single pass.

Example:
    python eva_scripts/compute_head_body_fvd_stats.py \
        --head_videos /path/to/head/videos \
        --body_videos /path/to/body/videos \
        --head_output /tmp/head_stats.npz \
        --body_output /tmp/body_stats.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

from FVD_helper import compute_fvd_stats, save_fvd_stats


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Precompute FVD stats for head and body folders.")
    ap.add_argument("--head_videos", required=True, help="Directory containing the head video set.")
    ap.add_argument("--body_videos", required=True, help="Directory containing the body video set.")
    ap.add_argument("--head_output", required=True, help="Output .npz file to store head stats.")
    ap.add_argument("--body_output", required=True, help="Output .npz file to store body stats.")
    ap.add_argument("--num_frames", type=int, default=64, help="Frames sampled per clip before encoding.")
    ap.add_argument("--frame_height", type=int, default=224, help="Resize height before feature extraction.")
    ap.add_argument("--frame_width", type=int, default=224, help="Resize width before feature extraction.")
    ap.add_argument("--device", type=str, default=None, help="torch device for feature extraction (e.g., cpu or cuda:0).")
    ap.add_argument("--encoder_frames", type=int, default=32, help="Temporal frames expected by the encoder (keep 32 for R(2+1)D).")
    ap.add_argument("--encoder_frame_size", type=int, default=112, help="Spatial size expected by the encoder (keep 112 for R(2+1)D).")
    return ap.parse_args()


def compute_and_save(
    tag: str,
    video_dir: Path,
    out_path: Path,
    args: argparse.Namespace,
) -> Dict[str, str | int]:
    if not video_dir.is_dir():
        raise SystemExit(f"{tag} folder does not exist: {video_dir}")

    mu, sigma, meta = compute_fvd_stats(
        video_dir,
        num_frames=args.num_frames,
        frame_height=args.frame_height,
        frame_width=args.frame_width,
        device=args.device,
        encoder_frames=args.encoder_frames,
        encoder_frame_size=args.encoder_frame_size,
    )
    meta_out: Dict[str, str | int] = dict(meta)
    meta_out["videos_path"] = str(video_dir)
    meta_out["output"] = str(out_path)
    meta_out["split"] = tag
    save_fvd_stats(out_path, mu, sigma, meta_out)
    return meta_out


def main() -> None:
    args = parse_args()
    head_meta = compute_and_save("head", Path(args.head_videos), Path(args.head_output), args)
    body_meta = compute_and_save("body", Path(args.body_videos), Path(args.body_output), args)
    print(json.dumps({"head": head_meta, "body": body_meta}, indent=2))


if __name__ == "__main__":
    main()
