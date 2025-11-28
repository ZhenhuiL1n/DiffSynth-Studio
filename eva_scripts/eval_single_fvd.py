#!/usr/bin/env python3
"""
Compute FVD for a single prediction folder (e.g., an OOD set) against GT videos,
optionally using precomputed real stats (.npz).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from FVD_helper import compute_fvd, load_fvd_stats


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute FVD for one prediction folder.")
    ap.add_argument("--pred", required=True, help="Directory with generated videos (e.g., OOD set).")
    ap.add_argument("--gt", required=True, help="Directory with GT/reference videos.")
    ap.add_argument("--stats", default=None, help="Optional .npz stats file for GT (mu/sigma).")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_frames", type=int, default=64)
    ap.add_argument("--frame_height", type=int, default=768)
    ap.add_argument("--frame_width", type=int, default=768)
    ap.add_argument("--encoder_frames", type=int, default=32)
    ap.add_argument("--encoder_frame_size", type=int, default=112)
    ap.add_argument("--device", default=None, help="torch device (e.g., cuda:0).")
    ap.add_argument("--json_output", default=None, help="Optional JSON path to save the score.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    pred_dir = Path(args.pred)
    gt_dir = Path(args.gt)
    if not pred_dir.is_dir():
        raise SystemExit(f"Prediction folder does not exist: {pred_dir}")
    if not gt_dir.is_dir():
        raise SystemExit(f"GT folder does not exist: {gt_dir}")

    real_stats = None
    if args.stats:
        stats_path = Path(args.stats)
        if not stats_path.is_file():
            raise SystemExit(f"Stats file not found: {stats_path}")
        mu, sigma, _ = load_fvd_stats(stats_path)
        real_stats = (mu, sigma)

    fvd_value = compute_fvd(
        pred_dir,
        gt_dir,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        frame_height=args.frame_height,
        frame_width=args.frame_width,
        device=args.device,
        real_stats=real_stats,
        encoder_frames=args.encoder_frames,
        encoder_frame_size=args.encoder_frame_size,
    )

    payload = {"pred": str(pred_dir), "gt": str(gt_dir), "FVD": float(fvd_value)}
    print(json.dumps(payload, indent=2))
    if args.json_output:
        out_path = Path(args.json_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"[INFO] Saved summary to {out_path}")


if __name__ == "__main__":
    main()
