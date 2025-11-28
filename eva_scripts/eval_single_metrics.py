#!/usr/bin/env python3
"""
Compute PSNR/SSIM/LPIPS/CSIM for a single prediction vs GT folder and emit JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from metrics_helper import evaluate_folders


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute metrics for one GT/pred pair.")
    ap.add_argument("--gt", required=True, help="GT video folder.")
    ap.add_argument("--pred", required=True, help="Prediction video folder.")
    ap.add_argument("--force_size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"), default=None)
    ap.add_argument("--num_frames", type=int, default=49, help="Frames sampled from each video.")
    ap.add_argument("--csv", default=None, help="Optional CSV path for per-sample metrics.")
    ap.add_argument("--json_output", default=None, help="Optional JSON path for summary metrics.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    gt_dir = Path(args.gt)
    pred_dir = Path(args.pred)
    if not gt_dir.is_dir():
        raise SystemExit(f"GT folder does not exist: {gt_dir}")
    if not pred_dir.is_dir():
        raise SystemExit(f"Prediction folder does not exist: {pred_dir}")

    csv_path = Path(args.csv) if args.csv else None
    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)

    force_size = tuple(args.force_size) if args.force_size else None
    summary = evaluate_folders(gt_dir, pred_dir, save_csv=csv_path, num_frames=args.num_frames, force_size=force_size)

    payload = {
        "gt": str(gt_dir),
        "pred": str(pred_dir),
        "num_frames": args.num_frames,
        "force_size": list(force_size) if force_size else None,
        "metrics": summary,
    }
    print(json.dumps(payload, indent=2))
    if args.json_output:
        out_path = Path(args.json_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"[INFO] Saved summary to {out_path}")


if __name__ == "__main__":
    main()
