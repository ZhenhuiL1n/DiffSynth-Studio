#!/usr/bin/env python3
"""
Evaluate metrics (PSNR/SSIM/LPIPS/CSIM) on in_dist splits and FVD on ood splits
for the forget_gain body/head experiments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

from FVD_helper import compute_fvd, load_fvd_stats
from metrics_helper import evaluate_folders


def parse_args() -> argparse.Namespace:
    root_default = "/home/longnhat/Lin_workspace/8TB2/Lin/801_Project/805/DiffSynth-Studio"
    scripts_root = f"{root_default}/eva_scripts"
    ap = argparse.ArgumentParser(description="Evaluate forget_gain body/head outputs.")
    ap.add_argument("--body_pred", required=True, help="Path to forget_gain_body output folder.")
    ap.add_argument("--head_pred", required=True, help="Path to forget_gain_head output folder.")
    ap.add_argument("--gt_body", default=f"{root_default}/baseline_results/GT/body", help="GT videos for body.")
    ap.add_argument("--gt_head", default=f"{root_default}/baseline_results/GT/head", help="GT videos for head.")
    ap.add_argument("--body_force_size", nargs=2, type=int, default=[768, 768], metavar=("HEIGHT", "WIDTH"))
    ap.add_argument("--head_force_size", nargs=2, type=int, default=[768, 768], metavar=("HEIGHT", "WIDTH"))
    ap.add_argument("--body_stats", default=f"{scripts_root}/body_stats_768.npz", help="Precomputed stats for body OOD.")
    ap.add_argument("--head_stats", default=f"{scripts_root}/head_stats_768.npz", help="Precomputed stats for head OOD.")
    ap.add_argument("--num_frames", type=int, default=49, help="Frames sampled for metrics evaluation.")
    ap.add_argument("--fvd_num_frames", type=int, default=64, help="Frames sampled per clip for FVD.")
    ap.add_argument("--encoder_frames", type=int, default=32)
    ap.add_argument("--encoder_frame_size", type=int, default=112)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--device", default=None, help="torch device for evaluation.")
    ap.add_argument("--metrics_csv_dir", default=None, help="Optional directory to save per-sample metric CSVs.")
    ap.add_argument("--json_output", default=None, help="Optional JSON output path.")
    return ap.parse_args()


def ensure_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise SystemExit(f"{label} directory not found: {path}")


def eval_metrics(gt_dir: Path, pred_dir: Path, force_size: Tuple[int, int], num_frames: int, csv_path: Optional[Path]) -> Dict[str, float]:
    return evaluate_folders(gt_dir, pred_dir, save_csv=csv_path, num_frames=num_frames, force_size=force_size)


def eval_fvd(pred_dir: Path, gt_dir: Path, stats_path: Path, args: argparse.Namespace) -> float:
    mu, sigma, _ = load_fvd_stats(stats_path)
    return compute_fvd(
        pred_dir,
        gt_dir,
        batch_size=args.batch_size,
        num_frames=args.fvd_num_frames,
        frame_height=args.body_force_size[0] if "body" in pred_dir.parts else args.head_force_size[0],
        frame_width=args.body_force_size[1] if "body" in pred_dir.parts else args.head_force_size[1],
        device=args.device,
        real_stats=(mu, sigma),
        encoder_frames=args.encoder_frames,
        encoder_frame_size=args.encoder_frame_size,
    )


def main() -> None:
    args = parse_args()
    body_pred_root = Path(args.body_pred)
    head_pred_root = Path(args.head_pred)
    gt_body = Path(args.gt_body)
    gt_head = Path(args.gt_head)

    ensure_dir(body_pred_root, "body predictions root")
    ensure_dir(head_pred_root, "head predictions root")
    ensure_dir(gt_body, "body GT")
    ensure_dir(gt_head, "head GT")

    results: Dict[str, Dict[str, float]] = {}
    csv_dir = Path(args.metrics_csv_dir) if args.metrics_csv_dir else None
    if csv_dir:
        csv_dir.mkdir(parents=True, exist_ok=True)

    # Body metrics (in_dist)
    body_metrics_csv = csv_dir / "body_metrics.csv" if csv_dir else None
    body_metrics = eval_metrics(
        gt_body,
        body_pred_root / "in_dist",
        tuple(args.body_force_size),
        args.num_frames,
        body_metrics_csv,
    )
    results["body_metrics"] = body_metrics

    # Head metrics (in_dist)
    head_metrics_csv = csv_dir / "head_metrics.csv" if csv_dir else None
    head_metrics = eval_metrics(
        gt_head,
        head_pred_root / "in_dist",
        tuple(args.head_force_size),
        args.num_frames,
        head_metrics_csv,
    )
    results["head_metrics"] = head_metrics

    # Body FVD (ood)
    body_fvd = compute_fvd(
        body_pred_root / "ood",
        gt_body,
        batch_size=args.batch_size,
        num_frames=args.fvd_num_frames,
        frame_height=args.body_force_size[0],
        frame_width=args.body_force_size[1],
        device=args.device,
        real_stats=load_fvd_stats(Path(args.body_stats))[:2],
        encoder_frames=args.encoder_frames,
        encoder_frame_size=args.encoder_frame_size,
    )
    results["body_fvd"] = float(body_fvd)

    # Head FVD (ood)
    head_fvd = compute_fvd(
        head_pred_root / "ood",
        gt_head,
        batch_size=args.batch_size,
        num_frames=args.fvd_num_frames,
        frame_height=args.head_force_size[0],
        frame_width=args.head_force_size[1],
        device=args.device,
        real_stats=load_fvd_stats(Path(args.head_stats))[:2],
        encoder_frames=args.encoder_frames,
        encoder_frame_size=args.encoder_frame_size,
    )
    results["head_fvd"] = float(head_fvd)

    print(json.dumps(results, indent=2))
    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2))
        print(f"[INFO] Saved summary to {output_path}")


if __name__ == "__main__":
    main()
