#!/usr/bin/env python3
"""
Compute FVD for the baseline mixed experiment (body/head) similar to the metrics CLI.

- `in_dist` folders are paired directly with GT videos.
- `ood` folders are compared against precomputed stats (.npz).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from FVD_helper import compute_fvd, load_fvd_stats


def parse_args() -> argparse.Namespace:
    default_root = "/home/longnhat/Lin_workspace/8TB2/Lin/801_Project/805/DiffSynth-Studio/baseline_results"
    default_scripts = "/home/longnhat/Lin_workspace/8TB2/Lin/801_Project/805/DiffSynth-Studio/eva_scripts"
    ap = argparse.ArgumentParser(description="Compute FVD for a baseline experiment (mixed/forgetting/pure_fintune).")
    ap.add_argument("--root", default=default_root, help="Root containing experiments (mixed, forgetting, pure_fintune) and GT.")
    ap.add_argument("--experiment", default="mixed", help="Name of the experiment folder under --root.")
    ap.add_argument("--body_subdir", default="body", help="Subdirectory name for body predictions inside the experiment.")
    ap.add_argument("--head_subdir", default="heads", help="Subdirectory name for head predictions inside the experiment.")
    ap.add_argument("--gt_body", default=None, help="Path to GT body videos (defaults to root/GT/body).")
    ap.add_argument("--gt_head", default=None, help="Path to GT head videos (defaults to root/GT/head).")
    ap.add_argument(
        "--body_stats",
        default=f"{default_scripts}/body_stats_768.npz",
        help="Precomputed FVD stats (.npz) for body OOD evaluation.",
    )
    ap.add_argument(
        "--head_stats",
        default=f"{default_scripts}/head_stats_768.npz",
        help="Precomputed FVD stats (.npz) for head OOD evaluation.",
    )
    ap.add_argument("--batch_size", type=int, default=4, help="Batch size for feature extraction.")
    ap.add_argument("--num_frames", type=int, default=64, help="Frames sampled per clip before encoding.")
    ap.add_argument("--frame_height", type=int, default=224, help="Resize height before feature extraction.")
    ap.add_argument("--frame_width", type=int, default=224, help="Resize width before feature extraction.")
    ap.add_argument("--device", type=str, default=None, help="torch device (e.g., cuda:0).")
    ap.add_argument("--encoder_frames", type=int, default=32, help="Temporal frames expected by the encoder.")
    ap.add_argument("--encoder_frame_size", type=int, default=112, help="Spatial size expected by the encoder.")
    ap.add_argument("--json_output", default=None, help="Optional JSON file to store the aggregated results.")
    return ap.parse_args()


def ensure_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise SystemExit(f"{label} directory not found: {path}")


def load_stats(path: Path) -> Tuple:
    if not path.is_file():
        raise SystemExit(f"Stats file not found: {path}")
    mu, sigma, _ = load_fvd_stats(path)
    return mu, sigma


def eval_split(
    name: str,
    pred_root: Path,
    gt_dir: Path,
    stats_tuple: Tuple,
    args: argparse.Namespace,
) -> List[Dict[str, object]]:
    ensure_dir(pred_root, f"{name} prediction root")
    kwargs = dict(
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        frame_height=args.frame_height,
        frame_width=args.frame_width,
        device=args.device,
        encoder_frames=args.encoder_frames,
        encoder_frame_size=args.encoder_frame_size,
    )
    results: List[Dict[str, object]] = []

    in_dir = pred_root / "in_dist"
    if in_dir.is_dir():
        try:
            fvd_value = compute_fvd(in_dir, gt_dir, **kwargs)
            results.append(
                {
                    "label": name,
                    "mode": "in_dist",
                    "pred_dir": str(in_dir),
                    "gt_dir": str(gt_dir),
                    "FVD": float(fvd_value),
                }
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                {
                    "label": name,
                    "mode": "in_dist",
                    "pred_dir": str(in_dir),
                    "error": str(exc),
                }
            )

    ood_dir = pred_root / "ood"
    if ood_dir.is_dir():
        try:
            fvd_value = compute_fvd(ood_dir, gt_dir, real_stats=stats_tuple, **kwargs)
            results.append(
                {
                    "label": name,
                    "mode": "ood",
                    "pred_dir": str(ood_dir),
                    "gt_stats": args.head_stats if name == "head" else args.body_stats,
                    "FVD": float(fvd_value),
                }
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                {
                    "label": name,
                    "mode": "ood",
                    "pred_dir": str(ood_dir),
                    "error": str(exc),
                }
            )

    return results


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    ensure_dir(root, "baseline root")
    exp_dir = root / args.experiment
    ensure_dir(exp_dir, f"experiment {args.experiment}")

    gt_body = Path(args.gt_body) if args.gt_body else root / "GT" / "body"
    gt_head = Path(args.gt_head) if args.gt_head else root / "GT" / "head"
    ensure_dir(gt_body, "GT body")
    ensure_dir(gt_head, "GT head")

    body_stats = load_stats(Path(args.body_stats))
    head_stats = load_stats(Path(args.head_stats))

    body_preds = exp_dir / args.body_subdir
    head_preds = exp_dir / args.head_subdir

    all_results: List[Dict[str, object]] = []
    if body_preds.is_dir():
        all_results.extend(
            eval_split(
                "body",
                body_preds,
                gt_body,
                body_stats,
                args,
            )
        )
    if head_preds.is_dir():
        all_results.extend(
            eval_split(
                "head",
                head_preds,
                gt_head,
                head_stats,
                args,
            )
        )

    payload = {"experiment": args.experiment, "results": all_results}
    print(json.dumps(payload, indent=2))

    if args.json_output:
        out_path = Path(args.json_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    main()
