#!/usr/bin/env python3
"""
Compute PSNR/SSIM/LPIPS/CSIM for the forgetting experiments (head/body) using GT videos.

Only the `in_dist` folders are evaluated while frames are resized to 758x758 by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

from metrics_helper import evaluate_folders


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate forgetting experiments (body/head) against GT videos.")
    ap.add_argument(
        "--root",
        default="/home/longnhat/Lin_workspace/8TB2/Lin/801_Project/805/DiffSynth-Studio/baseline_results",
        help="Root directory that contains `forgetting` and `GT` subfolders.",
    )
    ap.add_argument("--num_frames", type=int, default=49, help="Frames sampled per video for metrics.")
    ap.add_argument(
        "--force_size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=[758, 758],
        help="Resize GT/pred frames to this size before computing metrics (default: 758 758).",
    )
    ap.add_argument(
        "--csv_dir",
        default=None,
        help="Optional directory to dump per-sample CSVs (body.csv, head.csv).",
    )
    ap.add_argument(
        "--json_output",
        default=None,
        help="Optional JSON file to store aggregated metrics.",
    )
    return ap.parse_args()


def ensure_dir(path: Path, kind: str) -> None:
    if not path.is_dir():
        raise SystemExit(f"{kind} directory not found: {path}")


def evaluate_split(
    gt_dir: Path,
    pred_dir: Path,
    label: str,
    num_frames: int,
    force_size: Optional[Tuple[int, int]],
    csv_dir: Optional[Path],
) -> Dict[str, object]:
    ensure_dir(gt_dir, f"GT {label}")
    ensure_dir(pred_dir, f"{label} predictions")
    csv_path = None
    if csv_dir is not None:
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = csv_dir / f"{label}.csv"
    summary = evaluate_folders(
        gt_dir,
        pred_dir,
        save_csv=csv_path,
        num_frames=num_frames,
        force_size=force_size,
    )
    return {
        "label": label,
        "gt_dir": str(gt_dir),
        "pred_dir": str(pred_dir),
        **summary,
    }


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    forgetting_root = root / "forgetting"
    gt_root = root / "GT"
    ensure_dir(forgetting_root, "forgetting root")
    ensure_dir(gt_root, "GT root")

    force_size: Optional[Tuple[int, int]] = tuple(args.force_size) if args.force_size else None
    csv_dir = Path(args.csv_dir) if args.csv_dir else None

    body_result = evaluate_split(
        gt_root / "body",
        forgetting_root / "body" / "in_dist",
        "body",
        args.num_frames,
        force_size,
        csv_dir,
    )
    head_result = evaluate_split(
        gt_root / "head",
        forgetting_root / "head" / "in_dist",
        "head",
        args.num_frames,
        force_size,
        csv_dir,
    )

    payload = {"results": [body_result, head_result]}
    print(json.dumps(payload, indent=2))

    if args.json_output:
        out_path = Path(args.json_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    main()
