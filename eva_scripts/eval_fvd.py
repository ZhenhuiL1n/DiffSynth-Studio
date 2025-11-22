#!/usr/bin/env python3
"""
Batch FVD evaluation for every experiment folder under `results/`.

Usage:
    python results/eval_fvd.py \
        --results_root results \
        --num_frames 64 \
        --frame_height 224 \
        --frame_width 224 \
        --batch_size 8 \
        --device cpu \
        --include_substring ood

All folders except `GT`, `__pycache__`, and dot-prefixed entries are treated
as experiments and are each compared to `GT` via FVD. The script prints a JSON
summary and writes individual numbers to `<experiment>/fvd.txt`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from FVD_helper import compute_fvd, load_fvd_stats


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute FVD for every experiment folder.")
    ap.add_argument("--results_root", type=str, default="results", help="Root directory holding GT/ and experiment folders.")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_frames", type=int, default=64)
    ap.add_argument("--frame_height", type=int, default=224)
    ap.add_argument("--frame_width", type=int, default=224)
    ap.add_argument("--device", type=str, default=None, help="torch device string, e.g. cpu or cuda:0.")
    ap.add_argument("--real_stats", type=str, default=None, help="Optional .npz file with precomputed real-data FVD stats.")
    ap.add_argument("--include_substring", type=str, default=None, help="Only evaluate experiment folders whose name contains this substring (case-insensitive).")
    ap.add_argument("--output_json", type=str, default=None, help="Optional path to dump all FVD scores as JSON.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.results_root).resolve()
    gt_dir = root / "GT"
    if not gt_dir.is_dir():
        raise SystemExit(f"GT directory not found: {gt_dir}")

    experiment_dirs = sorted(
        p for p in root.iterdir()
        if p.is_dir()
        and p.name.lower() not in {"gt", "__pycache__", "fvd_stats"}
        and not p.name.startswith(".")
    )

    if not experiment_dirs:
        raise SystemExit("No experiment folders next to GT.")

    scores = {}
    stats_tuple = None
    if args.real_stats:
        stats_path = Path(args.real_stats).expanduser().resolve()
        if not stats_path.is_file():
            raise SystemExit(f"real_stats file not found: {stats_path}")
        mu_r, sig_r, meta = load_fvd_stats(stats_path)
        stats_tuple = (mu_r, sig_r)
        info = f"{meta.get('num_samples', 'unknown')} samples" if meta else "unknown samples"
        print(f"Loaded real stats from {stats_path} ({info}).")

    for exp_dir in experiment_dirs:
        print(f"Computing FVD for {exp_dir.name} vs GT …")
        score = compute_fvd(
            gen_dir=str(exp_dir),
            real_dir=str(gt_dir),
            batch_size=args.batch_size,
            num_frames=args.num_frames,
            frame_height=args.frame_height,
            frame_width=args.frame_width,
            device=args.device,
            real_stats=stats_tuple,
            include_pred_substring=args.include_substring,
        )
        scores[exp_dir.name] = float(score)
        out_txt = exp_dir / "fvd.txt"
        out_txt.write_text(f"{score:.6f}\n")
        print(f"  -> FVD {score:.4f} (saved to {out_txt})")

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(scores, indent=2))
    else:
        print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
