#!/usr/bin/env python3
"""
Scan experiment folders for `in_dist` video directories and compute
PSNR/SSIM/LPIPS/CSIM metrics against the provided GT sets.

Example:
    python eva_scripts/eval_in_dist_metrics.py \
        --search_root out \
        --gt_bodys /path/to/eval_samples_805/bodys \
        --gt_heads /path/to/eval_samples_805/heads \
        --csv_dir metrics_csvs \
        --json_output metrics_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from metrics_helper import evaluate_folders


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Batch-evaluate PSNR/SSIM/LPIPS/CSIM for every in_dist folder under a root.")
    ap.add_argument("--search_root", default="out", help="Root to scan for experiment folders containing an 'in_dist' directory.")
    ap.add_argument("--gt_bodys", required=True, help="GT MP4 folder for body experiments.")
    ap.add_argument("--gt_heads", required=True, help="GT MP4 folder for head experiments.")
    ap.add_argument("--num_frames", type=int, default=49, help="Max frames sampled from each video.")
    ap.add_argument("--csv_dir", default=None, help="Optional directory to dump per-sample CSVs (mirrors experiment structure).")
    ap.add_argument("--json_output", default=None, help="Optional JSON file to store the aggregated summary.")
    ap.add_argument(
        "--map",
        action="append",
        default=[],
        help="Explicit substring mapping in the form 'needle=head' or 'needle=body' used to resolve GT selection.",
    )
    ap.add_argument(
        "--default_label",
        choices=["head", "body"],
        default=None,
        help="Fallback GT label if an in_dist path does not match heuristics or --map entries.",
    )
    ap.add_argument(
        "--include",
        nargs="*",
        default=None,
        help="Optional substrings; only evaluate in_dist paths containing at least one of them.",
    )
    ap.add_argument(
        "--exclude",
        nargs="*",
        default=None,
        help="Optional substrings; skip any in_dist paths containing one of them.",
    )
    ap.add_argument(
        "--force_size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=None,
        help="Resize GT/pred frames to this size before computing metrics (e.g., 758 758).",
    )
    return ap.parse_args()


def find_in_dist_dirs(root: Path) -> List[Path]:
    """Return every child directory named 'in_dist' under the root."""
    if not root.is_dir():
        raise SystemExit(f"Search root does not exist: {root}")
    return sorted(p for p in root.rglob("in_dist") if p.is_dir())


def parse_mapping(entries: Sequence[str]) -> List[Tuple[str, str]]:
    mapping: List[Tuple[str, str]] = []
    for entry in entries:
        if "=" not in entry:
            raise SystemExit(f"Invalid --map '{entry}' (expected format substring=head/body).")
        needle, label = entry.split("=", 1)
        label_lower = label.strip().lower()
        if label_lower not in {"head", "body"}:
            raise SystemExit(f"Unsupported label '{label}' in --map '{entry}'.")
        mapping.append((needle.strip().lower(), label_lower))
    return mapping


def infer_label(path: Path, mapping: List[Tuple[str, str]], default_label: Optional[str]) -> str:
    lower_path = str(path).lower()
    for needle, label in mapping:
        if needle and needle in lower_path:
            return label
    if "head" in lower_path:
        return "head"
    if any(tok in lower_path for tok in ("body", "bodies", "bodys")):
        return "body"
    if default_label:
        return default_label
    raise RuntimeError(f"Could not infer GT label for {path}. Use --map or --default_label.")


def should_skip(path: Path, includes: Optional[Sequence[str]], excludes: Optional[Sequence[str]]) -> bool:
    lower_path = str(path).lower()
    if excludes and any(token.lower() in lower_path for token in excludes):
        return True
    if includes is None:
        return False
    includes_lower = [token.lower() for token in includes]
    return not any(token in lower_path for token in includes_lower)


def sanitize_rel_path(path: Path) -> str:
    return "_".join(path.parts)


def main() -> None:
    args = parse_args()
    search_root = Path(args.search_root)
    gt_body_dir = Path(args.gt_bodys)
    gt_head_dir = Path(args.gt_heads)
    for dir_path, label in [(gt_body_dir, "body"), (gt_head_dir, "head")]:
        if not dir_path.is_dir():
            raise SystemExit(f"GT folder for {label} does not exist: {dir_path}")

    mapping = parse_mapping(args.map)
    force_size = tuple(args.force_size) if args.force_size else None
    in_dist_dirs = find_in_dist_dirs(search_root)
    results: List[Dict[str, object]] = []
    errors: List[Dict[str, str]] = []

    csv_root = Path(args.csv_dir) if args.csv_dir else None
    if csv_root:
        csv_root.mkdir(parents=True, exist_ok=True)

    for idx, pred_dir in enumerate(in_dist_dirs, 1):
        if should_skip(pred_dir, args.include, args.exclude):
            continue
        try:
            label = infer_label(pred_dir, mapping, args.default_label)
        except RuntimeError as exc:
            errors.append({"pred_dir": str(pred_dir), "error": str(exc)})
            continue
        gt_dir = gt_head_dir if label == "head" else gt_body_dir
        rel_path = pred_dir.relative_to(search_root)
        csv_path = None
        if csv_root is not None:
            csv_path = csv_root / f"{sanitize_rel_path(rel_path)}.csv"
        try:
            summary = evaluate_folders(
                gt_dir,
                pred_dir,
                save_csv=csv_path,
                num_frames=args.num_frames,
                force_size=force_size,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append({"pred_dir": str(pred_dir), "error": str(exc)})
            continue
        result_row: Dict[str, object] = {
            "experiment": str(rel_path),
            "pred_dir": str(pred_dir),
            "gt_label": label,
            **summary,
        }
        results.append(result_row)
        print(f"[{idx}/{len(in_dist_dirs)}] {rel_path} ({label}) -> {summary}")

    payload = {"results": results, "errors": errors, "total_evaluated": len(results)}
    print(json.dumps(payload, indent=2))

    if args.json_output:
        out_path = Path(args.json_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    main()
