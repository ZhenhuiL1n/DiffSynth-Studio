#!/usr/bin/env python3
"""
Batch-compute FVD for every in_dist/ood folder under a root directory.

In-distribution folders are compared directly against GT MP4s.
Out-of-distribution folders are compared against precomputed FVD stats.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from FVD_helper import compute_fvd, load_fvd_stats


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute FVD for all in_dist/ood folders under a root.")
    ap.add_argument("--search_root", default="out", help="Root to scan for experiment folders.")
    ap.add_argument("--gt_bodys", required=True, help="GT MP4 folder for body experiments.")
    ap.add_argument("--gt_heads", required=True, help="GT MP4 folder for head experiments.")
    ap.add_argument("--body_stats", required=True, help="Precomputed FVD stats (.npz) for body OOD evaluation.")
    ap.add_argument("--head_stats", required=True, help="Precomputed FVD stats (.npz) for head OOD evaluation.")
    ap.add_argument("--batch_size", type=int, default=4, help="Batch size for feature extraction.")
    ap.add_argument("--num_frames", type=int, default=64, help="Frames sampled per clip before encoding.")
    ap.add_argument("--frame_height", type=int, default=224, help="Resize height before feature extraction.")
    ap.add_argument("--frame_width", type=int, default=224, help="Resize width before feature extraction.")
    ap.add_argument("--encoder_frames", type=int, default=32, help="Temporal frames expected by the encoder.")
    ap.add_argument("--encoder_frame_size", type=int, default=112, help="Spatial size expected by the encoder.")
    ap.add_argument("--device", type=str, default=None, help="torch device (e.g., cuda:0).")
    ap.add_argument("--json_output", default=None, help="Optional JSON summary path.")
    ap.add_argument(
        "--map",
        action="append",
        default=[],
        help="Manual mapping substring=head/body to resolve ambiguous experiment names (can repeat).",
    )
    ap.add_argument(
        "--default_label",
        choices=["head", "body"],
        default=None,
        help="Fallback label if heuristics + map cannot determine head/body.",
    )
    ap.add_argument(
        "--include",
        nargs="*",
        default=None,
        help="Only evaluate folders whose path contains at least one of these substrings.",
    )
    ap.add_argument(
        "--exclude",
        nargs="*",
        default=None,
        help="Skip folders whose path contains any of these substrings.",
    )
    return ap.parse_args()


def find_named_dirs(root: Path, name: str) -> List[Path]:
    return sorted(p for p in root.rglob(name) if p.is_dir())


def parse_mapping(entries: Sequence[str]) -> List[Tuple[str, str]]:
    mapping: List[Tuple[str, str]] = []
    for entry in entries:
        if "=" not in entry:
            raise SystemExit(f"Invalid --map '{entry}'. Expected substring=head/body.")
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
    raise RuntimeError(f"Could not infer head/body label for {path}. Use --map or --default_label.")


def should_skip(path: Path, includes: Optional[Sequence[str]], excludes: Optional[Sequence[str]]) -> bool:
    lower_path = str(path).lower()
    if excludes and any(token.lower() in lower_path for token in excludes):
        return True
    if not includes:
        return False
    include_tokens = [token.lower() for token in includes]
    return not any(token in lower_path for token in include_tokens)


def load_stats(path: Path) -> Tuple:
    mu, sigma, _ = load_fvd_stats(path)
    return (mu, sigma)


def main() -> None:
    args = parse_args()
    search_root = Path(args.search_root)
    gt_bodys = Path(args.gt_bodys)
    gt_heads = Path(args.gt_heads)
    body_stats_path = Path(args.body_stats)
    head_stats_path = Path(args.head_stats)

    for folder, label in [(gt_bodys, "body GT"), (gt_heads, "head GT")]:
        if not folder.is_dir():
            raise SystemExit(f"{label} folder does not exist: {folder}")
    if not body_stats_path.is_file():
        raise SystemExit(f"Body stats file not found: {body_stats_path}")
    if not head_stats_path.is_file():
        raise SystemExit(f"Head stats file not found: {head_stats_path}")

    mapping = parse_mapping(args.map)
    in_dist_dirs = find_named_dirs(search_root, "in_dist")
    ood_dirs = find_named_dirs(search_root, "ood")

    body_stats = load_stats(body_stats_path)
    head_stats = load_stats(head_stats_path)

    summary_rows: List[Dict[str, object]] = []
    errors: List[Dict[str, str]] = []

    def eval_folder(pred_dir: Path, mode: str) -> None:
        nonlocal summary_rows, errors
        if should_skip(pred_dir, args.include, args.exclude):
            return
        try:
            label = infer_label(pred_dir, mapping, args.default_label)
        except RuntimeError as exc:
            errors.append({"pred_dir": str(pred_dir), "error": str(exc), "mode": mode})
            return
        gt_dir = gt_heads if label == "head" else gt_bodys
        stats_tuple = head_stats if label == "head" else body_stats
        kwargs = dict(
            batch_size=args.batch_size,
            num_frames=args.num_frames,
            frame_height=args.frame_height,
            frame_width=args.frame_width,
            device=args.device,
            encoder_frames=args.encoder_frames,
            encoder_frame_size=args.encoder_frame_size,
        )
        try:
            if mode == "in_dist":
                score = compute_fvd(pred_dir, gt_dir, **kwargs)
            else:
                score = compute_fvd(pred_dir, gt_dir, real_stats=stats_tuple, **kwargs)
        except Exception as exc:  # noqa: BLE001
            errors.append({"pred_dir": str(pred_dir), "error": str(exc), "mode": mode})
            return
        row = {
            "mode": mode,
            "pred_dir": str(pred_dir),
            "experiment": str(pred_dir.relative_to(search_root)),
            "gt_label": label,
            "FVD": float(score),
        }
        summary_rows.append(row)
        print(f"[{mode}] {row['experiment']} ({label}) -> FVD {score:.4f}")

    for folder in in_dist_dirs:
        eval_folder(folder, "in_dist")
    for folder in ood_dirs:
        eval_folder(folder, "ood")

    payload = {"results": summary_rows, "errors": errors}
    print(json.dumps(payload, indent=2))

    if args.json_output:
        out_path = Path(args.json_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    main()
