#!/usr/bin/env python3
"""
Create metadata.csv for FLUX.2-klein camera adapter training.

Pairing rule (default):
- For each selected sequence folder
- Evenly sample N frame_* folders
- Use source view fixed at base azimuth (default 160.0 => view_0016_az160.0_el0.0.png)
- Build one target pair per requested relative angle

Default total rows with user settings:
10 folders * 20 frames * 9 angles = 1800
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


VIEW_RE = re.compile(r"^view_(\d+)_az(-?\d+(?:\.\d+)?)_el(-?\d+(?:\.\d+)?)\.png$")


@dataclass
class ViewEntry:
    path: Path
    view_id: int
    azimuth: float
    elevation: float


def parse_angles(text: str) -> List[float]:
    parts = [p.strip() for p in text.split(",") if p.strip() != ""]
    if not parts:
        raise ValueError("No angles found in --angles")
    return [float(p) for p in parts]


def parse_folder_list(text: str) -> List[str]:
    parts = [p.strip() for p in text.split(",") if p.strip() != ""]
    if not parts:
        raise ValueError("No folders found in --folders")
    return parts


def parse_view_file(path: Path) -> ViewEntry | None:
    m = VIEW_RE.match(path.name)
    if m is None:
        return None
    view_id = int(m.group(1))
    az = float(m.group(2))
    el = float(m.group(3))
    return ViewEntry(path=path, view_id=view_id, azimuth=az, elevation=el)


def list_frames(sequence_dir: Path) -> List[Path]:
    frame_dirs = [p for p in sequence_dir.iterdir() if p.is_dir() and p.name.startswith("frame_")]

    def frame_id(p: Path) -> int:
        return int(p.name.split("_")[-1])

    frame_dirs.sort(key=frame_id)
    return frame_dirs


def evenly_sample(items: Sequence[Path], k: int) -> List[Path]:
    if k <= 0:
        raise ValueError("k must be positive")
    n = len(items)
    if n == 0:
        return []
    if k >= n:
        return list(items)

    # Rounded linspace without numpy.
    raw_indices = [round(i * (n - 1) / (k - 1)) for i in range(k)]
    # Deduplicate while keeping order (can happen on very short sequences).
    used = set()
    indices = []
    for idx in raw_indices:
        idx = int(idx)
        if idx not in used:
            used.add(idx)
            indices.append(idx)

    # If dedup reduced length, fill gaps greedily.
    if len(indices) < k:
        for idx in range(n):
            if idx not in used:
                used.add(idx)
                indices.append(idx)
                if len(indices) == k:
                    break
        indices.sort()

    return [items[i] for i in indices[:k]]


def load_views(rgb_dir: Path) -> List[ViewEntry]:
    views: List[ViewEntry] = []
    for p in rgb_dir.iterdir():
        if not p.is_file():
            continue
        entry = parse_view_file(p)
        if entry is not None:
            views.append(entry)
    views.sort(key=lambda x: x.view_id)
    return views


def build_az_index(views: List[ViewEntry]) -> Dict[float, ViewEntry]:
    # Round azimuth keys for robust float matching.
    return {round(v.azimuth, 4): v for v in views}


def find_source_view(views: List[ViewEntry], base_azimuth: float, base_elevation: float) -> ViewEntry:
    for v in views:
        if abs(v.azimuth - base_azimuth) < 1e-4 and abs(v.elevation - base_elevation) < 1e-4:
            return v
    raise ValueError(
        f"Cannot find base source view az={base_azimuth}, el={base_elevation}. "
        f"Available az range: {[x.azimuth for x in views[:3]]} ... {[x.azimuth for x in views[-3:]]}"
    )


def normalize_az(angle: float) -> float:
    x = angle % 360.0
    if abs(x - 360.0) < 1e-6:
        x = 0.0
    return round(x, 4)


def choose_target_view(views: List[ViewEntry], target_abs_az: float, target_el: float) -> ViewEntry:
    # First try exact match.
    for v in views:
        if abs(v.azimuth - target_abs_az) < 1e-4 and abs(v.elevation - target_el) < 1e-4:
            return v

    # Fallback: nearest azimuth at same elevation.
    same_el = [v for v in views if abs(v.elevation - target_el) < 1e-4]
    if not same_el:
        same_el = views
    same_el.sort(key=lambda v: abs(normalize_az(v.azimuth) - normalize_az(target_abs_az)))
    return same_el[0]


def relpath_str(path: Path, base: Path) -> str:
    return str(path.relative_to(base)).replace("\\", "/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate metadata.csv for FLUX2 camera adapter training")
    parser.add_argument("--data_root", type=Path, default=Path("data"), help="Dataset root containing sequence folders")
    parser.add_argument(
        "--folders",
        type=str,
        default="0008_01,0012_09,0019_06,0022_10,0025_11,0031_03,0034_04,0047_01,0047_12,0094_02",
        help="Comma-separated sequence folder names",
    )
    parser.add_argument("--frames_per_folder", type=int, default=20, help="Evenly sampled frame count per sequence")
    parser.add_argument(
        "--angles",
        type=str,
        default="0,40,90,130,180,220,270,310,360",
        help="Comma-separated relative azimuth angles in degrees",
    )
    parser.add_argument(
        "--base_azimuth",
        type=float,
        default=160.0,
        help="Absolute source azimuth that defines relative 0°",
    )
    parser.add_argument("--base_elevation", type=float, default=0.0, help="Absolute source elevation")
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("data/camera_dataset/metadata.csv"),
        help="Output metadata CSV path",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a woman in traditional Chinese hanfu dress, black background",
        help="Prompt text written into CSV",
    )
    args = parser.parse_args()

    folders = parse_folder_list(args.folders)
    rel_angles = parse_angles(args.angles)

    rows: List[Dict[str, str]] = []

    for folder in folders:
        seq_dir = args.data_root / folder
        if not seq_dir.exists():
            raise FileNotFoundError(f"Missing sequence folder: {seq_dir}")

        all_frames = list_frames(seq_dir)
        sampled_frames = evenly_sample(all_frames, args.frames_per_folder)
        if len(sampled_frames) != args.frames_per_folder:
            raise RuntimeError(
                f"Expected {args.frames_per_folder} sampled frames for {folder}, got {len(sampled_frames)}"
            )

        for frame_dir in sampled_frames:
            rgb_dir = frame_dir / "rgb"
            if not rgb_dir.exists():
                raise FileNotFoundError(f"Missing rgb folder: {rgb_dir}")

            views = load_views(rgb_dir)
            if not views:
                raise RuntimeError(f"No view_*.png found in {rgb_dir}")

            source = find_source_view(views, args.base_azimuth, args.base_elevation)

            for rel_angle in rel_angles:
                target_abs_az = normalize_az(args.base_azimuth + rel_angle)
                target = choose_target_view(views, target_abs_az, args.base_elevation)

                rows.append(
                    {
                        # Training columns
                        "image": relpath_str(target.path, args.data_root),
                        "edit_image": relpath_str(source.path, args.data_root),
                        "prompt": args.prompt,
                        # Camera labels (delta mode preferred)
                        "camera_delta_azimuth": f"{rel_angle:.4f}",
                        "camera_delta_elevation": "0.0000",
                        # Extra debug / traceability columns
                        "sequence_id": folder,
                        "frame_id": frame_dir.name,
                        "source_azimuth": f"{source.azimuth:.4f}",
                        "source_elevation": f"{source.elevation:.4f}",
                        "target_azimuth": f"{target.azimuth:.4f}",
                        "target_elevation": f"{target.elevation:.4f}",
                        "target_rel_azimuth_request": f"{rel_angle:.4f}",
                        "source_view_file": source.path.name,
                        "target_view_file": target.path.name,
                    }
                )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output_csv}")
    expected = len(folders) * args.frames_per_folder * len(rel_angles)
    print(f"Expected rows: {expected}")


if __name__ == "__main__":
    main()
