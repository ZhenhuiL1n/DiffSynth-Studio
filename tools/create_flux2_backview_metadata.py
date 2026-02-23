#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


VIEW_NAME_RE = re.compile(r"^view_(\d+)_az(-?\d+(?:\.\d+)?)_el(-?\d+(?:\.\d+)?)\.png$")


def parse_args():
    parser = argparse.ArgumentParser(description="Create FLUX.2 back-view metadata.csv")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("data"),
        help="Dataset root that contains sequence folders (e.g., data/0008_01/frame_0/rgb/...)",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("data/back_view_dataset/metadata.csv"),
        help="Output metadata.csv path",
    )
    parser.add_argument(
        "--target_azimuth",
        type=float,
        default=340.0,
        help="Target azimuth used to define back view",
    )
    parser.add_argument(
        "--target_elevation",
        type=float,
        default=0.0,
        help="Target elevation used to define back view",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="Tolerance for float match on azimuth/elevation",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Back view of a woman in traditional Chinese hanfu dress, black background",
        help="Prompt written to every row",
    )
    parser.add_argument(
        "--sequence_start",
        type=str,
        default=None,
        help="Optional inclusive lower bound of sequence id (e.g., 0008_01)",
    )
    parser.add_argument(
        "--sequence_end",
        type=str,
        default=None,
        help="Optional inclusive upper bound of sequence id (e.g., 0133_07)",
    )
    return parser.parse_args()


def is_match(azimuth: float, elevation: float, target_azimuth: float, target_elevation: float, tol: float) -> bool:
    return abs(azimuth - target_azimuth) <= tol and abs(elevation - target_elevation) <= tol


def in_sequence_range(sequence_id: str, sequence_start: str | None, sequence_end: str | None) -> bool:
    if sequence_start is not None and sequence_id < sequence_start:
        return False
    if sequence_end is not None and sequence_id > sequence_end:
        return False
    return True


def build_rows(
    data_root: Path,
    prompt: str,
    target_azimuth: float,
    target_elevation: float,
    tol: float,
    sequence_start: str | None,
    sequence_end: str | None,
):
    rows = []
    rgb_files = sorted(data_root.glob("*/frame_*/rgb/*.png"))
    for image_path in rgb_files:
        match = VIEW_NAME_RE.match(image_path.name)
        if match is None:
            continue
        sequence_id = image_path.parts[-4]
        if not in_sequence_range(sequence_id, sequence_start, sequence_end):
            continue
        _, az_str, el_str = match.groups()
        azimuth = float(az_str)
        elevation = float(el_str)
        if not is_match(azimuth, elevation, target_azimuth, target_elevation, tol):
            continue

        rel_image = image_path.relative_to(data_root).as_posix()
        frame_id = image_path.parts[-3]
        _ = sequence_id, frame_id, azimuth, elevation  # kept for future extension
        rows.append({"image": rel_image, "prompt": prompt})
    return rows


def main():
    args = parse_args()
    data_root = args.data_root.resolve()
    output_csv = args.output_csv.resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = build_rows(
        data_root=data_root,
        prompt=args.prompt,
        target_azimuth=args.target_azimuth,
        target_elevation=args.target_elevation,
        tol=args.tol,
        sequence_start=args.sequence_start,
        sequence_end=args.sequence_end,
    )

    if len(rows) == 0:
        raise RuntimeError(
            f"No matching images found under {data_root} for az={args.target_azimuth}, el={args.target_elevation}."
        )

    fieldnames = ["image", "prompt"]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to: {output_csv}")
    print(f"Data root: {data_root}")
    print(f"Back view target: az={args.target_azimuth}, el={args.target_elevation}")
    if args.sequence_start is not None or args.sequence_end is not None:
        print(f"Sequence range: [{args.sequence_start or '-inf'}, {args.sequence_end or '+inf'}] (inclusive)")


if __name__ == "__main__":
    main()
