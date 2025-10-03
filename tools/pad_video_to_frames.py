#!/usr/bin/env python3
"""Pad or trim videos to a fixed frame count (default 17)."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import imageio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Video files or directories containing videos.")
    parser.add_argument(
        "--target-frames",
        type=int,
        default=17,
        help="Desired frame count. Clips shorter than this will be padded by cloning the last frame; longer clips are trimmed.",
    )
    parser.add_argument(
        "--glob",
        default="*.mp4",
        help="Glob pattern when walking directories (default: *.mp4).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write padded clips. Defaults to overwriting alongside the source with a _pad suffix.",
    )
    parser.add_argument(
        "--suffix",
        default="_pad",
        help="Suffix to insert before the extension when writing next to the source (default: _pad). Ignored if --output-dir is given.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing files in the destination.",
    )
    return parser.parse_args()


def iter_video_paths(inputs: Iterable[str], pattern: str) -> Iterable[Path]:
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            yield from sorted(path.rglob(pattern))
        elif path.is_file():
            yield path
        else:
            print(f"Skipping missing path: {path}")


def load_video(path: Path) -> tuple[list, float]:
    reader = imageio.get_reader(path)
    try:
        fps = reader.get_meta_data().get("fps", 24)
        frames = [frame.copy() for frame in reader]
    finally:
        reader.close()
    return frames, fps


def pad_or_trim(frames: list, target: int) -> list:
    if not frames:
        raise ValueError("Video has zero frames; cannot pad.")
    if target <= 0:
        raise ValueError("target must be positive")
    if len(frames) == target:
        return frames
    if len(frames) > target:
        return frames[:target]
    pad_frame = frames[-1]
    frames.extend([pad_frame.copy() for _ in range(target - len(frames))])
    return frames


def write_video(frames: list, fps: float, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Destination exists: {path}. Use --overwrite to replace.")
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(path, fps=fps)
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def destination_for(src: Path, output_dir: Path | None, suffix: str) -> Path:
    if output_dir is not None:
        return output_dir / src.name
    return src.with_name(f"{src.stem}{suffix}{src.suffix}")


def main() -> None:
    args = parse_args()
    for video_path in iter_video_paths(args.inputs, args.glob):
        dest = destination_for(video_path, args.output_dir, args.suffix)
        print(f"Processing {video_path} -> {dest}")
        frames, fps = load_video(video_path)
        frames = pad_or_trim(frames, args.target_frames)
        write_video(frames, fps, dest, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
