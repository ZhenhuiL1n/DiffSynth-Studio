#!/usr/bin/env python3
"""
Batch-crop & resize images in a folder using the same center-crop algorithm
from your snippet (crop to match the target aspect ratio, then resize).

Usage:
  python batch_crop_resize.py --input /path/to/images --output /path/to/out \
      --width 768 --height 432

Options:
  --recursive                Process images in subfolders as well.
  --formats .jpg .jpeg .png  File extensions to include (default: common types).
  --keep-structure           Mirror the input folder structure under the output folder.
  --overwrite                Overwrite existing files in the output folder.
  --jpeg-quality 90          JPEG save quality when saving .jpg/.jpeg files.
  --workers 0                Number of parallel workers (0 or 1 = no parallelism).
  --dry-run                  Show what would be processed without writing files.
"""

import argparse
import os
import sys
import concurrent.futures
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageOps


# ---- Crop & Resize algorithm (your original) ----
def crop_and_resize(image: Image.Image, height: int, width: int) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    arr = np.array(image)
    image_height, image_width, _ = arr.shape

    if image_height * width < height * image_width:
        cropped_width = int(image_height / height * width)
        left = (image_width - cropped_width) // 2
        arr = arr[:, left:left + cropped_width]
        result = Image.fromarray(arr).resize((width, height), Image.LANCZOS)
    else:
        cropped_height = int(image_width / width * height)
        top = (image_height - cropped_height) // 2
        arr = arr[top:top + cropped_height, :]
        result = Image.fromarray(arr).resize((width, height), Image.LANCZOS)
    return result


# ---- Utilities ----
def split_file_name(file_name: str) -> Tuple:
    result = []
    number = -1
    for ch in file_name:
        if ord("0") <= ord(ch) <= ord("9"):
            if number == -1:
                number = 0
            number = number * 10 + ord(ch) - ord("0")
        else:
            if number != -1:
                result.append(number)
                number = -1
            result.append(ch)
    if number != -1:
        result.append(number)
    return tuple(result)


def discover_images(folder: str, exts: Iterable[str], recursive: bool) -> List[str]:
    exts = {e.lower() for e in exts}
    paths: List[str] = []
    if recursive:
        for root, _, files in os.walk(folder):
            for f in files:
                if os.path.splitext(f)[1].lower() in exts:
                    paths.append(os.path.join(root, f))
    else:
        for f in os.listdir(folder):
            if os.path.splitext(f)[1].lower() in exts:
                paths.append(os.path.join(folder, f))

    paths = sorted(paths, key=lambda p: split_file_name(os.path.basename(p)))
    return paths


def out_path_for(in_path: str, input_root: str, output_root: str, keep_structure: bool) -> str:
    base = os.path.basename(in_path)
    if keep_structure:
        rel = os.path.relpath(os.path.dirname(in_path), input_root)
        return os.path.join(output_root, rel, base)
    return os.path.join(output_root, base)


def process_one(args) -> Tuple[str, str]:
    (in_path, out_path, width, height, overwrite, jpeg_quality, dry_run) = args
    try:
        if (not overwrite) and os.path.exists(out_path):
            return (in_path, f"skip (exists) -> {out_path}")
        if dry_run:
            return (in_path, f"dry-run -> {out_path}")

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with Image.open(in_path) as im:
            processed = crop_and_resize(im, height=height, width=width)

            ext = os.path.splitext(out_path)[1].lower()
            if ext in (".jpg", ".jpeg"):
                processed.save(out_path, quality=jpeg_quality, optimize=True)
            elif ext == ".png":
                processed.save(out_path, compress_level=6, optimize=True)
            else:
                processed.save(out_path)

        return (in_path, f"ok -> {out_path}")
    except Exception as e:
        return (in_path, f"ERROR: {e}")


# ---- Main CLI ----
def main() -> int:
    parser = argparse.ArgumentParser(description="Batch center-crop & resize images in a folder.")
    parser.add_argument("--input", required=True, help="Path to input image folder")
    parser.add_argument("--output", required=True, help="Path to output folder")
    parser.add_argument("--width", type=int, required=True, help="Target width in pixels")
    parser.add_argument("--height", type=int, required=True, help="Target height in pixels")
    parser.add_argument("--recursive", action="store_true", help="Process images in subfolders too")
    parser.add_argument("--keep-structure", action="store_true", help="Mirror folder structure under output")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite files that already exist in output")
    parser.add_argument("--formats", nargs="*", default=[".jpg", ".jpeg", ".png", ".webp"],
                        help="List of file extensions to include (default: .jpg .jpeg .png .webp)")
    parser.add_argument("--jpeg-quality", type=int, default=90, help="JPEG quality for .jpg/.jpeg saves (default: 90)")
    parser.add_argument("--workers", type=int, default=0, help="Parallel workers (0 or 1 = no parallelism)")
    parser.add_argument("--dry-run", action="store_true", help="List planned outputs without writing files")
    args = parser.parse_args()

    if args.width <= 0 or args.height <= 0:
        print("Width and height must be positive integers.", file=sys.stderr)
        return 2

    inputs = discover_images(args.input, args.formats, args.recursive)
    if not inputs:
        print("No images found. Check --input and --formats.", file=sys.stderr)
        return 1

    todo = []
    for in_path in inputs:
        out_path = out_path_for(in_path, args.input, args.output, args.keep_structure)
        in_ext = os.path.splitext(in_path)[1]
        out_root, _ = os.path.splitext(out_path)
        out_path = out_root + in_ext
        todo.append((in_path, out_path, args.width, args.height, args.overwrite, args.jpeg_quality, args.dry_run))

    print(f"Found {len(todo)} image(s). Processing to {args.width}x{args.height} ...")

    if args.workers and args.workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            for src, status in pool.map(process_one, todo):
                print(src, "->", status)
    else:
        for item in todo:
            src, status = process_one(item)
            print(src, "->", status)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

