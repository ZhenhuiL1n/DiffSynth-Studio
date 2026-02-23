#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw


def parse_args():
    parser = argparse.ArgumentParser(
        description="Copy corresponding input/GT images to eval output folder and build 3-column grids."
    )
    parser.add_argument(
        "--eval_root",
        type=Path,
        required=True,
        help="Eval output root (e.g., outputs/eval_backview_full_epoch4_sample_multiid)",
    )
    parser.add_argument(
        "--dataset_base",
        type=Path,
        default=Path("data"),
        help="Dataset base directory",
    )
    parser.add_argument(
        "--metadata_path",
        type=Path,
        default=None,
        help="Metadata CSV path. Default: <eval_root>/metadata_snapshot.csv",
    )
    parser.add_argument(
        "--pred_subdir",
        type=str,
        default="full",
        help="Prediction subdir under eval_root (e.g., full or lora)",
    )
    parser.add_argument(
        "--input_view_file",
        type=str,
        default="view_0016_az160.0_el0.0.png",
        help="Input view filename to fetch from dataset for each frame",
    )
    parser.add_argument(
        "--gt_view_file",
        type=str,
        default="view_0034_az340.0_el0.0.png",
        help="GT back view filename to fetch from dataset for each frame",
    )
    parser.add_argument(
        "--grid_name",
        type=str,
        default="grid_input_pred_gt",
        help="Grid output folder name under eval_root",
    )
    return parser.parse_args()


def parse_row_image(rel_image: str) -> Tuple[str, str]:
    parts = Path(rel_image).parts
    if len(parts) < 4:
        raise ValueError(f"Unexpected image path format: {rel_image}")
    seq = parts[0]
    frame = parts[1]
    return seq, frame


def read_rows(metadata_path: Path) -> List[str]:
    rows: List[str] = []
    with metadata_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "image" not in reader.fieldnames:
            raise RuntimeError(f"'image' column not found in {metadata_path}")
        for row in reader:
            rel = row["image"].strip()
            if rel:
                rows.append(rel)
    if not rows:
        raise RuntimeError(f"No rows found in metadata: {metadata_path}")
    return rows


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def draw_label(image: Image.Image, text: str) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    draw.rectangle((8, 8, 210, 58), fill=(0, 0, 0, 160))
    draw.text((18, 18), text, fill=(255, 255, 255))
    return img


def make_grid(input_img: Path, pred_img: Path, gt_img: Path, out_path: Path):
    inp = Image.open(input_img).convert("RGB")
    pred = Image.open(pred_img).convert("RGB")
    gt = Image.open(gt_img).convert("RGB")

    if pred.size != inp.size:
        pred = pred.resize(inp.size, Image.BICUBIC)
    if gt.size != inp.size:
        gt = gt.resize(inp.size, Image.BICUBIC)

    inp = draw_label(inp, "Input")
    pred = draw_label(pred, "Pred")
    gt = draw_label(gt, "GT")

    w, h = inp.size
    grid = Image.new("RGB", (w * 3, h))
    grid.paste(inp, (0, 0))
    grid.paste(pred, (w, 0))
    grid.paste(gt, (w * 2, 0))

    ensure_parent(out_path)
    grid.save(out_path)


def main():
    args = parse_args()
    eval_root = args.eval_root.resolve()
    dataset_base = args.dataset_base.resolve()
    metadata_path = (args.metadata_path or (eval_root / "metadata_snapshot.csv")).resolve()

    pred_root = eval_root / args.pred_subdir
    input_root = eval_root / "input"
    gt_root = eval_root / "gt"
    grid_root = eval_root / args.grid_name

    if not pred_root.exists():
        raise RuntimeError(f"Prediction folder not found: {pred_root}")
    if not metadata_path.exists():
        raise RuntimeError(f"Metadata path not found: {metadata_path}")

    rows = read_rows(metadata_path)
    ok = 0
    missing = 0
    for rel in rows:
        seq, frame = parse_row_image(rel)

        pred_path = pred_root / seq / frame / "rgb" / args.gt_view_file
        input_src = dataset_base / seq / frame / "rgb" / args.input_view_file
        gt_src = dataset_base / seq / frame / "rgb" / args.gt_view_file

        input_dst = input_root / seq / frame / "rgb" / args.input_view_file
        gt_dst = gt_root / seq / frame / "rgb" / args.gt_view_file
        grid_out = grid_root / f"{seq}_{frame}.png"

        if not pred_path.exists():
            print(f"[WARN] Missing pred: {pred_path}")
            missing += 1
            continue
        if not input_src.exists():
            print(f"[WARN] Missing input src: {input_src}")
            missing += 1
            continue
        if not gt_src.exists():
            print(f"[WARN] Missing gt src: {gt_src}")
            missing += 1
            continue

        ensure_parent(input_dst)
        ensure_parent(gt_dst)
        Image.open(input_src).save(input_dst)
        Image.open(gt_src).save(gt_dst)
        make_grid(input_dst, pred_path, gt_dst, grid_out)
        ok += 1

    print(f"Done. grids={ok}, missing={missing}")
    print(f"Pred root: {pred_root}")
    print(f"Input copied to: {input_root}")
    print(f"GT copied to: {gt_root}")
    print(f"Grid saved to: {grid_root}")


if __name__ == "__main__":
    main()
