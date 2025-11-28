# metrics_helper.py
from __future__ import annotations
import os, csv, glob, math, re
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

import lpips as lpips_lib
from tqdm.auto import tqdm


# ---------------------------
# I/O + preprocessing helpers
# ---------------------------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


def list_files(folder: str | Path, allowed_exts: set[str]) -> List[Path]:
    p = Path(folder)
    return sorted([q for q in p.iterdir() if q.is_file() and q.suffix.lower() in allowed_exts])


def list_images(folder: str | Path) -> List[Path]:
    return list_files(folder, IMG_EXTS)


def list_videos(folder: str | Path) -> List[Path]:
    return list_files(folder, VIDEO_EXTS)


def infer_media_type(folder: str | Path) -> Optional[str]:
    for entry in Path(folder).iterdir():
        if not entry.is_file():
            continue
        ext = entry.suffix.lower()
        if ext in IMG_EXTS:
            return "image"
        if ext in VIDEO_EXTS:
            return "video"
    return None


def read_rgb(path: str | Path) -> np.ndarray:
    """Read an image as float32 RGB in [0,1]."""
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return (im.astype(np.float32) / 255.0).clip(0, 1)


def to_torch_chw(x01: np.ndarray) -> torch.Tensor:
    """HWC [0,1] → 1xCxHxW float32 tensor."""
    t = torch.from_numpy(x01.transpose(2, 0, 1)).unsqueeze(0).float()
    return t


def resize_to(img01: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    """Resize to (H, W) with area interpolation (good for downscale)."""
    h, w = size_hw
    return cv2.resize(img01, (w, h), interpolation=cv2.INTER_AREA)

CANON_PATTERNS = [
    r"(in_distr[_\-\w]*\d+)$",
    r"(ood[_\-\w]*\d+)$",
]


def canon_stem(p: Path) -> str:
    """Canonicalize filenames so GT and predictions can be paired."""
    s = p.stem
    s = re.sub(r"^gt[_\-]+", "", s, flags=re.IGNORECASE)
    for pat in CANON_PATTERNS:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            s = m.group(1)
            break
    # normalize '_video' segments -> remove to align GT vs predictions
    s = re.sub(r"_video(?=_|$)", "", s, flags=re.IGNORECASE)
    return s


def read_video_frames(path: str | Path, max_frames: int = 49) -> List[np.ndarray]:
    """Decode up to `max_frames` RGB frames as float32 arrays in [0,1]."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    frames: List[np.ndarray] = []
    try:
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((frame.astype(np.float32) / 255.0).clip(0, 1))
    finally:
        cap.release()
    return frames


# ---------------------------
# Metric implementations
# ---------------------------
def compute_psnr(gt01: np.ndarray, pred01: np.ndarray) -> float:
    """PSNR in dB (higher is better)."""
    if pred01.shape != gt01.shape:
        pred01 = resize_to(pred01, gt01.shape[:2])
    return float(peak_signal_noise_ratio(gt01, pred01, data_range=1.0))

def compute_ssim(gt01: np.ndarray, pred01: np.ndarray) -> float:
    """SSIM in [0,1] typically (higher is better)."""
    if pred01.shape != gt01.shape:
        pred01 = resize_to(pred01, gt01.shape[:2])
    return float(structural_similarity(gt01, pred01, data_range=1.0, channel_axis=2))

class LPIPS:
    """LPIPS distance (lower is better)."""
    def __init__(self, net: str = "vgg", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = lpips_lib.LPIPS(net=net).to(self.device).eval()

    @torch.inference_mode()
    def __call__(self, gt01: np.ndarray, pred01: np.ndarray) -> float:
        if pred01.shape != gt01.shape:
            pred01 = resize_to(pred01, gt01.shape[:2])
        # lpips expects [-1,1] tensors, NCHW
        gt_t = to_torch_chw(gt01).to(self.device) * 2.0 - 1.0
        pr_t = to_torch_chw(pred01).to(self.device) * 2.0 - 1.0
        d = self.model.forward(gt_t, pr_t)
        return float(d.item())

class CSIM:
    """Cosine similarity of deep features (higher is better).
    Uses ResNet-50 global-average pooled features by default.
    """
    def __init__(self, backbone: str = "resnet50", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if backbone == "resnet50":
            net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            # remove classifier -> features before FC (2048-d after avgpool)
            self.backbone = nn.Sequential(*(list(net.children())[:-1])).to(self.device).eval()
            self.feat_dim = 2048
            self.input_size = 224
            self.normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def _prep(self, x01: np.ndarray) -> torch.Tensor:
        # Resize to model input and normalize
        x = cv2.resize(x01, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        t = to_torch_chw(x).to(self.device)
        t = self.normalize(t / 1.0)  # already [0,1]
        return t

    @torch.inference_mode()
    def __call__(self, gt01: np.ndarray, pred01: np.ndarray) -> float:
        gt_t = self._prep(gt01)
        pr_t = self._prep(pred01)
        # N=1, pass through backbone -> (1, C, 1, 1)
        f_gt = self.backbone(gt_t).flatten(1)  # (1, C)
        f_pr = self.backbone(pr_t).flatten(1)
        # cosine similarity
        f_gt = F.normalize(f_gt, dim=1)
        f_pr = F.normalize(f_pr, dim=1)
        sim = (f_gt * f_pr).sum(dim=1)  # (1,)
        return float(sim.item())


# ---------------------------
# Public API
# ---------------------------
class MetricBundle:
    """Holds metric computers so you can reuse heavy nets across many images."""
    def __init__(self, device: Optional[str] = None, lpips_net: str = "vgg"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.lpips = LPIPS(net=lpips_net, device=self.device)
        self.csim = CSIM(backbone="resnet50", device=self.device)

    def compute_all(self, gt01: np.ndarray, pred01: np.ndarray) -> Dict[str, float]:
        return {
            "PSNR": compute_psnr(gt01, pred01),
            "SSIM": compute_ssim(gt01, pred01),
            "LPIPS": self.lpips(gt01, pred01),
            "CSIM": self.csim(gt01, pred01),
        }

def compute_metrics_pair(gt_path: str | Path, pred_path: str | Path, bundle: Optional[MetricBundle] = None) -> Dict[str, float]:
    gt = read_rgb(gt_path)
    pr = read_rgb(pred_path)
    own = False
    if bundle is None:
        bundle = MetricBundle()
        own = True
    try:
        out = bundle.compute_all(gt, pr)
    finally:
        # keep bundle alive for reuse; no cleanup needed
        pass
    return out


def compute_metrics_video_pair(
    gt_path: str | Path,
    pred_path: str | Path,
    bundle: Optional[MetricBundle] = None,
    num_frames: int = 49,
    force_size: Optional[Tuple[int, int]] = None,
) -> Tuple[Dict[str, float], int]:
    """Compute metrics for two videos by averaging frame-level scores."""
    gt_frames = read_video_frames(gt_path, max_frames=num_frames)
    pr_frames = read_video_frames(pred_path, max_frames=num_frames)
    usable = min(len(gt_frames), len(pr_frames), num_frames)
    if usable == 0:
        raise RuntimeError(f"Could not decode overlapping frames for {gt_path} vs {pred_path}")

    if force_size is not None:
        gt_frames = [resize_to(frame, force_size) for frame in gt_frames[:usable]]
        pr_frames = [resize_to(frame, force_size) for frame in pr_frames[:usable]]
    else:
        gt_frames = gt_frames[:usable]
        pr_frames = pr_frames[:usable]

    if bundle is None:
        bundle = MetricBundle()
    metrics_per_frame = [
        bundle.compute_all(gt_frames[i], pr_frames[i])
        for i in range(usable)
    ]

    avg_metrics = {
        key: float(np.mean([m[key] for m in metrics_per_frame]))
        for key in metrics_per_frame[0].keys()
    }
    return avg_metrics, usable

def _pair_by_stem(gt_dir: Path, pred_dir: Path, valid_exts: set[str]) -> List[Tuple[Path, Path]]:
    """Pair images by normalized stem.
    - In GT, a leading 'gt_' is ignored for matching.
    - If multiple with same stem exist, the first sorted is used.
    """
    gt_files = list_files(gt_dir, valid_exts)
    pr_files = list_files(pred_dir, valid_exts)

    gt_map: Dict[str, Path] = {}
    for p in gt_files:
        gt_map.setdefault(canon_stem(p), p)

    pairs: List[Tuple[Path, Path]] = []
    for q in pr_files:
        key = canon_stem(q)
        if key in gt_map:
            pairs.append((gt_map[key], q))
    return pairs

def _evaluate_image_folders(gt_dir: Path, pred_dir: Path, save_csv: Optional[str | Path] = None) -> Dict[str, float]:
    pairs = _pair_by_stem(gt_dir, pred_dir, IMG_EXTS)
    if not pairs:
        raise RuntimeError("No image pairs found. Check filenames and extensions.")

    bundle = MetricBundle()
    rows = []
    for gt_p, pr_p in tqdm(pairs, desc=f"Images {pred_dir.name}", leave=False):
        m = compute_metrics_pair(gt_p, pr_p, bundle)
        rows.append({
            "name": pr_p.stem,
            "gt": str(gt_p.name),
            "pred": str(pr_p.name),
            **m
        })

    summary = {
        "PSNR": float(np.mean([r["PSNR"] for r in rows])),
        "SSIM": float(np.mean([r["SSIM"] for r in rows])),
        "LPIPS": float(np.mean([r["LPIPS"] for r in rows])),
        "CSIM": float(np.mean([r["CSIM"] for r in rows])),
    }

    if save_csv is not None:
        save_csv = Path(save_csv)
        save_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(save_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    return summary


def _evaluate_video_folders(
    gt_dir: Path,
    pred_dir: Path,
    save_csv: Optional[str | Path] = None,
    num_frames: int = 49,
    force_size: Optional[Tuple[int, int]] = None,
) -> Dict[str, float]:
    print("gt dir:", gt_dir, "pred dir:", pred_dir)
    pairs = _pair_by_stem(gt_dir, pred_dir, VIDEO_EXTS)
    
    if not pairs:
        raise RuntimeError("No video pairs found. Check filenames and extensions.")

    bundle = MetricBundle()
    rows = []
    for gt_p, pr_p in tqdm(pairs, desc=f"Videos {pred_dir.name}", leave=False):
        metrics, used_frames = compute_metrics_video_pair(
            gt_p,
            pr_p,
            bundle=bundle,
            num_frames=num_frames,
            force_size=force_size,
        )
        rows.append({
            "name": pr_p.stem,
            "gt": str(gt_p.name),
            "pred": str(pr_p.name),
            "frames": used_frames,
            **metrics,
        })

    summary = {
        "PSNR": float(np.mean([r["PSNR"] for r in rows])),
        "SSIM": float(np.mean([r["SSIM"] for r in rows])),
        "LPIPS": float(np.mean([r["LPIPS"] for r in rows])),
        "CSIM": float(np.mean([r["CSIM"] for r in rows])),
    }

    if save_csv is not None:
        save_csv = Path(save_csv)
        save_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(save_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    return summary


def evaluate_folders(
    gt_dir: str | Path,
    pred_dir: str | Path,
    save_csv: Optional[str | Path] = None,
    num_frames: int = 49,
    force_size: Optional[Tuple[int, int]] = None,
) -> Dict[str, float]:
    """Evaluate matched pairs (images or videos) in two folders and optionally write a CSV."""
    gt_dir, pred_dir = Path(gt_dir), Path(pred_dir)
    media_type = infer_media_type(gt_dir)
    if media_type is None:
        raise RuntimeError(f"Could not determine media type in {gt_dir}")
    pred_media_type = infer_media_type(pred_dir)
    if pred_media_type != media_type:
        raise RuntimeError(f"GT dir ({media_type}) and pred dir ({pred_media_type}) differ in media type.")

    if media_type == "image":
        return _evaluate_image_folders(gt_dir, pred_dir, save_csv=save_csv)
    if media_type == "video":
        return _evaluate_video_folders(
            gt_dir,
            pred_dir,
            save_csv=save_csv,
            num_frames=num_frames,
            force_size=force_size,
        )
    raise RuntimeError(f"Unsupported media type: {media_type}")


# ---------------------------
# Minimal CLI for convenience
# ---------------------------
if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(description="Compute PSNR/SSIM/LPIPS/CSIM for images or folders.")
    ap.add_argument("--gt", required=True, help="GT image or folder")
    ap.add_argument("--pred", required=True, help="Prediction image or folder")
    ap.add_argument("--csv", default=None, help="Optional CSV path for per-sample metrics")
    ap.add_argument("--num_frames", type=int, default=49, help="Max frames per video when evaluating mp4s.")
    ap.add_argument(
        "--force_size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=None,
        help="Optional height/width to resize both GT and prediction frames before computing metrics.",
    )
    args = ap.parse_args()

    gt_path, pred_path = Path(args.gt), Path(args.pred)
    force_size = tuple(args.force_size) if args.force_size else None

    if gt_path.is_dir() and pred_path.is_dir():
        summary = evaluate_folders(
            gt_path,
            pred_path,
            save_csv=args.csv,
            num_frames=args.num_frames,
            force_size=force_size,
        )
        print(json.dumps({"mode": "folder", "summary": summary}, indent=2))
    elif gt_path.is_file() and pred_path.is_file():
        if gt_path.suffix.lower() in VIDEO_EXTS:
            res, frames_used = compute_metrics_video_pair(
                gt_path,
                pred_path,
                num_frames=args.num_frames,
                force_size=force_size,
            )
            print(json.dumps({"mode": "video_pair", "frames": frames_used, "metrics": res}, indent=2))
        else:
            res = compute_metrics_pair(gt_path, pred_path)
            print(json.dumps({"mode": "pair", "metrics": res}, indent=2))
    else:
        raise SystemExit("Both --gt and --pred must be either files or folders.")
