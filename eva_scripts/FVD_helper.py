"""
PyTorch-only Fréchet Video Distance helper.
Uses torchvision's R(2+1)D-18 (Kinetics-400) encoder to extract features and computes
the Fréchet distance between GT and prediction embeddings.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18
from scipy import linalg
from tqdm.auto import tqdm

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
CANON_PATTERNS = [
    r"(in_distr[_\-\w]*\d+)$",
    r"(ood[_\-\w]*\d+)$",
]


# ---------------- math ----------------
def frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray, eps: float = 1e-6) -> float:
    """Fréchet distance between two multivariate Gaussians."""
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset), disp=False)[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean)


def gaussian_stats(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


# ---------------- pairing helpers ----------------
def list_videos(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXTS])


def canon_stem(path: Path) -> str:
    s = path.stem
    s = re.sub(r"^gt[_\-]+", "", s, flags=re.IGNORECASE)
    for pat in CANON_PATTERNS:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            s = m.group(1)
            break
    s = re.sub(r"_video(?=_|$)", "", s, flags=re.IGNORECASE)
    return s


def pair_videos(gt_dir: Path, pred_dir: Path) -> List[Tuple[Path, Path]]:
    gt_map: Dict[str, Path] = {}
    for vid in list_videos(gt_dir):
        gt_map.setdefault(canon_stem(vid), vid)
    pairs: List[Tuple[Path, Path]] = []
    for vid in list_videos(pred_dir):
        key = canon_stem(vid)
        if key in gt_map:
            pairs.append((gt_map[key], vid))
    return pairs


# ---------------- decoding ----------------
def read_video_clip(
    path: Path,
    num_frames: int,
    frame_height: int,
    frame_width: int,
) -> Optional[torch.Tensor]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 1:
        cap.release()
        return None
    idxs = np.linspace(0, max(0, total - 1), num_frames).astype(int)
    frames: List[torch.Tensor] = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
        frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        frames.append(frame)
    cap.release()
    if len(frames) != num_frames:
        return None
    clip = torch.stack(frames, dim=0)  # T,C,H,W
    return clip


def sample_temporal(clip: torch.Tensor, target_frames: int) -> torch.Tensor:
    """Uniformly sample/duplicate frames to reach `target_frames`."""
    t = clip.shape[0]
    if t == target_frames:
        return clip
    idxs = torch.linspace(0, max(0, t - 1), target_frames).long()
    return clip[idxs]


# ---------------- feature extractor ----------------
class VideoFeatureExtractor:
    def __init__(self, device: Optional[str] = None, input_frames: int = 32, frame_size: int = 112):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        weights = R2Plus1D_18_Weights.KINETICS400_V1
        backbone = r2plus1d_18(weights=weights)
        backbone.fc = torch.nn.Identity()
        self.model = backbone.to(self.device).eval()
        self.transforms = weights.transforms()
        self.input_frames = input_frames
        self.frame_size = frame_size

    @torch.no_grad()
    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        # clip: T,C,H,W in [0,1]
        clip = sample_temporal(clip, self.input_frames)  # T,C,H,W
        clip = self.transforms(clip)
        feats = self.model(clip.unsqueeze(0).to(self.device))  # (1, 512)
        return feats.squeeze(0).cpu()


# ---------------- stats helpers ----------------
def compute_fvd_stats(
    folder: str | Path,
    num_frames: int = 64,
    frame_height: int = 224,
    frame_width: int = 224,
    device: Optional[str] = None,
    encoder_frames: int = 32,
    encoder_frame_size: int = 112,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    folder = Path(folder)
    extractor = VideoFeatureExtractor(device=device, input_frames=encoder_frames, frame_size=encoder_frame_size)
    feats: List[torch.Tensor] = []
    videos = list_videos(folder)
    for vid in tqdm(videos, desc=f"Stats {folder.name}", leave=False):
        clip = read_video_clip(vid, num_frames=num_frames, frame_height=frame_height, frame_width=frame_width)
        if clip is None:
            continue
        feats.append(extractor(clip))
    if not feats:
        raise RuntimeError(f"No usable clips found in {folder} to compute FVD stats.")
    mat = torch.stack(feats).numpy()
    mu, sigma = gaussian_stats(mat)
    meta = {
        "num_videos": len(videos),
        "num_samples": len(feats),
        "num_frames": num_frames,
        "frame_height": frame_height,
        "frame_width": frame_width,
        "encoder_frames": encoder_frames,
        "encoder_frame_size": encoder_frame_size,
    }
    return mu, sigma, meta


def save_fvd_stats(path: str | Path, mu: np.ndarray, sigma: np.ndarray, meta: Optional[Dict[str, int]] = None) -> None:
    meta = {} if meta is None else meta
    np.savez(path, mu=mu, sigma=sigma, **meta)


def load_fvd_stats(path: str | Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    data = np.load(path, allow_pickle=True)
    mu = data["mu"]
    sigma = data["sigma"]
    meta: Dict[str, float] = {}
    for key in data.files:
        if key in {"mu", "sigma"}:
            continue
        value = data[key]
        if isinstance(value, np.ndarray) and value.shape == ():
            value = value.item()
        meta[key] = value
    return mu, sigma, meta


# ---------------- main API ----------------
def compute_fvd(
    gen_dir: str | Path,
    real_dir: str | Path,
    batch_size: int = 4,
    num_frames: int = 64,
    frame_height: int = 224,
    frame_width: int = 224,
    device: Optional[str] = None,
    real_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    real_stats_path: Optional[str | Path] = None,
    encoder_frames: int = 32,
    encoder_frame_size: int = 112,
    include_pred_substring: Optional[str] = None,
) -> float:
    gen_dir = Path(gen_dir)
    real_dir = Path(real_dir)
    # optionally load stats from disk
    if real_stats is None and real_stats_path is not None:
        mu_tmp, sig_tmp, meta = load_fvd_stats(real_stats_path)
        real_stats = (mu_tmp, sig_tmp)
        if meta:
            if "encoder_frames" in meta and meta["encoder_frames"] != encoder_frames:
                print(f"[FVD] Warning: stats computed with encoder_frames={meta['encoder_frames']} but current={encoder_frames}.")
            if "encoder_frame_size" in meta and meta["encoder_frame_size"] != encoder_frame_size:
                print(f"[FVD] Warning: stats computed with encoder_frame_size={meta['encoder_frame_size']} but current={encoder_frame_size}.")
            if "num_frames" in meta and meta["num_frames"] != num_frames:
                print(f"[FVD] Warning: stats used {meta['num_frames']} frames per clip but current={num_frames}.")
            if ("frame_height" in meta and meta["frame_height"] != frame_height) or ("frame_width" in meta and meta["frame_width"] != frame_width):
                print("[FVD] Warning: stats computed with different frame height/width.")

    use_pairs = real_stats is None
    if use_pairs:
        pairs = pair_videos(real_dir, gen_dir)
        if not pairs:
            raise RuntimeError(f"No video pairs between {real_dir} and {gen_dir}.")
        if include_pred_substring:
            needle = include_pred_substring.lower()
            pairs = [pair for pair in pairs if needle in pair[1].name.lower()]
            if not pairs:
                raise RuntimeError(f"No prediction files containing '{include_pred_substring}' found in {gen_dir}.")
    else:
        pred_files = list_videos(gen_dir)
        if include_pred_substring:
            needle = include_pred_substring.lower()
            pred_files = [p for p in pred_files if needle in p.name.lower()]
        if not pred_files:
            raise RuntimeError(f"No prediction videos (after filtering) found in {gen_dir}.")

    extractor = VideoFeatureExtractor(device=device, input_frames=encoder_frames, frame_size=encoder_frame_size)
    feats_real: List[torch.Tensor] = []
    feats_pred: List[torch.Tensor] = []

    if use_pairs:
        iterator = tqdm(pairs, desc=f"FVD {gen_dir.name}", leave=False)
        for gt_path, pred_path in iterator:
            pred_clip = read_video_clip(pred_path, num_frames=num_frames, frame_height=frame_height, frame_width=frame_width)
            if pred_clip is None:
                continue
            feats_pred.append(extractor(pred_clip))
            gt_clip = read_video_clip(gt_path, num_frames=num_frames, frame_height=frame_height, frame_width=frame_width)
            if gt_clip is None:
                continue
            feats_real.append(extractor(gt_clip))
    else:
        iterator = tqdm(pred_files, desc=f"FVD {gen_dir.name}", leave=False)
        for pred_path in iterator:
            pred_clip = read_video_clip(pred_path, num_frames=num_frames, frame_height=frame_height, frame_width=frame_width)
            if pred_clip is None:
                continue
            feats_pred.append(extractor(pred_clip))

    if not feats_pred:
        raise RuntimeError("Failed to extract any prediction features for FVD.")

    pred_mat = torch.stack(feats_pred).numpy()
    mu_p, sig_p = gaussian_stats(pred_mat)

    if real_stats is None:
        if not feats_real:
            raise RuntimeError("Failed to extract any real features for FVD.")
        real_mat = torch.stack(feats_real).numpy()
        mu_r, sig_r = gaussian_stats(real_mat)
    else:
        mu_r, sig_r = real_stats

    return frechet_distance(mu_p, sig_p, mu_r, sig_r)
