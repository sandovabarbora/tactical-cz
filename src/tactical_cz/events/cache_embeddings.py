"""Run V-JEPA2 once over all clips in a split, write embeddings + labels parquet.

Phase 2 architectural choice (2026-05-18): we freeze the V-JEPA2-L
backbone for the entire Phase 2. Re-encoding 100K clips every epoch
of head training is pointless cost — encode once, train the head over
the cached vectors. Side benefits:

    * Head training is seconds per epoch (BCE over (B, 1024) → (B, 12),
      no cv2, no GPU mem for activations of a frozen model).
    * Hyperparam search over loss weights, thresholds, head designs is
      free in iteration time.
    * Whole pipeline runs on Apple-silicon MPS overnight. Zero cloud
      bill for Phase 2 baseline.

Schema (one row per clip window):

    match_id       str   "england_efl/2019-2020/2019-10-01 - Mboro - PNE"
    clip_id        str   "{match_id}@{frame_idx}"
    video_path     str   absolute path to 720p.mp4 (or 224p.mp4 if --resolution 224p)
    frame_idx      int   center frame of the clip window
    embedding      list[float]   1024-d V-JEPA2-L mean-pooled token vector
    labels         list[int]     length-12 multi-hot over BAS_CLASSES

Clip window discovery:
    For each match's Labels-ball.json, we emit one window CENTERED on
    every annotated event, plus one extra "background" window every
    ``negatives_per_minute`` seconds (default 6 → roughly 1 negative
    per 10 seconds of match time). Negatives have all-zero label
    vectors. Both are mixed when training.

Usage:
    uv run python -m tactical_cz.events.cache_embeddings \\
        --split valid --resolution 224p \\
        --out data/processed/embeddings_valid.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from tactical_cz.config import (
    PROCESSED_DIR,
    configure_logging,
    ensure_dirs,
    get_device,
)
from tactical_cz.events.dataset import discover_matches, load_match_annotations
from tactical_cz.events.model import BAS_LABEL_TO_ID, BAS_NUM_CLASSES

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """All knobs for the embedding cache pass."""

    split_dir: Path
    out_parquet: Path
    backbone: str = "facebook/vjepa2-vitl-fpc64-256"
    resolution: str = "224p"               # "224p" or "720p"
    clip_seconds: float = 4.0
    fps: int = 25
    resize: int = 256
    label_window_seconds: float = 2.0      # event @ T → labels for any window centered in [T-2s, T+2s]
    negatives_per_minute: int = 6          # ~1 negative per 10s of match
    batch_size: int = 4
    flush_every_n_matches: int = 5         # write partial parquet so resume after crash works


# Labels-ball.json's gameTime field is e.g. "1 - 0:23.480" (period - mm:ss.ms)
_GT_PATTERN = re.compile(r"(\d+)\s*-\s*(\d+):(\d+(?:\.\d+)?)")


def _gametime_to_seconds(gametime: str, period_seconds: float = 45 * 60) -> float:
    """Parse 'period - mm:ss(.ms)' → seconds from match start.

    Assumes 45-minute halves, no injury time accounting (close enough
    for matching to video frames since videos are full half recordings).
    """
    m = _GT_PATTERN.search(gametime)
    if not m:
        return float("nan")
    period, mm, ss = int(m.group(1)), int(m.group(2)), float(m.group(3))
    return (period - 1) * period_seconds + mm * 60 + ss


def _load_video_meta(video_path: Path) -> tuple[float, int]:
    """Returns (fps, total_frames)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total


def _decode_clip(video_path: Path, center_frame: int, clip_frames: int, resize: int) -> np.ndarray:
    """Seek to a window centered at center_frame, decode clip_frames RGB frames, resize."""
    half = clip_frames // 2
    start = max(0, center_frame - half)
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames: list[np.ndarray] = []
    while len(frames) < clip_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (resize, resize))
        frames.append(frame)
    cap.release()
    if len(frames) < clip_frames:
        # Pad with the last good frame (edge of video case)
        if not frames:
            raise ValueError(f"No frames at {video_path} around {center_frame}")
        while len(frames) < clip_frames:
            frames.append(frames[-1])
    return np.stack(frames, axis=0)        # (T, H, W, 3) uint8


def _build_clip_windows_for_match(
    match_dir: Path,
    cfg: CacheConfig,
) -> list[tuple[int, np.ndarray]]:
    """Decide which windows we'll embed for a match.

    Returns list of (center_frame, label_vector) tuples. Each event in
    the match's Labels-ball.json contributes one positive window;
    negatives are sampled uniformly across the match every
    60 / cfg.negatives_per_minute seconds.
    """
    video_path = match_dir / f"{cfg.resolution.replace('p', 'p')}.mp4"
    if not video_path.exists():
        # Try the other resolution as fallback
        alt = match_dir / ("720p.mp4" if cfg.resolution == "224p" else "224p.mp4")
        if alt.exists():
            video_path = alt
        else:
            logger.warning("No video found in %s", match_dir)
            return []
    fps, total_frames = _load_video_meta(video_path)

    anns = load_match_annotations(match_dir).get("annotations", [])
    label_window_frames = int(cfg.label_window_seconds * fps)
    clip_frames = int(cfg.clip_seconds * fps)

    # Index events by their center frame for cheap proximity queries later
    event_frames: list[tuple[int, str]] = []
    for a in anns:
        t = _gametime_to_seconds(a.get("gameTime", ""))
        if np.isnan(t):
            continue
        label = a.get("label")
        if label not in BAS_LABEL_TO_ID:
            continue
        event_frames.append((int(t * fps), label))

    # Pick window centers: every event becomes one positive window;
    # negatives uniformly sampled across the match.
    centers: set[int] = set()
    for fidx, _ in event_frames:
        centers.add(max(clip_frames // 2, min(fidx, total_frames - clip_frames // 2 - 1)))
    neg_step = int(60 / cfg.negatives_per_minute * fps)
    for fidx in range(clip_frames // 2, total_frames - clip_frames // 2, neg_step):
        centers.add(fidx)

    # Build label vectors for each window
    windows = []
    for center in sorted(centers):
        labels = np.zeros(BAS_NUM_CLASSES, dtype=np.int8)
        for ev_frame, ev_label in event_frames:
            if abs(ev_frame - center) <= label_window_frames:
                labels[BAS_LABEL_TO_ID[ev_label]] = 1
        windows.append((center, labels))
    return windows, video_path, fps


@torch.no_grad()
def _embed_batch(
    backbone,
    clips: np.ndarray,           # (B, T, H, W, 3) uint8
    device: str,
) -> np.ndarray:                 # (B, hidden_dim) float32
    """Single forward pass through V-JEPA2 → mean-pooled token embeddings."""
    arr = clips.astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    x = torch.from_numpy(arr).permute(0, 1, 4, 2, 3).to(device)    # (B, T, 3, H, W)
    out = backbone(pixel_values_videos=x)
    pooled = out.last_hidden_state.mean(dim=1)                     # (B, hidden_dim)
    return pooled.cpu().float().numpy()


def cache_embeddings(cfg: CacheConfig) -> Path:
    """Run V-JEPA2 over a split, write embeddings + labels parquet."""
    device = get_device()
    logger.info("Embedding pass on device=%s", device)

    from transformers import VJEPA2Model
    backbone = VJEPA2Model.from_pretrained(cfg.backbone).to(device).train(False)
    hidden_dim = backbone.config.hidden_size
    logger.info("Loaded backbone %s (hidden_dim=%d)", cfg.backbone, hidden_dim)

    matches = discover_matches(cfg.split_dir)
    if not matches:
        raise RuntimeError(
            f"No matches in {cfg.split_dir}. Extract first:\n"
            f"  uv run python -m tactical_cz.events.downloader extract --split {cfg.split_dir.name}"
        )

    cfg.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    clip_frames_per_window = int(cfg.clip_seconds * cfg.fps)

    for m_idx, match_dir in enumerate(matches, 1):
        windows, video_path, fps = _build_clip_windows_for_match(match_dir, cfg)
        match_id = str(match_dir.relative_to(cfg.split_dir))
        logger.info("[%d/%d] %s: %d windows", m_idx, len(matches), match_id, len(windows))

        # Batch the windows for one forward at a time
        for batch_start in range(0, len(windows), cfg.batch_size):
            batch = windows[batch_start: batch_start + cfg.batch_size]
            clips = np.stack([
                _decode_clip(video_path, center, clip_frames_per_window, cfg.resize)
                for center, _ in batch
            ])
            embeddings = _embed_batch(backbone, clips, device)
            for (center, labels), embed in zip(batch, embeddings, strict=True):
                all_rows.append({
                    "match_id": match_id,
                    "clip_id": f"{match_id}@{center}",
                    "video_path": str(video_path),
                    "frame_idx": int(center),
                    "embedding": embed.tolist(),
                    "labels": labels.tolist(),
                })

        if m_idx % cfg.flush_every_n_matches == 0:
            df = pd.DataFrame(all_rows)
            df.to_parquet(cfg.out_parquet, index=False)
            logger.info("Flushed %d rows → %s", len(df), cfg.out_parquet)

    df = pd.DataFrame(all_rows)
    df.to_parquet(cfg.out_parquet, index=False)
    logger.info("Done: %d total rows → %s", len(df), cfg.out_parquet)
    return cfg.out_parquet


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=["train", "valid", "test"])
    parser.add_argument(
        "--split-dir-base",
        type=Path, default=Path("data/raw/soccernet/spotting-ball-2025"),
        help="Parent containing {train,valid,test}/ extracted dirs",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output parquet (default: data/processed/embeddings_{split}.parquet)",
    )
    parser.add_argument("--backbone", default="facebook/vjepa2-vitl-fpc64-256")
    parser.add_argument("--resolution", choices=["224p", "720p"], default="224p",
                        help="Which mp4 to decode; 224p ≈ 10x faster, fits MPS easily")
    parser.add_argument("--clip-seconds", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--negatives-per-minute", type=int, default=6)
    args = parser.parse_args(argv)

    configure_logging()
    ensure_dirs()

    split_dir = args.split_dir_base / args.split
    out = args.out or (PROCESSED_DIR / f"embeddings_{args.split}.parquet")
    cfg = CacheConfig(
        split_dir=split_dir,
        out_parquet=out,
        backbone=args.backbone,
        resolution=args.resolution,
        clip_seconds=args.clip_seconds,
        batch_size=args.batch_size,
        negatives_per_minute=args.negatives_per_minute,
    )
    cache_embeddings(cfg)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
