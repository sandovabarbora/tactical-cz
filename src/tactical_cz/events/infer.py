"""Run a fine-tuned BAS model over a tactical-cz video / vision parquet.

Closes the loop from Phase 1 vision (which gave us per-frame player +
ball positions) to Phase 2 events (per-frame event timeline). Output
schema mirrors Phase 1 to make joins downstream trivial.

Usage:
    uv run python -m tactical_cz.events.infer \\
        --checkpoint checkpoints/bas-04-0.211.ckpt \\
        --source data/raw/sparta_latest_30s.mp4 \\
        --out data/processed/events_timeline.parquet

Sliding-window strategy:
    Each window = ``clip_seconds`` × ``fps`` consecutive frames.
    Windows step by ``stride_seconds`` (default 1.0s, overlapping so
    brief events aren't missed at window boundaries).
    Per window we run the model once and emit one row per BAS class
    above ``threshold`` confidence.

Output parquet columns:
    frame_idx        center frame of the window
    second           seconds from clip start (frame_idx / fps)
    event_type       string name from BAS_CLASSES
    confidence       sigmoid of logit (multi-label decoder)
    source_video     basename of the input video
"""

from __future__ import annotations

import argparse
import logging
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
from tactical_cz.events.model import (
    BAS_CLASSES,
    BAS_NUM_CLASSES,
    BASModel,
    BASModelConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class InferConfig:
    """Inference knobs."""

    source: Path
    checkpoint: Path | None = None
    out: Path = PROCESSED_DIR / "events_timeline.parquet"
    backbone: str = "facebook/vjepa2-vitl-fpc64-256"
    clip_seconds: float = 4.0
    stride_seconds: float = 1.0
    resize: int = 256
    threshold: float = 0.3
    frames_per_clip: int = 8               # subsample inside the window; matches cache_embeddings default


def _read_clip_windows(
    source: Path,
    window_frames: int,
    stride_frames: int,
    resize: int,
    sample_frames: int = 8,
):
    """Generator of (center_frame_idx, tensor[sample_frames, 3, H, W]) windows.

    Reads the full window but emits only ``sample_frames`` evenly-spaced
    indices. V-JEPA2 attention scales O(T²), so dense 100-frame clips
    (4s @ 25fps) at 256×256 OOM on MPS. Sparse temporal sampling is the
    standard pattern (see V-JEPA2 paper sec. 4).
    """
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise IOError(f"Could not open {source}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Indices within the rolling window that we'll keep
    if sample_frames >= window_frames:
        keep = set(range(window_frames))
    else:
        keep = set(np.linspace(0, window_frames - 1, sample_frames).astype(int).tolist())

    frame_buf: list[np.ndarray] = []     # ring buffer of resized RGB frames, length window_frames
    frame_idx = 0
    next_emit = window_frames - 1
    while frame_idx < total:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (resize, resize))
        frame_buf.append(frame)
        if len(frame_buf) > window_frames:
            frame_buf.pop(0)
        if len(frame_buf) == window_frames and frame_idx >= next_emit:
            sampled = np.stack([frame_buf[i] for i in sorted(keep)], axis=0)
            arr = sampled.astype(np.float32) / 255.0
            arr = (arr - 0.5) / 0.5
            t = torch.from_numpy(arr).permute(0, 3, 1, 2)
            center = frame_idx - window_frames // 2
            yield center, t
            next_emit += stride_frames
        frame_idx += 1
    cap.release()


def _load_model(cfg: InferConfig, device: str) -> BASModel:
    """Build a BASModel and load weights from Lightning checkpoint if given."""
    model = BASModel(BASModelConfig(
        backbone=cfg.backbone,
        num_classes=BAS_NUM_CLASSES,
    ))
    if cfg.checkpoint:
        ckpt = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
        # Two formats supported:
        #   (a) full BASModel checkpoint — keys like "backbone.*" / "head.*",
        #       optionally "model." prefixed if saved by Lightning
        #   (b) head-only checkpoint from events.train (post-2026-05-18
        #       embedding-cache refactor) — keys are just "net.0.weight" etc.
        sd = {k.removeprefix("model."): v for k, v in sd.items()}
        is_head_only = not any(k.startswith("backbone.") for k in sd)
        if is_head_only:
            # BASHead (train.py) wraps its layers in self.net = Sequential(...)
            # → keys look like "net.0.weight". BASModel.head is the Sequential
            # directly (no inner attribute) → keys are "head.0.weight". We
            # strip the BASHead "net." prefix and add the BASModel "head." one.
            sd = {f"head.{k.removeprefix('net.')}": v for k, v in sd.items()}
            logger.info("Detected head-only checkpoint; backbone stays at pretrained weights.")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            logger.warning("Missing keys when loading checkpoint: %s", missing[:5])
        if unexpected:
            logger.warning("Unexpected keys when loading checkpoint: %s", unexpected[:5])
        logger.info("Loaded checkpoint %s", cfg.checkpoint)
    else:
        logger.warning(
            "No checkpoint provided. Inference uses random-init head; "
            "results are meaningless. Use --checkpoint <path>."
        )
    return model.to(device).train(False)  # eval mode without triggering hook regex


def run_inference(cfg: InferConfig) -> Path:
    """Run BAS inference over a video, write events_timeline.parquet."""
    device = get_device()
    logger.info("Inference device: %s", device)
    model = _load_model(cfg, device)

    cap = cv2.VideoCapture(str(cfg.source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    clip_frames = max(1, int(cfg.clip_seconds * fps))
    stride_frames = max(1, int(cfg.stride_seconds * fps))
    logger.info(
        "Window: %d frames (%.1fs) stride=%d frames (%.1fs) @ %.1f fps",
        clip_frames, cfg.clip_seconds, stride_frames, cfg.stride_seconds, fps,
    )

    rows: list[dict] = []
    for center_idx, clip in _read_clip_windows(
        cfg.source, clip_frames, stride_frames, cfg.resize, cfg.frames_per_clip,
    ):
        with torch.no_grad():
            logits = model(clip.unsqueeze(0).to(device))
        probs = torch.sigmoid(logits)[0].cpu().numpy()
        for class_id, p in enumerate(probs):
            if p >= cfg.threshold:
                rows.append({
                    "frame_idx": int(center_idx),
                    "second": float(center_idx / fps),
                    "event_type": BAS_CLASSES[class_id],
                    "confidence": float(p),
                    "source_video": cfg.source.name,
                })

    df = pd.DataFrame(rows)
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cfg.out, index=False)
    logger.info("Wrote %d events to %s", len(df), cfg.out)
    return cfg.out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Path to broadcast video (mp4)")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Lightning checkpoint from events.train (omit for random-init smoke)")
    parser.add_argument("--out", type=Path,
                        default=PROCESSED_DIR / "events_timeline.parquet")
    parser.add_argument("--backbone", default="facebook/vjepa2-vitl-fpc64-256")
    parser.add_argument("--clip-seconds", type=float, default=4.0)
    parser.add_argument("--stride-seconds", type=float, default=1.0)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Sigmoid confidence floor (multi-label decoder)")
    args = parser.parse_args(argv)

    configure_logging()
    ensure_dirs()
    cfg = InferConfig(**vars(args))
    out = run_inference(cfg)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
