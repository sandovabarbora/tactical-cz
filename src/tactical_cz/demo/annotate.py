"""Overlay event probabilities onto a broadcast clip.

Reads events_timeline.parquet (the output of events.infer) plus the
source mp4, writes a new mp4 with:
    * top-left: current frame counter + clock
    * bottom strip: horizontal bar per BAS class, filled to its
      probability at the nearest window center; top-1 class
      highlighted in accent
    * top-right: top-1 prediction label + confidence when above
      ``highlight_threshold``

Designed to be visually parsable at a glance: someone scrubbing the
video can see probabilities rise and fall in sync with the action.
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

from tactical_cz.config import OUTPUTS_DIR, configure_logging, ensure_dirs
from tactical_cz.events.model import BAS_CLASSES

logger = logging.getLogger(__name__)

# Visual constants (BGR for OpenCV).
BG = (28, 26, 24)                  # near-black overlay base
TEXT_PRIMARY = (240, 240, 235)     # warm white
TEXT_MUTED = (160, 160, 155)
BAR_BG = (60, 56, 52)
BAR_FILL = (110, 145, 110)         # desaturated grass green
ACCENT = (95, 145, 240)            # warm amber/orange (BGR) for top-1
STRIP_HEIGHT = 88                  # bottom strip total height
STRIP_PAD_TOP = 8
LABEL_W = 110                      # left text column inside the strip


@dataclass
class AnnotateConfig:
    source_video: Path
    events_parquet: Path
    out_video: Path
    highlight_threshold: float = 0.5      # show top-1 label above this
    label_top_k: int = 1                  # how many top labels to print in corner


def _events_per_frame(events: pd.DataFrame, total_frames: int) -> dict[int, dict[str, float]]:
    """Index events_timeline by nearest-window-center → {class_name: prob} dict.

    The infer output has rows keyed by window center frame. For each
    actual video frame, we find the closest center and reuse its
    probabilities. Returns a sparse dict keyed by center frames.
    """
    by_center: dict[int, dict[str, float]] = {}
    for center, group in events.groupby("frame_idx"):
        by_center[int(center)] = dict(zip(group["event_type"], group["confidence"]))
    return by_center


def _nearest_center(frame_idx: int, sorted_centers: list[int]) -> int:
    """Binary search for the closest center frame in O(log N)."""
    import bisect
    pos = bisect.bisect_left(sorted_centers, frame_idx)
    if pos == 0:
        return sorted_centers[0]
    if pos == len(sorted_centers):
        return sorted_centers[-1]
    before, after = sorted_centers[pos - 1], sorted_centers[pos]
    return after if (after - frame_idx) < (frame_idx - before) else before


def _draw_probability_strip(
    frame: np.ndarray,
    probs: dict[str, float],
    threshold: float,
) -> np.ndarray:
    """Composite a probability strip across the bottom of the frame.

    Each BAS class gets one cell with: class name + horizontal bar
    filled to its probability. Top-1 class highlighted in accent.
    """
    h, w = frame.shape[:2]
    strip_y = h - STRIP_HEIGHT
    strip = np.full((STRIP_HEIGHT, w, 3), BG, dtype=np.uint8)

    cell_w = w // len(BAS_CLASSES)
    bar_h = 6
    bar_y = STRIP_HEIGHT - 14
    top1_class = max(probs, key=probs.get) if probs else None

    for i, cls in enumerate(BAS_CLASSES):
        x0 = i * cell_w
        p = probs.get(cls, 0.0)
        is_top = cls == top1_class and p >= threshold

        # Class abbreviated to fit
        label = cls.split()[0][:5] if " " in cls else cls[:6]
        text_color = ACCENT if is_top else TEXT_MUTED
        cv2.putText(
            strip, label,
            (x0 + 6, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, text_color, 1, cv2.LINE_AA,
        )

        # Probability number above the bar
        cv2.putText(
            strip, f"{p:.2f}",
            (x0 + 6, 44),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35,
            TEXT_PRIMARY if is_top else TEXT_MUTED, 1, cv2.LINE_AA,
        )

        # Bar background + fill
        bx0 = x0 + 6
        bx1 = x0 + cell_w - 6
        cv2.rectangle(strip, (bx0, bar_y), (bx1, bar_y + bar_h), BAR_BG, -1)
        fill_px = int((bx1 - bx0) * p)
        if fill_px > 0:
            color = ACCENT if is_top else BAR_FILL
            cv2.rectangle(strip, (bx0, bar_y), (bx0 + fill_px, bar_y + bar_h), color, -1)

    # 50% alpha-blend strip with the frame so action below stays partly visible
    blended = cv2.addWeighted(frame[strip_y:], 0.15, strip, 0.85, 0)
    frame[strip_y:] = blended
    return frame


def annotate_video(cfg: AnnotateConfig) -> Path:
    """Render the annotated mp4. Returns output path."""
    cfg.out_video.parent.mkdir(parents=True, exist_ok=True)

    events = pd.read_parquet(cfg.events_parquet)
    if events.empty:
        raise ValueError(f"Empty events parquet: {cfg.events_parquet}")
    centers_index = _events_per_frame(events, total_frames=0)
    sorted_centers = sorted(centers_index.keys())

    cap = cv2.VideoCapture(str(cfg.source_video))
    if not cap.isOpened():
        raise IOError(f"Could not open {cfg.source_video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(cfg.out_video), fourcc, fps, (w, h))

    logger.info(
        "Annotating %d frames @ %.1f fps → %s (%d event windows from %s)",
        total, fps, cfg.out_video, len(centers_index), cfg.events_parquet,
    )

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Top-left: time + frame
        clock = f"{frame_idx / fps:5.2f}s  f{frame_idx:>4d}"
        cv2.putText(frame, clock, (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_PRIMARY, 2, cv2.LINE_AA)

        # Resolve current-window probabilities (nearest center)
        center = _nearest_center(frame_idx, sorted_centers)
        probs = centers_index[center]

        # Top-right: top-K label if above threshold
        top_sorted = sorted(probs.items(), key=lambda kv: -kv[1])[:cfg.label_top_k]
        y0 = 30
        for cls, p in top_sorted:
            if p < cfg.highlight_threshold:
                break
            text = f"{cls}  {p:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (w - tw - 24, y0 - th - 6),
                          (w - 12, y0 + 8), BG, -1)
            cv2.putText(frame, text, (w - tw - 18, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, ACCENT, 2, cv2.LINE_AA)
            y0 += th + 12

        # Bottom: probability strip
        frame = _draw_probability_strip(frame, probs, cfg.highlight_threshold)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    logger.info("Wrote %d annotated frames → %s", frame_idx, cfg.out_video)
    return cfg.out_video


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Path to broadcast video (mp4)")
    parser.add_argument("--events", type=Path, required=True,
                        help="events_timeline.parquet from events.infer")
    parser.add_argument("--out", type=Path,
                        default=OUTPUTS_DIR / "events_annotated.mp4")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Highlight top-1 label only when prob exceeds this")
    parser.add_argument("--label-top-k", type=int, default=1)
    args = parser.parse_args(argv)
    configure_logging()
    ensure_dirs()
    cfg = AnnotateConfig(
        source_video=args.source,
        events_parquet=args.events,
        out_video=args.out,
        highlight_threshold=args.threshold,
        label_top_k=args.label_top_k,
    )
    annotate_video(cfg)
    print(cfg.out_video)
    return 0


if __name__ == "__main__":
    sys.exit(main())
