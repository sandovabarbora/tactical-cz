"""Overlay event probabilities onto a broadcast clip.

Minimalist version (2026-05-19): drops the 12-bar bottom strip in
favour of a clean top-corner overlay that shows the model's TOP-3
predictions with full class names. The strip was visually noisy and
implied "this is happening" rather than "this is what the model
thinks is happening" — important framing distinction when the model
is wrong (e.g. fires Drive during an actual goal).

Layout per frame:
    top-left  : time + frame counter (small, mono)
    top-right : "MODEL SEES:" header + top-3 class names with confidence
                bars (full names, larger text, accent on top-1)
    bottom-right (corner-pinned) : confidence-level chip
                                   "high / medium / low / uncertain"
"""

from __future__ import annotations

import argparse
import bisect
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

# Palette (BGR for OpenCV).
BG_DARK = (24, 22, 20)              # near-black panel base
BG_DARK_ALPHA = 0.78                # panel opacity
TEXT_PRIMARY = (245, 243, 238)      # warm white
TEXT_MUTED = (170, 168, 162)
ACCENT_TOP1 = (95, 145, 240)        # warm amber for top-1
ACCENT_TOP2 = (130, 155, 105)       # cool green for top-2 (grass)
ACCENT_TOP3 = (140, 130, 175)       # cool plum for top-3
BAR_BG = (60, 56, 52)
CHIP_HIGH = (95, 145, 240)
CHIP_MED = (130, 155, 105)
CHIP_LOW = (110, 110, 110)


@dataclass
class AnnotateConfig:
    source_video: Path
    events_parquet: Path
    out_video: Path
    top_k: int = 3                       # how many classes to show
    confidence_high: float = 0.6         # chip threshold for "high"
    confidence_low: float = 0.3          # chip threshold for "low"


def _events_per_frame(events: pd.DataFrame) -> dict[int, dict[str, float]]:
    """Index events_timeline by center frame → {class_name: prob} dict."""
    out: dict[int, dict[str, float]] = {}
    for center, group in events.groupby("frame_idx"):
        out[int(center)] = dict(zip(group["event_type"], group["confidence"]))
    return out


def _nearest_center(frame_idx: int, sorted_centers: list[int]) -> int:
    pos = bisect.bisect_left(sorted_centers, frame_idx)
    if pos == 0:
        return sorted_centers[0]
    if pos == len(sorted_centers):
        return sorted_centers[-1]
    before, after = sorted_centers[pos - 1], sorted_centers[pos]
    return after if (after - frame_idx) < (frame_idx - before) else before


def _confidence_chip(top1_p: float, cfg: AnnotateConfig) -> tuple[str, tuple[int, int, int]]:
    if top1_p >= cfg.confidence_high:
        return "high", CHIP_HIGH
    if top1_p >= cfg.confidence_low:
        return "medium", CHIP_MED
    return "uncertain", CHIP_LOW


def _draw_top_right_panel(
    frame: np.ndarray,
    probs_sorted: list[tuple[str, float]],
    cfg: AnnotateConfig,
) -> None:
    """Top-right semi-transparent panel: 'MODEL SEES:' + top-K rows."""
    h, w = frame.shape[:2]
    panel_w = 320
    row_h = 30
    header_h = 28
    panel_h = header_h + cfg.top_k * row_h + 14
    pad = 14
    x0 = w - panel_w - pad
    y0 = pad

    # Alpha-blend dark panel
    sub = frame[y0: y0 + panel_h, x0: x0 + panel_w].copy()
    panel = np.full_like(sub, BG_DARK)
    blended = cv2.addWeighted(sub, 1 - BG_DARK_ALPHA, panel, BG_DARK_ALPHA, 0)
    frame[y0: y0 + panel_h, x0: x0 + panel_w] = blended

    # Header
    cv2.putText(
        frame, "MODEL SEES",
        (x0 + 12, y0 + 19),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT_MUTED, 1, cv2.LINE_AA,
    )

    # Rows
    accent_colors = [ACCENT_TOP1, ACCENT_TOP2, ACCENT_TOP3]
    bar_x0 = x0 + 12
    bar_w_max = panel_w - 24
    bar_h = 4
    for i, (cls, p) in enumerate(probs_sorted[: cfg.top_k]):
        row_y = y0 + header_h + i * row_h
        accent = accent_colors[i] if i < len(accent_colors) else TEXT_PRIMARY
        # Full class name (no abbreviation)
        cv2.putText(
            frame, cls,
            (bar_x0, row_y + 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, accent if i == 0 else TEXT_PRIMARY,
            1 if i > 0 else 2, cv2.LINE_AA,
        )
        # Probability number, right-aligned within panel
        p_str = f"{p:.2f}"
        (text_w, _), _ = cv2.getTextSize(p_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(
            frame, p_str,
            (x0 + panel_w - 12 - text_w, row_y + 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_MUTED, 1, cv2.LINE_AA,
        )
        # Thin probability bar below the row
        bar_y = row_y + 19
        cv2.rectangle(frame, (bar_x0, bar_y), (bar_x0 + bar_w_max, bar_y + bar_h), BAR_BG, -1)
        fill_w = int(bar_w_max * p)
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x0, bar_y), (bar_x0 + fill_w, bar_y + bar_h), accent, -1)


def _draw_clock(frame: np.ndarray, frame_idx: int, fps: float) -> None:
    """Top-left dark chip with frame counter + elapsed seconds."""
    text = f"{frame_idx / fps:5.2f}s  ·  f{frame_idx:>4d}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    pad_x, pad_y = 12, 14
    x0, y0 = 14, 14
    cv2.rectangle(frame, (x0, y0), (x0 + tw + 2 * pad_x, y0 + th + 2 * pad_y),
                  BG_DARK, -1)
    cv2.putText(frame, text, (x0 + pad_x, y0 + pad_y + th - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_PRIMARY, 1, cv2.LINE_AA)


def _draw_confidence_chip(frame: np.ndarray, top1_p: float, cfg: AnnotateConfig) -> None:
    """Bottom-right chip showing overall confidence level."""
    h, w = frame.shape[:2]
    label, color = _confidence_chip(top1_p, cfg)
    text = f"top-1 {label} (P={top1_p:.2f})"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    pad_x, pad_y = 10, 8
    box_w = tw + 2 * pad_x + 16
    box_h = th + 2 * pad_y
    x0 = w - box_w - 16
    y0 = h - box_h - 16
    cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), BG_DARK, -1)
    cv2.circle(frame, (x0 + 12, y0 + box_h // 2), 4, color, -1)
    cv2.putText(frame, text, (x0 + 22, y0 + box_h - pad_y - 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_PRIMARY, 1, cv2.LINE_AA)


def annotate_video(cfg: AnnotateConfig) -> Path:
    """Render the annotated mp4. Returns output path."""
    cfg.out_video.parent.mkdir(parents=True, exist_ok=True)

    events = pd.read_parquet(cfg.events_parquet)
    if events.empty:
        raise ValueError(f"Empty events parquet: {cfg.events_parquet}")
    centers_index = _events_per_frame(events)
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
        "Annotating %d frames @ %.1f fps → %s (%d event windows; top-%d display)",
        total, fps, cfg.out_video, len(centers_index), cfg.top_k,
    )

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resolve nearest-window probabilities
        center = _nearest_center(frame_idx, sorted_centers)
        probs = centers_index[center]
        probs_sorted = sorted(probs.items(), key=lambda kv: -kv[1])

        # Compose overlays
        _draw_clock(frame, frame_idx, fps)
        _draw_top_right_panel(frame, probs_sorted, cfg)
        if probs_sorted:
            _draw_confidence_chip(frame, probs_sorted[0][1], cfg)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    logger.info("Wrote %d annotated frames → %s", frame_idx, cfg.out_video)
    return cfg.out_video


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--out", type=Path,
                        default=OUTPUTS_DIR / "events_annotated.mp4")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args(argv)
    configure_logging()
    ensure_dirs()
    cfg = AnnotateConfig(
        source_video=args.source,
        events_parquet=args.events,
        out_video=args.out,
        top_k=args.top_k,
    )
    annotate_video(cfg)
    print(cfg.out_video)
    return 0


if __name__ == "__main__":
    sys.exit(main())
