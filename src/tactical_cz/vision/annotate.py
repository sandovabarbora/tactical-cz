"""Debug-overlay video writer: bbox + tracker_id + team colour + minimap inset.

Reads the per-frame tracking parquet produced by VisionPipeline and the
source video, renders an annotated mp4 next to it. Use to visually QA
the pipeline before any quantitative downstream work.

Annotations per frame:
    * Coloured ellipse under each player (team_id → red / blue, GK
      detected via class_id → green, unfit → grey)
    * Small "T#42" label above each bbox
    * 2D minimap inset (bottom-right corner): pitch outline + dots for
      every projected player from this frame
    * Frame counter top-left
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from tactical_cz.config import OUTPUTS_DIR, configure_logging, ensure_dirs
from tactical_cz.vision.detector import PLAYER_CLASS_GOALKEEPER
from tactical_cz.vision.pitch import PITCH_LENGTH_M, PITCH_WIDTH_M

logger = logging.getLogger(__name__)

# Team colours (BGR for OpenCV). Generic — won't be exactly the kit
# but readable on broadcast video. Sparta-red reserved for accents.
TEAM_COLOURS_BGR: dict[int, tuple[int, int, int]] = {
    0: (200, 80, 60),     # blue-ish
    1: (60, 60, 200),     # red-ish
    -1: (140, 140, 140),  # grey for unassigned (before TeamClassifier fit)
}
GK_COLOUR_BGR: tuple[int, int, int] = (60, 200, 90)  # green for goalkeepers
TEXT_COLOUR_BGR: tuple[int, int, int] = (255, 255, 255)

# Minimap inset geometry
MINIMAP_W: int = 320
MINIMAP_H: int = int(MINIMAP_W * (PITCH_WIDTH_M / PITCH_LENGTH_M))
MINIMAP_BG_BGR: tuple[int, int, int] = (30, 110, 50)  # turf green
MINIMAP_LINE_BGR: tuple[int, int, int] = (220, 220, 220)
MINIMAP_PADDING: int = 16


def _draw_minimap_base() -> np.ndarray:
    """Return a fresh minimap canvas with pitch outline."""
    mm = np.full((MINIMAP_H, MINIMAP_W, 3), MINIMAP_BG_BGR, dtype=np.uint8)
    # Pitch outline
    cv2.rectangle(mm, (1, 1), (MINIMAP_W - 2, MINIMAP_H - 2),
                  MINIMAP_LINE_BGR, 1)
    # Halfway line
    cx = MINIMAP_W // 2
    cv2.line(mm, (cx, 1), (cx, MINIMAP_H - 2), MINIMAP_LINE_BGR, 1)
    # Centre circle
    cy = MINIMAP_H // 2
    radius_m_to_px = MINIMAP_W / PITCH_LENGTH_M
    cv2.circle(mm, (cx, cy), int(9.15 * radius_m_to_px),
               MINIMAP_LINE_BGR, 1)
    return mm


def _pitch_to_minimap(pitch_xy: np.ndarray) -> np.ndarray:
    """Project (N, 2) pitch metres → (N, 2) minimap pixel coords."""
    if len(pitch_xy) == 0:
        return np.empty((0, 2), dtype=int)
    px = (pitch_xy[:, 0] / PITCH_LENGTH_M) * MINIMAP_W
    py = (pitch_xy[:, 1] / PITCH_WIDTH_M) * MINIMAP_H
    return np.column_stack([px, py]).astype(int)


def _draw_ellipse(frame: np.ndarray, bbox: np.ndarray,
                  colour: tuple[int, int, int]) -> None:
    """Draw a flat ellipse under a player's bbox foot (FIFA-broadcast style)."""
    x1, y1, x2, y2 = bbox.astype(int)
    cx = (x1 + x2) // 2
    width = (x2 - x1)
    cv2.ellipse(
        frame, (cx, y2), (width // 2, max(8, width // 6)),
        angle=0.0, startAngle=-45, endAngle=235,
        color=colour, thickness=2, lineType=cv2.LINE_AA,
    )


def _team_colour(class_id: int, team_id: int) -> tuple[int, int, int]:
    if class_id == PLAYER_CLASS_GOALKEEPER:
        return GK_COLOUR_BGR
    return TEAM_COLOURS_BGR.get(int(team_id), TEAM_COLOURS_BGR[-1])


def render_annotated_video(
    source_video: Path | str,
    tracking_parquet: Path | str,
    output_video: Path | str | None = None,
    max_frames: int | None = None,
) -> Path:
    """Render annotated mp4 from a video + the pipeline's tracking parquet.

    Returns the output path.
    """
    source_video = Path(source_video)
    tracking_parquet = Path(tracking_parquet)
    if output_video is None:
        output_video = OUTPUTS_DIR / f"{source_video.stem}_annotated.mp4"
    output_video = Path(output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(tracking_parquet)
    if df.empty:
        raise ValueError(f"Empty tracking parquet: {tracking_parquet}")

    cap = cv2.VideoCapture(str(source_video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

    by_frame = {int(idx): g for idx, g in df.groupby("frame_idx")}
    frame_idx = 0
    written = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames is not None and written >= max_frames:
            break

        # Annotate detections present this frame (we only have stride-N
        # frames in the parquet; on stride-skipped frames just pass-through)
        frame_rows = by_frame.get(frame_idx)
        if frame_rows is not None and len(frame_rows) > 0:
            for _, row in frame_rows.iterrows():
                bbox = np.array([row.x1, row.y1, row.x2, row.y2])
                colour = _team_colour(int(row.class_id), int(row.team_id))
                _draw_ellipse(frame, bbox, colour)
                cv2.putText(
                    frame, f"T{int(row.tracker_id)}",
                    (int(row.x1), max(15, int(row.y1) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOUR_BGR, 1, cv2.LINE_AA,
                )

            # Minimap inset (bottom-right)
            pitch_xy = frame_rows[["pitch_x_m", "pitch_y_m"]].dropna().values
            mm = _draw_minimap_base()
            if len(pitch_xy) > 0:
                for px, py in _pitch_to_minimap(pitch_xy):
                    if 0 <= px < MINIMAP_W and 0 <= py < MINIMAP_H:
                        cv2.circle(mm, (px, py), 4, (250, 250, 250), -1, cv2.LINE_AA)
            # Blit minimap into frame's bottom-right corner
            y0 = height - MINIMAP_H - MINIMAP_PADDING
            x0 = width - MINIMAP_W - MINIMAP_PADDING
            if y0 > 0 and x0 > 0:
                frame[y0: y0 + MINIMAP_H, x0: x0 + MINIMAP_W] = mm

        # Frame counter top-left
        cv2.putText(
            frame, f"frame {frame_idx}",
            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (240, 240, 240), 2, cv2.LINE_AA,
        )

        writer.write(frame)
        written += 1
        frame_idx += 1

    cap.release()
    writer.release()
    logger.info("Wrote %d annotated frames → %s", written, output_video)
    return output_video


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", "-s", type=str, required=True,
                        help="Path to source video (mp4)")
    parser.add_argument("--tracking", "-t", type=str, required=True,
                        help="Path to vision_tracking.parquet from VisionPipeline")
    parser.add_argument("--out", "-o", type=str, default=None,
                        help="Output mp4 path (default: outputs/<stem>_annotated.mp4)")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args(argv)
    configure_logging()
    ensure_dirs()
    render_annotated_video(args.source, args.tracking, args.out, args.max_frames)
    return 0


if __name__ == "__main__":
    sys.exit(main())
