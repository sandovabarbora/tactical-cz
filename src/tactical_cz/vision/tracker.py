"""ByteTrack wrapper for per-frame detections → persistent track IDs.

Wraps ``supervision.ByteTrack`` because:
    * BoxMOT is the broader tracker zoo, but supervision ships ByteTrack
      tightly integrated with its ``Detections`` type — fewer adapters.
    * Roboflow sports uses this exact path in their tutorials.
    * Switching to BoxMOT later for re-id (BotSORT, DeepOcSort) is a
      one-file change behind this interface.

For each frame:
    1. Detect (PlayerDetector).
    2. Pass detections through tracker → adds tracker_id to each box.
    3. Persist per-frame DataFrame for downstream analytics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd
import supervision as sv

logger = logging.getLogger(__name__)


@dataclass
class TrackerConfig:
    """ByteTrack hyperparameters. Defaults from supervision docs."""

    track_activation_threshold: float = 0.25
    lost_track_buffer: int = 30                # frames before declaring track lost
    minimum_matching_threshold: float = 0.8
    frame_rate: int = 25                       # broadcast football ≈ 25fps


@dataclass
class TrackedFrame:
    """One frame's tracked detections + the source frame index."""

    frame_idx: int
    detections: sv.Detections


class PlayerTracker:
    """Wraps sv.ByteTrack with frame-index bookkeeping + pandas export."""

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self.tracker = sv.ByteTrack(
            track_activation_threshold=self.config.track_activation_threshold,
            lost_track_buffer=self.config.lost_track_buffer,
            minimum_matching_threshold=self.config.minimum_matching_threshold,
            frame_rate=self.config.frame_rate,
        )
        self._rows: list[dict] = []

    def update(self, frame_idx: int, detections: sv.Detections) -> sv.Detections:
        """Update tracker with new detections; return Detections with tracker_id."""
        tracked = self.tracker.update_with_detections(detections)
        # Persist to internal buffer for later .to_dataframe()
        if len(tracked) > 0 and tracked.tracker_id is not None:
            for i in range(len(tracked)):
                x1, y1, x2, y2 = tracked.xyxy[i]
                self._rows.append({
                    "frame_idx": frame_idx,
                    "tracker_id": int(tracked.tracker_id[i]),
                    "class_id": int(tracked.class_id[i]) if tracked.class_id is not None else -1,
                    "confidence": float(tracked.confidence[i]) if tracked.confidence is not None else float("nan"),
                    "x1": float(x1), "y1": float(y1),
                    "x2": float(x2), "y2": float(y2),
                    "cx": float((x1 + x2) / 2),  # foot point uses cx, y2
                    "cy_foot": float(y2),
                })
        return tracked

    def reset(self) -> None:
        """Clear tracker state. Use between matches."""
        self.tracker.reset()
        self._rows.clear()

    def to_dataframe(self) -> pd.DataFrame:
        """Materialize all tracked frames into a long-format DataFrame."""
        if not self._rows:
            return pd.DataFrame(columns=[
                "frame_idx", "tracker_id", "class_id", "confidence",
                "x1", "y1", "x2", "y2", "cx", "cy_foot",
            ])
        return pd.DataFrame(self._rows)
