"""BoxMOT-backed tracker: per-frame detections → persistent track IDs.

Migrated from supervision.ByteTrack to boxmot 18.0 (sv.ByteTrack is now
deprecated upstream). BoxMOT's per-tracker classes live at
``boxmot.trackers.{name}.{name}`` and share a common API:

    tracker.update(dets, img, embs=None) → TrackResults (N, 8) ndarray
    where the row layout is [x1, y1, x2, y2, track_id, conf, cls, det_ind].

We default to BotSort because it is the BoxMOT default for sports and
slots in appearance re-ID (OSNet) cleanly. The re-ID model is OFF by
default to keep imports + tests offline-safe; enable it via
``TrackerConfig(with_reid=True)`` in production VisionPipeline use.

This file returns ``sv.Detections`` from update() so downstream code in
pipeline.py that already speaks sv.Detections does not change shape.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import supervision as sv
from boxmot.trackers.botsort.botsort import BotSort
from boxmot.trackers.bytetrack.bytetrack import ByteTrack

logger = logging.getLogger(__name__)


@dataclass
class TrackerConfig:
    """Tracker hyperparameters. Defaults tuned for 25 fps broadcast football."""

    backend: str = "botsort"            # "botsort" | "bytetrack"
    with_reid: bool = False             # OSNet appearance re-ID (BotSort only)
    reid_weights: str | None = None     # None → boxmot default (osnet_x0_25_msmt17.pt, auto-download)
    reid_device: str = "cpu"            # "cpu" | "mps" | "cuda" — MPS works for inference

    # Common
    frame_rate: int = 25
    track_buffer: int = 30              # frames before declaring track lost

    # ByteTrack
    track_thresh: float = 0.45          # high-confidence detection gate
    match_thresh: float = 0.8           # IoU match threshold

    # BotSort
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    proximity_thresh: float = 0.5
    appearance_thresh: float = 0.25


def _sv_to_boxmot_dets(detections: sv.Detections) -> np.ndarray:
    """sv.Detections → (N, 6) ndarray for BoxMOT [x1, y1, x2, y2, conf, cls]."""
    if len(detections) == 0:
        return np.empty((0, 6), dtype=np.float32)
    conf = detections.confidence if detections.confidence is not None else np.ones(len(detections))
    cls = detections.class_id if detections.class_id is not None else np.zeros(len(detections), dtype=int)
    return np.column_stack([
        detections.xyxy.astype(np.float32),
        conf.astype(np.float32),
        cls.astype(np.float32),
    ])


def _boxmot_to_sv_dets(out: np.ndarray) -> sv.Detections:
    """BoxMOT TrackResults (N, 8) → sv.Detections.

    TrackResults rows: [x1, y1, x2, y2, track_id, conf, cls, det_ind].
    Empty results return an empty sv.Detections in the expected shape.
    """
    if out is None or len(out) == 0:
        return sv.Detections(
            xyxy=np.empty((0, 4), dtype=np.float32),
            class_id=np.empty((0,), dtype=int),
            confidence=np.empty((0,), dtype=np.float32),
            tracker_id=np.empty((0,), dtype=int),
        )
    return sv.Detections(
        xyxy=out[:, :4].astype(np.float32),
        tracker_id=out[:, 4].astype(int),
        confidence=out[:, 5].astype(np.float32),
        class_id=out[:, 6].astype(int),
    )


class PlayerTracker:
    """BoxMOT BotSort / ByteTrack wrapper with frame-index bookkeeping + pandas export."""

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self._rows: list[dict] = []
        self.tracker = self._build_tracker()

    def _build_tracker(self):
        cfg = self.config
        if cfg.backend == "bytetrack":
            return ByteTrack(
                track_thresh=cfg.track_thresh,
                match_thresh=cfg.match_thresh,
                track_buffer=cfg.track_buffer,
                frame_rate=cfg.frame_rate,
            )
        if cfg.backend == "botsort":
            reid_model = None
            if cfg.with_reid:
                from boxmot.reid.core.reid import ReID
                # BoxMOT trackers call ``.get_features(xyxys, img)`` directly on
                # the model object. The outer ``ReID`` instance is a loader
                # wrapper, its ``.model`` (PyTorchBackend) is the thing that
                # actually exposes ``get_features``. The engine path uses the
                # same indirection — see boxmot/engine/inference.py:163.
                reid_model = ReID(
                    path=cfg.reid_weights,        # None → boxmot default OSNet x0_25 MSMT17 (3 MB)
                    device=cfg.reid_device,
                ).model
                logger.info("BotSort re-ID enabled (OSNet, device=%s)", cfg.reid_device)
            return BotSort(
                reid_model=reid_model,
                with_reid=cfg.with_reid,
                track_high_thresh=cfg.track_high_thresh,
                track_low_thresh=cfg.track_low_thresh,
                new_track_thresh=cfg.new_track_thresh,
                track_buffer=cfg.track_buffer,
                match_thresh=cfg.match_thresh,
                proximity_thresh=cfg.proximity_thresh,
                appearance_thresh=cfg.appearance_thresh,
                frame_rate=cfg.frame_rate,
            )
        raise ValueError(f"Unknown tracker backend: {cfg.backend!r}")

    def update(
        self,
        frame_idx: int,
        detections: sv.Detections,
        frame: np.ndarray,
    ) -> sv.Detections:
        """Update tracker with new detections + the source frame image.

        BoxMOT needs the image (for re-ID crops + camera motion compensation).
        ByteTrack ignores it but the interface stays uniform.
        """
        dets_arr = _sv_to_boxmot_dets(detections)
        out = self.tracker.update(dets_arr, frame)
        tracked = _boxmot_to_sv_dets(out)

        for i in range(len(tracked)):
            x1, y1, x2, y2 = tracked.xyxy[i]
            self._rows.append({
                "frame_idx": frame_idx,
                "tracker_id": int(tracked.tracker_id[i]),
                "class_id": int(tracked.class_id[i]),
                "confidence": float(tracked.confidence[i]),
                "x1": float(x1), "y1": float(y1),
                "x2": float(x2), "y2": float(y2),
                "cx": float((x1 + x2) / 2),
                "cy_foot": float(y2),
            })
        return tracked

    def reset(self) -> None:
        """Clear tracker state + persistence buffer. Use between matches."""
        self.tracker = self._build_tracker()
        self._rows.clear()

    def to_dataframe(self) -> pd.DataFrame:
        """Materialize all tracked frames into a long-format DataFrame."""
        if not self._rows:
            return pd.DataFrame(columns=[
                "frame_idx", "tracker_id", "class_id", "confidence",
                "x1", "y1", "x2", "y2", "cx", "cy_foot",
            ])
        return pd.DataFrame(self._rows)
