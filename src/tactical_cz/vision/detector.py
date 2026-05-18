"""Detector wrappers around Ultralytics YOLOv8 + Roboflow sports weights.

Three detectors:
    * PlayerDetector — 4-class (player, goalkeeper, referee, ball)
    * BallDetector   — ball-only, higher recall on the tiny fast object
    * PitchDetector  — 32 pitch keypoints (corners, centre, penalty spots)

All three return ``supervision.Detections`` objects so downstream
tracking + annotation can use the same library. First call to each
downloads the .pt weights into ``models/`` and caches.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import supervision as sv

from tactical_cz.config import (
    DEFAULT_BALL_MODEL,
    DEFAULT_PITCH_MODEL,
    DEFAULT_PLAYER_MODEL,
    MODELS_DIR,
    ROBOFLOW_SPORTS_MODELS,
    get_device,
    model_path,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Class IDs for the Roboflow football detector
# ---------------------------------------------------------------------------
# Indexed by Roboflow's training set order. Verified against the
# weights' metadata on first load and logged once for sanity.
PLAYER_CLASS_BALL: int = 0
PLAYER_CLASS_GOALKEEPER: int = 1
PLAYER_CLASS_PLAYER: int = 2
PLAYER_CLASS_REFEREE: int = 3


@dataclass(frozen=True)
class DetectionConfig:
    """Per-detector inference knobs. Defaults match Roboflow's recipe."""

    conf_threshold: float = 0.3       # min confidence
    iou_threshold: float = 0.5         # NMS overlap
    img_size: int = 1280               # YOLO input resolution
    max_det: int = 300                 # cap detections per frame


# ---------------------------------------------------------------------------
# Weight download
# ---------------------------------------------------------------------------


def _ensure_weights(model_key: str) -> Path:
    """Download model weights if not cached. Returns local path.

    Roboflow publishes weights via Google Drive (see their setup.sh).
    We shell out to ``gdown`` because it handles the Drive interstitial
    confirmation page that plain urllib doesn't.
    """
    target = model_path(model_key)
    if target.exists():
        return target
    if model_key not in ROBOFLOW_SPORTS_MODELS:
        raise KeyError(
            f"Unknown model key: {model_key!r}. "
            f"Available: {sorted(ROBOFLOW_SPORTS_MODELS)}"
        )
    gdrive_id = ROBOFLOW_SPORTS_MODELS[model_key]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading model %s (gdrive id=%s) → %s", model_key, gdrive_id, target)
    # `uv run gdown ...` uses the project's pinned gdown
    subprocess.run(
        ["uv", "run", "gdown", "-O", str(target),
         f"https://drive.google.com/uc?id={gdrive_id}"],
        check=True,
        cwd=str(target.parent.parent.parent),  # project root for `uv run`
    )
    size_mb = target.stat().st_size / (1024 * 1024)
    logger.info("Cached %s (%.1f MB)", target.name, size_mb)
    return target


# ---------------------------------------------------------------------------
# Base detector
# ---------------------------------------------------------------------------


class _YoloDetector:
    """Internal base: load YOLO model once, run per-frame inference."""

    def __init__(self, model_key: str, config: DetectionConfig | None = None) -> None:
        from ultralytics import YOLO  # heavy import; defer

        self.model_key = model_key
        self.config = config or DetectionConfig()
        self.device = get_device()
        self._weights = _ensure_weights(model_key)
        self.model = YOLO(str(self._weights))
        # Touching .names confirms weights loaded
        logger.info(
            "Loaded %s on %s — classes: %s",
            self._weights.name, self.device, self.model.names,
        )

    def __call__(self, frame: np.ndarray) -> sv.Detections:
        """Run inference on one BGR frame (numpy H×W×3). Returns Detections."""
        result = self.model(
            frame,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.img_size,
            max_det=self.config.max_det,
            device=self.device,
            verbose=False,
        )[0]
        return sv.Detections.from_ultralytics(result)


# ---------------------------------------------------------------------------
# Public detectors
# ---------------------------------------------------------------------------


class PlayerDetector(_YoloDetector):
    """4-class football detector: ball, goalkeeper, player, referee."""

    def __init__(self, config: DetectionConfig | None = None) -> None:
        super().__init__(DEFAULT_PLAYER_MODEL, config)

    def players_only(self, detections: sv.Detections) -> sv.Detections:
        """Filter to outfield players + goalkeepers (drops ball + referees)."""
        mask = np.isin(
            detections.class_id, [PLAYER_CLASS_PLAYER, PLAYER_CLASS_GOALKEEPER]
        )
        return detections[mask]

    def referees(self, detections: sv.Detections) -> sv.Detections:
        mask = detections.class_id == PLAYER_CLASS_REFEREE
        return detections[mask]


class BallDetector(_YoloDetector):
    """Ball-only detector. Higher recall than the multi-class for the tiny object."""

    def __init__(self, config: DetectionConfig | None = None) -> None:
        # Ball is small + fast; default to lower confidence to catch motion-blurred frames
        if config is None:
            config = DetectionConfig(conf_threshold=0.2)
        super().__init__(DEFAULT_BALL_MODEL, config)


class PitchDetector(_YoloDetector):
    """32-keypoint pitch detector for camera calibration / homography.

    The underlying YOLOv8-pose model emits one detection per frame (the
    pitch as a whole) with 32 keypoints inside it (corners, centre marks,
    penalty spots, goalposts). We return a ``supervision.KeyPoints`` so
    downstream code can use it with ``compute_homography(keypoints=…)``.
    """

    def __init__(self, config: DetectionConfig | None = None) -> None:
        super().__init__(DEFAULT_PITCH_MODEL, config)

    def __call__(self, frame: np.ndarray) -> sv.KeyPoints:  # type: ignore[override]
        result = self.model(
            frame,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.img_size,
            max_det=self.config.max_det,
            device=self.device,
            verbose=False,
        )[0]
        return sv.KeyPoints.from_ultralytics(result)
