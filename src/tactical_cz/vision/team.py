"""KMeans team classification from jersey colours.

For each tracked player crop, take the central torso region (avoids
short colours, hair, shoes), reduce to a dominant HSV value, then
fit KMeans-2 over a sample of players. The two clusters correspond to
the two teams. Goalkeepers are typically one-off outliers — handled
separately by class_id from the detector.

Caveats this is *intentionally* loose:
    * Black/grey kits vs ref colours can conflate; use class_id from
      detector to drop refs before clustering.
    * Indoor low-light or evening kick-off can shift hues — refit per
      match, not per league.
    * Better recipes (Roboflow's `team` notebook) use SigLIP embeddings
      of crops + KMeans on those embeddings; this is the cheap baseline.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import supervision as sv
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


def _crop_torso(frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Take the centre-vertical 30-60% strip of a player bbox.

    Avoids head (hair/skin) and feet (shoes/socks/turf). The 30-60%
    band typically lands on torso for both standing and running players.
    """
    x1, y1, x2, y2 = bbox.astype(int)
    h = y2 - y1
    y_top = y1 + int(h * 0.30)
    y_bot = y1 + int(h * 0.60)
    # Add small horizontal padding to avoid edge halo
    w = x2 - x1
    x_left = x1 + int(w * 0.10)
    x_right = x2 - int(w * 0.10)
    # Clamp
    y_top = max(0, y_top); y_bot = min(frame.shape[0], y_bot)
    x_left = max(0, x_left); x_right = min(frame.shape[1], x_right)
    if y_bot <= y_top or x_right <= x_left:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return frame[y_top:y_bot, x_left:x_right]


def _dominant_hsv(crop: np.ndarray) -> np.ndarray:
    """Return a 3-vector (H, S, V) summarising the crop's dominant colour."""
    if crop.size == 0:
        return np.array([0, 0, 0], dtype=np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # Drop very dark (likely shadow/black background) and very low-sat (greys)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (v > 40) & (s > 30)
    if mask.sum() < 10:
        return hsv.reshape(-1, 3).mean(axis=0).astype(np.float32)
    return hsv[mask].mean(axis=0).astype(np.float32)


class TeamClassifier:
    """Fit once per match, then classify per-detection."""

    def __init__(self, seed: int = 42) -> None:
        self.kmeans: KMeans | None = None
        self.seed = seed

    def fit(self, frames: list[np.ndarray], detections_list: list[sv.Detections]) -> None:
        """Build a kit-colour KMeans from a sample of frames + detections.

        ``frames`` and ``detections_list`` must be parallel (frame N has
        detections_list[N]). Use ~50-100 frames spread across the half.
        Detections must already be filtered to outfield players + GKs.
        """
        if len(frames) != len(detections_list):
            raise ValueError("frames and detections_list must be parallel")
        feats: list[np.ndarray] = []
        for frame, dets in zip(frames, detections_list, strict=True):
            if dets is None or len(dets) == 0:
                continue
            for bbox in dets.xyxy:
                crop = _crop_torso(frame, bbox)
                feats.append(_dominant_hsv(crop))
        if len(feats) < 4:
            raise ValueError(
                f"Need ≥4 player samples to fit, got {len(feats)}. "
                "Check detector confidence + sample frame coverage."
            )
        X = np.stack(feats)
        self.kmeans = KMeans(n_clusters=2, random_state=self.seed, n_init=10)
        self.kmeans.fit(X)
        logger.info(
            "TeamClassifier fit on %d crops; centroids (HSV): %s",
            len(X), self.kmeans.cluster_centers_.round(1).tolist(),
        )

    def predict(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """Return team_id ∈ {0, 1} per detection."""
        if self.kmeans is None:
            raise RuntimeError("Call .fit() before .predict()")
        if len(detections) == 0:
            return np.array([], dtype=int)
        feats = np.stack([
            _dominant_hsv(_crop_torso(frame, bbox))
            for bbox in detections.xyxy
        ])
        return self.kmeans.predict(feats).astype(int)
