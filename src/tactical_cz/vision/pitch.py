"""Pitch keypoint → 2D minimap homography.

The PitchDetector returns up to 32 known landmarks (penalty area
corners, centre spot, etc.). We map those image-space points to a
fixed real-world pitch coordinate system (105×68 m, IFAB standard)
via OpenCV homography. Any player bounding box on the same frame
can then be projected to the minimap by transforming its foot point
through the same matrix.

Coordinate convention (IFAB-aligned):
    * origin = bottom-left corner of attacking-left team
    * x-axis right (0 → 105 m, attacking direction)
    * y-axis up (0 → 68 m)

Reference for landmark IDs and their real-world coordinates follows
the Roboflow `sports` pitch detection model labelling.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import supervision as sv

logger = logging.getLogger(__name__)

# Pitch dimensions in metres, matching the Roboflow `sports`
# SoccerPitchConfiguration (120 m × 70 m — slightly larger than the IFAB
# minimum, but it's what their model was trained against). Keep these
# constants synchronised with that config so the homography lands on
# the same coordinate system Roboflow's helpers expect.
PITCH_LENGTH_M: float = 120.0
PITCH_WIDTH_M: float = 70.0

# Geometry constants in metres (cm / 100 from the upstream config)
_PENALTY_BOX_WIDTH: float = 41.0
_PENALTY_BOX_LENGTH: float = 20.15
_GOAL_BOX_WIDTH: float = 18.32
_GOAL_BOX_LENGTH: float = 5.5
_CENTRE_CIRCLE_RADIUS: float = 9.15
_PENALTY_SPOT_DISTANCE: float = 11.0

_W: float = PITCH_WIDTH_M
_L: float = PITCH_LENGTH_M

# Real-world coordinates for the 32 pitch landmarks, in the EXACT order
# emitted by Roboflow's pitch keypoint model (so class_id == array index
# after we drop unconfident points). Origin = one corner, x = length axis,
# y = width axis. Indexes shown 1-based in the upstream config; we use
# 0-based here. Source:
# https://github.com/roboflow/sports/blob/main/sports/configs/soccer.py
PITCH_LANDMARKS_M: np.ndarray = np.array(
    [
        (0.0, 0.0),                                              # 0  corner L-bottom
        (0.0, (_W - _PENALTY_BOX_WIDTH) / 2),                    # 1  L penalty box outer bottom
        (0.0, (_W - _GOAL_BOX_WIDTH) / 2),                       # 2  L goal box outer bottom
        (0.0, (_W + _GOAL_BOX_WIDTH) / 2),                       # 3  L goal box outer top
        (0.0, (_W + _PENALTY_BOX_WIDTH) / 2),                    # 4  L penalty box outer top
        (0.0, _W),                                               # 5  corner L-top
        (_GOAL_BOX_LENGTH, (_W - _GOAL_BOX_WIDTH) / 2),          # 6  L goal box inner bottom
        (_GOAL_BOX_LENGTH, (_W + _GOAL_BOX_WIDTH) / 2),          # 7  L goal box inner top
        (_PENALTY_SPOT_DISTANCE, _W / 2),                        # 8  L penalty spot
        (_PENALTY_BOX_LENGTH, (_W - _PENALTY_BOX_WIDTH) / 2),    # 9  L penalty box inner bottom
        (_PENALTY_BOX_LENGTH, (_W - _GOAL_BOX_WIDTH) / 2),       # 10 L penalty area bottom mark
        (_PENALTY_BOX_LENGTH, (_W + _GOAL_BOX_WIDTH) / 2),       # 11 L penalty area top mark
        (_PENALTY_BOX_LENGTH, (_W + _PENALTY_BOX_WIDTH) / 2),    # 12 L penalty box inner top
        (_L / 2, 0.0),                                           # 13 centre bottom
        (_L / 2, _W / 2 - _CENTRE_CIRCLE_RADIUS),                # 14 centre circle bottom
        (_L / 2, _W / 2 + _CENTRE_CIRCLE_RADIUS),                # 15 centre circle top
        (_L / 2, _W),                                            # 16 centre top
        (_L - _PENALTY_BOX_LENGTH, (_W - _PENALTY_BOX_WIDTH) / 2),   # 17 R penalty box inner bottom
        (_L - _PENALTY_BOX_LENGTH, (_W - _GOAL_BOX_WIDTH) / 2),      # 18 R penalty area bottom mark
        (_L - _PENALTY_BOX_LENGTH, (_W + _GOAL_BOX_WIDTH) / 2),      # 19 R penalty area top mark
        (_L - _PENALTY_BOX_LENGTH, (_W + _PENALTY_BOX_WIDTH) / 2),   # 20 R penalty box inner top
        (_L - _PENALTY_SPOT_DISTANCE, _W / 2),                       # 21 R penalty spot
        (_L - _GOAL_BOX_LENGTH, (_W - _GOAL_BOX_WIDTH) / 2),         # 22 R goal box inner bottom
        (_L - _GOAL_BOX_LENGTH, (_W + _GOAL_BOX_WIDTH) / 2),         # 23 R goal box inner top
        (_L, 0.0),                                                   # 24 corner R-bottom
        (_L, (_W - _PENALTY_BOX_WIDTH) / 2),                         # 25 R penalty box outer bottom
        (_L, (_W - _GOAL_BOX_WIDTH) / 2),                            # 26 R goal box outer bottom
        (_L, (_W + _GOAL_BOX_WIDTH) / 2),                            # 27 R goal box outer top
        (_L, (_W + _PENALTY_BOX_WIDTH) / 2),                         # 28 R penalty box outer top
        (_L, _W),                                                    # 29 corner R-top
        (_L / 2 - _CENTRE_CIRCLE_RADIUS, _W / 2),                    # 30 centre circle left
        (_L / 2 + _CENTRE_CIRCLE_RADIUS, _W / 2),                    # 31 centre circle right
    ],
    dtype=np.float32,
)
assert PITCH_LANDMARKS_M.shape == (32, 2), "Pitch landmark table must be 32×2"


def compute_homography(
    keypoints: sv.KeyPoints | None = None,
    image_pts: np.ndarray | None = None,
    world_pts: np.ndarray | None = None,
    min_correspondences: int = 4,
    min_keypoint_confidence: float = 0.5,
    max_reprojection_px: float = 5.0,
    min_inlier_ratio: float = 0.6,
) -> np.ndarray | None:
    """Compute the 3×3 homography from image to pitch-metres.

    Two calling conventions:
        * pass a supervision KeyPoints object from PitchDetector output,
        * or pass raw image_pts + world_pts arrays directly.

    Confidence gates:
        * ``min_correspondences``: minimum keypoints above confidence
          threshold required before attempting a fit.
        * ``min_keypoint_confidence``: per-point confidence floor.
        * ``min_inlier_ratio``: fraction of RANSAC inliers among the
          fitted points; below this we treat the fit as unreliable and
          return None. Catches the case where 6+ keypoints triangulate
          a wonky homography because most were noisy.

    Returns None if any gate fails.
    """
    if keypoints is not None:
        if len(keypoints) == 0:
            return None
        xy = keypoints.xy[0]                       # (K, 2)
        conf = (
            keypoints.confidence[0]
            if keypoints.confidence is not None
            else np.ones(len(xy))
        )
        mask = conf > min_keypoint_confidence
        if mask.sum() < min_correspondences:
            return None
        image_pts = xy[mask].astype(np.float32)
        class_ids = np.arange(len(xy))[mask]
        world_pts = PITCH_LANDMARKS_M[class_ids]
    elif image_pts is None or world_pts is None:
        raise ValueError("Provide either keypoints OR (image_pts + world_pts)")

    if len(image_pts) < min_correspondences:
        return None

    H, ransac_mask = cv2.findHomography(
        srcPoints=image_pts.reshape(-1, 1, 2),
        dstPoints=world_pts.reshape(-1, 1, 2),
        method=cv2.RANSAC,
        ransacReprojThreshold=max_reprojection_px,
    )
    if H is None or ransac_mask is None:
        return None
    # Inlier ratio guard — too few inliers means the fit is unstable
    inlier_ratio = float(ransac_mask.sum()) / len(ransac_mask)
    if inlier_ratio < min_inlier_ratio:
        return None
    return H


def project_to_pitch(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Project image-space points (N, 2) → pitch-metre coordinates (N, 2).

    For player positions, pass the FOOT point of each bbox
    ((x1+x2)/2, y2), not the centre — the foot is where the player
    actually stands on the pitch plane.
    """
    if homography is None or len(points) == 0:
        return np.empty((0, 2), dtype=np.float32)
    pts = points.reshape(-1, 1, 2).astype(np.float32)
    projected = cv2.perspectiveTransform(pts, homography)
    return projected.reshape(-1, 2)
