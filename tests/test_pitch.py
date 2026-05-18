"""Offline tests for pitch homography math. No GPU, no model download."""

from __future__ import annotations

import numpy as np
import pytest

from tactical_cz.vision.pitch import (
    PITCH_LANDMARKS_M,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
    compute_homography,
    project_to_pitch,
)


def test_landmarks_table_shape() -> None:
    assert PITCH_LANDMARKS_M.shape == (32, 2)
    assert PITCH_LANDMARKS_M.dtype == np.float32


def test_landmarks_within_pitch_bounds() -> None:
    # All landmarks must fall inside or on the pitch boundary
    xs = PITCH_LANDMARKS_M[:, 0]
    ys = PITCH_LANDMARKS_M[:, 1]
    assert xs.min() >= 0.0 and xs.max() <= PITCH_LENGTH_M
    assert ys.min() >= 0.0 and ys.max() <= PITCH_WIDTH_M


def test_compute_homography_minimum_correspondences() -> None:
    # 3 points is too few — should return None
    image_pts = np.array([[0, 0], [100, 0], [0, 100]], dtype=np.float32)
    world_pts = np.array([[0, 0], [10, 0], [0, 10]], dtype=np.float32)
    H = compute_homography(image_pts=image_pts, world_pts=world_pts)
    assert H is None


def test_compute_homography_identity_projects_correctly() -> None:
    # 4 corners of a 100×100 image mapping to a 10×10 m pitch square.
    # The homography should be a pure scale; projecting back should recover.
    image_pts = np.array(
        [[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32
    )
    world_pts = np.array(
        [[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32
    )
    H = compute_homography(image_pts=image_pts, world_pts=world_pts)
    assert H is not None
    # Project the centre of the image; should land at (5, 5) m
    centre = np.array([[50, 50]], dtype=np.float32)
    projected = project_to_pitch(centre, H)
    assert projected.shape == (1, 2)
    np.testing.assert_allclose(projected[0], [5.0, 5.0], atol=0.1)


def test_project_empty_points_returns_empty() -> None:
    H = np.eye(3, dtype=np.float32)
    out = project_to_pitch(np.empty((0, 2), dtype=np.float32), H)
    assert out.shape == (0, 2)


def test_project_with_none_homography_returns_empty() -> None:
    pts = np.array([[100, 200]], dtype=np.float32)
    out = project_to_pitch(pts, None)
    assert out.shape == (0, 2)
