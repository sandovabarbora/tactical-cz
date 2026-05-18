"""Offline tests for TeamClassifier helpers. No model download, no GPU."""

from __future__ import annotations

import numpy as np

from tactical_cz.vision.team import _crop_torso, _dominant_hsv


def test_crop_torso_takes_middle_third() -> None:
    # 100×100 black image with a coloured strip in the centre
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[40:60, :] = (0, 0, 255)  # red horizontal strip
    bbox = np.array([0, 0, 100, 100])
    crop = _crop_torso(frame, bbox)
    # Should grab from y=30 to y=60, so the red strip (40-60) lies fully inside
    assert crop.shape[0] > 0 and crop.shape[1] > 0
    # Crop should be predominantly red
    red_pixels = (crop[..., 2] > 200).sum()
    total = crop.shape[0] * crop.shape[1]
    assert red_pixels / total > 0.5


def test_dominant_hsv_returns_3_vector() -> None:
    crop = np.full((20, 20, 3), 200, dtype=np.uint8)  # bright grey
    hsv = _dominant_hsv(crop)
    assert hsv.shape == (3,)
    assert hsv.dtype == np.float32


def test_dominant_hsv_skips_dark_pixels() -> None:
    # Half dark (V<40), half bright cyan
    crop = np.zeros((20, 20, 3), dtype=np.uint8)
    crop[:, 10:] = (200, 200, 50)  # blue-green BGR
    hsv = _dominant_hsv(crop)
    # The bright pixels dominate; V should be high
    assert hsv[2] > 100, f"Expected high V, got {hsv}"


def test_dominant_hsv_handles_all_dark() -> None:
    # All pixels below the mask threshold — should fall back to mean
    crop = np.zeros((10, 10, 3), dtype=np.uint8)
    hsv = _dominant_hsv(crop)
    assert hsv.shape == (3,)
    # Mean of all-zero HSV is (0, 0, 0)
    np.testing.assert_allclose(hsv, [0, 0, 0], atol=1e-3)
