"""Smoke tests for the demo modules.

Don't render a real video (cv2 is slow + needs a source mp4). Instead
check imports + the small pure functions that wrangle data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_demo_modules_import() -> None:
    from tactical_cz.demo import annotate, render
    assert hasattr(annotate, "AnnotateConfig")
    assert hasattr(annotate, "annotate_video")
    assert hasattr(render, "RenderConfig")
    assert hasattr(render, "render_demo")


def test_annotate_nearest_center() -> None:
    from tactical_cz.demo.annotate import _nearest_center
    centers = [10, 50, 100, 150]
    assert _nearest_center(0, centers) == 10
    assert _nearest_center(40, centers) == 50
    assert _nearest_center(75, centers) == 50      # equidistant → before
    assert _nearest_center(76, centers) == 100     # tip to after
    assert _nearest_center(200, centers) == 150


def test_render_per_class_table_sorts_desc() -> None:
    from tactical_cz.demo.render import _per_class_table
    df = pd.DataFrame([
        {"frame_idx": 10, "second": 0.4, "event_type": "Pass", "confidence": 0.3},
        {"frame_idx": 20, "second": 0.8, "event_type": "Pass", "confidence": 0.9},
        {"frame_idx": 10, "second": 0.4, "event_type": "Shot", "confidence": 0.95},
        {"frame_idx": 20, "second": 0.8, "event_type": "Shot", "confidence": 0.05},
    ])
    table = _per_class_table(df)
    # Top of table = the class with the single highest max prob
    assert table[0]["cls"] == "Shot"
    assert table[0]["max_p"] == pytest.approx(0.95)
    assert table[0]["at_frame"] == 10
    assert table[1]["cls"] == "Pass"
    assert table[1]["max_p"] == pytest.approx(0.9)


def test_render_chart_returns_inline_svg() -> None:
    from tactical_cz.demo.render import _render_shot_timeline_svg
    # Synthetic events with a peak inside the fake band
    rows = []
    for f in range(0, 800, 25):
        prob = 0.95 if 300 <= f <= 400 else 0.05
        rows.append({"frame_idx": f, "second": f / 25.0, "event_type": "Shot",
                     "confidence": prob, "source_video": "test.mp4"})
    df = pd.DataFrame(rows)
    svg = _render_shot_timeline_svg(df, fake_low=300, fake_high=400)
    assert "<svg" in svg
    assert "</svg>" in svg
    assert "P(Shot)" in svg


def test_annotate_cli_help() -> None:
    from tactical_cz.demo.annotate import main
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_render_cli_help() -> None:
    from tactical_cz.demo.render import main
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
