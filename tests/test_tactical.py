"""Tests for tactical feature derivation + SVG visualization.

Synthetic vision + events parquets; no real video, no MPS. Verifies the
join + per-zone counting + SVG generation work end-to-end.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tactical_cz.tactical.features import (
    PITCH_LENGTH_M,
    TacticalFeatures,
    compute_all,
    field_tilt_per_frame,
    shot_moments,
    zone_distribution,
)
from tactical_cz.tactical.visualize import (
    render_minimap_at_frame,
    render_tilt_timeline,
    render_zone_chart,
)


def _synthetic_vision(n_frames: int = 50, n_players: int = 8) -> pd.DataFrame:
    """Mock Phase 1 tracking parquet — 8 players over n_frames frames."""
    rng = np.random.default_rng(42)
    rows = []
    for f in range(n_frames):
        for p in range(n_players):
            # First half of clip: defensive third; second half: attacking third
            base_x = 20.0 if f < n_frames // 2 else 100.0
            rows.append({
                "frame_idx": f * 2,                  # stride 2
                "tracker_id": p,
                "class_id": 2 if p > 0 else 1,       # 1 GK + 7 players
                "team_id": p % 2,
                "confidence": 0.9,
                "x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 10.0,
                "cx": 5.0, "cy_foot": 10.0,
                "pitch_x_m": base_x + rng.uniform(-10, 10),
                "pitch_y_m": rng.uniform(10, 60),
            })
    return pd.DataFrame(rows)


def _synthetic_events(n_frames: int = 50) -> pd.DataFrame:
    """Mock Phase 2 events_timeline — one row per (frame, class)."""
    classes = ["Shot", "Drive", "Pass", "Cross"]
    rng = np.random.default_rng(0)
    rows = []
    for f in range(0, n_frames * 2, 2):              # match vision stride
        for cls in classes:
            rows.append({
                "frame_idx": f,
                "second": f / 25.0,
                "event_type": cls,
                "confidence": float(rng.uniform(0.05, 0.95)),
                "source_video": "synthetic.mp4",
            })
    return pd.DataFrame(rows)


def test_field_tilt_per_frame_returns_one_row_per_frame() -> None:
    vis = _synthetic_vision(n_frames=10, n_players=4)
    tilt = field_tilt_per_frame(vis)
    assert len(tilt) == 10
    assert {"frame_idx", "tilt_m", "n_players", "second"}.issubset(tilt.columns)
    # Player count includes 1 GK (class 1) + 3 outfield (class 2)
    assert (tilt["n_players"] == 4).all()


def test_field_tilt_shifts_with_synthetic_attacking_motion() -> None:
    """First half clip has base_x=20m, second half=100m. Mean tilt
    in second half should be higher."""
    vis = _synthetic_vision(n_frames=40, n_players=6)
    tilt = field_tilt_per_frame(vis)
    first_half = tilt.iloc[: len(tilt) // 2]["tilt_m"].mean()
    second_half = tilt.iloc[len(tilt) // 2 :]["tilt_m"].mean()
    assert second_half > first_half + 30, (
        f"expected attacking shift; got {first_half:.1f} → {second_half:.1f}"
    )


def test_zone_distribution_three_zones() -> None:
    vis = _synthetic_vision(n_frames=40, n_players=6)
    evt = _synthetic_events(n_frames=40)
    zd = zone_distribution(vis, evt, target_classes=("Shot", "Drive"))
    assert set(zd.keys()) <= {"Shot", "Drive"}
    for cls, counts in zd.items():
        assert set(counts.keys()) == {"defensive", "midfield", "attacking"}
        assert sum(counts.values()) > 0


def test_shot_moments_returns_at_most_top_k() -> None:
    vis = _synthetic_vision(n_frames=40, n_players=6)
    evt = _synthetic_events(n_frames=40)
    moments = shot_moments(vis, evt, top_k=3)
    assert len(moments) <= 3
    for m in moments:
        assert m["action_zone"] in {"defensive", "midfield", "attacking"}
        assert 0 <= m["p_shot"] <= 1
        assert m["n_players_visible"] >= 0


def test_compute_all_returns_tactical_features() -> None:
    vis = _synthetic_vision(n_frames=40, n_players=6)
    evt = _synthetic_events(n_frames=40)
    tact = compute_all(vis, evt)
    assert isinstance(tact, TacticalFeatures)
    assert tact.summary["n_shot_moments"] > 0
    assert tact.summary["n_players_per_frame_mean"] > 0
    assert 0 <= tact.summary["attacking_zone_share"] <= 1


def test_compute_all_handles_empty_vision_gracefully() -> None:
    vis = pd.DataFrame(columns=[
        "frame_idx", "tracker_id", "class_id", "team_id", "confidence",
        "x1", "y1", "x2", "y2", "cx", "cy_foot", "pitch_x_m", "pitch_y_m",
    ])
    evt = _synthetic_events(n_frames=10)
    tact = compute_all(vis, evt)
    assert tact.field_tilt_timeline == []
    assert tact.shot_moments == []


def test_render_minimap_returns_svg() -> None:
    vis = _synthetic_vision(n_frames=10, n_players=6)
    svg = render_minimap_at_frame(vis, frame_idx=4)
    assert "<svg" in svg and "</svg>" in svg
    # Each player should be one circle
    assert svg.count("<circle") >= 6


def test_render_zone_chart_returns_svg() -> None:
    vis = _synthetic_vision(n_frames=40, n_players=6)
    evt = _synthetic_events(n_frames=40)
    zd = zone_distribution(vis, evt, target_classes=("Shot", "Drive"))
    svg = render_zone_chart(zd)
    assert "<svg" in svg and "</svg>" in svg


def test_render_tilt_timeline_handles_empty() -> None:
    svg = render_tilt_timeline([])
    assert "no tilt timeline" in svg
