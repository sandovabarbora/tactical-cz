"""Tactical features derived from Phase 1 tracking × Phase 2 events.

The join key is ``frame_idx``. Phase 1 tracking has one row per (frame,
player); Phase 2 events have one row per (window, class) where window
is centered on a specific frame. We snap events to the nearest tracked
frame in the vision parquet (typically within stride/2 = 1 frame).

Features are intentionally team-agnostic where possible (since the
auto-detected team_id is just a K-means HSV clustering, not a club-
identity claim). Team-aware features (defensive line, ball-side
density) live in :func:`compute_team_aware`.

Schema produced by :func:`compute_all`:

    {
      "field_tilt": list[(second, mean_pitch_x_m)]      # temporal
      "tilt_summary": {"mean": float, "p10": float, "p90": float}
      "zone_distribution": {
          "Shot":  {"defensive": int, "midfield": int, "attacking": int},
          "Drive": ...
      }
      "shot_moments": [
          {"second": float, "frame": int, "p_shot": float,
           "n_players_visible": int, "action_centroid_x": float,
           "action_zone": "attacking" | "midfield" | "defensive"}
      ]
      "summary": {
          "n_events_total": int,
          "n_shot_moments": int,
          "mean_field_tilt_m": float,
          "attacking_zone_share": float,    # fraction of Shot+Drive events that fired in attacking third
          "n_players_per_frame_mean": float,
      }
    }
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Pitch dimensions matching tactical_cz.vision.pitch (Roboflow soccer
# config = 120 × 70 m). Thirds are 40 m each on the long axis.
PITCH_LENGTH_M: float = 120.0
PITCH_WIDTH_M: float = 70.0
DEFENSIVE_THIRD_X: float = PITCH_LENGTH_M / 3            # 0..40 m
ATTACKING_THIRD_X: float = 2 * PITCH_LENGTH_M / 3        # 80..120 m

# Class IDs from the Roboflow `sports` player detector. Used to drop
# non-player rows when computing field tilt etc.
CLASS_BALL: int = 0
CLASS_GOALKEEPER: int = 1
CLASS_PLAYER: int = 2
CLASS_REFEREE: int = 3
PLAYER_CLASSES: tuple[int, ...] = (CLASS_GOALKEEPER, CLASS_PLAYER)


@dataclass(frozen=True)
class TacticalFeatures:
    """All derived tactical features for a single clip."""

    field_tilt_timeline: list[tuple[float, float]]   # (second, mean_pitch_x_m)
    tilt_summary: dict[str, float]
    zone_distribution: dict[str, dict[str, int]]
    shot_moments: list[dict]
    summary: dict


def _zone(pitch_x_m: float) -> str:
    if pitch_x_m < DEFENSIVE_THIRD_X:
        return "defensive"
    if pitch_x_m < ATTACKING_THIRD_X:
        return "midfield"
    return "attacking"


def _player_rows(vision: pd.DataFrame) -> pd.DataFrame:
    """Filter vision parquet to player + GK rows with valid pitch coords."""
    is_player = vision["class_id"].isin(PLAYER_CLASSES)
    has_pitch = vision["pitch_x_m"].notna() & vision["pitch_y_m"].notna()
    in_bounds = (
        vision["pitch_x_m"].between(0, PITCH_LENGTH_M)
        & vision["pitch_y_m"].between(0, PITCH_WIDTH_M)
    )
    return vision[is_player & has_pitch & in_bounds].copy()


def field_tilt_per_frame(vision: pd.DataFrame, fps: float = 25.0) -> pd.DataFrame:
    """For each frame with player detections, the mean pitch_x_m.

    Field tilt = how high up the pitch the action is happening. Values
    above 60 m mean the action centroid is in the attacking half; below
    60 m it's in the defending half. Direction is whichever side the
    Roboflow homography labels as 'x=0'; for our purposes the relative
    shape (rises/falls) is what matters more than absolute orientation.
    """
    players = _player_rows(vision)
    if players.empty:
        return pd.DataFrame(columns=["frame_idx", "second", "tilt_m", "n_players"])
    grouped = players.groupby("frame_idx").agg(
        tilt_m=("pitch_x_m", "mean"),
        n_players=("tracker_id", "size"),
    ).reset_index()
    grouped["second"] = grouped["frame_idx"] / fps
    return grouped


def zone_distribution(
    vision: pd.DataFrame,
    events: pd.DataFrame,
    target_classes: tuple[str, ...] = ("Shot", "Drive", "Pass", "Cross", "Header"),
    fps: float = 25.0,
) -> dict[str, dict[str, int]]:
    """For each target event class, how its high-confidence windows split
    across the three pitch thirds (by action centroid x).
    """
    tilt = field_tilt_per_frame(vision, fps=fps)
    if tilt.empty:
        return {}

    # Pre-sort tilt frames for fast nearest-frame lookup
    tilt_frames = tilt["frame_idx"].values
    tilt_xm = tilt["tilt_m"].values

    out: dict[str, dict[str, int]] = {}
    for cls in target_classes:
        sub = events[events["event_type"] == cls]
        # Threshold each class at its own 75th percentile of confidence,
        # which collapses to "top-quartile windows for this class" — a
        # class-agnostic way to pick "model thinks this is happening".
        if len(sub) == 0:
            continue
        thr = sub["confidence"].quantile(0.75)
        hot = sub[sub["confidence"] >= thr]
        counts = {"defensive": 0, "midfield": 0, "attacking": 0}
        for _, row in hot.iterrows():
            idx = np.searchsorted(tilt_frames, row["frame_idx"])
            if idx == len(tilt_frames):
                idx = len(tilt_frames) - 1
            if idx > 0 and (idx == len(tilt_frames) or
                            abs(tilt_frames[idx-1] - row["frame_idx"]) <
                            abs(tilt_frames[idx] - row["frame_idx"])):
                idx = idx - 1
            counts[_zone(float(tilt_xm[idx]))] += 1
        out[cls] = counts
    return out


def shot_moments(
    vision: pd.DataFrame,
    events: pd.DataFrame,
    top_k: int = 8,
    fps: float = 25.0,
) -> list[dict]:
    """Top-K highest-confidence Shot windows with their tactical context.

    For each: action centroid, zone, number of players visible, the
    moment in seconds. These feed the minimap snapshot panels on the
    demo page.
    """
    shots = events[events["event_type"] == "Shot"].copy()
    if shots.empty:
        return []
    shots = shots.sort_values("confidence", ascending=False).head(top_k)
    tilt = field_tilt_per_frame(vision, fps=fps)
    if tilt.empty:
        return []

    tilt_frames = tilt["frame_idx"].values
    tilt_xm = tilt["tilt_m"].values
    n_players = tilt["n_players"].values

    out = []
    for _, row in shots.iterrows():
        idx = int(np.argmin(np.abs(tilt_frames - row["frame_idx"])))
        cx = float(tilt_xm[idx])
        out.append({
            "frame": int(row["frame_idx"]),
            "second": float(row["frame_idx"] / fps),
            "p_shot": float(row["confidence"]),
            "n_players_visible": int(n_players[idx]),
            "action_centroid_x": cx,
            "action_zone": _zone(cx),
        })
    return sorted(out, key=lambda x: x["frame"])


def compute_all(
    vision: pd.DataFrame,
    events: pd.DataFrame,
    fps: float = 25.0,
) -> TacticalFeatures:
    """Compute the bundle of tactical features used by the demo page."""
    tilt = field_tilt_per_frame(vision, fps=fps)
    if tilt.empty:
        logger.warning("No player rows with valid pitch coords; tactical layer empty")
        return TacticalFeatures(
            field_tilt_timeline=[],
            tilt_summary={},
            zone_distribution={},
            shot_moments=[],
            summary={"n_events_total": 0},
        )

    # Downsample tilt timeline to ~30 points for plotting
    if len(tilt) > 30:
        idx = np.linspace(0, len(tilt) - 1, 30, dtype=int)
        tilt_sample = tilt.iloc[idx]
    else:
        tilt_sample = tilt
    field_tilt_timeline = [
        (float(r.second), float(r.tilt_m)) for r in tilt_sample.itertuples()
    ]

    tilt_summary = {
        "mean": float(tilt["tilt_m"].mean()),
        "p10": float(tilt["tilt_m"].quantile(0.10)),
        "p90": float(tilt["tilt_m"].quantile(0.90)),
        "p50": float(tilt["tilt_m"].quantile(0.50)),
    }

    zd = zone_distribution(vision, events, fps=fps)
    sm = shot_moments(vision, events, top_k=8, fps=fps)

    # Summary block: keep it small and decision-relevant
    shot_drive = []
    for cls in ("Shot", "Drive"):
        if cls in zd:
            shot_drive.append(zd[cls])
    n_total = sum(sum(d.values()) for d in shot_drive) if shot_drive else 0
    n_attacking = sum(d.get("attacking", 0) for d in shot_drive) if shot_drive else 0
    attacking_share = n_attacking / n_total if n_total > 0 else 0.0

    summary = {
        "n_events_total": int(len(events.event_type.unique())),
        "n_shot_moments": int(len(sm)),
        "mean_field_tilt_m": tilt_summary["mean"],
        "median_field_tilt_m": tilt_summary["p50"],
        "attacking_zone_share": attacking_share,
        "n_players_per_frame_mean": float(tilt["n_players"].mean()),
    }

    return TacticalFeatures(
        field_tilt_timeline=field_tilt_timeline,
        tilt_summary=tilt_summary,
        zone_distribution=zd,
        shot_moments=sm,
        summary=summary,
    )
