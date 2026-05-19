"""SVG renderers for the tactical layer of the demo page.

Three primitives:
    1. ``render_minimap_at_frame`` — pitch outline + player dots at one frame
    2. ``render_zone_chart`` — horizontal bar chart of attacking-zone split
    3. ``render_tilt_timeline`` — line chart of field tilt over time

All return inline SVG strings ready to drop into the Jinja2 template via
``{{ svg | safe }}``. Palette matches demo.html.j2 (warm paper + grass-green
accent).
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from tactical_cz.tactical.features import (
    ATTACKING_THIRD_X,
    DEFENSIVE_THIRD_X,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
    PLAYER_CLASSES,
)

logger = logging.getLogger(__name__)

# Palette (must match templates/demo.html.j2 :root variables)
COLOR_BG = "#f1eee6"
COLOR_PITCH = "#dfe6d4"     # pitch green-tint, lighter than chart bg
COLOR_LINE = "#8b9080"      # pitch markings
COLOR_INK = "#221f1c"
COLOR_MUTED = "#76706a"
COLOR_RULE = "#d8d2c4"
COLOR_ACCENT = "#4a7c4a"
COLOR_ACCENT_SOFT = "#cce0c8"
COLOR_TEAM_A = "#4a7c4a"    # accent green for team 0
COLOR_TEAM_B = "#9b5a3a"    # warm brown for team 1 (NOT slop red/blue)
COLOR_OTHER = "#76706a"     # GK / no team assigned


def _svg_pitch_base(width_px: int = 360) -> tuple[str, float]:
    """Return SVG <svg ...> opening + the pitch lines, plus the px/m
    scale factor. Caller is expected to add player dots and close </svg>.
    """
    height_px = int(width_px * PITCH_WIDTH_M / PITCH_LENGTH_M)
    scale = width_px / PITCH_LENGTH_M

    # Pitch outline + markings
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width_px} {height_px}" '
        f'style="background:{COLOR_PITCH}">',
        # Outer rectangle
        f'<rect x="2" y="2" width="{width_px-4}" height="{height_px-4}" '
        f'fill="none" stroke="{COLOR_LINE}" stroke-width="1.5"/>',
        # Halfway line
        f'<line x1="{width_px/2}" y1="2" x2="{width_px/2}" y2="{height_px-2}" '
        f'stroke="{COLOR_LINE}" stroke-width="1"/>',
        # Centre circle (9.15 m radius)
        f'<circle cx="{width_px/2}" cy="{height_px/2}" r="{9.15*scale:.1f}" '
        f'fill="none" stroke="{COLOR_LINE}" stroke-width="1"/>',
        # Penalty boxes (~40.32 m wide × 16.5 m deep — Roboflow config uses
        # 41 × 20.15 m, close enough for visual)
        f'<rect x="2" y="{(height_px - 41*scale)/2:.1f}" '
        f'width="{20.15*scale:.1f}" height="{41*scale:.1f}" '
        f'fill="none" stroke="{COLOR_LINE}" stroke-width="1"/>',
        f'<rect x="{width_px - 2 - 20.15*scale:.1f}" '
        f'y="{(height_px - 41*scale)/2:.1f}" '
        f'width="{20.15*scale:.1f}" height="{41*scale:.1f}" '
        f'fill="none" stroke="{COLOR_LINE}" stroke-width="1"/>',
    ]
    return "\n".join(parts), scale


def render_minimap_at_frame(
    vision: pd.DataFrame,
    frame_idx: int,
    width_px: int = 320,
    annotation: str | None = None,
) -> str:
    """Pitch outline + player dots at one frame. Returns inline SVG."""
    base, scale = _svg_pitch_base(width_px=width_px)
    height_px = int(width_px * PITCH_WIDTH_M / PITCH_LENGTH_M)

    # Players at this frame
    rows = vision[vision["frame_idx"] == frame_idx]
    is_player = rows["class_id"].isin(PLAYER_CLASSES)
    has_pitch = rows["pitch_x_m"].notna() & rows["pitch_y_m"].notna()
    in_bounds = (
        rows["pitch_x_m"].between(0, PITCH_LENGTH_M)
        & rows["pitch_y_m"].between(0, PITCH_WIDTH_M)
    )
    rows = rows[is_player & has_pitch & in_bounds]

    dots = []
    for _, r in rows.iterrows():
        cx_px = float(r.pitch_x_m) * scale
        cy_px = float(r.pitch_y_m) * scale
        team = int(r.team_id) if "team_id" in r and r.team_id >= 0 else -1
        if int(r.class_id) == 1:    # GK
            color = COLOR_OTHER
        elif team == 0:
            color = COLOR_TEAM_A
        elif team == 1:
            color = COLOR_TEAM_B
        else:
            color = COLOR_OTHER
        dots.append(
            f'<circle cx="{cx_px:.1f}" cy="{cy_px:.1f}" r="4" '
            f'fill="{color}" stroke="{COLOR_BG}" stroke-width="1"/>'
        )

    extras = []
    if annotation:
        extras.append(
            f'<text x="{width_px-8}" y="{height_px-8}" '
            f'text-anchor="end" fill="{COLOR_MUTED}" '
            f'style="font:11px ui-monospace,monospace">{annotation}</text>'
        )

    return base + "\n" + "\n".join(dots + extras) + "\n</svg>"


def render_zone_chart(
    zone_distribution: dict[str, dict[str, int]],
    width_px: int = 480,
    row_height_px: int = 32,
) -> str:
    """Horizontal stacked bar: per class, fraction of high-confidence
    windows in defensive / midfield / attacking third.

    Stacked left-to-right: defensive (warm-grey), midfield (paper),
    attacking (accent green). Easy to read "where the action is" at a
    glance for each event class.
    """
    if not zone_distribution:
        return "<!-- no zone distribution -->"

    classes = list(zone_distribution.keys())
    n_rows = len(classes)
    label_w = 130
    bar_x = label_w + 8
    bar_w = width_px - bar_x - 8
    height_px = row_height_px * n_rows + 28      # +28 for legend

    color_zones = {
        "defensive": "#a89b88",
        "midfield": "#d8d2c4",
        "attacking": COLOR_ACCENT,
    }

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width_px} {height_px}" '
        f'style="background:{COLOR_BG}">',
        # Legend
        f'<g transform="translate({bar_x}, 6)">',
    ]
    legend_offset = 0
    for label, color in color_zones.items():
        out.append(
            f'<rect x="{legend_offset}" y="2" width="10" height="10" fill="{color}"/>'
            f'<text x="{legend_offset+14}" y="11" fill="{COLOR_MUTED}" '
            f'style="font:10px ui-monospace,monospace">{label}</text>'
        )
        legend_offset += 90
    out.append("</g>")

    for i, cls in enumerate(classes):
        counts = zone_distribution[cls]
        total = sum(counts.values())
        if total == 0:
            continue
        y = 28 + i * row_height_px
        # Label
        out.append(
            f'<text x="{label_w-8}" y="{y + row_height_px/2 + 4:.1f}" '
            f'text-anchor="end" fill="{COLOR_INK}" '
            f'style="font:13px Source\\ Sans\\ 3,sans-serif">{cls}</text>'
        )
        # Stacked bar
        cursor = bar_x
        for zone in ("defensive", "midfield", "attacking"):
            n = counts.get(zone, 0)
            seg_w = (n / total) * bar_w
            if seg_w > 0:
                out.append(
                    f'<rect x="{cursor:.1f}" y="{y + 6}" '
                    f'width="{seg_w:.1f}" height="{row_height_px - 12}" '
                    f'fill="{color_zones[zone]}"/>'
                )
                # Inline count if segment is wide enough
                if seg_w > 28:
                    out.append(
                        f'<text x="{cursor + seg_w/2:.1f}" '
                        f'y="{y + row_height_px/2 + 4:.1f}" '
                        f'text-anchor="middle" fill="{COLOR_INK}" '
                        f'style="font:10px ui-monospace,monospace">{n}</text>'
                    )
                cursor += seg_w
    out.append("</svg>")
    return "\n".join(out)


def render_tilt_timeline(
    field_tilt_timeline: list[tuple[float, float]],
    width_px: int = 720,
    height_px: int = 180,
) -> str:
    """Line chart of mean pitch_x (action centroid) over the clip."""
    if not field_tilt_timeline:
        return "<!-- no tilt timeline -->"

    pad_l, pad_r, pad_t, pad_b = 44, 12, 12, 28
    plot_w = width_px - pad_l - pad_r
    plot_h = height_px - pad_t - pad_b

    seconds = np.array([s for s, _ in field_tilt_timeline])
    tilts = np.array([t for _, t in field_tilt_timeline])
    s_min, s_max = float(seconds.min()), float(seconds.max())
    # y axis: 0..120 m (pitch length)
    y_min, y_max = 0.0, PITCH_LENGTH_M

    def x_px(s: float) -> float:
        return pad_l + (s - s_min) / max(s_max - s_min, 1e-6) * plot_w

    def y_px(t: float) -> float:
        return pad_t + (1 - (t - y_min) / (y_max - y_min)) * plot_h

    points = " ".join(f"{x_px(s):.1f},{y_px(t):.1f}" for s, t in zip(seconds, tilts))

    third_lines = []
    for y_m, label in [(DEFENSIVE_THIRD_X, "1/3"), (ATTACKING_THIRD_X, "2/3")]:
        third_lines.append(
            f'<line x1="{pad_l}" x2="{width_px-pad_r}" '
            f'y1="{y_px(y_m):.1f}" y2="{y_px(y_m):.1f}" '
            f'stroke="{COLOR_RULE}" stroke-width="0.6" stroke-dasharray="3 3"/>'
        )

    return f'''<svg xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 {width_px} {height_px}"
    style="background:{COLOR_BG}">
  {''.join(third_lines)}
  <polyline points="{points}" fill="none" stroke="{COLOR_ACCENT}" stroke-width="2"/>
  <text x="{pad_l-6}" y="{y_px(60):.1f}" text-anchor="end" fill="{COLOR_MUTED}"
        style="font:10px ui-monospace,monospace">60 m</text>
  <text x="{pad_l-6}" y="{y_px(20):.1f}" text-anchor="end" fill="{COLOR_MUTED}"
        style="font:10px ui-monospace,monospace">20 m</text>
  <text x="{pad_l-6}" y="{y_px(100):.1f}" text-anchor="end" fill="{COLOR_MUTED}"
        style="font:10px ui-monospace,monospace">100 m</text>
  <text x="{(pad_l + width_px - pad_r)/2:.1f}" y="{height_px-8}" text-anchor="middle"
        fill="{COLOR_MUTED}" style="font:11px ui-monospace,monospace">seconds</text>
</svg>'''
