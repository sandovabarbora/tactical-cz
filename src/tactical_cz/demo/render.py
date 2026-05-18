"""Render the Phase 2 demo page (index.html) from the events parquet + annotated mp4.

Single-file output: an HTML page that embeds the annotated mp4 (relative
path) and inlines a matplotlib SVG chart showing P(Shot) over time.
Ships to docs/ for GitHub Pages or stays in outputs/ for local preview.

The template is in templates/demo.html.j2 and uses the same warm-paper +
single-accent palette as the Sparta report (institutional cousin, not
copy-paste) — keeps the brand family coherent across both portfolio pieces.
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from tactical_cz.config import (
    OUTPUTS_DIR,
    PROJECT_ROOT,
    configure_logging,
    ensure_dirs,
    get_device,
)

logger = logging.getLogger(__name__)

TEMPLATES_DIR = PROJECT_ROOT / "templates"

# Visual constants for matplotlib (match the HTML page palette).
COLOR_BG = "#f1eee6"          # paper
COLOR_INK = "#221f1c"
COLOR_MUTED = "#76706a"
COLOR_ACCENT = "#4a7c4a"      # grass green
COLOR_ACCENT_SOFT = "#cce0c8"
COLOR_RULE = "#d8d2c4"


@dataclass
class RenderConfig:
    events_parquet: Path
    annotated_video: Path
    source_video: Path
    out_html: Path
    title: str = "Phase 2 trained-head demo on a Sparta broadcast clip"
    lede: str = (
        "End-to-end V-JEPA2-L → BAS head pipeline running on a 30 s Sparta "
        "highlight, fully on Apple-silicon MPS. Proves the pipeline executes "
        "on real Czech broadcast video before SoccerNet NDA labels arrive."
    )
    repo_url: str = "https://github.com/sandovabarbora/tactical-cz"
    fake_label_frame: int = 350
    fake_label_window_frames: int = 50
    highlight_threshold: float = 0.5


def _render_shot_timeline_svg(
    df: pd.DataFrame,
    fake_low: int, fake_high: int,
) -> str:
    """Inline matplotlib SVG: P(Shot) over time with fake-label band shaded."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shots = df[df.event_type == "Shot"].sort_values("frame_idx")
    if shots.empty:
        return "<!-- no Shot rows in events parquet -->"

    fig, ax = plt.subplots(figsize=(8.4, 3.2))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor("#fafaf7")

    # Shade fake-label band
    ax.axvspan(fake_low, fake_high, color=COLOR_ACCENT_SOFT, alpha=0.6,
               label=f"fake label window")

    # Probability curve with markers
    ax.plot(shots.frame_idx, shots.confidence, color=COLOR_ACCENT,
            linewidth=2.0, marker="o", markersize=5, markerfacecolor=COLOR_ACCENT,
            markeredgecolor=COLOR_BG, markeredgewidth=0.7, zorder=3)

    ax.axhline(0.5, color=COLOR_MUTED, linewidth=0.8,
               linestyle="--", alpha=0.5, zorder=1)
    ax.text(shots.frame_idx.min(), 0.52, "0.5 threshold",
            color=COLOR_MUTED, fontsize=8)

    ax.set_xlabel("frame index", color=COLOR_MUTED, fontsize=9)
    ax.set_ylabel("P(Shot)", color=COLOR_MUTED, fontsize=9)
    ax.tick_params(colors=COLOR_MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(COLOR_RULE)
        spine.set_linewidth(0.6)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(False)

    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue().split("<?xml", 1)[-1].split("?>", 1)[-1]  # strip XML prolog


def _per_class_table(df: pd.DataFrame) -> list[dict]:
    """Top-confidence frame per BAS class, sorted by max prob descending."""
    rows = []
    for cls, group in df.groupby("event_type"):
        top = group.loc[group.confidence.idxmax()]
        rows.append({
            "cls": cls,
            "max_p": float(top.confidence),
            "at_frame": int(top.frame_idx),
            "at_second": float(top.second),
        })
    return sorted(rows, key=lambda r: -r["max_p"])


def render_demo(cfg: RenderConfig) -> Path:
    """Render demo index.html. Annotated mp4 is copied next to it so the
    page works as a self-contained directory (GitHub Pages friendly)."""
    cfg.out_html.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(cfg.events_parquet)
    if df.empty:
        raise ValueError(f"Empty events parquet: {cfg.events_parquet}")

    fake_low = cfg.fake_label_frame - cfg.fake_label_window_frames
    fake_high = cfg.fake_label_frame + cfg.fake_label_window_frames
    fake_second = cfg.fake_label_frame / 25.0   # FIXME pass fps if not 25

    # Signal ratio
    shots = df[df.event_type == "Shot"]
    lab = shots[(shots.frame_idx >= fake_low) & (shots.frame_idx <= fake_high)]
    unl = shots[(shots.frame_idx < fake_low) | (shots.frame_idx > fake_high)]
    signal_ratio = (lab.confidence.mean() / max(unl.confidence.mean(), 1e-6)) if not lab.empty else float("nan")

    n_windows = df.frame_idx.nunique()

    # Copy annotated mp4 next to the html (so relative <video src=...> works)
    annotated_target = cfg.out_html.parent / cfg.annotated_video.name
    if annotated_target.resolve() != cfg.annotated_video.resolve():
        shutil.copy2(cfg.annotated_video, annotated_target)
    video_size_mb = round(annotated_target.stat().st_size / (1024 * 1024), 1)

    cap = cv2.VideoCapture(str(annotated_target))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    video_duration = round(total / fps, 1)

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(default=True),
    )
    template = env.get_template("demo.html.j2")
    html = template.render(
        title=cfg.title,
        lede=cfg.lede,
        repo_url=cfg.repo_url,
        generated_at=dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M UTC"),
        source_video=cfg.source_video.name,
        annotated_video_rel=annotated_target.name,
        video_size_mb=video_size_mb,
        video_duration=video_duration,
        n_windows=n_windows,
        device=get_device(),
        clip_seconds=4.0,
        highlight_threshold=cfg.highlight_threshold,
        fake_label_frame=cfg.fake_label_frame,
        fake_label_second=int(round(fake_second)),
        fake_label_low=fake_low,
        fake_label_high=fake_high,
        signal_ratio=f"{signal_ratio:.1f}",
        train_loss_final="0.0007",
        shot_chart_svg=_render_shot_timeline_svg(df, fake_low, fake_high),
        per_class_table=_per_class_table(df),
    )
    cfg.out_html.write_text(html, encoding="utf-8")
    logger.info("Wrote demo page: %s (%d bytes)", cfg.out_html, len(html))
    return cfg.out_html


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--annotated", type=Path, required=True,
                        help="Annotated mp4 from demo.annotate")
    parser.add_argument("--source", type=Path, required=True,
                        help="Original source mp4 (for label in the byline)")
    parser.add_argument("--out", type=Path, default=OUTPUTS_DIR / "demo" / "index.html")
    parser.add_argument("--fake-label-frame", type=int, default=350)
    parser.add_argument("--fake-label-window", type=int, default=50)
    args = parser.parse_args(argv)
    configure_logging()
    ensure_dirs()
    cfg = RenderConfig(
        events_parquet=args.events,
        annotated_video=args.annotated,
        source_video=args.source,
        out_html=args.out,
        fake_label_frame=args.fake_label_frame,
        fake_label_window_frames=args.fake_label_window,
    )
    render_demo(cfg)
    print(cfg.out_html)
    return 0


if __name__ == "__main__":
    sys.exit(main())
