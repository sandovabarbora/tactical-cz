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
    vision_parquet: Path | None = None     # Phase 1 tracking parquet → enables tactical layer
    transcript_parquet: Path | None = None # Phase 3 ASR transcript → enables commentary layer
    agent_qa_json: Path | None = None      # Phase 4 pre-computed Claude Q&A → enables agent layer
    title: str = "Phase 2 baseline: V-JEPA2-L + BAS head on a Sparta goal compilation"
    lede: str = (
        "Frozen V-JEPA2-L (Meta, MIT-licensed) encoder, small linear head "
        "trained on 4 EFL matches from SoccerNet Ball Action Spotting 2025 "
        "(~8K windows, 12 event classes), evaluated on a 2-minute Sparta "
        "all-goals-of-2025 compilation from the official @acspartapraha "
        "channel. Out-of-distribution evaluation: Czech league + multi-match "
        "splice + different camera operators. Honest baseline showing what "
        "transfers, what doesn't, and why."
    )
    repo_url: str = "https://github.com/sandovabarbora/tactical-cz"
    fake_label_frame: int = 350
    fake_label_window_frames: int = 50
    highlight_threshold: float = 0.5
    # Real-labels mode flips the methodology section + chart + headline
    # to describe a SoccerNet-trained run instead of the original
    # fake-labels demo. Default real-labels going forward.
    mode: str = "real-labels"               # "fake-labels" | "real-labels"
    train_loss_final: str = "0.18"
    val_loss_final: str = "0.21"


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
    """Calibration-aware per-class summary.

    For each BAS class: max P (peak signal), mean P (calibration bias
    proxy — high mean ≈ fires high almost always), where peak landed.

    The calibration column is the diagnostic that matters: ``mean / max``
    near 1.0 means the class is over-confident across all windows;
    near 0 means it spikes only at specific moments (the right behaviour
    for event spotting).
    """
    rows = []
    for cls, group in df.groupby("event_type"):
        top = group.loc[group.confidence.idxmax()]
        rows.append({
            "cls": cls,
            "max_p": float(top.confidence),
            "mean_p": float(group.confidence.mean()),
            "at_frame": int(top.frame_idx),
            "at_second": float(top.second),
            "calibration_ratio": float(group.confidence.mean() / max(top.confidence, 1e-6)),
        })
    return sorted(rows, key=lambda r: -r["max_p"])


def _top1_timeline(df: pd.DataFrame, max_rows: int = 30) -> list[dict]:
    """Top-1 class + runner-up per window. For long clips (>30 windows),
    sub-sample uniformly so the table stays readable. The real signal:
    does the model localize events to specific moments?"""
    piv = df.pivot_table(index="frame_idx", columns="event_type", values="confidence")
    if len(piv) > max_rows:
        # Uniform stride sampling — keeps temporal coverage across the clip
        stride = len(piv) // max_rows
        piv = piv.iloc[::stride].head(max_rows)
    rows = []
    for f in piv.index:
        sorted_classes = piv.loc[f].sort_values(ascending=False)
        rows.append({
            "frame": int(f),
            "second": float(f / 25.0),
            "top1_cls": str(sorted_classes.index[0]),
            "top1_p": float(sorted_classes.iloc[0]),
            "top2_cls": str(sorted_classes.index[1]),
            "top2_p": float(sorted_classes.iloc[1]),
        })
    return rows


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

    # Tactical layer (Phase 1 vision × Phase 2 events join)
    tactical_summary = None
    tactical_zone_svg = None
    tactical_tilt_svg = None
    tactical_minimaps: list[dict] = []
    if cfg.vision_parquet and cfg.vision_parquet.exists():
        from tactical_cz.tactical.features import compute_all
        from tactical_cz.tactical.visualize import (
            render_minimap_at_frame,
            render_tilt_timeline,
            render_zone_chart,
        )
        vision = pd.read_parquet(cfg.vision_parquet)
        tact = compute_all(vision, df, fps=fps)
        tactical_summary = tact.summary
        tactical_zone_svg = render_zone_chart(tact.zone_distribution)
        tactical_tilt_svg = render_tilt_timeline(tact.field_tilt_timeline)
        # Render minimaps at the top shot moments; cap at 6 for page weight
        for moment in tact.shot_moments[:6]:
            svg = render_minimap_at_frame(
                vision, moment["frame"],
                annotation=f"t={moment['second']:.1f}s · {moment['action_zone']}",
            )
            tactical_minimaps.append({
                **moment,
                "minimap_svg": svg,
            })
        logger.info("Tactical layer computed: %d shot moments, %d zone classes",
                    len(tact.shot_moments), len(tact.zone_distribution))
    else:
        logger.info("No vision_parquet provided; tactical layer skipped")

    # Commentary layer (Phase 3 ASR × Phase 2 events alignment)
    commentary_events = None
    commentary_summary = None
    if cfg.transcript_parquet and cfg.transcript_parquet.exists():
        from tactical_cz.audio.align import align as align_commentary
        transcript = pd.read_parquet(cfg.transcript_parquet)
        events_at = align_commentary(transcript, df, fps=fps)
        commentary_events = [
            {
                "start_s": e.start_s,
                "midpoint_s": e.midpoint_s,
                "text": e.text,
                "matched_stems": ", ".join(e.matched_stems),
                "bas_classes": ", ".join(e.bas_classes),
                "model_p_max": e.model_p_max,
                "model_top_class": e.model_top_class or "—",
                "model_top_p": e.model_top_p,
                "agrees": e.agrees,
            }
            for e in events_at
        ]
        n_agree = sum(1 for e in events_at if e.agrees)
        n_goal_calls = sum(1 for e in events_at if "Goal" in e.bas_classes)
        n_goal_agreements = sum(
            1 for e in events_at if "Goal" in e.bas_classes and e.agrees
        )
        commentary_summary = {
            "n_segments": int(len(transcript)),
            "n_tagged": int(len(events_at)),
            "n_agree": int(n_agree),
            "agree_share": n_agree / max(len(events_at), 1),
            "n_goal_calls": int(n_goal_calls),
            "n_goal_agreements": int(n_goal_agreements),
        }
        logger.info(
            "Commentary layer: %d segments, %d tagged, %d agree with events model",
            commentary_summary["n_segments"], commentary_summary["n_tagged"], n_agree,
        )
    else:
        logger.info("No transcript_parquet provided; commentary layer skipped")

    # Agent layer (Phase 4: pre-computed Claude Q&A)
    agent_qa = None
    if cfg.agent_qa_json and cfg.agent_qa_json.exists():
        import json as _json
        import markdown as _markdown
        raw = _json.loads(cfg.agent_qa_json.read_text(encoding="utf-8"))
        # Convert each answer from Markdown to HTML (tables, lists, bold)
        agent_qa = [
            {
                "question": item["question"],
                "answer_html": _markdown.markdown(
                    item["answer"],
                    extensions=["tables", "fenced_code", "sane_lists"],
                ),
            }
            for item in raw
        ]
        logger.info("Agent layer: %d pre-computed Q&A pairs", len(agent_qa))
    else:
        logger.info("No agent_qa_json provided; agent layer skipped")

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
        mode=cfg.mode,
        fake_label_frame=cfg.fake_label_frame,
        fake_label_second=int(round(fake_second)),
        fake_label_low=fake_low,
        fake_label_high=fake_high,
        signal_ratio=f"{signal_ratio:.1f}",
        train_loss_final=cfg.train_loss_final,
        val_loss_final=cfg.val_loss_final,
        shot_chart_svg=_render_shot_timeline_svg(df, fake_low, fake_high),
        per_class_table=_per_class_table(df),
        top1_timeline=_top1_timeline(df),
        tactical_summary=tactical_summary,
        tactical_zone_svg=tactical_zone_svg,
        tactical_tilt_svg=tactical_tilt_svg,
        tactical_minimaps=tactical_minimaps,
        commentary_summary=commentary_summary,
        commentary_events=commentary_events,
        agent_qa=agent_qa,
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
    parser.add_argument("--vision", type=Path, default=None,
                        help="Phase 1 tracking parquet (enables tactical layer)")
    parser.add_argument("--transcript", type=Path, default=None,
                        help="Phase 3 ASR transcript parquet (enables commentary layer)")
    parser.add_argument("--agent-qa", type=Path, default=None,
                        help="Phase 4 pre-computed Q&A JSON (enables agent layer)")
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
        vision_parquet=args.vision,
        transcript_parquet=args.transcript,
        agent_qa_json=args.agent_qa,
        fake_label_frame=args.fake_label_frame,
        fake_label_window_frames=args.fake_label_window,
    )
    render_demo(cfg)
    print(cfg.out_html)
    return 0


if __name__ == "__main__":
    sys.exit(main())
