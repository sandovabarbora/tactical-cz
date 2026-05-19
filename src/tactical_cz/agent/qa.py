"""Phase 4: natural language Q&A over the multimodal pipeline outputs.

Loads the three parquet artefacts (vision tracking, events timeline,
ASR transcript), summarises them into a compact text context, and lets
Claude answer questions about a specific clip.

The agent has access to:
    * which classes the events model fired at each second
    * mean field tilt + zone distribution from the vision layer
    * Czech commentary text with timestamps + which BAS classes each
      segment was tagged with
    * per-class calibration summary (mean / max / ratio)

It is NOT given raw frames or audio — only the derived parquets. That's
the right level of abstraction for a tactical-tool agent: it reasons
over event timelines, not pixels.

Usage:
    # CLI: one-off question
    uv run python -m tactical_cz.agent.qa \\
        --events data/processed/events_timeline_goals.parquet \\
        --vision data/processed/vision_tracking_goals.parquet \\
        --transcript data/processed/transcript_goals.parquet \\
        --question "When does Sparta attack most aggressively?"

    # Python API: pre-compute several Q&A for the demo page
    from tactical_cz.agent.qa import ClipQA
    qa = ClipQA(events_path, vision_path, transcript_path)
    answer = qa.ask("Where in the clip does the model see Goal moments?")
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tactical_cz.config import PROCESSED_DIR, configure_logging, ensure_dirs

logger = logging.getLogger(__name__)

# Sonnet 4.6 is the right balance for this: handles a few KB of pipeline
# context comfortably, fast enough for interactive demo, and the cost is
# basically nothing (~$3/MTok input, ~$15/MTok output, our context is
# ~3 KB and answers ~500 tokens).
DEFAULT_MODEL: str = "claude-sonnet-4-6"
SYSTEM_PROMPT: str = """You are a tactical-analysis assistant for a multimodal AI pipeline analysing Czech football broadcasts.

You receive THREE inputs derived from a clip:
1. EVENTS TIMELINE — per-window probabilities for 12 BAS classes (Pass, Drive, Shot, Goal, Header, Cross, etc.) from a V-JEPA2-L encoder + small classifier head trained on SoccerNet Ball Action Spotting 2025.
2. TACTICAL FEATURES — vision-derived signals: field tilt (action centroid x in metres on 120×70 m pitch), zone distribution per class, top shot moments.
3. CZECH COMMENTARY — Whisper-transcribed segments with timestamps, tagged where keywords map to BAS classes.

Your job is to answer questions about the clip in a tactical-coach voice: specific, evidence-based, willing to admit when the data doesn't support a confident answer.

Rules:
- Cite specific seconds when possible: "at t=18.5s, P(Shot)=0.83".
- When data conflicts (e.g. commentary says "gol!" but model fires Drive), call it out as a known limitation: model's Goal class has only 16 training examples.
- If asked something the data can't support, say so explicitly. Don't fabricate.
- Use English unless the user writes in Czech; mirror their language.
- Keep answers under 200 words unless asked for detail.
- Use specific numbers from the context. No empty phrases like "the model performs well".
"""


@dataclass
class ClipContext:
    """Compact text summary of all three modalities for one clip."""

    events_summary: str
    tactical_summary: str
    commentary_summary: str
    clip_duration_s: float

    def as_prompt(self) -> str:
        return (
            f"# Clip context ({self.clip_duration_s:.0f}s)\n\n"
            f"## EVENTS MODEL (V-JEPA2 + BAS head)\n{self.events_summary}\n\n"
            f"## TACTICAL FEATURES (vision × events join)\n{self.tactical_summary}\n\n"
            f"## CZECH COMMENTARY (Whisper)\n{self.commentary_summary}\n"
        )


def _events_summary(events: pd.DataFrame) -> str:
    """Compact text: per-class max P (and when) + mean P. Sorted by max."""
    rows = []
    for cls, group in events.groupby("event_type"):
        peak_row = group.loc[group["confidence"].idxmax()]
        rows.append((float(peak_row["confidence"]), cls, float(peak_row["second"]),
                     float(group["confidence"].mean())))
    rows.sort(reverse=True)
    lines = ["Per-class stats across all sliding windows:"]
    for max_p, cls, peak_s, mean_p in rows:
        lines.append(
            f"  {cls}: max={max_p:.3f} (at t={peak_s:.1f}s), mean={mean_p:.3f}"
        )
    return "\n".join(lines)


def _tactical_summary_text(vision: pd.DataFrame, events: pd.DataFrame, fps: float = 25.0) -> str:
    """Use the tactical.features.compute_all bundle as the summary source."""
    from tactical_cz.tactical.features import compute_all
    tact = compute_all(vision, events, fps=fps)
    if not tact.field_tilt_timeline:
        return "Tactical layer unavailable (no in-pitch detections)."
    lines = [
        f"Median field tilt: {tact.summary['median_field_tilt_m']:.1f} m on 120 m pitch",
        f"Mean players visible per frame: {tact.summary['n_players_per_frame_mean']:.1f}",
        f"Attacking-zone share of Shot+Drive top-quartile windows: {tact.summary['attacking_zone_share']:.0%}",
        "",
        "Zone distribution (defensive / midfield / attacking):",
    ]
    for cls, counts in tact.zone_distribution.items():
        total = sum(counts.values())
        if total == 0:
            continue
        d_pct = counts.get("defensive", 0) / total * 100
        m_pct = counts.get("midfield", 0) / total * 100
        a_pct = counts.get("attacking", 0) / total * 100
        lines.append(
            f"  {cls}: def {d_pct:.0f}% / mid {m_pct:.0f}% / att {a_pct:.0f}%"
        )
    lines.append("")
    lines.append("Top Shot moments (highest model confidence + pitch context):")
    for m in tact.shot_moments[:6]:
        lines.append(
            f"  t={m['second']:.1f}s: P(Shot)={m['p_shot']:.2f}, "
            f"action in {m['action_zone']} third, {m['n_players_visible']} players visible"
        )
    return "\n".join(lines)


def _commentary_summary_text(transcript: pd.DataFrame, events: pd.DataFrame) -> str:
    """Tagged commentary segments with model agreement context."""
    from tactical_cz.audio.align import align
    aligned = align(transcript, events)
    if not aligned:
        return "No commentary segments tagged with BAS keywords."
    lines = [
        f"Whisper transcribed {len(transcript)} segments; "
        f"{len(aligned)} contain BAS-class keywords.",
        "",
        "Tagged commentary moments (commentator's words + which event class they map to + what the events model thought):",
    ]
    for e in aligned:
        marker = "AGREE" if e.agrees else "disagree"
        lines.append(
            f"  t={e.midpoint_s:.1f}s [{marker}] tagged={e.bas_classes}: "
            f'"{e.text.strip()}" '
            f"(model max P={e.model_p_max:.2f}, top: {e.model_top_class}({e.model_top_p:.2f}))"
        )
    return "\n".join(lines)


def build_context(
    events_path: Path,
    vision_path: Path | None = None,
    transcript_path: Path | None = None,
    fps: float = 25.0,
) -> ClipContext:
    """Assemble the multimodal context summary."""
    events = pd.read_parquet(events_path)
    clip_duration_s = float(events["second"].max() - events["second"].min()) if not events.empty else 0.0

    ev_text = _events_summary(events)
    if vision_path and vision_path.exists():
        vision = pd.read_parquet(vision_path)
        tact_text = _tactical_summary_text(vision, events, fps=fps)
    else:
        tact_text = "Vision tracking parquet not provided; tactical layer unavailable."

    if transcript_path and transcript_path.exists():
        transcript = pd.read_parquet(transcript_path)
        com_text = _commentary_summary_text(transcript, events)
    else:
        com_text = "Transcript parquet not provided; commentary layer unavailable."

    return ClipContext(
        events_summary=ev_text,
        tactical_summary=tact_text,
        commentary_summary=com_text,
        clip_duration_s=clip_duration_s,
    )


class ClipQA:
    """Stateful Q&A: one ClipContext + repeated questions against Claude."""

    def __init__(
        self,
        events_path: Path,
        vision_path: Path | None = None,
        transcript_path: Path | None = None,
        model: str = DEFAULT_MODEL,
        fps: float = 25.0,
    ) -> None:
        # Lazy .env load (mirrors downloader.py pattern)
        from dotenv import load_dotenv
        load_dotenv()
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Add it to .env at project root."
            )
        from anthropic import Anthropic
        self.client = Anthropic()
        self.model = model
        self.context = build_context(events_path, vision_path, transcript_path, fps=fps)

    def ask(self, question: str, max_tokens: int = 600) -> str:
        """Run a single question through Claude with the clip context."""
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT + "\n\n" + self.context.as_prompt(),
            messages=[{"role": "user", "content": question}],
        )
        # Concatenate text blocks (Claude returns a list of content blocks)
        return "".join(b.text for b in resp.content if hasattr(b, "text")).strip()

    def ask_many(self, questions: list[str]) -> list[dict]:
        """Run multiple questions, return [{question, answer}] list."""
        out = []
        for q in questions:
            logger.info("Asking: %s", q[:80])
            a = self.ask(q)
            out.append({"question": q, "answer": a})
        return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--vision", type=Path, default=None)
    parser.add_argument("--transcript", type=Path, default=None)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args(argv)
    configure_logging()
    ensure_dirs()

    qa = ClipQA(
        events_path=args.events,
        vision_path=args.vision,
        transcript_path=args.transcript,
        model=args.model,
    )
    print(qa.ask(args.question))
    return 0


if __name__ == "__main__":
    sys.exit(main())
