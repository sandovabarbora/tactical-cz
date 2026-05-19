"""Whisper transcription of Czech football broadcast commentary.

Uses faster-whisper (CTranslate2 backend, ~4× faster than openai-whisper
on CPU; matters for M-series MPS where openai-whisper falls back to
CPU anyway). Czech is in Whisper's training set; we explicitly pass
``language="cs"`` to skip language detection and avoid drift to Slovak/
Polish on short clips.

Output schema (parquet, one row per Whisper segment):

    start_s   float   segment start, seconds from clip start
    end_s     float   segment end
    text      str     transcribed Czech text
    duration  float   end_s - start_s
    n_words   int     word count proxy for speech density

Segments are typically 2-15 seconds each; Whisper's segmenter cuts at
silence + sentence boundaries. For event spotting we'll later align
segment midpoints with the events_timeline parquet (within ±2s window).
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from tactical_cz.config import (
    PROCESSED_DIR,
    configure_logging,
    ensure_dirs,
)

logger = logging.getLogger(__name__)


@dataclass
class TranscribeConfig:
    source: Path
    out_parquet: Path
    model_size: str = "medium"       # tiny|base|small|medium|large-v3
    language: str = "cs"
    compute_type: str = "int8"       # CPU-friendly; "float16" needs GPU
    beam_size: int = 5


def transcribe(cfg: TranscribeConfig) -> Path:
    """Run faster-whisper on a clip, write segments parquet. Returns path."""
    from faster_whisper import WhisperModel

    logger.info("Loading faster-whisper %s (~first run downloads, cached after)",
                cfg.model_size)
    model = WhisperModel(cfg.model_size, device="cpu", compute_type=cfg.compute_type)

    logger.info("Transcribing %s (language=%s, beam_size=%d)",
                cfg.source, cfg.language, cfg.beam_size)
    segments, info = model.transcribe(
        str(cfg.source),
        language=cfg.language,
        beam_size=cfg.beam_size,
        vad_filter=True,                # drop silence; cleaner segment boundaries
    )
    logger.info("Detected duration: %.1fs, language probability: %.2f",
                info.duration, info.language_probability)

    rows = []
    for seg in segments:
        text = seg.text.strip()
        rows.append({
            "start_s": float(seg.start),
            "end_s": float(seg.end),
            "duration": float(seg.end - seg.start),
            "text": text,
            "n_words": int(len(text.split())),
        })
        if len(rows) % 10 == 0:
            logger.info("  %d segments so far …", len(rows))

    df = pd.DataFrame(rows)
    cfg.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cfg.out_parquet, index=False)
    logger.info("Wrote %d segments → %s", len(df), cfg.out_parquet)
    return cfg.out_parquet


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Path to broadcast video (mp4)")
    parser.add_argument("--out", type=Path,
                        default=PROCESSED_DIR / "transcript.parquet")
    parser.add_argument("--model-size", default="medium",
                        choices=["tiny", "base", "small", "medium", "large-v3"])
    parser.add_argument("--language", default="cs")
    parser.add_argument("--compute-type", default="int8",
                        choices=["int8", "float16", "float32"])
    args = parser.parse_args(argv)
    configure_logging()
    ensure_dirs()
    cfg = TranscribeConfig(
        source=args.source,
        out_parquet=args.out,
        model_size=args.model_size,
        language=args.language,
        compute_type=args.compute_type,
    )
    transcribe(cfg)
    print(cfg.out_parquet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
