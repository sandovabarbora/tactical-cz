"""Inference stub — events on a tactical-cz vision parquet.

Phase 2 implementation. Reads the vision_tracking.parquet from Phase 1's
VisionPipeline, runs the fine-tuned BAS model on the corresponding source
video, and writes events_timeline.parquet alongside.
"""

from __future__ import annotations

from pathlib import Path


def infer_events(source_video: Path, vision_parquet: Path, out_parquet: Path) -> Path:
    """Stub. Implemented in Phase 2."""
    raise NotImplementedError(
        "infer_events stub. Requires Phase 2 trained model. See docs/PHASE2_KICKOFF.md § 5."
    )
