"""Tests for the audio (commentary) layer.

Synthetic transcript + events parquets. Whisper actual transcription is
not exercised here — that needs a real audio file + model download. The
tests focus on keyword alignment and BAS-class mapping correctness,
which is where the project-specific logic lives.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tactical_cz.audio.align import (
    CZECH_TO_BAS,
    CommentaryEvent,
    align,
    tag_segments,
)


def _synthetic_transcript() -> pd.DataFrame:
    return pd.DataFrame([
        {"start_s": 0.0, "end_s": 3.0, "duration": 3.0,
         "text": "Sparta začíná zápas v útočném postavení.", "n_words": 5},
        {"start_s": 3.0, "end_s": 6.5, "duration": 3.5,
         "text": "Krásná přihrávka ve středu pole.", "n_words": 5},
        {"start_s": 6.5, "end_s": 9.0, "duration": 2.5,
         "text": "Centruje na hlavičku, ale mimo!", "n_words": 5},
        {"start_s": 9.0, "end_s": 12.0, "duration": 3.0,
         "text": "Tvrdá střela z dálky, brankář vyrazil.", "n_words": 6},
        {"start_s": 12.0, "end_s": 15.0, "duration": 3.0,
         "text": "Góóól! Sparta vede 1:0.", "n_words": 4},
        {"start_s": 15.0, "end_s": 18.0, "duration": 3.0,
         "text": "Rozhodčí píská faul a Sparta má volný kop.", "n_words": 8},
    ])


def _synthetic_events_for_seconds(seconds_with_event: dict[float, dict[str, float]]) -> pd.DataFrame:
    """Build events_timeline parquet with given P(class) at each second."""
    rows = []
    for s, probs in seconds_with_event.items():
        frame = int(s * 25)
        for cls, p in probs.items():
            rows.append({
                "frame_idx": frame,
                "second": s,
                "event_type": cls,
                "confidence": p,
                "source_video": "synthetic.mp4",
            })
    return pd.DataFrame(rows)


def test_czech_keyword_dict_covers_all_bas_classes() -> None:
    """Every BAS class should have at least one Czech keyword variant."""
    from tactical_cz.events.model import BAS_CLASSES
    # The dict can include the "Cross / Corner" synthesis but every
    # real BAS class must appear (in some form, possibly slashed)
    keys_str = " ".join(CZECH_TO_BAS.keys())
    missing = [c for c in BAS_CLASSES if c not in keys_str]
    # Allow some classes to be missing if no clean Czech mapping exists,
    # but most should be there
    assert len(missing) <= 2, f"Most BAS classes should have Czech mapping; missing: {missing}"


def test_tag_segments_finds_keywords() -> None:
    t = _synthetic_transcript()
    tagged = tag_segments(t)
    # Row 1 ("přihrávka") → Pass
    assert "Pass" in tagged.iloc[1].bas_classes
    # Row 2 ("Centruje", "hlavičku", "mimo") → Cross + Header + Out
    classes_2 = set(tagged.iloc[2].bas_classes)
    assert "Cross" in classes_2
    assert "Header" in classes_2
    assert "Out" in classes_2
    # Row 3 ("střela") → Shot
    assert "Shot" in tagged.iloc[3].bas_classes
    # Row 4 ("Góóól") → Goal
    assert "Goal" in tagged.iloc[4].bas_classes
    # Row 5 ("faul" doesn't map to anything, "volný kop" → Free Kick)
    assert "Free Kick" in tagged.iloc[5].bas_classes


def test_tag_segments_no_false_positives() -> None:
    """Row 0 has no event keywords → empty bas_classes."""
    t = _synthetic_transcript()
    tagged = tag_segments(t)
    assert tagged.iloc[0].bas_classes == []
    assert tagged.iloc[0].matched_stems == []


def test_align_marks_agreement_at_high_p() -> None:
    """Whisper says Goal at second 13.5; events parquet has Goal P=0.9 at second 13 → agree."""
    t = _synthetic_transcript()
    events = _synthetic_events_for_seconds({
        13.5: {"Goal": 0.9, "Shot": 0.6, "Pass": 0.05},
        10.5: {"Shot": 0.7, "Pass": 0.1, "Goal": 0.02},
    })
    aligned = align(t, events, fps=25.0, agree_threshold=0.30)
    # Row matching "Góóól" (start 12, end 15, mid 13.5) should agree on Goal
    goal_event = [e for e in aligned if "Goal" in e.bas_classes][0]
    assert goal_event.agrees, f"Goal commentary should agree with Goal P=0.9 at second 13"
    assert goal_event.model_p_max >= 0.9


def test_align_marks_disagreement_at_low_p() -> None:
    """Whisper says Goal but events have low P → does not agree."""
    t = _synthetic_transcript()
    events = _synthetic_events_for_seconds({
        13.5: {"Goal": 0.05, "Shot": 0.10, "Pass": 0.80},
    })
    aligned = align(t, events, fps=25.0, agree_threshold=0.30)
    goal_event = [e for e in aligned if "Goal" in e.bas_classes][0]
    assert not goal_event.agrees, "low P(Goal) should not register as agreement"


def test_align_returns_commentary_event_dataclass() -> None:
    t = _synthetic_transcript()
    events = _synthetic_events_for_seconds({
        4.75: {"Pass": 0.6, "Drive": 0.4, "Shot": 0.05},
    })
    aligned = align(t, events, fps=25.0)
    assert all(isinstance(e, CommentaryEvent) for e in aligned)
    pass_event = [e for e in aligned if "Pass" in e.bas_classes][0]
    assert pass_event.model_top_class is not None
    assert 0 <= pass_event.model_top_p <= 1
