"""Czech commentary → BAS event class keyword alignment.

Two-step alignment:

1. **Keyword tagging**: scan each Whisper segment for Czech keywords
   that map to BAS event classes. Multiple keywords per segment are
   allowed (a commentator can say "střela na bránu po centru" in one
   breath, tagging both Shot and Cross).

2. **Cross-reference with events_timeline**: at each tagged segment's
   midpoint, look up the model's per-class probability and report
   agreement / disagreement. The interesting moments are where both
   modalities agree at high confidence — they're the "real" events.

Czech-keyword dictionary is intentionally minimal and football-specific.
We use Czech word *stems* (lower-cased substrings) so we catch
inflections without needing morphological analysis: "střela", "střelu",
"střelil", "střílí" all match "střel".
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _ascii_fold(s: str) -> str:
    """Lowercase + strip Czech diacritics. Whisper's transcription of
    Czech football commentary frequently drops or mangles diacritics
    (góóól → gol, střela → srela / sřela, přihrávka → prihravka),
    so we match on the diacritic-free form on both sides.
    """
    return "".join(
        c for c in unicodedata.normalize("NFD", s.lower())
        if unicodedata.category(c) != "Mn"
    )


# Czech keyword stems → BAS class. Both keys (stems) and incoming text
# are ASCII-folded before matching, so diacritics on either side don't
# matter. Stems must be lower-case.
CZECH_TO_BAS: dict[str, list[str]] = {
    # NB: "branka" intentionally NOT here — substring matches "brankář"
    # (goalkeeper). Football commentators say "gol" 99% of the time
    # for actual goals; we anchor on that.
    "Goal": ["goool", "gol", "gola", "goly", "golu",
             "skore", "skoruje", "do site", "dohral do site",
             "dava gol", "je to gol"],
    "Shot": ["strel", "srel",                           # střela / sřela (Whisper drops ř)
             "vystrel", "vystrelil",
             "rana", "ranil",
             "zakonceni", "zakoncit", "zakoncil",
             "strelec", "srelec", "strelecke", "srelecke", "strelecky",
             "vyrazil", "chyt", "pokus"],
    "Cross": ["centr", "centruj", "centrovan"],
    "Header": ["hlavick", "hlavic", "hlavou"],
    "Pass": ["prihravk", "prihrav", "prihraj", "prihral", "nahral",
             "nahravk", "balon", "podan"],
    "High Pass": ["dlouhy mic", "dlouha prihrav", "dlouhy balon"],
    "Throw In": ["vhazovani", "vhodit"],
    "Free Kick": ["volny kop", "primy kop", "trestny kop", "rohovy kop", "rohak"],
    "Out": ["mimo", "do autu", "do bocniho"],
    "Ball Player Block": ["zblokov", "zablokov", "zblok"],
    "Player Successful Tackle": ["odebr", "skluz", "vyhral souboj"],
    "Drive": ["pronik", "ujizdi", "rozjizdi", "rozjel", "brejk", "rychly utok"],
}

# Reverse lookup: stem → set of BAS classes that contain it
_STEM_TO_CLASSES: dict[str, set[str]] = {}
for cls, stems in CZECH_TO_BAS.items():
    for stem in stems:
        _STEM_TO_CLASSES.setdefault(stem, set()).add(cls)


@dataclass(frozen=True)
class CommentaryEvent:
    """One commentary-tagged moment with model agreement signal."""

    start_s: float
    end_s: float
    midpoint_s: float
    text: str
    matched_stems: list[str]
    bas_classes: list[str]
    model_p_max: float                  # max P over matched classes at midpoint
    model_top_class: str | None         # top-1 model class at midpoint (for context)
    model_top_p: float                  # top-1 P
    agrees: bool                        # True if any matched_class has P >= 0.30 at midpoint


def tag_segments(transcript: pd.DataFrame) -> pd.DataFrame:
    """Add `matched_stems` and `bas_classes` columns to a transcript DataFrame.

    Returns the same DataFrame with two new columns of list[str].
    Segments with no matches get empty lists.
    """
    out = transcript.copy()
    matched_stems: list[list[str]] = []
    bas_classes: list[list[str]] = []
    for text in transcript["text"].fillna(""):
        folded = _ascii_fold(text)
        hits_stems: list[str] = []
        hits_classes: set[str] = set()
        for stem, classes in _STEM_TO_CLASSES.items():
            if stem in folded:
                hits_stems.append(stem)
                hits_classes.update(classes)
        matched_stems.append(hits_stems)
        bas_classes.append(sorted(hits_classes))
    out["matched_stems"] = matched_stems
    out["bas_classes"] = bas_classes
    return out


def _nearest_event_probs(
    events: pd.DataFrame,
    target_frame: int,
) -> dict[str, float]:
    """Per-class probability at the events row closest to target_frame.

    Events parquet has multiple rows per frame_idx (one per class). We
    pivot on frame_idx first, then pick the nearest frame.
    """
    if events.empty:
        return {}
    frames = events["frame_idx"].unique()
    nearest = int(frames[np.argmin(np.abs(frames - target_frame))])
    sub = events[events["frame_idx"] == nearest]
    return dict(zip(sub["event_type"], sub["confidence"]))


def align(
    transcript: pd.DataFrame,
    events: pd.DataFrame,
    fps: float = 25.0,
    agree_threshold: float = 0.30,
) -> list[CommentaryEvent]:
    """For each transcript segment with a BAS-class match, look up the
    model's per-class P at the segment midpoint frame. Return a list
    of CommentaryEvent rows.
    """
    tagged = tag_segments(transcript)
    out: list[CommentaryEvent] = []
    for _, row in tagged.iterrows():
        bas_classes: list[str] = row["bas_classes"]
        if not bas_classes:
            continue
        mid_s = (row["start_s"] + row["end_s"]) / 2
        target_frame = int(mid_s * fps)
        probs = _nearest_event_probs(events, target_frame)
        # Map "Cross / Corner" → just "Cross" for probability lookup
        # (it's not a real BAS class, just a Czech-language synthesis)
        effective_classes = [c.split(" / ")[0] for c in bas_classes]
        ps = [probs.get(c, 0.0) for c in effective_classes]
        p_max = max(ps) if ps else 0.0
        if probs:
            top_cls = max(probs, key=probs.get)
            top_p = float(probs[top_cls])
        else:
            top_cls = None
            top_p = 0.0
        out.append(CommentaryEvent(
            start_s=float(row["start_s"]),
            end_s=float(row["end_s"]),
            midpoint_s=float(mid_s),
            text=str(row["text"]),
            matched_stems=row["matched_stems"],
            bas_classes=bas_classes,
            model_p_max=float(p_max),
            model_top_class=top_cls,
            model_top_p=top_p,
            agrees=p_max >= agree_threshold,
        ))
    return out
