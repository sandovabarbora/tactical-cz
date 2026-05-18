"""SoccerNet BAS PyTorch Dataset.

Schema we're consuming (verified against valid.zip on 2026-05-18):

    {split}/
      {league}/{season}/{date - home - away}/
        ├── 224p.mp4          ~250 MB per match, low-res for fast iter
        ├── 720p.mp4          ~1.8 GB per match, full quality
        └── Labels-ball.json  per-match annotations

Labels-ball.json structure (per BAS task convention):
    {
      "UrlLocal": "...",
      "annotations": [
        {"gameTime": "1 - 0:23", "label": "Pass", "position": "...", ...},
        ...
      ]
    }

This module is intentionally a SKELETON until SoccerNet NDA password
arrives (extraction is gated; see downloader.py). The class signature
is the load-bearing contract — once data exists on disk we just fill
in __getitem__.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class BASClip:
    """One annotated clip window from a SoccerNet match.

    Encodes (match_video_path, start_frame, end_frame, label) so the
    Dataset can lazily decode just the frames it needs instead of
    materializing whole-match tensors.
    """

    video_path: Path
    start_frame: int
    end_frame: int
    label: str          # event class string
    label_id: int       # mapped int for one-hot / cross-entropy
    match_id: str       # for split-aware sampling


def discover_matches(split_dir: Path) -> list[Path]:
    """Walk a {split} dir and return all match directories (containing 720p.mp4)."""
    if not split_dir.exists():
        return []
    matches = [p.parent for p in split_dir.rglob("Labels-ball.json")]
    logger.info("Discovered %d match dirs in %s", len(matches), split_dir)
    return matches


def load_match_annotations(match_dir: Path) -> dict:
    """Parse one Labels-ball.json. Returns the raw dict."""
    labels_path = match_dir / "Labels-ball.json"
    if not labels_path.exists():
        raise FileNotFoundError(labels_path)
    with labels_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def event_class_inventory(split_dir: Path) -> dict[str, int]:
    """Frequency table of event labels across all matches in a split.

    Helps decide which classes to train on vs collapse (some events
    are rare enough that BAS leaderboard reports just a subset).
    """
    from collections import Counter
    counter: Counter = Counter()
    for match_dir in discover_matches(split_dir):
        anns = load_match_annotations(match_dir).get("annotations", [])
        for a in anns:
            counter[a.get("label", "<unknown>")] += 1
    return dict(counter.most_common())


class SoccerNetBASDataset:
    """PyTorch Dataset over BAS clip windows.

    Skeleton until extraction lands. Interface:
        ds = SoccerNetBASDataset(split_dir, clip_seconds=4.0, fps=25,
                                  resolution="224p")
        len(ds)  → number of clip windows across all matches in split
        ds[i]    → (frames: Tensor[T, 3, H, W], label_id: int)
    """

    def __init__(
        self,
        split_dir: Path | str,
        clip_seconds: float = 4.0,
        fps: int = 25,
        resolution: str = "224p",
    ) -> None:
        self.split_dir = Path(split_dir)
        self.clip_seconds = clip_seconds
        self.fps = fps
        self.resolution = resolution
        self.matches = discover_matches(self.split_dir)
        if not self.matches:
            logger.warning(
                "No matches found in %s. Either the split is not yet extracted "
                "(see downloader.extract_bas_split) or the path is wrong.",
                self.split_dir,
            )

    def __len__(self) -> int:
        raise NotImplementedError("Wait for BAS extraction, then index annotations into clip windows.")

    def __getitem__(self, idx: int):
        raise NotImplementedError("Implement after extraction; reads frames via decord or PyAV.")
