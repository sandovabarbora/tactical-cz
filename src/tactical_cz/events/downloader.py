"""SoccerNet Ball Action Spotting (BAS) downloader stub.

Implementation deferred until SoccerNet NDA is approved
(see docs/PHASE2_KICKOFF.md § 1).

When implemented, this module wraps SoccerNet.Downloader.SoccerNetDownloader
and caches the BAS 2025 train/valid/test splits into data/raw/soccernet/.
"""

from __future__ import annotations

from tactical_cz.config import RAW_DIR

SOCCERNET_DIR = RAW_DIR / "soccernet"


def download_bas_2025(password: str, splits: tuple[str, ...] = ("train", "valid", "test")) -> None:
    """Download SoccerNet Ball Action Spotting 2025 package.

    Requires NDA password from soccernet@uliege.be approval email.
    """
    raise NotImplementedError(
        "SoccerNet BAS downloader stub. See docs/PHASE2_KICKOFF.md § 1 for setup."
    )
