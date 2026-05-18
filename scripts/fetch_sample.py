"""Download a Roboflow sports sample broadcast clip into data/raw/.

These are short football TV clips (~30-60 seconds at 25 fps, 720p)
that Roboflow publishes alongside their `sports` notebooks. Useful
for smoke-testing the vision pipeline before pointing it at real
Czech-league broadcast material.

Usage:
    uv run python scripts/fetch_sample.py            # default clip
    uv run python scripts/fetch_sample.py --key 121364_0
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from tactical_cz.config import (
    RAW_DIR,
    ROBOFLOW_SPORTS_SAMPLES,
    configure_logging,
    ensure_dirs,
)

logger = logging.getLogger(__name__)


def fetch_sample(key: str = "121364_0") -> Path:
    """Download (or cache) one Roboflow sample clip. Returns the local path."""
    if key not in ROBOFLOW_SPORTS_SAMPLES:
        raise KeyError(
            f"Unknown sample key {key!r}. "
            f"Available: {sorted(ROBOFLOW_SPORTS_SAMPLES)}"
        )
    ensure_dirs()
    target = RAW_DIR / f"{key}.mp4"
    if target.exists():
        size_mb = target.stat().st_size / (1024 * 1024)
        logger.info("Cached: %s (%.1f MB)", target, size_mb)
        return target

    gdrive_id = ROBOFLOW_SPORTS_SAMPLES[key]
    logger.info("Downloading sample %s (gdrive id=%s) → %s", key, gdrive_id, target)
    subprocess.run(
        ["uv", "run", "gdown", "-O", str(target),
         f"https://drive.google.com/uc?id={gdrive_id}"],
        check=True,
    )
    size_mb = target.stat().st_size / (1024 * 1024)
    logger.info("Cached: %s (%.1f MB)", target, size_mb)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--key", default="121364_0",
        choices=list(ROBOFLOW_SPORTS_SAMPLES),
        help="Which Roboflow sample to fetch.",
    )
    args = parser.parse_args(argv)
    configure_logging()
    path = fetch_sample(args.key)
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
