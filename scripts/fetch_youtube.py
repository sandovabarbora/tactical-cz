"""Download a short broadcast clip via yt-dlp into data/raw/.

Use ONLY legitimately-published clips:
    * Official Fortuna Liga (@FortunaLigaCZ) YouTube channel highlights
    * Official club channels (Sparta, Slavia, Plzeň, etc.)
    * O2 TV Sport official clips
    * FAČR (Czech FA) channel

Avoid full match uploads from unofficial accounts (copyright risk +
project's reputation).

Usage:
    uv run python scripts/fetch_youtube.py URL [--name custom_slug] [--start 0:00] [--end 0:45]

The optional --start / --end let you grab a precise window inside a
longer highlight reel without downloading the whole video.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from tactical_cz.config import RAW_DIR, configure_logging, ensure_dirs

logger = logging.getLogger(__name__)


def fetch_clip(
    url: str,
    name: str | None = None,
    start: str | None = None,
    end: str | None = None,
    fmt: str = "best[height<=720]",
) -> Path:
    """Download a YouTube clip into data/raw/{name}.mp4. Returns path."""
    ensure_dirs()
    slug = name or url.rsplit("=", 1)[-1].rsplit("/", 1)[-1][:32]
    target = RAW_DIR / f"{slug}.mp4"

    if target.exists():
        size_mb = target.stat().st_size / (1024 * 1024)
        logger.info("Cached: %s (%.1f MB)", target, size_mb)
        return target

    cmd = ["uv", "run", "yt-dlp", "-f", fmt, "-o", str(target), url]
    if start and end:
        # yt-dlp's --download-sections needs ffmpeg, which our supervision
        # / opencv path already installs
        cmd.extend(["--download-sections", f"*{start}-{end}"])
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    size_mb = target.stat().st_size / (1024 * 1024)
    logger.info("Cached: %s (%.1f MB)", target, size_mb)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="YouTube URL (official channel only)")
    parser.add_argument("--name", default=None, help="Override output stem")
    parser.add_argument("--start", default=None, help="Start timestamp (e.g. 0:30)")
    parser.add_argument("--end", default=None, help="End timestamp (e.g. 1:15)")
    parser.add_argument("--format", default="best[height<=720]",
                        help="yt-dlp format selector")
    args = parser.parse_args(argv)
    configure_logging()
    path = fetch_clip(args.url, args.name, args.start, args.end, args.format)
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
