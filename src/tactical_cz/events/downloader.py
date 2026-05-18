"""SoccerNet Ball Action Spotting (BAS) 2025 downloader + AES extractor.

Reality check (verified 2026-05-18 against the HF dataset card):

Two-stage gate:
    1. Download: PUBLIC. The HF dataset ``SoccerNet/SN-BAS-2025`` is
       fetchable by anyone, no auth, no NDA. License: GPL-3.0.
    2. Extract: NDA-GATED. The zip contents (videos + labels) are
       AES-encrypted with a password obtained from the SoccerNet NDA
       Google Form: https://docs.google.com/forms/d/e/1FAIpQLSfYFqjZNm4IgwGnyJXDPk2Ko_lZcbVtYX73w5lf6din5nxfmA/viewform

The README hand-waves this with "videos from the original SoccerNet
dataset, which is password-protected". Concretely it means Python's
stdlib ``zipfile`` cannot extract these — they use AES (compress_type=99),
not ZipCrypto-Legacy. We use ``pyzipper`` for extraction.

Total dataset size: ~19 GB (train 8.5, valid 2.0, test 4.5, challenge 4.2).

Per-match layout inside a split's zip:
    {league}/{season}/{date - home - away}/
        ├── 224p.mp4         (low-res for fast iteration)
        ├── 720p.mp4         (full quality)
        └── Labels-ball.json (frame-level event annotations)

Typical usage:
    # 1. Fetch (no password needed)
    download_bas_2025(splits=("valid",))

    # 2. Extract (password from NDA Google Form)
    extract_bas_split("valid", password="<nda password>")
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from tactical_cz.config import RAW_DIR, configure_logging, ensure_dirs

logger = logging.getLogger(__name__)

SOCCERNET_DIR: Path = RAW_DIR / "soccernet"
BAS_TASK: str = "spotting-ball-2025"
VALID_SPLITS: tuple[str, ...] = ("train", "valid", "test", "challenge")
NDA_FORM_URL: str = (
    "https://docs.google.com/forms/d/e/"
    "1FAIpQLSfYFqjZNm4IgwGnyJXDPk2Ko_lZcbVtYX73w5lf6din5nxfmA/viewform"
)
# Env var to read NDA password from (preferred over CLI args so it never
# leaks into shell history or git diff).
NDA_PASSWORD_ENV: str = "SOCCERNET_PASSWORD"


def download_bas_2025(
    splits: tuple[str, ...] = ("train", "valid", "test"),
    local_dir: Path | None = None,
    verbose: bool = True,
) -> Path:
    """Download SoccerNet Ball Action Spotting 2025 splits from HuggingFace.

    No password required for this stage. Returns the directory containing
    the downloaded zip(s). Idempotent; SoccerNet's downloader skips files
    already present.
    """
    unknown = set(splits) - set(VALID_SPLITS)
    if unknown:
        raise ValueError(f"Unknown splits: {unknown}. Valid: {VALID_SPLITS}")

    local_dir = Path(local_dir) if local_dir else SOCCERNET_DIR
    local_dir.mkdir(parents=True, exist_ok=True)

    from SoccerNet.Downloader import SoccerNetDownloader

    d = SoccerNetDownloader(LocalDirectory=str(local_dir))
    logger.info(
        "Fetching BAS 2025 splits=%s → %s (public HF dataset SoccerNet/SN-BAS-2025)",
        splits, local_dir,
    )
    d.downloadDataTask(task=BAS_TASK, split=list(splits), verbose=verbose)
    task_dir = local_dir / BAS_TASK
    logger.info("Downloaded splits cached at %s", task_dir)
    return task_dir


def extract_bas_split(
    split: str,
    password: str | None = None,
    local_dir: Path | None = None,
) -> Path:
    """AES-decrypt one BAS split's zip into a sibling directory.

    Password resolution: explicit argument > SOCCERNET_PASSWORD env var.
    Get the password by filling the NDA Google Form (NDA_FORM_URL above).

    Skips files that already exist on disk so re-running is safe.
    Returns the extracted directory path.
    """
    if password is None:
        password = os.environ.get(NDA_PASSWORD_ENV)
    if not password:
        raise RuntimeError(
            f"No password provided. Set ${NDA_PASSWORD_ENV} or pass "
            f"password=..., obtained from the SoccerNet NDA form:\n  {NDA_FORM_URL}"
        )

    local_dir = Path(local_dir) if local_dir else SOCCERNET_DIR
    zip_path = local_dir / BAS_TASK / f"{split}.zip"
    if not zip_path.exists():
        raise FileNotFoundError(
            f"{zip_path} not found. Run download_bas_2025(splits=({split!r},)) first."
        )

    out_dir = local_dir / BAS_TASK / split
    out_dir.mkdir(parents=True, exist_ok=True)

    import pyzipper

    extracted = 0
    skipped = 0
    with pyzipper.AESZipFile(zip_path) as zf:
        zf.setpassword(password.encode("utf-8"))
        for info in zf.infolist():
            target = out_dir / info.filename
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.exists() and target.stat().st_size == info.file_size:
                skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as dst:
                # 8 MB chunks so a single mp4 doesn't pin all of RAM
                while True:
                    chunk = src.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
            extracted += 1
    logger.info(
        "Extracted %s/%s.zip: %d new files, %d already present → %s",
        BAS_TASK, split, extracted, skipped, out_dir,
    )
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_dl = sub.add_parser("download", help="Fetch zip from HF (no password)")
    p_dl.add_argument("--splits", nargs="+", default=["valid"],
                      choices=list(VALID_SPLITS))
    p_dl.add_argument("--local-dir", type=str, default=None)

    p_ex = sub.add_parser("extract", help="AES-decrypt a downloaded split")
    p_ex.add_argument("--split", required=True, choices=list(VALID_SPLITS))
    p_ex.add_argument("--password", type=str, default=None,
                      help=f"NDA password (or set ${NDA_PASSWORD_ENV} env var)")
    p_ex.add_argument("--local-dir", type=str, default=None)

    args = parser.parse_args(argv)
    configure_logging()
    ensure_dirs()

    if args.cmd == "download":
        out = download_bas_2025(
            splits=tuple(args.splits),
            local_dir=Path(args.local_dir) if args.local_dir else None,
        )
    else:
        out = extract_bas_split(
            split=args.split,
            password=args.password,
            local_dir=Path(args.local_dir) if args.local_dir else None,
        )
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
