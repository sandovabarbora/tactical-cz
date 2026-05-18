"""Central config: paths, device selection, model URLs, logging.

Imported by every module. No business logic lives here.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PACKAGE_ROOT: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = PACKAGE_ROOT.parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
CONFIGS_DIR: Path = PROJECT_ROOT / "configs"


def ensure_dirs() -> None:
    """Create on-disk scaffolding. Idempotent."""
    for d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------


def get_device() -> str:
    """Return best available torch device: cuda > mps > cpu.

    Apple Silicon → "mps" (M-series GPUs). NVIDIA → "cuda". Fallback "cpu".
    Override via TACTICAL_DEVICE env var (useful for forcing CPU on tests).
    """
    override = os.environ.get("TACTICAL_DEVICE")
    if override:
        return override
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Roboflow sports pretrained models
# ---------------------------------------------------------------------------
# These are YOLOv8 weights Roboflow released on their `sports` repo
# (https://github.com/roboflow/sports). We mirror to the user-side
# models/ directory on first use. ~50MB each, cached after first download.

# Google Drive file IDs published by Roboflow at
# https://github.com/roboflow/sports/blob/main/examples/soccer/setup.sh
# Stored as IDs (not direct URLs) because gdown takes the ID form natively.
ROBOFLOW_SPORTS_MODELS: dict[str, str] = {
    # Player + GK + referee + ball, 4-class football detector
    "player_detection": "17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q",
    # Ball-only detector, higher recall on small object
    "ball_detection":   "1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V",
    # 32-keypoint pitch detector (corners, centre marks, penalty spots)
    "pitch_detection":  "1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf",
}

# Sample broadcast clips, also published via Roboflow's setup.sh
ROBOFLOW_SPORTS_SAMPLES: dict[str, str] = {
    "0bfacc_0": "12TqauVZ9tLAv8kWxTTBFWtgt2hNQ4_ZF",
    "2e57b9_0": "19PGw55V8aA6GZu5-Aac5_9mCy3fNxmEf",
    "08fd33_0": "1OG8K6wqUw9t7lp9ms1M48DxRhwTYciK-",
    "573e61_0": "1yYPKuXbHsCxqjA9G-S6aeR2Kcnos8RPU",
    "121364_0": "1vVwjW1dE1drIdd4ZSILfbCGPD4weoNiu",
}

DEFAULT_PLAYER_MODEL: str = "player_detection"
DEFAULT_BALL_MODEL: str = "ball_detection"
DEFAULT_PITCH_MODEL: str = "pitch_detection"


def model_path(model_key: str) -> Path:
    """Local cache path for a pretrained model."""
    return MODELS_DIR / f"{model_key}.pt"


# ---------------------------------------------------------------------------
# Whisper config (Phase 3)
# ---------------------------------------------------------------------------

WHISPER_MODEL_DEFAULT: str = "large-v3"
WHISPER_LANGUAGE_CZ: str = "cs"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG_FORMAT: str = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

_logging_configured: bool = False


def configure_logging(level: int = logging.INFO, log_file: Path | None = None) -> None:
    """Set up root logging once per process."""
    global _logging_configured
    if _logging_configured:
        return

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    fmt = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
    stream = logging.StreamHandler(stream=sys.stdout)
    stream.setFormatter(fmt)
    root.addHandler(stream)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)

    for noisy in ("urllib3", "PIL.PngImagePlugin", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _logging_configured = True
