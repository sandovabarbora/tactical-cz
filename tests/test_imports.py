"""Smoke: every top-level module imports without optional-heavy errors."""

from __future__ import annotations

import importlib

import pytest

MODULES = [
    "tactical_cz",
    "tactical_cz.config",
    "tactical_cz.vision",
    "tactical_cz.vision.detector",
    "tactical_cz.vision.tracker",
    "tactical_cz.vision.pitch",
    "tactical_cz.vision.team",
    "tactical_cz.vision.pipeline",
    "tactical_cz.events",
    "tactical_cz.audio",
    "tactical_cz.multimodal",
    "tactical_cz.agent",
    "tactical_cz.demo",
]


@pytest.mark.parametrize("modname", MODULES)
def test_import(modname: str) -> None:
    importlib.import_module(modname)
