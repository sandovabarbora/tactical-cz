"""Smoke tests for events.train + events.infer CLI scaffolds.

These don't actually train or run inference (those need data + GPU).
They verify the modules import, the CLI parsers accept expected args,
and the config dataclasses round-trip correctly. That's enough to
catch the everyday breakage (rename of a kwarg, dropped CLI flag).
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_train_module_imports() -> None:
    from tactical_cz.events import train
    assert hasattr(train, "TrainConfig")
    assert hasattr(train, "build_module")
    assert hasattr(train, "build_dataloaders")
    assert hasattr(train, "train")
    assert hasattr(train, "main")


def test_train_config_defaults_sensible() -> None:
    from tactical_cz.events.train import TrainConfig
    cfg = TrainConfig()
    assert cfg.epochs >= 1
    assert cfg.batch_size >= 1
    assert 0 < cfg.lr < 1
    assert 0 <= cfg.weight_decay < 1
    assert 0 < cfg.warmup_fraction < 1
    assert cfg.backbone.startswith("facebook/vjepa2")
    assert cfg.freeze_backbone is True   # Phase 2 default; flip in Phase 3


def test_train_cli_parses_required_flags() -> None:
    from tactical_cz.events.train import main
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_train_cli_overrides() -> None:
    from tactical_cz.events.train import TrainConfig
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(["--epochs", "3", "--batch-size", "2"])
    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size)
    assert cfg.epochs == 3
    assert cfg.batch_size == 2


def test_infer_module_imports() -> None:
    from tactical_cz.events import infer
    assert hasattr(infer, "InferConfig")
    assert hasattr(infer, "run_inference")
    assert hasattr(infer, "main")


def test_infer_config_defaults() -> None:
    from tactical_cz.events.infer import InferConfig
    cfg = InferConfig(source=Path("/nonexistent.mp4"))
    assert cfg.clip_seconds > 0
    assert cfg.stride_seconds > 0
    assert 0 < cfg.threshold < 1
    assert cfg.resize >= 64


def test_infer_cli_help() -> None:
    from tactical_cz.events.infer import main
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_infer_cli_requires_source() -> None:
    from tactical_cz.events.infer import main
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code != 0
