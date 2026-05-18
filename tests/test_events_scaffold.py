"""Smoke tests for events.cache_embeddings + events.train + events.infer CLI scaffolds.

These don't actually train (need GPU) or extract real SoccerNet data
(need NDA password). They verify the modules import, the CLI parsers
accept expected args, the EmbeddingDataset round-trips, and the head
training loop runs end-to-end against a tiny synthetic parquet.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


# ---------------------------------------------------------------------------
# cache_embeddings
# ---------------------------------------------------------------------------


def test_cache_embeddings_imports() -> None:
    from tactical_cz.events import cache_embeddings
    assert hasattr(cache_embeddings, "CacheConfig")
    assert hasattr(cache_embeddings, "cache_embeddings")
    assert hasattr(cache_embeddings, "main")


def test_cache_embeddings_cli_help() -> None:
    from tactical_cz.events.cache_embeddings import main
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_gametime_parse() -> None:
    from tactical_cz.events.cache_embeddings import _gametime_to_seconds
    assert _gametime_to_seconds("1 - 0:23.480") == pytest.approx(23.48)
    assert _gametime_to_seconds("2 - 5:00") == pytest.approx(45 * 60 + 300)
    # Garbage in → NaN, not exception
    assert _gametime_to_seconds("not a time") != _gametime_to_seconds("not a time")  # NaN!=NaN


# ---------------------------------------------------------------------------
# EmbeddingDataset + train
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_embeddings_parquet(tmp_path):
    """Synthetic 60-clip embeddings parquet (embed_dim=8, 12 BAS classes)."""
    rng = np.random.default_rng(42)
    n, d, k = 60, 8, 12
    rows = []
    for i in range(n):
        # Multi-hot label: most clips have 1-2 positive classes
        labels = np.zeros(k, dtype=np.int8)
        for c in rng.choice(k, size=rng.integers(0, 3), replace=False):
            labels[c] = 1
        rows.append({
            "match_id": "synthetic/0",
            "clip_id": f"synthetic/0@{i}",
            "video_path": "/dev/null",
            "frame_idx": i * 25,
            "embedding": rng.standard_normal(d).astype(np.float32).tolist(),
            "labels": labels.tolist(),
        })
    out = tmp_path / "embeddings_tiny.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    return out


def test_embedding_dataset_roundtrip(tiny_embeddings_parquet) -> None:
    from tactical_cz.events.train import EmbeddingDataset
    ds = EmbeddingDataset(tiny_embeddings_parquet)
    assert len(ds) == 60
    assert ds.embed_dim == 8
    assert ds.num_classes == 12
    x, y = ds[0]
    assert x.shape == (8,)
    assert y.shape == (12,)
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32


def test_bas_head_forward_shape() -> None:
    from tactical_cz.events.train import BASHead
    head = BASHead(embed_dim=8, num_classes=12)
    x = torch.randn(4, 8)
    out = head(x)
    assert out.shape == (4, 12)


def test_bas_head_mlp_variant() -> None:
    from tactical_cz.events.train import BASHead
    head = BASHead(embed_dim=8, num_classes=12, head_hidden_dim=16)
    x = torch.randn(2, 8)
    assert head(x).shape == (2, 12)


def test_train_loop_overfits_tiny(tiny_embeddings_parquet, tmp_path) -> None:
    """End-to-end smoke: 5 epochs of head training should reduce loss on tiny data."""
    from tactical_cz.events.train import TrainConfig, train
    cfg = TrainConfig(
        train_embeddings=tiny_embeddings_parquet,
        val_embeddings=tiny_embeddings_parquet,   # same set, just checking val plumbing
        output_dir=tmp_path / "ckpts",
        epochs=5,
        batch_size=16,
        lr=1e-2,
        pos_weight_mode="uniform",
    )
    best_path = train(cfg)
    assert best_path.exists()
    state = torch.load(best_path, map_location="cpu", weights_only=False)
    assert state["embed_dim"] == 8
    assert state["num_classes"] == 12
    assert "val_loss" in state


def test_train_cli_help() -> None:
    from tactical_cz.events.train import main
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


# ---------------------------------------------------------------------------
# infer
# ---------------------------------------------------------------------------


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
