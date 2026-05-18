"""Offline architecture test for events.model.

We don't download V-JEPA2-L (326M params, ~650 MB). Instead we
construct a tiny VJEPA2Model from a hand-rolled config and monkey-patch
the from_pretrained call so the wrapper's plumbing — pooling, head,
forward shape contract — is exercised end-to-end without network.
"""

from __future__ import annotations

import pytest
import torch

from tactical_cz.events.model import (
    BAS_CLASSES,
    BAS_NUM_CLASSES,
    BASModel,
    BASModelConfig,
)


@pytest.fixture
def tiny_vjepa2(monkeypatch):
    """Replace VJEPA2Model.from_pretrained with a tiny random-weights stand-in."""
    from transformers import VJEPA2Config, VJEPA2Model

    cfg = VJEPA2Config(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        crop_size=64,        # smaller frame
        frames_per_clip=8,
    )
    tiny = VJEPA2Model(cfg)
    monkeypatch.setattr(
        "transformers.VJEPA2Model.from_pretrained",
        classmethod(lambda cls, *args, **kwargs: tiny),
    )
    return tiny


def test_bas_classes_consistent() -> None:
    assert len(BAS_CLASSES) == BAS_NUM_CLASSES == 12
    assert BAS_CLASSES[0] == "Pass"
    assert BAS_CLASSES[-1] == "Goal"
    # No duplicates
    assert len(set(BAS_CLASSES)) == BAS_NUM_CLASSES


def test_bas_model_linear_head_shape(tiny_vjepa2) -> None:
    model = BASModel(BASModelConfig(
        backbone="ignored-by-monkeypatch",
        num_classes=12,
        head_hidden_dim=None,        # linear head
    ))
    B, T, C, H, W = 2, 8, 3, 64, 64
    x = torch.randn(B, T, C, H, W)
    logits = model(x)
    assert logits.shape == (B, 12), f"expected (B, 12), got {logits.shape}"


def test_bas_model_mlp_head_shape(tiny_vjepa2) -> None:
    model = BASModel(BASModelConfig(
        backbone="ignored-by-monkeypatch",
        num_classes=12,
        head_hidden_dim=32,           # MLP head
        head_dropout=0.0,
    ))
    x = torch.randn(1, 8, 3, 64, 64)
    logits = model(x)
    assert logits.shape == (1, 12)


def test_bas_model_backbone_frozen_by_default(tiny_vjepa2) -> None:
    model = BASModel(BASModelConfig(backbone="ignored"))
    backbone_grads = [p.requires_grad for p in model.backbone.parameters()]
    head_grads = [p.requires_grad for p in model.head.parameters()]
    assert not any(backbone_grads), "backbone should be frozen by default"
    assert all(head_grads), "head must be trainable"


def test_bas_model_backbone_unfreezable(tiny_vjepa2) -> None:
    model = BASModel(BASModelConfig(backbone="ignored", freeze_backbone=False))
    assert all(p.requires_grad for p in model.backbone.parameters())
