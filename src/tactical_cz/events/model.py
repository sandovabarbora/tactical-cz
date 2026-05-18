"""V-JEPA2 backbone + Ball Action Spotting classification head.

Architecture decision per PHASE2_KICKOFF.md § 3 decision gate:
MatchVision and SoccerMaster papers exist but weights were not released
on HuggingFace Hub (checked 2026-05-18). Fallback chosen:
``facebook/vjepa2-vitl-fpc64-256`` (326M params, MIT, 915K downloads,
197 likes on HF), wrapped behind a small classification head.

Default backbone is V-JEPA2-L. Swap to V-JEPA2-G (giant) via
``BASModelConfig(backbone="facebook/vjepa2-vitg-fpc64-256")`` once
RunPod GPU is online.

Backbone is FROZEN by default (we train just the head). Unfreezing is
Phase 3 (LoRA via peft for parameter-efficient fine-tuning).

Forward contract:
    pixel_values: (B, T, 3, H, W)   # T frames per clip, RGB, 256x256
    returns:      (B, num_classes)  # raw logits, apply softmax / sigmoid downstream
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Canonical SoccerNet Ball Action Spotting class list. Order matches the
# BAS 2024/2025 leaderboard so confusion matrices line up with published
# baselines. Populated against the Labels-ball.json schema once we have
# the NDA password and can verify; this list is the spec we plan against.
BAS_CLASSES: tuple[str, ...] = (
    "Pass",
    "Drive",
    "Header",
    "High Pass",
    "Out",
    "Cross",
    "Throw In",
    "Shot",
    "Ball Player Block",
    "Player Successful Tackle",
    "Free Kick",
    "Goal",
)
BAS_NUM_CLASSES: int = len(BAS_CLASSES)
BAS_LABEL_TO_ID: dict[str, int] = {name: i for i, name in enumerate(BAS_CLASSES)}


@dataclass
class BASModelConfig:
    """All knobs for the BAS model. Defaults target laptop + single A100."""

    backbone: str = "facebook/vjepa2-vitl-fpc64-256"
    num_classes: int = BAS_NUM_CLASSES
    freeze_backbone: bool = True
    head_dropout: float = 0.1
    pool: str = "mean"                    # "mean" | "cls" (V-JEPA2 has no CLS, mean is the right default)
    head_hidden_dim: int | None = None    # None → linear head; int → MLP with one hidden layer


class BASModel(nn.Module):
    """V-JEPA2 video encoder + classification head over BAS event classes.

    Two-stage forward:
        1. Backbone embeds (B, T, 3, H, W) → (B, num_tokens, hidden_dim)
           where num_tokens ≈ (T/2) * (H/16) * (W/16) per V-JEPA2's
           tubelet patch grid.
        2. Pool tokens → (B, hidden_dim), then head → (B, num_classes).

    We do NOT load V-JEPA2's video_classifier head (it's trained for
    different label sets — Kinetics-700, SSv2, etc.). The backbone's
    ``last_hidden_state`` is the right tap point for transfer.
    """

    def __init__(self, config: BASModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or BASModelConfig()

        # Lazy heavy import — keeps `from events.model import BASModel`
        # snappy and CI-friendly (no HF round-trip at import time).
        from transformers import VJEPA2Model

        logger.info(
            "Loading V-JEPA2 backbone: %s (freeze=%s)",
            self.config.backbone, self.config.freeze_backbone,
        )
        self.backbone = VJEPA2Model.from_pretrained(self.config.backbone)
        self.hidden_dim = self.backbone.config.hidden_size

        if self.config.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        if self.config.head_hidden_dim is None:
            self.head = nn.Sequential(
                nn.Dropout(self.config.head_dropout),
                nn.Linear(self.hidden_dim, self.config.num_classes),
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(self.hidden_dim, self.config.head_hidden_dim),
                nn.GELU(),
                nn.Dropout(self.config.head_dropout),
                nn.Linear(self.config.head_hidden_dim, self.config.num_classes),
            )

    def _pool(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, num_tokens, hidden_dim) → (B, hidden_dim)."""
        if self.config.pool == "mean":
            return tokens.mean(dim=1)
        if self.config.pool == "cls":
            return tokens[:, 0]
        raise ValueError(f"Unknown pool strategy: {self.config.pool!r}")

    def forward(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        """Run a batch of clips through backbone + head.

        Argument name mirrors V-JEPA2's own forward kwarg
        (``pixel_values_videos``) so callers can swap our wrapper in
        wherever a raw VJEPA2Model fits.

        Shape: (B, T, 3, H, W). V-JEPA2's processor expects T=64 frames
        at 256x256 for the default model; we don't enforce it here so
        callers can override (the model interpolates).
        """
        if self.config.freeze_backbone:
            with torch.no_grad():
                out = self.backbone(pixel_values_videos=pixel_values_videos)
        else:
            out = self.backbone(pixel_values_videos=pixel_values_videos)
        pooled = self._pool(out.last_hidden_state)
        return self.head(pooled)


def build_bas_model(
    backbone: str = "facebook/vjepa2-vitl-fpc64-256",
    num_classes: int = BAS_NUM_CLASSES,
    **kwargs,
) -> BASModel:
    """Factory wrapping the config dance. Used by train.py / infer.py."""
    return BASModel(BASModelConfig(
        backbone=backbone, num_classes=num_classes, **kwargs,
    ))
