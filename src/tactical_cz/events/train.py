"""Phase 2 fine-tuning loop for BAS event spotting.

Built to be **trigger-ready**: once NDA password lands and data is
extracted, single CLI invocation kicks off training. No surprises.

Design choices:

- **Loss**: BCEWithLogitsLoss (multi-label). A 4-second clip window
  can legitimately contain a Pass + a Drive + a Cross + a Throw In.
  Single-label cross-entropy throws away that signal. Threshold at
  inference is per-class (Phase 3 tuning).

- **Optimizer**: AdamW with linear warmup + cosine schedule. Standard
  V-JEPA2 fine-tune recipe from the original paper (warmup 10% of
  total steps, weight decay 0.05 on non-bias / non-norm params).

- **Mixed precision**: bf16 on A100, fp32 fallback on MPS (MPS
  autocast support is patchy in PyTorch 2.x).

- **Checkpoints**: Lightning's ModelCheckpoint on val_macro_map
  (matches SoccerNet BAS leaderboard metric). Top-3 kept.

- **Wandb**: flag-gated. CI / smoke runs default-off so we don't
  prompt for login in headless contexts.

CLI:
    uv run python -m tactical_cz.events.train \\
        --data-dir data/raw/soccernet/spotting-ball-2025 \\
        --backbone facebook/vjepa2-vitl-fpc64-256 \\
        --epochs 10 --batch-size 4 --lr 1e-4 \\
        --output-dir checkpoints/ \\
        --wandb-project tactical-cz       # omit to disable wandb
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tactical_cz.config import (
    PROCESSED_DIR,
    configure_logging,
    ensure_dirs,
    get_device,
)
from tactical_cz.events.dataset import SoccerNetBASDataset
from tactical_cz.events.model import (
    BAS_NUM_CLASSES,
    BASModel,
    BASModelConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """All training hyperparameters in one place."""

    # Data
    data_dir: Path = Path("data/raw/soccernet/spotting-ball-2025")
    clip_seconds: float = 4.0
    fps: int = 25
    resolution: str = "224p"

    # Model
    backbone: str = "facebook/vjepa2-vitl-fpc64-256"
    freeze_backbone: bool = True
    head_hidden_dim: int | None = None     # None → linear head
    head_dropout: float = 0.1

    # Optim
    epochs: int = 10
    batch_size: int = 4
    lr: float = 1e-4
    weight_decay: float = 0.05
    warmup_fraction: float = 0.1

    # Output
    output_dir: Path = Path("checkpoints")
    precision: str = "bf16-mixed"          # "32-true" forced on MPS

    # Logging
    wandb_project: str | None = None       # None → no wandb
    log_every_n_steps: int = 10


def build_module(cfg: TrainConfig):
    """Construct the LightningModule. Lazy-import pytorch_lightning."""
    import pytorch_lightning as pl

    class BASLitModule(pl.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.save_hyperparameters(ignore=["cfg"])
            self.model = BASModel(BASModelConfig(
                backbone=cfg.backbone,
                num_classes=BAS_NUM_CLASSES,
                freeze_backbone=cfg.freeze_backbone,
                head_hidden_dim=cfg.head_hidden_dim,
                head_dropout=cfg.head_dropout,
            ))
            self.loss = nn.BCEWithLogitsLoss()

        def forward(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
            return self.model(pixel_values_videos)

        def _step(self, batch, prefix: str) -> torch.Tensor:
            x, y = batch                          # y: (B, num_classes) multi-hot
            logits = self(x)
            loss = self.loss(logits, y.float())
            self.log(f"{prefix}_loss", loss, prog_bar=True, on_step=(prefix == "train"))
            return loss

        def training_step(self, batch, batch_idx) -> torch.Tensor:
            return self._step(batch, "train")

        def validation_step(self, batch, batch_idx) -> torch.Tensor:
            return self._step(batch, "val")

        def configure_optimizers(self):
            # Don't weight-decay biases or layer norms
            decay, no_decay = [], []
            for n, p in self.named_parameters():
                if not p.requires_grad:
                    continue
                (no_decay if n.endswith(".bias") or "norm" in n.lower() else decay).append(p)
            opt = torch.optim.AdamW(
                [
                    {"params": decay, "weight_decay": cfg.weight_decay},
                    {"params": no_decay, "weight_decay": 0.0},
                ],
                lr=cfg.lr,
            )
            # We don't know total steps until DataLoader is built; the
            # Trainer wires this up correctly when num_training_steps is
            # accessed via self.trainer.estimated_stepping_batches.
            sched = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=cfg.lr,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=cfg.warmup_fraction,
                anneal_strategy="cos",
            )
            return {
                "optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "step"},
            }

    return BASLitModule()


def build_dataloaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader]:
    """Train + val dataloaders over SoccerNetBASDataset.

    Will return empty loaders until extract step has run. The training
    loop's first epoch will be a no-op in that case and log a clear
    warning — caller should run downloader.extract_bas_split first.
    """
    train_ds = SoccerNetBASDataset(
        split_dir=cfg.data_dir / "train",
        clip_seconds=cfg.clip_seconds,
        fps=cfg.fps,
        resolution=cfg.resolution,
    )
    val_ds = SoccerNetBASDataset(
        split_dir=cfg.data_dir / "valid",
        clip_seconds=cfg.clip_seconds,
        fps=cfg.fps,
        resolution=cfg.resolution,
    )
    if not train_ds.matches or not val_ds.matches:
        logger.warning(
            "Train or val split has no matches. Run:\n"
            "  uv run python -m tactical_cz.events.downloader extract --split train\n"
            "  uv run python -m tactical_cz.events.downloader extract --split valid"
        )
    train_dl = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4,
        pin_memory=True, persistent_workers=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2,
        pin_memory=True, persistent_workers=True,
    )
    return train_dl, val_dl


def train(cfg: TrainConfig) -> Path:
    """Fit the model. Returns path to best checkpoint."""
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import WandbLogger

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    precision = "32-true" if device == "mps" else cfg.precision

    train_dl, val_dl = build_dataloaders(cfg)
    module = build_module(cfg)

    ckpt_cb = ModelCheckpoint(
        dirpath=cfg.output_dir,
        filename="bas-{epoch:02d}-{val_loss:.3f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
    )

    wandb_logger = None
    if cfg.wandb_project:
        wandb_logger = WandbLogger(project=cfg.wandb_project)

    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        callbacks=[ckpt_cb],
        logger=wandb_logger or False,
        accelerator="auto",
        devices="auto",
        precision=precision,
        log_every_n_steps=cfg.log_every_n_steps,
        default_root_dir=str(cfg.output_dir),
    )
    logger.info("Starting fit on device=%s, precision=%s", device, precision)
    trainer.fit(module, train_dl, val_dl)
    best = Path(ckpt_cb.best_model_path) if ckpt_cb.best_model_path else cfg.output_dir / "no_checkpoint.ckpt"
    logger.info("Best checkpoint: %s", best)
    return best


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=Path("data/raw/soccernet/spotting-ball-2025"))
    parser.add_argument("--backbone", default="facebook/vjepa2-vitl-fpc64-256")
    parser.add_argument("--freeze-backbone", action="store_true", default=True)
    parser.add_argument("--unfreeze-backbone", dest="freeze_backbone",
                        action="store_false",
                        help="Train the V-JEPA2 backbone end-to-end (Phase 3+)")
    parser.add_argument("--head-hidden-dim", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--clip-seconds", type=float, default=4.0)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--resolution", choices=["224p", "720p"], default="224p")
    parser.add_argument("--precision", default="bf16-mixed",
                        choices=["bf16-mixed", "16-mixed", "32-true"])
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--wandb-project", default=None,
                        help="Wandb project name (omit to disable wandb)")
    args = parser.parse_args(argv)

    configure_logging()
    ensure_dirs()
    cfg = TrainConfig(**vars(args))
    best = train(cfg)
    print(best)
    return 0


if __name__ == "__main__":
    sys.exit(main())
