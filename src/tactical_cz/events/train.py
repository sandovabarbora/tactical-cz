"""Phase 2 head training over cached V-JEPA2 embeddings.

The 2026-05-18 refactor decoupled V-JEPA2 forward from head training:
``cache_embeddings.py`` runs the backbone once over a split and dumps
(embedding, labels) rows to parquet. This module trains the small head
on top, which means:

    * Seconds per epoch (BCE over (B, 1024) → (B, 12), no cv2,
      no GPU memory for backbone activations).
    * Iterating head designs (linear vs MLP, dropout, head_hidden_dim)
      is cheap — no re-encoding the dataset.
    * Whole training runs on laptop CPU (M-series MPS optional).

CLI:
    # 1. One-time embedding cache (slow, but only once per split)
    uv run python -m tactical_cz.events.cache_embeddings --split valid

    # 2. Fast head training (seconds per epoch)
    uv run python -m tactical_cz.events.train \\
        --train-embeddings data/processed/embeddings_train.parquet \\
        --val-embeddings   data/processed/embeddings_valid.parquet \\
        --epochs 30 --batch-size 256 --lr 3e-3

If you want end-to-end backbone fine-tuning (Phase 3 LoRA territory),
see the BASModel wrapper in events.model — pass it into your own
Lightning loop with cv2 in the hot path. That is intentionally NOT
this file's concern.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from tactical_cz.config import configure_logging, ensure_dirs, get_device
from tactical_cz.events.model import BAS_NUM_CLASSES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class EmbeddingDataset(Dataset):
    """Returns (embedding_tensor[D], labels_tensor[num_classes]) per index.

    Loads the entire parquet into RAM on construction; embeddings parquet
    for one BAS split is on the order of hundreds of MB at most.
    """

    def __init__(self, parquet_path: Path | str) -> None:
        self.path = Path(parquet_path)
        df = pd.read_parquet(self.path)
        if df.empty:
            raise ValueError(f"Empty embeddings parquet: {self.path}")
        # Eager-stack into one ndarray each so __getitem__ is O(1)
        self.embeddings = np.stack(df["embedding"].apply(np.asarray).to_numpy()).astype(np.float32)
        self.labels = np.stack(df["labels"].apply(np.asarray).to_numpy()).astype(np.float32)
        logger.info(
            "Loaded %d clips from %s (embed_dim=%d, num_classes=%d)",
            len(df), self.path, self.embeddings.shape[1], self.labels.shape[1],
        )

    @property
    def embed_dim(self) -> int:
        return int(self.embeddings.shape[1])

    @property
    def num_classes(self) -> int:
        return int(self.labels.shape[1])

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.embeddings[idx]),
            torch.from_numpy(self.labels[idx]),
        )


# ---------------------------------------------------------------------------
# Head
# ---------------------------------------------------------------------------


class BASHead(nn.Module):
    """The same head BASModel uses, lifted out of the V-JEPA2 wrapper.

    Linear or 1-layer MLP from embed_dim → num_classes. Lives standalone
    so train.py can fit it without ever instantiating the backbone.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int = BAS_NUM_CLASSES,
        head_hidden_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if head_hidden_dim is None:
            self.net = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(embed_dim, num_classes),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(embed_dim, head_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden_dim, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    train_embeddings: Path
    val_embeddings: Path | None = None
    output_dir: Path = Path("checkpoints")

    epochs: int = 30
    batch_size: int = 256
    lr: float = 3e-3
    weight_decay: float = 0.01

    head_hidden_dim: int | None = None
    head_dropout: float = 0.1

    # Per-class positive weight for BCE. None → uniform 1.0; "auto" →
    # inverse class frequency to handle rare events (Goal vs Pass).
    pos_weight_mode: str = "auto"          # "auto" | "uniform"


def _compute_pos_weight(labels: np.ndarray, mode: str) -> torch.Tensor | None:
    if mode != "auto":
        return None
    n = labels.shape[0]
    pos = labels.sum(axis=0)
    neg = n - pos
    # Avoid div-by-zero for classes never seen in this split
    pos_weight = np.where(pos > 0, neg / np.maximum(pos, 1), 1.0)
    return torch.from_numpy(pos_weight.astype(np.float32))


def train(cfg: TrainConfig) -> Path:
    """Fit the head over cached embeddings. Returns path to best checkpoint."""
    device = get_device()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = EmbeddingDataset(cfg.train_embeddings)
    val_ds = EmbeddingDataset(cfg.val_embeddings) if cfg.val_embeddings else None

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False) if val_ds else None

    head = BASHead(
        embed_dim=train_ds.embed_dim,
        num_classes=train_ds.num_classes,
        head_hidden_dim=cfg.head_hidden_dim,
        dropout=cfg.head_dropout,
    ).to(device)

    pos_weight = _compute_pos_weight(train_ds.labels, cfg.pos_weight_mode)
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)
        logger.info("pos_weight (inverse class freq): %s", pos_weight.cpu().numpy().round(2))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    best_val = float("inf")
    best_path = cfg.output_dir / "head_best.pt"
    for epoch in range(1, cfg.epochs + 1):
        head.train()
        train_losses = []
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            logits = head(x)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(loss.item())
        sched.step()
        tl = float(np.mean(train_losses))

        if val_dl is not None:
            head.train(False)
            val_losses = []
            with torch.no_grad():
                for x, y in val_dl:
                    x, y = x.to(device), y.to(device)
                    val_losses.append(loss_fn(head(x), y).item())
            vl = float(np.mean(val_losses))
            logger.info("epoch %3d  train=%.4f  val=%.4f  lr=%.2e",
                        epoch, tl, vl, sched.get_last_lr()[0])
            if vl < best_val:
                best_val = vl
                torch.save({
                    "epoch": epoch,
                    "state_dict": head.state_dict(),
                    "embed_dim": train_ds.embed_dim,
                    "num_classes": train_ds.num_classes,
                    "val_loss": vl,
                }, best_path)
        else:
            logger.info("epoch %3d  train=%.4f", epoch, tl)
            torch.save({
                "epoch": epoch,
                "state_dict": head.state_dict(),
                "embed_dim": train_ds.embed_dim,
                "num_classes": train_ds.num_classes,
            }, best_path)

    logger.info("Best head: %s (val_loss=%.4f)", best_path, best_val)
    return best_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-embeddings", type=Path, required=True)
    parser.add_argument("--val-embeddings", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--head-hidden-dim", type=int, default=None)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--pos-weight", choices=["auto", "uniform"], default="auto")
    args = parser.parse_args(argv)

    configure_logging()
    ensure_dirs()
    cfg = TrainConfig(
        train_embeddings=args.train_embeddings,
        val_embeddings=args.val_embeddings,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        head_hidden_dim=args.head_hidden_dim,
        head_dropout=args.head_dropout,
        pos_weight_mode=args.pos_weight,
    )
    best = train(cfg)
    print(best)
    return 0


if __name__ == "__main__":
    sys.exit(main())
