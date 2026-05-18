"""Fine-tuning loop stub (Lightning + wandb).

Phase 2 implementation. Runs on RunPod A100, logs to wandb, checkpoints
to /workspace/checkpoints/ on the pod, syncs final model to HF Hub.
"""

from __future__ import annotations


def train(config_path: str) -> None:
    """Stub. Implemented in Phase 2 after GPU access."""
    raise NotImplementedError(
        "train stub. Requires `uv sync --extra train` and GPU. See docs/PHASE2_KICKOFF.md § 2."
    )
