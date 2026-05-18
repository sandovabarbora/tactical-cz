"""SoccerNet BAS PyTorch Dataset stub.

Phase 2 implementation. Wraps the SoccerNet BAS clip directory into a
torch.utils.data.Dataset that yields (frames_tensor, label_tensor) pairs
suitable for fine-tuning a video classifier or action-spotting head.
"""

from __future__ import annotations


class SoccerNetBASDataset:
    """Stub. Implemented in Phase 2 after NDA + GPU access.

    Expected interface:
        ds = SoccerNetBASDataset(root, split="train", clip_seconds=4.0, fps=25)
        len(ds) → ~50k clips
        ds[i] → (frames: Tensor[T, 3, H, W], labels: Tensor[num_classes])
    """

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "SoccerNetBASDataset stub. See docs/PHASE2_KICKOFF.md."
        )
