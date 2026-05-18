"""Action-spotting model wrapper stub.

Phase 2 implementation. Either wraps MatchVision / SoccerMaster from HF
(if weights released), or a V-JEPA2 / VideoMAE2 backbone + custom BAS head.
Decision gate documented in docs/PHASE2_KICKOFF.md § 3.
"""

from __future__ import annotations


def build_model(backbone: str = "auto", num_classes: int = 12):
    """Stub. Implemented in Phase 2.

    backbone ∈ {"auto", "matchvision", "soccermaster", "vjepa2", "videomae2"}.
    "auto" picks the best available open-weights option at runtime.
    """
    raise NotImplementedError(
        "build_model stub. See docs/PHASE2_KICKOFF.md § 3 for the decision gate."
    )
