"""Phase 2 — Events layer.

Action spotting on broadcast video. See docs/PHASE2_KICKOFF.md for the
external prerequisites (SoccerNet NDA, RunPod) and the implementation
sequence (downloader → dataset → model → train → infer).

This package is intentionally empty at end of Phase 1. The CLI shim below
fails fast with a pointer so users hitting `make events` before the
prerequisites are met get a helpful message instead of an ImportError.
"""

from __future__ import annotations

import sys


def main() -> int:
    msg = (
        "tactical-cz events module is not implemented yet.\n"
        "This is the Phase 2 entry point. Before running, complete the\n"
        "external prerequisites in docs/PHASE2_KICKOFF.md:\n"
        "  1. Request SoccerNet NDA access at https://www.soccer-net.org/data\n"
        "  2. Create a RunPod account + A100 instance\n"
        "  3. Verify MatchVision / SoccerMaster weights on HuggingFace\n"
    )
    print(msg, file=sys.stderr)
    return 2
