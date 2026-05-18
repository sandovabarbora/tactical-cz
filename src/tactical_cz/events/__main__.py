"""Enable `python -m tactical_cz.events` — runs the stub CLI shim."""

from __future__ import annotations

import sys

from tactical_cz.events import main

if __name__ == "__main__":
    sys.exit(main())
