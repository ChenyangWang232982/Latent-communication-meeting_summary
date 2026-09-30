"""Run Prefix-KV + StateBridge communication-cost tests without content output.

This wrapper keeps the complete specialist pipeline, including optional
backfill rounds, but asks ``run_prefix_cache`` to write only per-sample and
aggregate natural-language versus latent communication counts.
"""

from __future__ import annotations

import sys

from .run_prefix_cache import main


if __name__ == "__main__":
    if "--communication-metrics-only" not in sys.argv:
        sys.argv.append("--communication-metrics-only")
    main()
