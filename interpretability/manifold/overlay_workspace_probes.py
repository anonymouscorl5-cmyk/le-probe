#!/usr/bin/env python3
"""Deprecated alias — use ``visualize_workspace_probe_latents.py`` (500 probes only)."""

from __future__ import annotations

import warnings

warnings.warn(
    "overlay_workspace_probes.py is deprecated; use visualize_workspace_probe_latents.py",
    DeprecationWarning,
    stacklevel=1,
)

from interpretability.manifold.visualize_workspace_probe_latents import (  # noqa: E402
    embed_probes,
    main,
    run_overlay,
    visualize_probe_latents,
)

__all__ = ["embed_probes", "main", "run_overlay", "visualize_probe_latents"]

if __name__ == "__main__":
    main()
