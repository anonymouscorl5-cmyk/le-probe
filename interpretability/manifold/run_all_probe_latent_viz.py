#!/usr/bin/env python3
"""Run B6 latent-space probe visualizations (all variants × PCA / UMAP / t-SNE)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interpretability.manifold.run_all_probe_overlays import main

if __name__ == "__main__":
    main()
