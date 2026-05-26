#!/usr/bin/env python3
"""Run B6: 500 probe latents × 4 variants × PCA / UMAP / t-SNE (no training data)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
LE_PROBE_ROOT = CURRENT_FILE.parents[2]
VIZ_DIR = LE_PROBE_ROOT / "workspace_visualization"
if str(LE_PROBE_ROOT) not in sys.path:
    sys.path.insert(0, str(LE_PROBE_ROOT))

from interpretability.manifold.visualize_workspace_probe_latents import (
    visualize_probe_latents,
)

PROBE_DIR = LE_PROBE_ROOT / "datasets/workspace_probe_grasp"

VARIANTS = [
    ("single-view", PROBE_DIR / "workspace_probe_latents_singleview.pt", "singleview"),
    ("multi-view", PROBE_DIR / "workspace_probe_latents_multiview.pt", "multiview"),
    (
        "multi-view + skeleton",
        PROBE_DIR / "workspace_probe_latents_multiview_skeleton.pt",
        "multiview_skeleton",
    ),
    (
        "multi-view + skeleton + DINO",
        PROBE_DIR / "workspace_probe_latents_multiview_skeleton_dino.pt",
        "multiview_skeleton_dino",
    ),
]

DEFAULT_METHODS = ("umap", "tsne", "pca")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["pca", "umap", "tsne"],
        default=list(DEFAULT_METHODS),
    )
    parser.add_argument("--out-dir", type=str, default=str(VIZ_DIR))
    parser.add_argument(
        "--monochrome",
        action="store_true",
        help="Gray single-color plots (geometry only); writes *_geometry.* files",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_metrics: dict = {}

    for method in args.methods:
        method_metrics: dict = {}
        for label, probes, stem in VARIANTS:
            if not probes.exists():
                raise FileNotFoundError(f"Probe latents missing: {probes}")

            suffix = "_geometry" if args.monochrome else ""
            stem_file = f"workspace_probe_latent_{stem}_{method}{suffix}"
            out_png = out_dir / f"{stem_file}.png"
            out_html = out_dir / f"{stem_file}.html"
            print(
                f"\n=== {label} | {method.upper()}{' [mono]' if args.monochrome else ''} ==="
            )
            metrics = visualize_probe_latents(
                probes,
                method=method,
                out_png=out_png,
                out_html=out_html,
                variant_label=label,
                monochrome=args.monochrome,
            )
            (out_dir / f"{stem_file}.metrics.json").write_text(
                json.dumps(metrics, indent=2)
            )
            method_metrics[stem] = metrics
            print(f"✅ {out_png.name} + {out_html.name} (n={metrics['n_probes']})")

        all_metrics[method] = method_metrics
        (out_dir / f"workspace_probe_latent_all_{method}.json").write_text(
            json.dumps(method_metrics, indent=2)
        )

    (out_dir / "workspace_probe_latent_all_methods.json").write_text(
        json.dumps(all_metrics, indent=2)
    )
    print(f"\n✅ Done → {out_dir}")


if __name__ == "__main__":
    main()
