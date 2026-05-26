#!/usr/bin/env python3
"""B6: Overlay workspace probe latents on training manifold (fit reducer on training only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

try:
    import umap
except ImportError:
    umap = None

CURRENT_FILE = Path(__file__).resolve()
LE_PROBE_ROOT = CURRENT_FILE.parents[2]
if str(LE_PROBE_ROOT) not in sys.path:
    sys.path.insert(0, str(LE_PROBE_ROOT))

from dataset.task_workspace_probe.segments import SEGMENT_COLORS


def fit_reducer(method: str, train_latents: np.ndarray):
    method = method.lower()
    if method == "pca":
        r = PCA(n_components=3)
        r.fit(train_latents)
        return r
    if method == "umap":
        if umap is None:
            raise ImportError("umap-learn required for --method umap")
        r = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1)
        r.fit(train_latents)
        return r
    raise ValueError(f"Unsupported method: {method}")


def transform_reducer(reducer, x: np.ndarray) -> np.ndarray:
    if hasattr(reducer, "transform"):
        return reducer.transform(x)
    return reducer.transform(x)  # UMAP also has transform after fit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training",
        type=str,
        required=True,
        help="manifold_data.pt from harvest_manifold (training frames)",
    )
    parser.add_argument(
        "--probes",
        type=str,
        required=True,
        help="workspace_probe_latents_*.pt",
    )
    parser.add_argument("--method", choices=["pca", "umap"], default="pca")
    parser.add_argument(
        "--out",
        type=str,
        default=str(LE_PROBE_ROOT / "assets/workspace_probe_overlay.png"),
    )
    parser.add_argument(
        "--max_train_points",
        type=int,
        default=8000,
        help="Subsample training points for faster plots",
    )
    args = parser.parse_args()

    train = torch.load(args.training, map_location="cpu", weights_only=False)
    probe = torch.load(args.probes, map_location="cpu", weights_only=False)

    train_z = np.asarray(train["latents"], dtype=np.float64)
    probe_z = np.asarray(probe["latents"], dtype=np.float64)
    segments = probe.get("segment_hint") or ["unknown"] * len(probe_z)

    if len(train_z) > args.max_train_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(train_z), args.max_train_points, replace=False)
        train_z_fit = train_z[idx]
    else:
        train_z_fit = train_z

    reducer = fit_reducer(args.method, train_z_fit)
    train_3d = transform_reducer(reducer, train_z_fit)
    probe_3d = transform_reducer(reducer, probe_z)

    # Silhouette on probes only (need ≥2 segments with ≥2 points)
    seg_to_id = {k: i for i, k in enumerate(SEGMENT_COLORS)}
    labels = np.array([seg_to_id.get(s, -1) for s in segments])
    valid = labels >= 0
    sil = None
    if valid.sum() >= 10 and len(np.unique(labels[valid])) >= 2:
        try:
            sil = float(silhouette_score(probe_3d[valid], labels[valid]))
        except Exception:
            sil = None

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        train_3d[:, 0],
        train_3d[:, 1],
        train_3d[:, 2],
        c="#cccccc",
        s=1,
        alpha=0.15,
        label="training",
    )

    for seg in sorted(set(segments)):
        mask = np.array([s == seg for s in segments])
        if not mask.any():
            continue
        ax.scatter(
            probe_3d[mask, 0],
            probe_3d[mask, 1],
            probe_3d[mask, 2],
            c=SEGMENT_COLORS.get(seg, "#333"),
            s=40,
            alpha=0.9,
            label=f"probe:{seg}",
            edgecolors="black",
            linewidths=0.3,
        )

    title = f"Workspace probes on training {args.method.upper()}"
    if sil is not None:
        title += f" (probe silhouette={sil:.3f})"
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=7)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    metrics_path = out.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(
            {
                "method": args.method,
                "training": str(args.training),
                "probes": str(args.probes),
                "n_train_fit": int(len(train_z_fit)),
                "n_probes": int(len(probe_z)),
                "silhouette_probe_segments": sil,
            },
            indent=2,
        )
    )
    print(f"✅ Overlay → {out}")
    if sil is not None:
        print(f"   silhouette (probes) = {sil:.4f}")


if __name__ == "__main__":
    main()
