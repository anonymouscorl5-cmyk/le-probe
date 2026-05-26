#!/usr/bin/env python3
"""
B6: Visualize 500 workspace probe latents in 3D (PCA / UMAP / t-SNE).

Uses **only** ``workspace_probe_latents_*.pt`` (500 points). No training manifold.
World-frame sampling context: ``visualize_probe_ee_scatter.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

try:
    import umap
except ImportError:
    umap = None

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

CURRENT_FILE = Path(__file__).resolve()
LE_PROBE_ROOT = CURRENT_FILE.parents[2]
WORKSPACE_VIZ_DIR = LE_PROBE_ROOT / "workspace_visualization"
if str(LE_PROBE_ROOT) not in sys.path:
    sys.path.insert(0, str(LE_PROBE_ROOT))

from dataset.task_workspace_probe.segments import SEGMENT_COLORS, SEGMENT_ORDER


def _axis_limits(
    points: np.ndarray, margin: float = 0.08
) -> tuple[list[float], list[float]]:
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    pad = span * margin
    return (lo - pad).tolist(), (hi + pad).tolist()


def _dim_axis_titles(method: str) -> tuple[str, str, str]:
    m = method.upper()
    if m == "PCA":
        return "PC 1", "PC 2", "PC 3"
    return f"{m} 1", f"{m} 2", f"{m} 3"


def embed_probes(
    method: str,
    probe_latents: np.ndarray,
    *,
    random_state: int = 0,
) -> tuple[np.ndarray, str]:
    """Reduce **n probe** latents to 3D (n is typically 500)."""
    method = method.lower()
    n = len(probe_latents)

    if method == "pca":
        n_comp = min(3, probe_latents.shape[1], n)
        reducer = PCA(n_components=n_comp, random_state=random_state)
        out = reducer.fit_transform(probe_latents)
        if n_comp < 3:
            pad = np.zeros((n, 3 - n_comp), dtype=np.float64)
            out = np.hstack([out, pad])
        return out, f"PCA on {n} probes"

    if method == "umap":
        if umap is None:
            raise ImportError("umap-learn required for --method umap")
        n_neighbors = min(15, max(2, n - 1))
        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            random_state=random_state,
        )
        return reducer.fit_transform(probe_latents), f"UMAP on {n} probes"

    if method == "tsne":
        perplexity = min(30.0, max(5.0, (n - 1) / 3.0))
        reducer = TSNE(
            n_components=3,
            perplexity=perplexity,
            max_iter=1000,
            random_state=random_state,
            init="pca",
            learning_rate="auto",
        )
        return (
            reducer.fit_transform(probe_latents),
            f"t-SNE on {n} probes (perplexity={perplexity:.1f})",
        )

    raise ValueError(f"Unsupported method: {method} (use pca, umap, tsne)")


def _ordered_segments(segments) -> list[str]:
    present = set(segments)
    out = [s for s in SEGMENT_ORDER if s in present]
    out.extend(sorted(present - set(out)))
    return out


def _probe_silhouette(probe_3d: np.ndarray, segments) -> float | None:
    seg_to_id = {k: i for i, k in enumerate(SEGMENT_ORDER)}
    labels = np.array([seg_to_id.get(s, -1) for s in segments])
    valid = labels >= 0
    if valid.sum() < 10 or len(np.unique(labels[valid])) < 2:
        return None
    try:
        return float(silhouette_score(probe_3d[valid], labels[valid]))
    except Exception:
        return None


def save_png(
    probe_3d: np.ndarray,
    segments,
    *,
    method: str,
    title: str,
    out: Path,
    sil: float | None,
) -> None:
    lo, hi = _axis_limits(probe_3d)
    x0, x1 = lo[0], hi[0]
    y0, y1 = lo[1], hi[1]
    z0, z1 = lo[2], hi[2]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for seg in _ordered_segments(segments):
        mask = np.array([s == seg for s in segments])
        if not mask.any():
            continue
        ax.scatter(
            probe_3d[mask, 0],
            probe_3d[mask, 1],
            probe_3d[mask, 2],
            c=SEGMENT_COLORS.get(seg, "#333333"),
            s=40,
            alpha=0.9,
            label=seg,
            edgecolors="black",
            linewidths=0.3,
        )

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_zlim(z0, z1)
    try:
        ax.set_box_aspect((x1 - x0, y1 - y0, z1 - z0))
    except Exception:
        pass

    ax.set_xlabel(_dim_axis_titles(method)[0])
    ax.set_ylabel(_dim_axis_titles(method)[1])
    ax.set_zlabel(_dim_axis_titles(method)[2])

    full_title = title
    if sil is not None:
        full_title += f" (silhouette={sil:.3f})"
    ax.set_title(full_title)
    ax.legend(loc="upper left", fontsize=8)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


def save_html(
    probe_3d: np.ndarray,
    segments,
    probe_ids,
    *,
    method: str,
    title: str,
    out: Path,
    sil: float | None,
) -> None:
    if go is None:
        raise ImportError("plotly required for --html (pip install plotly)")

    lo, hi = _axis_limits(probe_3d)
    ax_titles = _dim_axis_titles(method)

    fig = go.Figure()
    for seg in _ordered_segments(segments):
        mask = np.array([s == seg for s in segments])
        if not mask.any():
            continue
        idx = np.where(mask)[0]
        hover = [
            f"probe {probe_ids[i]}<br>{seg}<br>"
            f"({probe_3d[i,0]:.3f}, {probe_3d[i,1]:.3f}, {probe_3d[i,2]:.3f})"
            for i in idx
        ]
        fig.add_trace(
            go.Scatter3d(
                x=probe_3d[mask, 0],
                y=probe_3d[mask, 1],
                z=probe_3d[mask, 2],
                mode="markers",
                name=seg,
                marker=dict(
                    size=5,
                    color=SEGMENT_COLORS.get(seg, "#888888"),
                    opacity=0.9,
                    line=dict(width=0.5, color="black"),
                ),
                text=hover,
                hoverinfo="text",
            )
        )

    full_title = title
    if sil is not None:
        full_title += f" (silhouette={sil:.3f})"
    fig.update_layout(
        title=full_title,
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(x=0.01, y=0.99),
        scene=dict(
            xaxis=dict(title=ax_titles[0], range=[lo[0], hi[0]]),
            yaxis=dict(title=ax_titles[1], range=[lo[1], hi[1]]),
            zaxis=dict(title=ax_titles[2], range=[lo[2], hi[2]]),
            aspectmode="cube",
        ),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs="cdn", config=dict(scrollZoom=True))


def visualize_probe_latents(
    probes_path: Path,
    *,
    method: str,
    out_png: Path,
    out_html: Path | None,
    variant_label: str,
) -> dict:
    """Reduce and plot **500** workspace probe latents only."""
    probe = torch.load(probes_path, map_location="cpu", weights_only=False)

    probe_z = np.asarray(probe["latents"], dtype=np.float64)
    segments = list(probe.get("segment_hint") or ["unknown"] * len(probe_z))
    probe_ids = probe.get("probe_ids")
    if probe_ids is not None:
        if hasattr(probe_ids, "numpy"):
            probe_ids = probe_ids.numpy()
        probe_ids = [int(x) for x in probe_ids]
    else:
        probe_ids = list(range(len(probe_z)))

    probe_3d, embed_note = embed_probes(method, probe_z)
    sil = _probe_silhouette(probe_3d, segments)

    title = f"500 probes — {variant_label} ({method.upper()})"
    save_png(probe_3d, segments, method=method, title=title, out=out_png, sil=sil)
    if out_html is not None:
        save_html(
            probe_3d,
            segments,
            probe_ids,
            method=method,
            title=title,
            out=out_html,
            sil=sil,
        )

    return {
        "method": method,
        "embedding": embed_note,
        "probes": str(probes_path),
        "variant": variant_label,
        "n_probes": int(len(probe_z)),
        "silhouette_probe_segments": sil,
        "segment_counts": {s: segments.count(s) for s in _ordered_segments(segments)},
    }


# Deprecated alias (old two-arg API ignored training if passed as first arg)
def run_overlay(training_path, probes_path, **kwargs):
    return visualize_probe_latents(Path(probes_path), **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="B6: 3D latent viz for 500 workspace probes only."
    )
    parser.add_argument(
        "--probes",
        type=str,
        required=True,
        help="workspace_probe_latents_*.pt from harvest_workspace_probes.py",
    )
    parser.add_argument("--method", choices=["pca", "umap", "tsne"], default="umap")
    parser.add_argument("--variant", type=str, default="")
    parser.add_argument(
        "--out",
        type=str,
        default=str(WORKSPACE_VIZ_DIR / "workspace_probe_latent_viz.png"),
    )
    parser.add_argument("--html", type=str, default=None)
    args = parser.parse_args()

    variant = args.variant or Path(args.probes).stem.replace(
        "workspace_probe_latents_", ""
    )
    out_png = Path(args.out)
    out_html = Path(args.html) if args.html else out_png.with_suffix(".html")

    metrics = visualize_probe_latents(
        Path(args.probes),
        method=args.method,
        out_png=out_png,
        out_html=out_html,
        variant_label=variant,
    )

    metrics_path = out_png.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"✅ Latent viz ({metrics['n_probes']} probes) PNG → {out_png}")
    if args.html or out_html.exists():
        print(f"✅ Latent viz HTML → {out_html}")
    if metrics["silhouette_probe_segments"] is not None:
        print(f"   silhouette = {metrics['silhouette_probe_segments']:.4f}")


if __name__ == "__main__":
    main()
