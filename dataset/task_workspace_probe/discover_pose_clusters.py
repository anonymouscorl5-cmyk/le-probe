#!/usr/bin/env python3
"""
Discover fuzzy pose groups for 500 workspace probes.

Uses **skeleton masks only** (no RGB): downsampled 1-channel masks → PCA features,
optionally all 5 cameras (``--all-views``). Optional ``--feature joints`` mixes
proprio — avoid if you want pure visual pose shape. GMM soft membership; ``k`` by BIC.

Writes ``workspace_probe_pose_clusters.json`` for ``relabel_probe_segments.py --scheme pose``.

Example::

    cd cortex-os
    uv run le-probe/dataset/task_workspace_probe/discover_pose_clusters.py \\
      --feature joints_skeleton --k-min 4 --k-max 10
    uv run le-probe/dataset/task_workspace_probe/relabel_probe_segments.py --scheme pose
    uv run le-probe/interpretability/manifold/run_all_probe_overlays.py \\
      --out-dir le-probe/workspace_visualization/pose_clusters
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
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from dataset.task_workspace_probe.probe_sim import CAM_NAMES, ProbeSimulator
from lewm.planning_constraints import RIGHT_ARM_NORM_SLICE

PROBE_DIR = REPO_DIR / "datasets/workspace_probe_grasp"
DEFAULT_BUNDLE = PROBE_DIR / "workspace_probe_bundle.pt"
DEFAULT_POSES = PROBE_DIR / "workspace_probe_poses.json"
DEFAULT_OUT = PROBE_DIR / "workspace_probe_pose_clusters.json"
CENTER_CAM_IDX = CAM_NAMES.index("world_center")

# Distinct from distance/lateral palette
POSE_CLUSTER_COLORS = [
    "#4C78A8",
    "#F58518",
    "#E45756",
    "#72B7B2",
    "#B279A2",
    "#FF9DA6",
    "#9D7660",
    "#59A14F",
    "#EDC948",
    "#AF7AA1",
    "#499894",
    "#86BCB6",
]


def _right_arm_features(state_norm: np.ndarray) -> np.ndarray:
    return state_norm[:, RIGHT_ARM_NORM_SLICE].astype(np.float64)


def _skeleton_features(
    bundle: dict,
    *,
    pca_dim: int = 24,
    downsample: int = 56,
    all_views: bool = False,
) -> np.ndarray:
    """
    Skeleton masks only (no RGB). ``all_views=False`` uses ``world_center``;
    ``all_views=True`` concatenates downsampled masks from all 5 cameras.
    """
    sk = bundle.get("skeleton")
    if sk is None:
        raise ValueError(
            "Bundle has no skeleton channel; re-run record_probe_snapshots.py --with_skeleton"
        )
    sk_np = sk.numpy().astype(np.float64) / 255.0  # (N, V, H, W, 1)
    n, v, h, w, _ = sk_np.shape
    step = max(1, h // downsample)
    view_indices = range(v) if all_views else (CENTER_CAM_IDX,)
    blocks = []
    for vi in view_indices:
        masks = sk_np[:, vi, :, :, 0]
        small = masks[:, ::step, ::step].reshape(n, -1)
        blocks.append(small)
    flat = np.hstack(blocks)
    pca = PCA(n_components=min(pca_dim, flat.shape[1], flat.shape[0] - 1))
    return pca.fit_transform(flat)


def _build_feature_matrix(
    bundle: dict,
    mode: str,
    *,
    skel_pca_dim: int,
    all_views: bool,
) -> tuple[np.ndarray, list[str]]:
    state = bundle["state_norm"].numpy().astype(np.float64)
    parts: list[np.ndarray] = []
    names: list[str] = []
    if mode in ("joints", "joints_skeleton"):
        parts.append(_right_arm_features(state))
        names.append("joints7")
    if mode in ("skeleton", "joints_skeleton"):
        sk = _skeleton_features(bundle, pca_dim=skel_pca_dim, all_views=all_views)
        parts.append(sk)
        view_tag = "5view" if all_views else "center"
        names.append(f"skeleton_pca{skel_pca_dim}_{view_tag}")
    if mode == "joints":
        pass
    elif mode == "skeleton":
        pass
    elif mode != "joints_skeleton":
        raise ValueError(f"Unknown feature mode: {mode}")
    if not parts:
        raise ValueError("No features selected")
    return np.hstack(parts), names


def _fuzzy_c_means(
    x: np.ndarray,
    k: int,
    *,
    m: float = 2.0,
    max_iter: int = 200,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (n,k) membership U, (k,d) centers, partition coefficient."""
    rng = np.random.default_rng(seed)
    n, d = x.shape
    u = rng.random((n, k))
    u /= u.sum(axis=1, keepdims=True)
    for _ in range(max_iter):
        um = u**m
        centers = (um.T @ x) / um.sum(axis=0, keepdims=True).T
        dist = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2) + 1e-12
        power = 2.0 / (m - 1.0)
        inv = (dist[:, :, None] / dist[:, None, :]) ** power
        u_new = 1.0 / inv.sum(axis=2)
        if np.linalg.norm(u_new - u) < 1e-5:
            u = u_new
            break
        u = u_new
    pc = float(np.sum(u**2) / n)
    return u, centers, pc


def _gmm_fit_metrics(x: np.ndarray, k: int, seed: int, *, n_init: int = 5) -> dict:
    gmm = GaussianMixture(
        n_components=k,
        covariance_type="full",
        n_init=n_init,
        random_state=seed,
    )
    gmm.fit(x)
    labels = gmm.predict(x)
    counts = np.bincount(labels, minlength=k)
    sil = None
    if len(np.unique(labels)) > 1:
        try:
            sil = float(silhouette_score(x, labels))
        except Exception:
            pass
    return {
        "k": k,
        "bic": float(gmm.bic(x)),
        "aic": float(gmm.aic(x)),
        "lower_bound": float(gmm.lower_bound_),
        "silhouette": sil,
        "min_cluster_size": int(counts.min()),
        "cluster_sizes": counts.tolist(),
    }


def _sweep_k_gmm(x: np.ndarray, k_min: int, k_max: int, seed: int) -> list[dict]:
    return [_gmm_fit_metrics(x, k, seed) for k in range(k_min, k_max + 1)]


def _pick_k_gmm(x: np.ndarray, k_min: int, k_max: int, seed: int) -> tuple[int, dict]:
    rows = _sweep_k_gmm(x, k_min, k_max, seed)
    records = {str(r["k"]): r for r in rows}
    best = min(rows, key=lambda r: r["bic"])
    return int(best["k"]), records


def _plot_k_sweep(
    rows: list[dict],
    out_png: Path,
    *,
    bic_pick_k: int | None,
    sil_pick_k: int | None,
) -> None:
    ks = [r["k"] for r in rows]
    bics = [r["bic"] for r in rows]
    sils = [r["silhouette"] if r["silhouette"] is not None else np.nan for r in rows]

    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(ks, bics, "o-", color="#4C78A8", label="BIC")
    ax1.set_xlabel("Number of clusters k")
    ax1.set_ylabel("BIC (lower better)", color="#4C78A8")
    ax1.tick_params(axis="y", labelcolor="#4C78A8")
    ax1.set_xticks(ks)

    ax2 = ax1.twinx()
    ax2.plot(ks, sils, "s-", color="#F58518", label="Silhouette")
    ax2.set_ylabel("Silhouette in feature space (higher better)", color="#F58518")
    ax2.tick_params(axis="y", labelcolor="#F58518")
    ax2.axhline(0.0, color="#888888", linewidth=0.8, linestyle="--")

    if bic_pick_k is not None:
        row = next(r for r in rows if r["k"] == bic_pick_k)
        ax1.axvline(bic_pick_k, color="#4C78A8", alpha=0.35, linestyle="--")
        ax1.scatter(
            [bic_pick_k],
            [row["bic"]],
            s=120,
            facecolors="none",
            edgecolors="#4C78A8",
            linewidths=2,
        )
    if sil_pick_k is not None and sil_pick_k != bic_pick_k:
        row = next(r for r in rows if r["k"] == sil_pick_k)
        ax2.axvline(sil_pick_k, color="#F58518", alpha=0.35, linestyle="--")
        ax2.scatter(
            [sil_pick_k],
            [row["silhouette"]],
            s=120,
            facecolors="none",
            edgecolors="#F58518",
            linewidths=2,
        )

    title_bits = []
    if bic_pick_k is not None:
        title_bits.append(f"BIC→k={bic_pick_k}")
    if sil_pick_k is not None:
        title_bits.append(f"max sil→k={sil_pick_k}")
    fig.suptitle("GMM sweep (skeleton features): " + ", ".join(title_bits), fontsize=11)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()


def run_k_sweep(
    x: np.ndarray,
    *,
    k_min: int,
    k_max: int,
    seed: int,
    sweep_png: Path,
    sweep_json: Path,
) -> tuple[int, int, list[dict]]:
    """Return (k_by_bic, k_by_silhouette, rows)."""
    rows = _sweep_k_gmm(x, k_min, k_max, seed)
    k_bic = int(min(rows, key=lambda r: r["bic"])["k"])
    sil_rows = [r for r in rows if r["silhouette"] is not None]
    k_sil = (
        int(max(sil_rows, key=lambda r: r["silhouette"])["k"]) if sil_rows else k_bic
    )

    _plot_k_sweep(rows, sweep_png, bic_pick_k=k_bic, sil_pick_k=k_sil)
    sweep_json.parent.mkdir(parents=True, exist_ok=True)
    sweep_json.write_text(
        json.dumps(
            {
                "k_min": k_min,
                "k_max": k_max,
                "k_by_bic": k_bic,
                "k_by_silhouette": k_sil,
                "rows": rows,
            },
            indent=2,
        )
    )
    return k_bic, k_sil, rows


def _render_cluster_grid(
    poses_doc: dict,
    labels: list[str],
    out_png: Path,
    *,
    max_per_cluster: int = 3,
) -> None:
    """Show example MuJoCo skeleton masks per pose cluster."""
    by_label: dict[str, list[dict]] = {}
    for pose, lab in zip(poses_doc["poses"], labels):
        by_label.setdefault(lab, []).append(pose)

    sim = ProbeSimulator()
    cube = np.asarray(poses_doc.get("cube_xyz"), dtype=np.float64)
    labs = sorted(
        by_label.keys(), key=lambda s: int(s.split("_")[1]) if "_" in s else s
    )
    nrows = len(labs)
    ncols = max_per_cluster
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    if nrows == 1:
        axes = np.asarray([axes])
    if ncols == 1:
        axes = axes.reshape(-1, 1)

    for ri, lab in enumerate(labs):
        examples = by_label[lab][:ncols]
        for ci in range(ncols):
            ax = axes[ri, ci]
            ax.axis("off")
            if ci >= len(examples):
                continue
            wire = np.asarray(examples[ci]["wire32_rad"], dtype=np.float64)
            sim.set_pose_from_wire32_rad(wire, cube_xyz=cube)
            mask = sim.render_skeleton_mask("world_center")
            ax.imshow(mask, cmap="gray")
            if ci == 0:
                ax.set_title(lab, fontsize=10)
    fig.suptitle("Pose cluster examples (center skeleton)", fontsize=12)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fuzzy pose clustering for workspace probes"
    )
    parser.add_argument("--bundle", type=str, default=str(DEFAULT_BUNDLE))
    parser.add_argument("--poses", type=str, default=str(DEFAULT_POSES))
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    parser.add_argument(
        "--feature",
        choices=("joints", "skeleton", "joints_skeleton"),
        default="skeleton",
        help="Clustering input (no RGB). Prefer skeleton-only; joints_* mixes proprio",
    )
    parser.add_argument(
        "--all-views",
        action="store_true",
        help="Use skeleton masks from all 5 cameras (default: world_center only)",
    )
    parser.add_argument("--k", type=int, default=0, help="Fixed k (0 = pick by BIC)")
    parser.add_argument("--k-min", type=int, default=4)
    parser.add_argument("--k-max", type=int, default=12)
    parser.add_argument(
        "--sweep-k",
        action="store_true",
        help="Sweep k in [k-min,k-max]; write BIC/silhouette curve PNG + JSON and exit",
    )
    parser.add_argument(
        "--sweep-png",
        type=str,
        default=str(REPO_DIR / "workspace_visualization/pose_clusters/k_sweep.png"),
    )
    parser.add_argument(
        "--sweep-json",
        type=str,
        default=str(REPO_DIR / "workspace_visualization/pose_clusters/k_sweep.json"),
    )
    parser.add_argument("--min-cluster-size", type=int, default=20)
    parser.add_argument(
        "--also-fcm",
        action="store_true",
        help="Run fuzzy c-means at chosen k (report partition coefficient)",
    )
    parser.add_argument("--skel-pca-dim", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--review-png",
        type=str,
        default=str(
            REPO_DIR / "workspace_visualization/pose_clusters/cluster_review.png"
        ),
    )
    args = parser.parse_args()

    bundle = torch.load(args.bundle, map_location="cpu", weights_only=False)
    poses_doc = json.loads(Path(args.poses).read_text())
    probe_ids = bundle["probe_ids"].numpy().astype(int).tolist()

    raw, feat_names = _build_feature_matrix(
        bundle,
        args.feature,
        skel_pca_dim=args.skel_pca_dim,
        all_views=args.all_views,
    )
    x = StandardScaler().fit_transform(raw)

    if args.sweep_k:
        k_bic, k_sil, rows = run_k_sweep(
            x,
            k_min=args.k_min,
            k_max=args.k_max,
            seed=args.seed,
            sweep_png=Path(args.sweep_png),
            sweep_json=Path(args.sweep_json),
        )
        print(
            f"📊 k sweep {args.k_min}–{args.k_max}  BIC→k={k_bic}  max silhouette→k={k_sil}"
        )
        print(f"✅ Curve → {args.sweep_png}")
        print(f"✅ Table → {args.sweep_json}")
        print("\n| k | BIC | silhouette | min cluster |")
        print("|---|-----|------------|-------------|")
        for r in rows:
            sil = r["silhouette"]
            sil_s = f"{sil:.4f}" if sil is not None else "—"
            print(f"| {r['k']} | {r['bic']:.0f} | {sil_s} | {r['min_cluster_size']} |")
        print(
            "\nRe-run with fixed k, e.g.  --k",
            k_sil,
            "  (silhouette) or  --k",
            k_bic,
            "  (BIC)",
        )
        return

    if args.k > 0:
        k = args.k
        bic_table = {}
    else:
        k, bic_table = _pick_k_gmm(x, args.k_min, args.k_max, args.seed)
        print(f"📊 BIC chose k={k} (range {args.k_min}–{args.k_max})")

    gmm = GaussianMixture(
        n_components=k,
        covariance_type="full",
        n_init=10,
        random_state=args.seed,
    )
    gmm.fit(x)
    membership = gmm.predict_proba(x)
    hard = gmm.predict(x)
    labels = [f"pose_{i}" for i in hard]

    # Drop tiny clusters by merging to nearest GMM center (rare with full cov)
    counts = np.bincount(hard, minlength=k)
    if (counts < args.min_cluster_size).any():
        print(f"⚠️  cluster sizes {counts.tolist()} (min={args.min_cluster_size})")

    sil = float(silhouette_score(x, hard)) if len(np.unique(hard)) > 1 else None
    print(
        f"✅ GMM k={k}  silhouette (feature space)={sil:.3f}"
        if sil is not None
        else f"✅ GMM k={k}"
    )

    fcm_info = None
    if args.also_fcm:
        u, _, pc = _fuzzy_c_means(x, k, seed=args.seed)
        fcm_info = {
            "partition_coefficient": pc,
            "max_membership_mean": float(u.max(axis=1).mean()),
        }
        print(f"   FCM partition coefficient={pc:.3f}")

    cluster_sizes = {f"pose_{i}": int(c) for i, c in enumerate(counts)}

    doc = {
        "version": 1,
        "feature_mode": args.feature,
        "feature_parts": feat_names,
        "n_probes": len(probe_ids),
        "n_clusters": k,
        "bic_by_k": bic_table,
        "gmm_bic": float(gmm.bic(x)),
        "silhouette_feature_space": sil,
        "fcm": fcm_info,
        "probe_ids": probe_ids,
        "segment_hint": labels,
        "membership": membership.tolist(),
        "cluster_sizes": cluster_sizes,
        "colors": {
            f"pose_{i}": POSE_CLUSTER_COLORS[i % len(POSE_CLUSTER_COLORS)]
            for i in range(k)
        },
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(doc, indent=2))
    print(f"✅ Wrote {out_path}")
    print("   counts:", cluster_sizes)

    if args.review_png:
        _render_cluster_grid(poses_doc, labels, Path(args.review_png))

    print("\nNext:")
    print(
        "  uv run le-probe/dataset/task_workspace_probe/relabel_probe_segments.py --scheme pose"
    )
    print("  uv run le-probe/interpretability/manifold/run_all_probe_overlays.py \\")
    print("       --out-dir le-probe/workspace_visualization/pose_clusters")


if __name__ == "__main__":
    main()
