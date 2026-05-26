#!/usr/bin/env python3
"""
Ask: on 500 workspace probes, does latent geometry track pose (joints), EE position, or cube distance?

Joins ``workspace_probe_bundle.pt`` (state_norm, EE) with ``workspace_probe_latents_*.pt``.
Reports correlations, kNN consistency, and silhouette under different labelings.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

CURRENT_FILE = Path(__file__).resolve()
LE_PROBE_ROOT = CURRENT_FILE.parents[2]
if str(LE_PROBE_ROOT) not in sys.path:
    sys.path.insert(0, str(LE_PROBE_ROOT))

from lewm.planning_constraints import RIGHT_ARM_NORM_SLICE

PROBE_DIR = LE_PROBE_ROOT / "datasets/workspace_probe_grasp"
DEFAULT_BUNDLE = PROBE_DIR / "workspace_probe_bundle.pt"

VARIANTS = [
    ("singleview", "workspace_probe_latents_singleview.pt"),
    ("multiview", "workspace_probe_latents_multiview.pt"),
    ("multiview_skeleton", "workspace_probe_latents_multiview_skeleton.pt"),
    ("multiview_skeleton_dino", "workspace_probe_latents_multiview_skeleton_dino.pt"),
]


def _align_bundle_latents(
    bundle: dict, latents_doc: dict
) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(latents_doc["latents"], dtype=np.float64)
    b_ids = bundle["probe_ids"].numpy().astype(int)
    l_ids = np.asarray(
        (
            latents_doc["probe_ids"].numpy()
            if hasattr(latents_doc["probe_ids"], "numpy")
            else latents_doc["probe_ids"]
        ),
        dtype=int,
    )
    if len(b_ids) != len(l_ids) or not np.array_equal(b_ids, l_ids):
        order = {int(pid): i for i, pid in enumerate(b_ids)}
        idx = np.array([order[int(p)] for p in l_ids])
        states = bundle["state_norm"].numpy()[idx]
        ee = bundle["ee_achieved_xyz"].numpy()[idx]
        dist = bundle["dist_to_cube_m"].numpy()[idx]
        segs = [bundle["segment_hint"][i] for i in idx]
    else:
        states = bundle["state_norm"].numpy()
        ee = bundle["ee_achieved_xyz"].numpy()
        dist = bundle["dist_to_cube_m"].numpy()
        segs = list(bundle["segment_hint"])
    return z, dict(state_norm=states, ee_xyz=ee, dist_cube=dist, segment_hint=segs)


def _right_arm_joints(state_norm: np.ndarray) -> np.ndarray:
    return state_norm[:, RIGHT_ARM_NORM_SLICE]


def _kmeans_labels(x: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    return km.fit_predict(StandardScaler().fit_transform(x))


def _silhouette_on_embedding(emb: np.ndarray, labels: np.ndarray) -> float | None:
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return None
    try:
        return float(silhouette_score(emb, labels))
    except Exception:
        return None


def _max_abs_corr(emb: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Max |corr| between each embedding dim and target (scalar or vector cols)."""
    target = np.asarray(target, dtype=np.float64)
    if target.ndim == 1:
        target = target.reshape(-1, 1)
    out: dict[str, float] = {}
    for j in range(target.shape[1]):
        col = target[:, j]
        if np.std(col) < 1e-9:
            continue
        corrs = []
        for d in range(emb.shape[1]):
            if np.std(emb[:, d]) < 1e-9:
                continue
            c = np.corrcoef(emb[:, d], col)[0, 1]
            if np.isfinite(c):
                corrs.append(abs(c))
        key = f"col{j}" if target.shape[1] > 1 else "scalar"
        if corrs:
            out[key] = float(max(corrs))
    return out


def _knn_consistency(
    latent: np.ndarray,
    joint: np.ndarray,
    ee: np.ndarray,
    k: int = 15,
) -> dict[str, float]:
    """If latent neighbors are closer in joint than EE space → pose-dominated."""
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(StandardScaler().fit_transform(latent))
    _, idx = nn.kneighbors(StandardScaler().fit_transform(latent))
    idx = idx[:, 1:]  # drop self

    j_scale = StandardScaler().fit_transform(joint)
    e_scale = StandardScaler().fit_transform(ee)
    j_dists, e_dists = [], []
    for i in range(len(latent)):
        j_dists.append(np.linalg.norm(j_scale[idx[i]] - j_scale[i], axis=1).mean())
        e_dists.append(np.linalg.norm(e_scale[idx[i]] - e_scale[i], axis=1).mean())
    j_mean, e_mean = float(np.mean(j_dists)), float(np.mean(e_dists))
    return {
        "mean_joint_dist_knn": j_mean,
        "mean_ee_dist_knn": e_mean,
        "ee_over_joint_ratio": e_mean / j_mean if j_mean > 1e-9 else None,
    }


def _linear_r2_joint_vs_ee(emb: np.ndarray, joint: np.ndarray, ee: np.ndarray) -> dict:
    """R² for predicting each PC from joints-only vs EE-only (OLS, standardized)."""
    from numpy.linalg import lstsq

    def r2_predict(y: np.ndarray, x: np.ndarray) -> float:
        x = np.column_stack([np.ones(len(x)), StandardScaler().fit_transform(x)])
        coef, _, _, _ = lstsq(x, y, rcond=None)
        pred = x @ coef
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return float(1 - ss_res / ss_tot) if ss_tot > 1e-9 else 0.0

    j = StandardScaler().fit_transform(joint)
    e = StandardScaler().fit_transform(ee)
    r2j, r2e = [], []
    for d in range(min(3, emb.shape[1])):
        y = emb[:, d]
        r2j.append(r2_predict(y, j))
        r2e.append(r2_predict(y, e))
    return {
        "pc_r2_right_arm": r2j,
        "pc_r2_ee_xyz": r2e,
        "pc_r2_joint_minus_ee": [a - b for a, b in zip(r2j, r2e)],
    }


def analyze_variant(
    bundle: dict,
    latents_path: Path,
    *,
    k_clusters: int = 4,
    pca_dims: int = 3,
) -> dict:
    lat_doc = torch.load(latents_path, map_location="cpu", weights_only=False)
    z, phys = _align_bundle_latents(bundle, lat_doc)
    joint = _right_arm_joints(phys["state_norm"])
    ee = phys["ee_xyz"]
    dist = phys["dist_cube"]

    pca = PCA(n_components=min(pca_dims, z.shape[1], z.shape[0]))
    emb = pca.fit_transform(z)

    ee_scalar = {
        "dist_cube": dist,
        "ee_x": ee[:, 0],
        "ee_y": ee[:, 1],
        "ee_z": ee[:, 2],
    }

    corr_joint = _max_abs_corr(emb, joint)
    corr_ee = _max_abs_corr(emb, ee)
    corr_dist = _max_abs_corr(emb, dist.reshape(-1, 1))

    labels = {
        "segment_hint": None,  # filled below if categorical
        f"kmeans_joint_k{k_clusters}": _kmeans_labels(joint, k_clusters),
        f"kmeans_ee_k{k_clusters}": _kmeans_labels(ee, k_clusters),
        f"kmeans_dist_k{k_clusters}": _kmeans_labels(dist, k_clusters),
    }
    # segment strings → ints
    segs = phys["segment_hint"]
    uniq = sorted(set(segs))
    seg_ids = np.array([uniq.index(s) for s in segs])
    labels["segment_hint"] = seg_ids

    silhouettes = {
        name: _silhouette_on_embedding(emb, lab) for name, lab in labels.items()
    }

    return {
        "latents_path": str(latents_path),
        "use_skeleton": bool(lat_doc.get("use_skeleton")),
        "use_dino": bool(lat_doc.get("use_dino")),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "max_abs_corr_with_pcs": {
            "right_arm_7d": corr_joint,
            "ee_xyz": corr_ee,
            "dist_cube": corr_dist.get("scalar"),
            "ee_z_only": _max_abs_corr(emb, ee[:, 2].reshape(-1, 1)).get("scalar"),
        },
        "best_joint_corr": max(corr_joint.values()) if corr_joint else None,
        "best_ee_corr": max(corr_ee.values()) if corr_ee else None,
        "linear_pc_predictors": _linear_r2_joint_vs_ee(emb, joint, ee),
        "knn_consistency_k15": _knn_consistency(emb, joint, ee, k=15),
        "silhouette_pca3": silhouettes,
        "segment_counts": {s: segs.count(s) for s in uniq},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=str, default=str(DEFAULT_BUNDLE))
    parser.add_argument("--out", type=str, default="")
    parser.add_argument(
        "--k", type=int, default=4, help="k-means clusters for pose/EE/dist"
    )
    args = parser.parse_args()

    bundle = torch.load(args.bundle, map_location="cpu", weights_only=False)
    report: dict = {
        "bundle": str(args.bundle),
        "n_probes": int(bundle["state_norm"].shape[0]),
    }

    for name, fname in VARIANTS:
        path = PROBE_DIR / fname
        if not path.exists():
            print(f"⚠️  skip {name}: missing {path}")
            continue
        report[name] = analyze_variant(bundle, path, k_clusters=args.k)
        m = report[name]
        knn = m["knn_consistency_k15"]
        print(f"\n=== {name} (skel={m['use_skeleton']}, dino={m['use_dino']}) ===")
        print(
            f"  best |corr| joint={m['best_joint_corr']:.3f}  ee={m['best_ee_corr']:.3f}  dist={m['max_abs_corr_with_pcs']['dist_cube']:.3f}"
        )
        print(
            f"  kNN mean dist  joint={knn['mean_joint_dist_knn']:.3f}  ee={knn['mean_ee_dist_knn']:.3f}  ratio(ee/joint)={knn['ee_over_joint_ratio']:.3f}"
        )
        sil = m["silhouette_pca3"]
        print(
            f"  silhouette  segment={sil.get('segment_hint')}  joint_k={sil.get(f'kmeans_joint_k{args.k}')}  ee_k={sil.get(f'kmeans_ee_k{args.k}')}  dist_k={sil.get(f'kmeans_dist_k{args.k}')}"
        )

    out = (
        Path(args.out)
        if args.out
        else LE_PROBE_ROOT / "workspace_visualization/probe_latent_driver_analysis.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\n✅ Wrote {out}")


if __name__ == "__main__":
    main()
