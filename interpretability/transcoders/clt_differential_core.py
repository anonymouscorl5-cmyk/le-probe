"""
Shared cluster-differential CLT scoring for static workspace probes.

Used by analyze_cluster_differential_features.py and build_neuronpedia_probe_playbook.py.
"""

from __future__ import annotations

import random
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

VARIANT_MAP = {
    "singleview": "lewm_grasp_baseline",
    "multiview": "lewm_grasp_multiview",
    "multiview_skeleton": "lewm_grasp_multiview_skeleton",
    "multiview_skeleton_dino": "lewm_grasp_multiview_skeleton_dino",
}

VARIANT_LABELS = {
    "singleview": "SV",
    "multiview": "MV",
    "multiview_skeleton": "MV+Skel",
    "multiview_skeleton_dino": "MV+Skel+DINO",
}


def jaccard(a: list[int] | set[int], b: list[int] | set[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def mean_pairwise_jaccard(feat_lists: list[list[int]], max_pairs: int = 2000) -> float:
    if len(feat_lists) < 2:
        return float("nan")
    pairs = list(combinations(range(len(feat_lists)), 2))
    if len(pairs) > max_pairs:
        rng = random.Random(0)
        pairs = rng.sample(pairs, max_pairs)
    return float(np.mean([jaccard(feat_lists[i], feat_lists[j]) for i, j in pairs]))


def load_clt_activations(z: torch.Tensor, clt_path: Path) -> np.ndarray:
    doc = torch.load(clt_path, map_location="cpu", weights_only=False)
    sd = doc["state_dict"]
    norm = doc["norm_stats"]
    w_enc = torch.as_tensor(sd["encoder.weight"], dtype=torch.float32).T
    b_enc = torch.as_tensor(sd["encoder.bias"], dtype=torch.float32)
    mean = torch.as_tensor(norm["mean"], dtype=torch.float32)
    std = torch.as_tensor(norm["std"], dtype=torch.float32)
    z_norm = (z - mean) / (std + 1e-8)
    feat = torch.relu(z_norm @ w_enc + b_enc)
    return feat.cpu().numpy().astype(np.float32)


def feature_masks(
    activations: np.ndarray,
    *,
    eps: float,
    p_ubiq: float,
) -> tuple[np.ndarray, dict]:
    n_feat = activations.shape[1]
    max_act = activations.max(axis=0)
    prevalence = (activations > eps).mean(axis=0)
    never_fired = max_act < eps
    ubiquitous = prevalence > p_ubiq
    valid = ~(never_fired | ubiquitous)
    stats = {
        "n_features_total": int(n_feat),
        "n_never_fired": int(never_fired.sum()),
        "n_ubiquitous": int(ubiquitous.sum()),
        "n_valid": int(valid.sum()),
        "n_probes": int(activations.shape[0]),
    }
    return valid, stats


def differential_scores(
    activations: np.ndarray, idx_list: list[int], valid_mask: np.ndarray
) -> np.ndarray:
    in_mask = np.zeros(activations.shape[0], dtype=bool)
    in_mask[idx_list] = True
    mean_in = activations[in_mask].mean(axis=0)
    mean_out = activations[~in_mask].mean(axis=0)
    scores = mean_in - mean_out
    scores[~valid_mask] = -np.inf
    return scores


def topk_from_scores(scores: np.ndarray, k: int, min_score: float = 0.0) -> list[int]:
    idx = np.argsort(scores)[::-1]
    out = []
    for i in idx:
        if len(out) >= k:
            break
        if scores[i] >= min_score and np.isfinite(scores[i]):
            out.append(int(i))
    return out


def probe_differential_topk(
    probe_act: np.ndarray,
    cluster_scores: np.ndarray,
    k: int,
) -> list[int]:
    combined = cluster_scores.copy()
    finite = np.isfinite(combined) & (combined > -np.inf)
    combined[finite] = combined[finite] * probe_act[finite]
    return topk_from_scores(combined, k, min_score=0.0)


def labels_for_scheme(
    scheme: str,
    probe_ids: np.ndarray,
    ee_xyz: np.ndarray,
    cube_xyz: np.ndarray,
    pose_labels: dict[int, str],
) -> dict[int, str]:
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from dataset.task_workspace_probe.segments import segment_hint as seg_hint_fn

    out: dict[int, str] = {}
    for i, pid in enumerate(probe_ids):
        if scheme == "lateral":
            out[int(pid)] = seg_hint_fn(ee_xyz[i], cube_xyz=cube_xyz, scheme="lateral")
        elif scheme == "distance":
            out[int(pid)] = seg_hint_fn(ee_xyz[i], cube_xyz=cube_xyz, scheme="distance")
        elif scheme == "pose":
            out[int(pid)] = pose_labels.get(int(pid), "unknown")
        else:
            raise ValueError(scheme)
    return out


def subsample(indices: list[int], max_n: int, rng: random.Random) -> list[int]:
    if len(indices) <= max_n:
        return indices
    return rng.sample(indices, max_n)


def analyze_variant_scheme(
    tag: str,
    activations: np.ndarray,
    probe_ids: np.ndarray,
    clusters: dict[str, list[int]],
    *,
    top_k: int,
    max_samples_per_cluster: int,
    eps: float,
    p_ubiq: float,
    rng: random.Random,
    export_per_probe: bool = False,
) -> dict:
    valid_mask, mask_stats = feature_masks(activations, eps=eps, p_ubiq=p_ubiq)

    cluster_scores: dict[str, np.ndarray] = {}
    for cluster, idx_list in clusters.items():
        if cluster == "unknown":
            continue
        cluster_scores[cluster] = differential_scores(activations, idx_list, valid_mask)

    probe_to_cluster: dict[int, str] = {}
    for cluster, idx_list in clusters.items():
        for i in idx_list:
            probe_to_cluster[i] = cluster

    per_probe_diff: dict[int, list[int]] = {}
    for i in range(activations.shape[0]):
        cluster = probe_to_cluster.get(i, "unknown")
        if cluster not in cluster_scores:
            per_probe_diff[i] = []
            continue
        per_probe_diff[i] = probe_differential_topk(
            activations[i], cluster_scores[cluster], top_k
        )

    per_cluster = {}
    intra_vals = []
    for cluster, idx_list in clusters.items():
        if cluster == "unknown":
            continue
        idxs = subsample(idx_list, max_samples_per_cluster, rng)
        feat_lists = [per_probe_diff[i] for i in idxs if per_probe_diff[i]]
        intra = mean_pairwise_jaccard(feat_lists)
        per_cluster[cluster] = {
            "n_probes": len(idx_list),
            "n_probes_used": len(idxs),
            "intra_jaccard": intra,
            "top_differential_features_cluster_level": topk_from_scores(
                cluster_scores[cluster], top_k, min_score=0.0
            ),
        }
        if feat_lists and intra == intra:
            intra_vals.append(intra)

    cluster_names = [c for c in sorted(clusters.keys()) if c != "unknown"]
    cross_vals = []
    for c1, c2 in combinations(cluster_names, 2):
        idx1 = subsample(clusters[c1], max(5, max_samples_per_cluster // 2), rng)
        idx2 = subsample(clusters[c2], max(5, max_samples_per_cluster // 2), rng)
        f1 = [per_probe_diff[i] for i in idx1 if per_probe_diff[i]]
        f2 = [per_probe_diff[i] for i in idx2 if per_probe_diff[i]]
        if f1 and f2:
            cross_vals.append(jaccard(rng.choice(f1), rng.choice(f2)))

    cross_mean = float(np.mean(cross_vals)) if cross_vals else float("nan")
    intra_mean = float(np.mean(intra_vals)) if intra_vals else float("nan")
    margin = (
        intra_mean - cross_mean
        if intra_mean == intra_mean and cross_mean == cross_mean
        else float("nan")
    )

    result = {
        "mask_stats": mask_stats,
        "per_cluster": per_cluster,
        "intra_cluster_jaccard_mean": intra_mean,
        "cross_cluster_jaccard_mean": cross_mean,
        "separation_margin": margin,
    }

    if export_per_probe:
        per_probe_export = {}
        for i in range(activations.shape[0]):
            cluster = probe_to_cluster.get(i, "unknown")
            per_probe_export[str(int(probe_ids[i]))] = {
                "bundle_index": int(i),
                "cluster": cluster,
                "differential_top_k": per_probe_diff.get(i, []),
            }
        result["per_probe"] = per_probe_export

    return result
