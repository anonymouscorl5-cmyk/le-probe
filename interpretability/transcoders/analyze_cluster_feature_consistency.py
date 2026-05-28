"""
Compare CLT top features across static-probe clusters, separately per probe scheme.

For each scheme (lateral / distance / pose) and each variant:
  - sample multiple probes per cluster
  - extract top-k encoder_L0 features per probe
  - measure within-cluster consistency vs across-cluster separation (same variant)
  - measure same-cluster alignment across variants (decoder-mapped Jaccard)
"""

from __future__ import annotations

import json
import random
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
    "singleview": "Single-View RGB",
    "multiview": "Multi-View RGB",
    "multiview_skeleton": "Multi-View RGB + Skeletal Priors",
    "multiview_skeleton_dino": "Multi-View RGB + Skeletal Priors + DINOv3 Waypoints",
}


def _jaccard(a, b) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _mean_pairwise_jaccard(feat_lists: list[list[int]], max_pairs: int = 2000) -> float:
    if len(feat_lists) < 2:
        return float("nan")
    pairs = list(combinations(range(len(feat_lists)), 2))
    if len(pairs) > max_pairs:
        rng = random.Random(0)
        pairs = rng.sample(pairs, max_pairs)
    vals = [_jaccard(feat_lists[i], feat_lists[j]) for i, j in pairs]
    return float(np.mean(vals))


def _load_clt(clt_path: Path):
    doc = torch.load(clt_path, map_location="cpu", weights_only=False)
    sd = doc["state_dict"]
    norm = doc["norm_stats"]
    w_enc = torch.as_tensor(sd["encoder.weight"], dtype=torch.float32).T
    b_enc = torch.as_tensor(sd["encoder.bias"], dtype=torch.float32)
    dec = torch.as_tensor(sd["decoder.weight"], dtype=torch.float32)
    dec = dec / (dec.norm(dim=0, keepdim=True) + 1e-8)
    return norm, w_enc, b_enc, dec


def _top_features_per_probe(
    z: torch.Tensor, norm: dict, w_enc: torch.Tensor, b_enc: torch.Tensor, top_k: int
) -> list[list[int]]:
    mean = torch.as_tensor(norm["mean"], dtype=torch.float32)
    std = torch.as_tensor(norm["std"], dtype=torch.float32)
    z_norm = (z - mean) / (std + 1e-8)
    feat = torch.relu(z_norm @ w_enc + b_enc)
    out = []
    for i in range(feat.shape[0]):
        out.append(
            torch.topk(feat[i], k=min(top_k, feat.shape[1])).indices.cpu().tolist()
        )
    return out


def _map_to_ref(
    features: list[int], dec: torch.Tensor, ref_dec: torch.Tensor
) -> list[int]:
    if not features:
        return []
    q = dec[:, features].T
    sims = q @ ref_dec
    return sims.argmax(dim=1).cpu().numpy().tolist()


def _labels_for_scheme(
    scheme: str,
    probe_ids: np.ndarray,
    ee_xyz: np.ndarray,
    cube_xyz: np.ndarray,
    segment_hint: np.ndarray,
    pose_labels: dict[int, str],
) -> dict[int, str]:
    import os
    import sys

    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from dataset.task_workspace_probe.segments import segment_hint as seg_hint_fn

    out = {}
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


def _subsample_indices(indices: list[int], max_n: int, rng: random.Random) -> list[int]:
    if len(indices) <= max_n:
        return indices
    return rng.sample(indices, max_n)


def analyze_scheme(
    scheme: str,
    probe_ids: np.ndarray,
    ee_xyz: np.ndarray,
    cube_xyz: np.ndarray,
    segment_hint: np.ndarray,
    pose_labels: dict[int, str],
    probe_dir: Path,
    checkpoints: Path,
    top_k: int,
    max_samples_per_cluster: int,
    ref_variant: str = "singleview",
) -> dict:
    labels_by_pid = _labels_for_scheme(
        scheme, probe_ids, ee_xyz, cube_xyz, segment_hint, pose_labels
    )

    # probe_id -> index in latent tensors
    pid_to_idx = {int(pid): i for i, pid in enumerate(probe_ids)}

    clusters: dict[str, list[int]] = defaultdict(list)
    for pid, lab in labels_by_pid.items():
        clusters[str(lab)].append(pid_to_idx[pid])

    rng = random.Random(0)
    per_variant_probe_feats: dict[str, list[list[int]]] = {}
    per_variant_dec: dict[str, torch.Tensor] = {}

    for tag, ckpt in VARIANT_MAP.items():
        lat_path = probe_dir / f"workspace_probe_latents_{tag}.pt"
        clt_path = (
            checkpoints
            / ckpt
            / "transcoder_weights_residual"
            / "encoder_L0_residual_clt.pt"
        )
        lat_doc = torch.load(lat_path, map_location="cpu", weights_only=False)
        z = torch.as_tensor(lat_doc["latents"], dtype=torch.float32)
        norm, w_enc, b_enc, dec = _load_clt(clt_path)
        per_variant_probe_feats[tag] = _top_features_per_probe(
            z, norm, w_enc, b_enc, top_k
        )
        per_variant_dec[tag] = dec

    ref_dec = per_variant_dec[ref_variant]

    # --- Within-variant metrics per cluster ---
    within_variant = {}
    for tag in VARIANT_MAP:
        within_variant[tag] = {}
        for cluster, idx_list in clusters.items():
            idxs = _subsample_indices(idx_list, max_samples_per_cluster, rng)
            feat_lists = [per_variant_probe_feats[tag][i] for i in idxs]
            within_variant[tag][cluster] = {
                "n_probes_used": len(idxs),
                "intra_cluster_jaccard_mean": _mean_pairwise_jaccard(feat_lists),
            }

        # across-cluster separation (different clusters, same variant)
        cluster_names = sorted(clusters.keys())
        cross_vals = []
        for c1, c2 in combinations(cluster_names, 2):
            idx1 = _subsample_indices(
                clusters[c1], max(5, max_samples_per_cluster // 2), rng
            )
            idx2 = _subsample_indices(
                clusters[c2], max(5, max_samples_per_cluster // 2), rng
            )
            f1 = [per_variant_probe_feats[tag][i] for i in idx1]
            f2 = [per_variant_probe_feats[tag][i] for i in idx2]
            if f1 and f2:
                cross_vals.append(_jaccard(rng.choice(f1), rng.choice(f2)))
        within_variant[tag]["_cross_cluster_jaccard_mean"] = (
            float(np.mean(cross_vals)) if cross_vals else float("nan")
        )

    # --- Cross-variant same-cluster alignment (decoder-mapped to ref) ---
    cross_variant_same_cluster = {}
    for cluster in clusters:
        cross_variant_same_cluster[cluster] = {}
        for tag in VARIANT_MAP:
            if tag == ref_variant:
                continue
            idxs = _subsample_indices(clusters[cluster], max_samples_per_cluster, rng)
            # aggregate mapped top features for this cluster in each variant
            agg_ref = []
            agg_other = []
            for i in idxs:
                f = per_variant_probe_feats[ref_variant][i]
                g = per_variant_probe_feats[tag][i]
                agg_ref.extend(_map_to_ref(f, per_variant_dec[ref_variant], ref_dec))
                agg_other.extend(_map_to_ref(g, per_variant_dec[tag], ref_dec))
            # cluster-level jaccard between aggregated mapped sets
            cross_variant_same_cluster[cluster][tag] = _jaccard(agg_ref, agg_other)

    return {
        "scheme": scheme,
        "clusters": {k: len(v) for k, v in clusters.items()},
        "within_variant": within_variant,
        "cross_variant_same_cluster_decoder_mapped": cross_variant_same_cluster,
    }


def _render_markdown(all_results: dict, out_md: Path):
    lines = [
        "# CLT Cluster Feature Consistency (Static Probes)",
        "",
        "Per probe scheme, we compare top-k encoder_L0 features:",
        "- **Within-cluster consistency** (same variant, same cluster, multiple probes)",
        "- **Across-cluster separation** (same variant, different clusters)",
        "- **Cross-variant same-cluster alignment** (decoder-mapped features vs Single-View RGB reference)",
        "",
    ]

    for scheme, res in all_results.items():
        lines.append(f"## Scheme: `{scheme}`")
        lines.append("")
        lines.append("### Within-variant (per cluster)")
        lines.append("")
        lines.append(
            "| Cluster | n | SV intra | MV intra | MV+Skel intra | MV+Skel+DINO intra |"
        )
        lines.append("| :-- | --: | --: | --: | --: | --: |")

        cluster_names = sorted(res["clusters"].keys())
        for c in cluster_names:
            row = [f"`{c}`", str(res["clusters"][c])]
            for tag in VARIANT_MAP:
                intra = (
                    res["within_variant"][tag]
                    .get(c, {})
                    .get("intra_cluster_jaccard_mean", float("nan"))
                )
                row.append(f"{intra:.3f}" if intra == intra else "nan")
            lines.append("| " + " | ".join(row) + " |")

        lines.append("")
        lines.append("Cross-cluster separation (lower = better separated clusters):")
        for tag, label in VARIANT_LABELS.items():
            v = res["within_variant"][tag].get(
                "_cross_cluster_jaccard_mean", float("nan")
            )
            lines.append(f"- **{label}**: {v:.3f}")
        lines.append("")

        lines.append("### Cross-variant same-cluster (decoder-mapped Jaccard vs SV)")
        lines.append("")
        lines.append("| Cluster | MV | MV+Skel | MV+Skel+DINO |")
        lines.append("| :-- | --: | --: | --: |")
        for c in cluster_names:
            mv = res["cross_variant_same_cluster_decoder_mapped"][c].get(
                "multiview", float("nan")
            )
            mvs = res["cross_variant_same_cluster_decoder_mapped"][c].get(
                "multiview_skeleton", float("nan")
            )
            mvsd = res["cross_variant_same_cluster_decoder_mapped"][c].get(
                "multiview_skeleton_dino", float("nan")
            )
            lines.append(f"| `{c}` | {mv:.3f} | {mvs:.3f} | {mvsd:.3f} |")
        lines.append("")

    out_md.write_text("\n".join(lines))


def main():
    repo_root = Path(__file__).resolve().parents[2]
    workspace = repo_root.parent
    probe_dir = repo_root / "datasets" / "workspace_probe_grasp"
    out_dir = repo_root / "workspace_visualization"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _as_np(x, dtype):
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        return np.asarray(x, dtype=dtype)

    bundle = torch.load(
        probe_dir / "workspace_probe_bundle.pt", map_location="cpu", weights_only=False
    )
    probe_ids = _as_np(bundle["probe_ids"], np.int64)
    ee_xyz = _as_np(bundle["ee_achieved_xyz"], np.float64)
    cube_xyz = _as_np(bundle.get("cube_xyz", bundle.get("cube_xyz_m")), np.float64)
    segment_hint = np.asarray(bundle["segment_hint"], dtype=object)

    pose_json = json.loads(
        (probe_dir / "workspace_probe_pose_clusters.json").read_text()
    )
    pose_labels = {
        int(pid): str(lab)
        for pid, lab in zip(pose_json["probe_ids"], pose_json["segment_hint"])
    }

    top_k = 20
    max_samples_per_cluster = 40

    all_results = {}
    for scheme in ("lateral", "distance", "pose"):
        all_results[scheme] = analyze_scheme(
            scheme=scheme,
            probe_ids=probe_ids,
            ee_xyz=ee_xyz,
            cube_xyz=cube_xyz,
            segment_hint=segment_hint,
            pose_labels=pose_labels,
            probe_dir=probe_dir,
            checkpoints=workspace / "checkpoints",
            top_k=top_k,
            max_samples_per_cluster=max_samples_per_cluster,
        )

    out_json = out_dir / "clt_cluster_feature_consistency.json"
    out_md = out_dir / "clt_cluster_feature_consistency.md"
    out_json.write_text(json.dumps(all_results, indent=2, default=str))
    _render_markdown(all_results, out_md)

    print(f"Saved: {out_json}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
