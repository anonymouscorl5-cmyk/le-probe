#!/usr/bin/env python3
"""
Build a shortlist of static probes for Neuronpedia / IG graph generation.

Picks canonical and borderline probes per scheme × variant × cluster using
cluster-differential L0 feature overlap (Tier A).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from interpretability.transcoders.clt_differential_core import (
    VARIANT_MAP,
    analyze_variant_scheme,
    labels_for_scheme,
    load_clt_activations,
)


def _as_np(x, dtype):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=dtype)


def _overlap_score(probe_feats: list[int], cluster_feats: list[int]) -> float:
    if not cluster_feats:
        return 0.0
    return len(set(probe_feats) & set(cluster_feats)) / len(set(cluster_feats))


def _borderline_score(
    probe_idx: int,
    cluster_indices: list[int],
    ee_xyz: np.ndarray,
    *,
    scheme: str,
) -> float:
    """Higher = closer to other clusters in EE space (within scheme)."""
    center = ee_xyz[cluster_indices].mean(axis=0)
    dist_own = float(np.linalg.norm(ee_xyz[probe_idx] - center))
    all_dists = [
        float(np.linalg.norm(ee_xyz[probe_idx] - ee_xyz[j])) for j in cluster_indices
    ]
    return dist_own / (np.mean(all_dists) + 1e-6)


def pick_playbook_entries(
    scheme: str,
    variant_tag: str,
    probe_ids: np.ndarray,
    ee_xyz: np.ndarray,
    analysis: dict,
    *,
    canonical_per_cluster: int = 2,
    borderline_per_scheme: int = 3,
) -> list[dict]:
    per_cluster = analysis["per_cluster"]
    per_probe = analysis.get("per_probe", {})
    clusters_map: dict[str, list[int]] = defaultdict(list)
    for pid_str, row in per_probe.items():
        clusters_map[row["cluster"]].append(int(pid_str))

    # invert: probe_id -> info
    by_pid = {int(k): v for k, v in per_probe.items()}
    entries: list[dict] = []

    for cluster, cluster_row in per_cluster.items():
        if cluster == "unknown":
            continue
        cluster_top = cluster_row.get("top_differential_features_cluster_level", [])
        pids_in_cluster = [
            int(pid) for pid, row in by_pid.items() if row["cluster"] == cluster
        ]
        ranked = []
        cluster_indices = [
            by_pid[pid]["bundle_index"] for pid in pids_in_cluster if pid in by_pid
        ]
        for pid in pids_in_cluster:
            row = by_pid[pid]
            ranked.append(
                (
                    _overlap_score(row["differential_top_k"], cluster_top),
                    pid,
                    row,
                )
            )
        ranked.sort(key=lambda x: (-x[0], x[1]))
        for rank, (_, pid, row) in enumerate(ranked[:canonical_per_cluster]):
            entries.append(
                {
                    "variant": variant_tag,
                    "scheme": scheme,
                    "cluster": cluster,
                    "role": "canonical",
                    "rank_in_cluster": rank,
                    "probe_id": int(pid),
                    "bundle_index": int(row["bundle_index"]),
                    "differential_top_k": row["differential_top_k"],
                    "overlap_with_cluster_top_k": _overlap_score(
                        row["differential_top_k"], cluster_top
                    ),
                    "ee_xyz": ee_xyz[row["bundle_index"]].tolist(),
                }
            )

    # Borderline: probes with low overlap but not unknown — one per cluster max
    borderline_candidates = []
    for pid, row in by_pid.items():
        if row["cluster"] == "unknown":
            continue
        cluster = row["cluster"]
        cluster_top = per_cluster.get(cluster, {}).get(
            "top_differential_features_cluster_level", []
        )
        ov = _overlap_score(row["differential_top_k"], cluster_top)
        idx = row["bundle_index"]
        cidxs = [
            by_pid[p]["bundle_index"] for p in by_pid if by_pid[p]["cluster"] == cluster
        ]
        borderline_candidates.append(
            (
                ov,
                _borderline_score(idx, cidxs, ee_xyz, scheme=scheme),
                int(pid),
                row,
                cluster,
            )
        )
    borderline_candidates.sort(key=lambda x: (x[0], -x[1]))
    seen_clusters = set()
    for ov, _, pid, row, cluster in borderline_candidates:
        if cluster in seen_clusters:
            continue
        if ov > 0.35:
            continue
        seen_clusters.add(cluster)
        entries.append(
            {
                "variant": variant_tag,
                "scheme": scheme,
                "cluster": cluster,
                "role": "borderline",
                "probe_id": int(pid),
                "bundle_index": int(row["bundle_index"]),
                "differential_top_k": row["differential_top_k"],
                "overlap_with_cluster_top_k": ov,
                "ee_xyz": ee_xyz[row["bundle_index"]].tolist(),
            }
        )
        if (
            len([e for e in entries if e["role"] == "borderline"])
            >= borderline_per_scheme
        ):
            break

    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--probe-dir",
        type=str,
        default=None,
        help="Directory with workspace_probe_bundle.pt",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default=None,
        help="Parent checkpoints directory",
    )
    parser.add_argument(
        "--schemes",
        nargs="+",
        default=["lateral", "distance", "pose"],
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(VARIANT_MAP.keys()),
    )
    parser.add_argument(
        "--pilot", action="store_true", help="distance + multiview_skeleton only"
    )
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    probe_dir = Path(args.probe_dir or repo_root / "datasets/workspace_probe_grasp")
    ckpt_root = Path(args.checkpoints or repo_root / "checkpoints")

    bundle = torch.load(
        probe_dir / "workspace_probe_bundle.pt", map_location="cpu", weights_only=False
    )
    probe_ids = _as_np(bundle["probe_ids"], np.int64)
    ee_xyz = _as_np(bundle["ee_achieved_xyz"], np.float64)
    cube = bundle.get("cube_xyz", bundle.get("cube_xyz_m"))
    cube_xyz = _as_np(cube, np.float64)
    pose_json = json.loads(
        (probe_dir / "workspace_probe_pose_clusters.json").read_text()
    )
    pose_labels = {
        int(pid): str(lab)
        for pid, lab in zip(pose_json["probe_ids"], pose_json["segment_hint"])
    }

    schemes = list(args.schemes)
    variants = list(args.variants)
    if args.pilot:
        schemes = ["distance"]
        variants = ["multiview_skeleton"]

    all_entries: list[dict] = []

    for scheme in schemes:
        labels_by_pid = labels_for_scheme(
            scheme, probe_ids, ee_xyz, cube_xyz, pose_labels
        )
        pid_to_idx = {int(pid): i for i, pid in enumerate(probe_ids)}
        clusters: dict[str, list[int]] = defaultdict(list)
        for pid, lab in labels_by_pid.items():
            clusters[str(lab)].append(pid_to_idx[pid])

        for tag in variants:
            if tag not in VARIANT_MAP:
                print(f"Skip unknown variant {tag}", file=sys.stderr)
                continue
            lat_path = probe_dir / f"workspace_probe_latents_{tag}.pt"
            clt_path = (
                ckpt_root
                / VARIANT_MAP[tag]
                / "transcoder_weights_residual"
                / "encoder_L0_residual_clt.pt"
            )
            if not lat_path.exists() or not clt_path.exists():
                print(f"Skip {tag}: missing latents or CLT", file=sys.stderr)
                continue
            lat_doc = torch.load(lat_path, map_location="cpu", weights_only=False)
            z = torch.as_tensor(lat_doc["latents"], dtype=torch.float32)
            activations = load_clt_activations(z, clt_path)
            analysis = analyze_variant_scheme(
                tag,
                activations,
                probe_ids,
                dict(clusters),
                top_k=20,
                max_samples_per_cluster=40,
                eps=1e-6,
                p_ubiq=0.8,
                rng=__import__("random").Random(0),
                export_per_probe=True,
            )
            entries = pick_playbook_entries(scheme, tag, probe_ids, ee_xyz, analysis)
            all_entries.extend(entries)
            print(f"{scheme} / {tag}: {len(entries)} playbook probes")

    out_path = Path(
        args.out
        or repo_root / "workspace_visualization/neuronpedia_probe_playbook.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "version": 1,
        "n_entries": len(all_entries),
        "entries": all_entries,
    }
    out_path.write_text(json.dumps(doc, indent=2))
    print(f"Saved {out_path} ({len(all_entries)} entries)")


if __name__ == "__main__":
    main()
