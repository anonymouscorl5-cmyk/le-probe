"""
Tier A: cluster-differential CLT features on static probes (encoder_L0 on 192-d latents).
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from interpretability.transcoders.clt_differential_core import (
    VARIANT_LABELS,
    VARIANT_MAP,
    analyze_variant_scheme,
    labels_for_scheme,
    load_clt_activations,
)


def _fmt(x: float) -> str:
    return f"{x:.3f}" if x == x else "nan"


def _render_markdown(all_results: dict, out_md: Path) -> None:
    p = all_results["lateral"]["params"]
    lines = [
        "# Tier A: Cluster-Differential CLT Features (Static Probes)",
        "",
        "Method: encoder_L0 CLT on 192-d probe latents. Features with "
        f"`max activation < {p['eps']}` (never-fired) or prevalence > {p['p_ubiquitous']:.0%} "
        "(ubiquitous) are excluded. Per cluster `c`, "
        "`score(f,c) = mean(A[f]|c) - mean(A[f]|¬c)`; each probe keeps top-"
        f"{p['top_k']} features by `score(f,c) × A[probe,f]` within its cluster.",
        "",
        "**Separation margin** = mean intra-cluster Jaccard − mean cross-cluster Jaccard "
        "(positive ⇒ cluster-specific differential overlap).",
        "",
        "Preliminary, non-causal; within-variant only.",
        "",
    ]

    lines.append("## Feature mask counts (per variant, all 500 probes)")
    lines.append("")
    lines.append("| Variant | Total | Never-fired | Ubiquitous | Valid |")
    lines.append("| :-- | --: | --: | --: | --: |")
    ref = all_results["lateral"]["variants"]
    for tag, short in VARIANT_LABELS.items():
        st = ref[tag]["mask_stats"]
        lines.append(
            f"| {short} | {st['n_features_total']} | {st['n_never_fired']} | "
            f"{st['n_ubiquitous']} | {st['n_valid']} |"
        )
    lines.append("")

    for scheme, res in all_results.items():
        lines.append(f"## Scheme: `{scheme}`")
        lines.append("")
        lines.append("### Separation margin (variant-level)")
        lines.append("")
        lines.append("| Variant | Intra | Cross | Margin |")
        lines.append("| :-- | --: | --: | --: |")
        for tag, short in VARIANT_LABELS.items():
            v = res["variants"][tag]
            lines.append(
                f"| {short} | {_fmt(v['intra_cluster_jaccard_mean'])} | "
                f"{_fmt(v['cross_cluster_jaccard_mean'])} | {_fmt(v['separation_margin'])} |"
            )
        lines.append("")
        lines.append("### Intra-cluster Jaccard (differential top-k, per cluster)")
        lines.append("")
        header = "| Cluster | n | " + " | ".join(VARIANT_LABELS.values()) + " |"
        lines.append(header)
        lines.append(
            "| :-- | --: | " + " | ".join(["--:"] * len(VARIANT_LABELS)) + " |"
        )
        cluster_names = sorted(res["clusters"].keys())
        for c in cluster_names:
            if c == "unknown":
                continue
            row = [f"`{c}`", str(res["clusters"][c])]
            for tag in VARIANT_MAP:
                intra = (
                    res["variants"][tag]["per_cluster"]
                    .get(c, {})
                    .get("intra_jaccard", float("nan"))
                )
                row.append(_fmt(intra))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")


def analyze_scheme(
    scheme: str,
    probe_ids: np.ndarray,
    ee_xyz: np.ndarray,
    cube_xyz: np.ndarray,
    pose_labels: dict[int, str],
    probe_dir: Path,
    checkpoints: Path,
    *,
    top_k: int,
    max_samples_per_cluster: int,
    eps: float,
    p_ubiq: float,
    export_per_probe: bool = True,
) -> dict:
    labels_by_pid = labels_for_scheme(scheme, probe_ids, ee_xyz, cube_xyz, pose_labels)
    pid_to_idx = {int(pid): i for i, pid in enumerate(probe_ids)}
    clusters: dict[str, list[int]] = defaultdict(list)
    for pid, lab in labels_by_pid.items():
        clusters[str(lab)].append(pid_to_idx[pid])

    rng = random.Random(0)
    per_variant = {}
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
        activations = load_clt_activations(z, clt_path)
        per_variant[tag] = analyze_variant_scheme(
            tag,
            activations,
            probe_ids,
            dict(clusters),
            top_k=top_k,
            max_samples_per_cluster=max_samples_per_cluster,
            eps=eps,
            p_ubiq=p_ubiq,
            rng=rng,
            export_per_probe=export_per_probe,
        )

    return {
        "scheme": scheme,
        "clusters": {k: len(v) for k, v in clusters.items()},
        "params": {
            "top_k": top_k,
            "eps": eps,
            "p_ubiquitous": p_ubiq,
            "max_samples_per_cluster": max_samples_per_cluster,
            "scoring": "mean_in_cluster - mean_out_cluster; probe rank = score * activation",
            "clt": "encoder_L0_residual on 192-d probe latents",
        },
        "variants": per_variant,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    probe_dir = repo_root / "datasets" / "workspace_probe_grasp"
    ckpt_root = repo_root / "checkpoints"
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
    pose_json = json.loads(
        (probe_dir / "workspace_probe_pose_clusters.json").read_text()
    )
    pose_labels = {
        int(pid): str(lab)
        for pid, lab in zip(pose_json["probe_ids"], pose_json["segment_hint"])
    }

    top_k = 20
    all_results = {}
    for scheme in ("lateral", "distance", "pose"):
        all_results[scheme] = analyze_scheme(
            scheme,
            probe_ids,
            ee_xyz,
            cube_xyz,
            pose_labels,
            probe_dir,
            ckpt_root,
            top_k=top_k,
            max_samples_per_cluster=40,
            eps=1e-6,
            p_ubiq=0.8,
            export_per_probe=True,
        )

    out_json = out_dir / "clt_cluster_differential.json"
    out_md = out_dir / "clt_cluster_differential.md"
    out_json.write_text(json.dumps(all_results, indent=2, default=str))
    _render_markdown(all_results, out_md)
    print(f"Saved: {out_json}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
