import json
from pathlib import Path

import numpy as np
import torch

VARIANT_MAP = {
    "singleview": "lewm_grasp_baseline",
    "multiview": "lewm_grasp_multiview",
    "multiview_skeleton": "lewm_grasp_multiview_skeleton",
    "multiview_skeleton_dino": "lewm_grasp_multiview_skeleton_dino",
}


def _load_pose_labels(pose_json_path: Path):
    payload = json.loads(pose_json_path.read_text())
    probe_ids = np.asarray(payload["probe_ids"], dtype=np.int64)
    membership = np.asarray(payload["membership"], dtype=np.float64)
    pose_ids = membership.argmax(axis=1).astype(np.int64)
    return dict(zip(probe_ids.tolist(), pose_ids.tolist()))


def _load_variant_top_features(
    latents_path: Path, clt_path: Path, top_k: int, bin_masks: dict
) -> tuple[dict, torch.Tensor]:
    lat_doc = torch.load(latents_path, map_location="cpu", weights_only=False)
    z = torch.as_tensor(lat_doc["latents"], dtype=torch.float32)

    clt_doc = torch.load(clt_path, map_location="cpu", weights_only=False)
    sd = clt_doc["state_dict"]
    norm = clt_doc["norm_stats"]

    mean = torch.as_tensor(norm["mean"], dtype=torch.float32)
    std = torch.as_tensor(norm["std"], dtype=torch.float32)

    # encoder.weight is [n_features, d_model] in these checkpoints
    w_enc = torch.as_tensor(sd["encoder.weight"], dtype=torch.float32).T
    b_enc = torch.as_tensor(sd["encoder.bias"], dtype=torch.float32)

    z_norm = (z - mean) / (std + 1e-8)
    feat = torch.relu(z_norm @ w_enc + b_enc)

    results = {}
    for bin_name, mask in bin_masks.items():
        if mask.sum() == 0:
            results[bin_name] = []
            continue
        avg = feat[mask].mean(dim=0)
        top_idx = torch.topk(avg, k=top_k).indices.cpu().numpy().tolist()
        results[bin_name] = top_idx
    dec = torch.as_tensor(sd["decoder.weight"], dtype=torch.float32)
    # Normalize feature vectors (columns)
    dec = dec / (dec.norm(dim=0, keepdim=True) + 1e-8)
    return results, dec


def _jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def main():
    repo_root = Path(__file__).resolve().parents[2]
    workspace = repo_root.parent

    probe_dir = repo_root / "datasets" / "workspace_probe_grasp"
    checkpoints = workspace / "checkpoints"

    top_k = 20
    ref_variant = "singleview"

    bundle = torch.load(
        probe_dir / "workspace_probe_bundle.pt", map_location="cpu", weights_only=False
    )
    probe_ids = bundle["probe_ids"].cpu().numpy().astype(np.int64)
    dist = bundle["dist_to_cube_m"].cpu().numpy().astype(np.float64)
    segment = np.asarray(bundle["segment_hint"], dtype=object)

    pose_by_probe = _load_pose_labels(probe_dir / "workspace_probe_pose_clusters.json")
    pose = np.asarray(
        [pose_by_probe.get(int(pid), -1) for pid in probe_ids], dtype=np.int64
    )

    near_thresh = np.quantile(dist, 0.33)
    far_thresh = np.quantile(dist, 0.66)

    bin_masks = {
        "distance_near": dist <= near_thresh,
        "distance_far": dist >= far_thresh,
    }

    # segment_hint currently stores pose_* buckets for these probe bundles.
    for seg_name in sorted(set(segment.tolist())):
        bin_masks[f"segment_{seg_name}"] = segment == seg_name

    for p in sorted(set(pose.tolist())):
        bin_masks[f"pose_cluster_{p}"] = pose == p

    per_variant = {}
    per_variant_decoder = {}
    for tag, ckpt_dir in VARIANT_MAP.items():
        latents = probe_dir / f"workspace_probe_latents_{tag}.pt"
        clt = (
            checkpoints
            / ckpt_dir
            / "transcoder_weights_residual"
            / "encoder_L0_residual_clt.pt"
        )
        if not latents.exists() or not clt.exists():
            raise FileNotFoundError(f"Missing artifacts for {tag}: {latents} | {clt}")
        tops, dec = _load_variant_top_features(latents, clt, top_k, bin_masks)
        per_variant[tag] = tops
        per_variant_decoder[tag] = dec

    ref = per_variant[ref_variant]
    ref_dec = per_variant_decoder[ref_variant]
    overlap = {}
    overlap_mapped = {}
    for bin_name in bin_masks:
        overlap[bin_name] = {}
        overlap_mapped[bin_name] = {}
        for tag in VARIANT_MAP:
            if tag == ref_variant:
                continue
            overlap[bin_name][tag] = _jaccard(ref[bin_name], per_variant[tag][bin_name])
            # Map features from other variant to nearest SV feature by decoder cosine.
            other_feats = per_variant[tag][bin_name]
            if not other_feats:
                mapped = []
            else:
                q = per_variant_decoder[tag][:, other_feats].T  # [k, d]
                sims = q @ ref_dec  # [k, n_ref_features]
                mapped = sims.argmax(dim=1).cpu().numpy().tolist()
            overlap_mapped[bin_name][tag] = _jaccard(ref[bin_name], mapped)

    out_dir = repo_root / "workspace_visualization"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "transcoder_feature_overlap.json"
    out_md = out_dir / "transcoder_feature_overlap_table.md"

    bin_counts = {k: int(v.sum()) for k, v in bin_masks.items()}
    payload = {
        "top_k": top_k,
        "reference": ref_variant,
        "bins": list(bin_masks.keys()),
        "bin_counts": bin_counts,
        "overlap_jaccard": overlap,
        "overlap_jaccard_decoder_mapped": overlap_mapped,
    }
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "| Bin | Raw MV/SV | Raw MV+Skel/SV | Raw MV+Skel+DINO/SV | Mapped MV/SV | Mapped MV+Skel/SV | Mapped MV+Skel+DINO/SV |",
        "| :-- | --: | --: | --: | --: | --: | --: |",
    ]
    for b in bin_masks:
        mv = overlap[b]["multiview"]
        mvs = overlap[b]["multiview_skeleton"]
        mvsd = overlap[b]["multiview_skeleton_dino"]
        mmv = overlap_mapped[b]["multiview"]
        mmvs = overlap_mapped[b]["multiview_skeleton"]
        mmvsd = overlap_mapped[b]["multiview_skeleton_dino"]
        lines.append(
            f"| {b} (n={bin_counts[b]}) | {mv:.3f} | {mvs:.3f} | {mvsd:.3f} | {mmv:.3f} | {mmvs:.3f} | {mmvsd:.3f} |"
        )
    out_md.write_text("\n".join(lines) + "\n")

    print(f"Saved JSON: {out_json}")
    print(f"Saved table: {out_md}")


if __name__ == "__main__":
    main()
