#!/usr/bin/env python3
"""Re-apply segment_hint to poses, bundle, and probe latent files (no re-encode)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from dataset.task_workspace_probe.segments import (
    LabelScheme,
    load_pose_cluster_labels,
    segment_hint,
)

PROBE_DIR = REPO_DIR / "datasets/workspace_probe_grasp"


def _update_latents(
    path: Path,
    cube_xyz,
    scheme: LabelScheme,
    pose_labels: dict[int, str] | None,
    probe_ids: list[int],
) -> Counter[str]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    ee = data.get("ee_achieved_xyz")
    if ee is None:
        raise KeyError(f"{path} missing ee_achieved_xyz")
    if hasattr(ee, "numpy"):
        ee_np = ee.numpy()
    else:
        ee_np = ee
    cube = data.get("cube_xyz", cube_xyz)
    if hasattr(cube, "numpy"):
        cube = cube.numpy()
    lat_pids = data.get("probe_ids")
    if lat_pids is not None:
        lat_pids = (
            lat_pids.numpy().astype(int).tolist()
            if hasattr(lat_pids, "numpy")
            else [int(x) for x in lat_pids]
        )
    else:
        lat_pids = probe_ids

    if scheme == "pose":
        segments = [pose_labels[int(lat_pids[i])] for i in range(len(ee_np))]
    else:
        segments = [
            segment_hint(ee_np[i], cube_xyz=cube, scheme=scheme)
            for i in range(len(ee_np))
        ]
    data["segment_hint"] = segments
    data["label_scheme"] = scheme
    if scheme == "pose" and pose_labels is not None:
        import json

        cluster_doc = json.loads(
            (PROBE_DIR / "workspace_probe_pose_clusters.json").read_text()
        )
        data["pose_membership"] = cluster_doc.get("membership")
    torch.save(data, path)
    return Counter(segments)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--poses",
        type=str,
        default=str(PROBE_DIR / "workspace_probe_poses.json"),
    )
    parser.add_argument(
        "--bundle",
        type=str,
        default=str(PROBE_DIR / "workspace_probe_bundle.pt"),
    )
    parser.add_argument(
        "--latents-glob",
        type=str,
        default=str(PROBE_DIR / "workspace_probe_latents_*.pt"),
    )
    parser.add_argument(
        "--scheme",
        choices=("lateral", "distance", "pose"),
        default="distance",
        help="Label scheme: lateral, distance, or pose (from discover_pose_clusters.py)",
    )
    args = parser.parse_args()
    scheme: LabelScheme = args.scheme

    poses_path = Path(args.poses)
    poses_doc = json.loads(poses_path.read_text())
    cube_xyz = poses_doc.get("cube_xyz")

    pose_labels: dict[int, str] | None = None
    if scheme == "pose":
        pose_labels, _ = load_pose_cluster_labels(PROBE_DIR)

    counts: Counter[str] = Counter()
    for pose in poses_doc["poses"]:
        if scheme == "pose":
            hint = pose_labels[int(pose["probe_id"])]
        else:
            hint = segment_hint(
                pose["ee_achieved_xyz"], cube_xyz=cube_xyz, scheme=scheme
            )
        pose["segment_hint"] = hint
        counts[hint] += 1

    poses_doc["label_scheme"] = scheme
    poses_path.write_text(json.dumps(poses_doc, indent=2))
    print(f"✅ Updated {len(poses_doc['poses'])} poses → {poses_path}")
    print("   counts:", dict(counts))

    bundle_path = Path(args.bundle)
    if bundle_path.exists():
        bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
        ee = bundle["ee_achieved_xyz"]
        ee_np = ee.numpy() if hasattr(ee, "numpy") else ee
        cube = bundle.get("cube_xyz", cube_xyz)
        if hasattr(cube, "numpy"):
            cube = cube.numpy()
        pids = bundle["probe_ids"].numpy().astype(int).tolist()
        if scheme == "pose":
            segments = [pose_labels[int(p)] for p in pids]
        else:
            segments = [
                segment_hint(ee_np[i], cube_xyz=cube, scheme=scheme)
                for i in range(len(ee_np))
            ]
        bundle["segment_hint"] = segments
        bundle["label_scheme"] = scheme
        torch.save(bundle, bundle_path)
        print(f"✅ Updated bundle → {bundle_path}")
        print("   counts:", dict(Counter(segments)))

    poses_pids = [int(p["probe_id"]) for p in poses_doc["poses"]]
    for lat_path in sorted(PROBE_DIR.glob("workspace_probe_latents_*.pt")):
        c = _update_latents(lat_path, cube_xyz, scheme, pose_labels, poses_pids)
        print(f"✅ Updated latents → {lat_path.name}")
        print("   counts:", dict(c))


if __name__ == "__main__":
    main()
