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

from dataset.task_workspace_probe.segments import segment_hint

PROBE_DIR = REPO_DIR / "datasets/workspace_probe_grasp"


def _update_latents(path: Path, cube_xyz) -> Counter[str]:
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
    segments = [segment_hint(ee_np[i], cube_xyz=cube) for i in range(len(ee_np))]
    data["segment_hint"] = segments
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
    args = parser.parse_args()

    poses_path = Path(args.poses)
    poses_doc = json.loads(poses_path.read_text())
    cube_xyz = poses_doc.get("cube_xyz")

    counts: Counter[str] = Counter()
    for pose in poses_doc["poses"]:
        hint = segment_hint(pose["ee_achieved_xyz"], cube_xyz=cube_xyz)
        pose["segment_hint"] = hint
        counts[hint] += 1

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
        segments = [segment_hint(ee_np[i], cube_xyz=cube) for i in range(len(ee_np))]
        bundle["segment_hint"] = segments
        torch.save(bundle, bundle_path)
        print(f"✅ Updated bundle → {bundle_path}")
        print("   counts:", dict(Counter(segments)))

    for lat_path in sorted(PROBE_DIR.glob("workspace_probe_latents_*.pt")):
        c = _update_latents(lat_path, cube_xyz)
        print(f"✅ Updated latents → {lat_path.name}")
        print("   counts:", dict(c))


if __name__ == "__main__":
    main()
