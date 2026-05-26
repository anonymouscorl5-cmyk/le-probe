#!/usr/bin/env python3
"""Re-apply segment_hint to poses JSON + bundle (no MuJoCo re-render)."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--poses",
        type=str,
        default=str(
            REPO_DIR / "datasets/workspace_probe_grasp/workspace_probe_poses.json"
        ),
    )
    parser.add_argument(
        "--bundle",
        type=str,
        default=str(
            REPO_DIR / "datasets/workspace_probe_grasp/workspace_probe_bundle.pt"
        ),
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
        bundle = torch.load(bundle_path, weights_only=False)
        ee = bundle["ee_achieved_xyz"]
        if hasattr(ee, "numpy"):
            ee_np = ee.numpy()
        else:
            ee_np = ee
        cube = bundle.get("cube_xyz", cube_xyz)
        if hasattr(cube, "numpy"):
            cube = cube.numpy()
        segments = [segment_hint(ee_np[i], cube_xyz=cube) for i in range(len(ee_np))]
        bundle["segment_hint"] = segments
        torch.save(bundle, bundle_path)
        print(f"✅ Updated bundle segment_hint → {bundle_path}")
        print("   counts:", dict(Counter(segments)))


if __name__ == "__main__":
    main()
