#!/usr/bin/env python3
"""B3: Record static probe snapshots (5 views + optional skeleton) into a .pt bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from dataset.task_workspace_probe.probe_sim import (
    CAM_NAMES,
    ProbeSimulator,
    RENDER_SIZE,
)


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
        "--out",
        type=str,
        default=str(
            REPO_DIR / "datasets/workspace_probe_grasp/workspace_probe_bundle.pt"
        ),
    )
    parser.add_argument(
        "--with_skeleton",
        action="store_true",
        help="Store skeleton masks (1ch) per view for MV+Skel encode path",
    )
    args = parser.parse_args()

    poses_doc = json.loads(Path(args.poses).read_text())
    cube_xyz = np.asarray(poses_doc.get("cube_xyz"), dtype=np.float64)

    sim = ProbeSimulator()
    sim.reset_probe_scene(lock_posture=True)

    n = len(poses_doc["poses"])
    V = len(CAM_NAMES)
    rgb_stack = np.zeros((n, V, RENDER_SIZE, RENDER_SIZE, 3), dtype=np.uint8)
    skel_stack = (
        np.zeros((n, V, RENDER_SIZE, RENDER_SIZE, 1), dtype=np.uint8)
        if args.with_skeleton
        else None
    )
    states = np.zeros((n, 32), dtype=np.float32)
    probe_ids = np.zeros(n, dtype=np.int32)
    ee_xyz = np.zeros((n, 3), dtype=np.float32)
    dist_cube = np.zeros(n, dtype=np.float32)
    segments = []

    for i, pose in enumerate(tqdm(poses_doc["poses"], desc="Record")):
        wire32 = np.asarray(pose["wire32_rad"], dtype=np.float64)
        sim.set_pose_from_wire32_rad(wire32, cube_xyz=cube_xyz)
        views = sim.render_rgb_views()
        for v, cam in enumerate(CAM_NAMES):
            rgb_stack[i, v] = views[cam]
            if skel_stack is not None:
                skel_stack[i, v, :, :, 0] = sim.render_skeleton_mask(cam)

        states[i] = np.asarray(pose["state_norm"], dtype=np.float32)
        probe_ids[i] = int(pose["probe_id"])
        achieved = np.asarray(pose["ee_achieved_xyz"], dtype=np.float32)
        ee_xyz[i] = achieved
        dist_cube[i] = float(np.linalg.norm(achieved - cube_xyz))
        segments.append(pose.get("segment_hint", "unknown"))

    bundle = {
        "version": 1,
        "cam_names": CAM_NAMES,
        "cube_xyz": cube_xyz.astype(np.float32),
        "rgb": torch.from_numpy(rgb_stack),  # (N, V, H, W, 3)
        "state_norm": torch.from_numpy(states),
        "probe_ids": torch.from_numpy(probe_ids),
        "ee_achieved_xyz": torch.from_numpy(ee_xyz),
        "dist_to_cube_m": torch.from_numpy(dist_cube),
        "segment_hint": segments,
        "with_skeleton": args.with_skeleton,
    }
    if skel_stack is not None:
        bundle["skeleton"] = torch.from_numpy(skel_stack)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out_path)
    print(f"✅ Saved bundle ({n} probes) → {out_path}")


if __name__ == "__main__":
    main()
