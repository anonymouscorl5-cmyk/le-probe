#!/usr/bin/env python3
"""B1: Sample reachable probe configs in joint space, FK to fingertip EE."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from dataset.task_workspace_probe.probe_sim import ProbeSimulator
from dataset.task_workspace_probe.sampling import (
    movable_indices_for_mode,
    sample_joint_space_configs,
)
from gr1_scene_sync import DEFAULT_CUBE_XYZ


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--joint-mode",
        choices=("wild", "ik"),
        default="wild",
        help="wild: indices 16–31 (dataset wild_reset); ik: ik_joints.txt only",
    )
    parser.add_argument(
        "--no-hull-filter",
        action="store_true",
        help="Keep all joint samples (EE may lie outside MPC hull)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(
            REPO_DIR / "datasets/workspace_probe_grasp/workspace_probe_targets.json"
        ),
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    movable = movable_indices_for_mode(args.joint_mode)
    sim = ProbeSimulator()
    configs, stats = sample_joint_space_configs(
        args.n,
        rng,
        sim,
        movable_indices=movable,
        hull_filter=not args.no_hull_filter,
    )
    cube = np.asarray(DEFAULT_CUBE_XYZ, dtype=np.float64)

    targets = []
    for i, cfg in enumerate(configs):
        ee = np.asarray(cfg["ee_xyz"], dtype=np.float64)
        targets.append(
            {
                "probe_id": i,
                "ee_xyz": cfg["ee_xyz"],
                "wire32_rad": cfg["wire32_rad"],
                "state_norm": cfg["state_norm"],
                "sample_method": f"joint_uniform_{args.joint_mode}",
                "dist_to_cube_m": float(np.linalg.norm(ee - cube)),
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 2,
        "ee_body": "R_index_tip_link",
        "cube_xyz": cube.tolist(),
        "joint_mode": args.joint_mode,
        "hull_filter": not args.no_hull_filter,
        "n_targets": len(targets),
        "sampling_stats": stats,
        "targets": targets,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(
        f"✅ Wrote {len(targets)} joint-space targets ({args.joint_mode}, "
        f"hull_filter={not args.no_hull_filter}) → {out_path}"
    )


if __name__ == "__main__":
    main()
