#!/usr/bin/env python3
"""B2: Validate joint-space targets (or IK legacy EE-only targets)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import numpy as np
from dataset.task_workspace_probe.probe_sim import ProbeSimulator
from dataset.task_workspace_probe.segments import segment_hint
from gr1_protocol import StandardScaler
from tqdm import tqdm


def _pose_from_wire32(
    sim: ProbeSimulator,
    probe_id: int,
    ee_target: list,
    wire32_rad: list,
    *,
    cube_xyz,
    require_hull: bool,
) -> tuple[dict | None, str]:
    wire32 = np.asarray(wire32_rad, dtype=np.float64)
    sim.set_pose_from_wire32_rad(wire32)
    achieved = sim.fingertip_xyz()
    viol = sim.hull_violation(achieved)
    if require_hull and viol > sim._feas_eps:
        return None, "hull_fail"

    scaler = StandardScaler()
    state_norm = scaler.scale_state(wire32.astype(np.float32))
    hint = segment_hint(achieved, cube_xyz=cube_xyz)
    dist_m = float(
        np.linalg.norm(
            np.asarray(achieved, dtype=np.float64)
            - np.asarray(cube_xyz, dtype=np.float64)
        )
    )
    err = float(np.linalg.norm(achieved - np.asarray(ee_target, dtype=np.float64)))
    return (
        {
            "probe_id": probe_id,
            "ee_target_xyz": ee_target,
            "ee_achieved_xyz": achieved.tolist(),
            "dist_to_cube_m": dist_m,
            "ik_error_m": err,
            "hull_violation": viol,
            "wire32_rad": wire32.astype(np.float64).tolist(),
            "state_norm": state_norm.astype(np.float64).tolist(),
            "segment_hint": hint,
            "status": "ok",
            "source": "joint_sample",
        },
        "ok",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--targets",
        type=str,
        default=str(
            REPO_DIR / "datasets/workspace_probe_grasp/workspace_probe_targets.json"
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(
            REPO_DIR / "datasets/workspace_probe_grasp/workspace_probe_poses.json"
        ),
    )
    parser.add_argument(
        "--force-ik",
        action="store_true",
        help="Run IK even when targets include wire32_rad (legacy)",
    )
    args = parser.parse_args()

    targets_doc = json.loads(Path(args.targets).read_text())
    cube_xyz = targets_doc.get("cube_xyz")
    hull_filter = targets_doc.get("hull_filter", True)

    sim = ProbeSimulator()
    sim.reset_probe_scene(lock_posture=True)

    poses = []
    ik_fail = 0
    hull_fail = 0
    joint_ok = 0

    n_targets = len(targets_doc["targets"])
    use_joint = not args.force_ik and any(
        "wire32_rad" in t for t in targets_doc["targets"]
    )
    desc = "poses (joint)" if use_joint else "IK"
    pbar = tqdm(targets_doc["targets"], desc=desc, total=n_targets)

    for t in pbar:
        probe_id = t["probe_id"]
        ee = t["ee_xyz"]

        if use_joint and "wire32_rad" in t:
            pose, status = _pose_from_wire32(
                sim,
                probe_id,
                ee,
                t["wire32_rad"],
                cube_xyz=cube_xyz,
                require_hull=hull_filter,
            )
            if pose is None:
                hull_fail += 1
            else:
                poses.append(pose)
                joint_ok += 1
        else:
            q, diag = sim.solve_probe_ik(ee)
            if q is None:
                st = diag.get("status", "fail")
                if st == "hull_fail":
                    hull_fail += 1
                else:
                    ik_fail += 1
            else:
                achieved = diag.get("ee_achieved_xyz", sim.fingertip_xyz().tolist())
                hint = segment_hint(achieved, cube_xyz=cube_xyz)
                dist_m = float(
                    np.linalg.norm(
                        np.asarray(achieved, dtype=np.float64)
                        - np.asarray(cube_xyz, dtype=np.float64)
                    )
                )
                poses.append(
                    {
                        "probe_id": probe_id,
                        "ee_target_xyz": t["ee_xyz"],
                        "ee_achieved_xyz": achieved,
                        "dist_to_cube_m": dist_m,
                        "ik_error_m": diag.get("ik_error_m"),
                        "hull_violation": diag.get("hull_violation"),
                        "wire32_rad": diag["wire32_rad"],
                        "state_norm": diag["state_norm"],
                        "segment_hint": hint,
                        "status": "ok",
                        "source": "ik",
                    }
                )

        n_ok = len(poses)
        postfix = dict(
            ok=n_ok,
            hull_fail=hull_fail,
            rate=f"{100.0 * n_ok / max(pbar.n, 1):.1f}%",
        )
        if use_joint:
            postfix["joint"] = joint_ok
        else:
            postfix["ik_fail"] = ik_fail
        pbar.set_postfix(**postfix, refresh=False)
        pbar.set_description(f"{desc} ({n_ok}/{n_targets} ok)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "cube_xyz": cube_xyz,
        "stats": {
            "requested": len(targets_doc["targets"]),
            "accepted": len(poses),
            "ik_fail": ik_fail,
            "hull_fail": hull_fail,
            "joint_pass_through": joint_ok if use_joint else 0,
        },
        "poses": poses,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(
        f"✅ Accepted {len(poses)}/{len(targets_doc['targets'])} poses → {out_path} "
        f"(ik_fail={ik_fail}, hull_fail={hull_fail})"
    )


if __name__ == "__main__":
    main()
