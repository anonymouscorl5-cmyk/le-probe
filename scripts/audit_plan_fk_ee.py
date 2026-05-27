#!/usr/bin/env python3
"""
Recompute ee_per_step_xyz from a lewm lifecycle audit JSON.

Uses the same FK as lewm_server / task_workspace (wire32 baseline + normalized plan).
"""

# --- Path Stabilization ---
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO_ROOT = os.path.abspath(os.path.join(ROOT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# --------------------------

import argparse
import json
from pathlib import Path

import numpy as np

from lewm.task_workspace import TaskWorkspaceMPCConstraint


def load_audit(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def ee_per_step_from_audit(
    audit: dict,
    *,
    use_stored_baseline: bool = True,
) -> list[list[float]]:
    plan_norm = np.asarray(audit["best_plan_norm"], dtype=np.float64)
    print(f"plan_norm: {plan_norm} {plan_norm.shape}")

    if use_stored_baseline and "request_wire32_rad" in audit:
        wire32_rad = np.asarray(audit["request_wire32_rad"], dtype=np.float64)
    else:
        raise KeyError(
            "Audit missing request_wire32_rad; re-run server with FK debug logging."
        )

    cube_xyz = None
    if audit.get("scene_cube_xyz") is not None:
        cube_xyz = np.asarray(audit["scene_cube_xyz"], dtype=np.float64)
    elif audit.get("fk_debug", {}).get("scene_cube_xyz") is not None:
        cube_xyz = np.asarray(audit["fk_debug"]["scene_cube_xyz"], dtype=np.float64)

    tw = TaskWorkspaceMPCConstraint()
    report = tw.fk_debug_report(
        wire32_rad, plan_norm, check_final_only=True, cube_xyz=cube_xyz
    )
    return report["ee_per_step_xyz"]


def main() -> None:
    default_audit = (
        Path(REPO_ROOT) / "inference_history_lewm" / "lewm_lifecycle_audit_35.json"
    )
    parser = argparse.ArgumentParser(
        description="FK ee_per_step_xyz for best_plan_norm in a lifecycle audit."
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=default_audit,
        help=f"Path to lifecycle audit JSON (default: {default_audit})",
    )
    parser.add_argument(
        "--compare-stored",
        action="store_true",
        help="If audit has fk_debug.ee_per_step_xyz, print max abs diff vs recompute.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Indent JSON output.",
    )
    args = parser.parse_args()

    audit = load_audit(args.audit.resolve())
    ee_steps = ee_per_step_from_audit(audit)

    out = {
        "audit": str(args.audit.resolve()),
        "n_plan_steps": len(ee_steps),
        "ee_per_step_xyz": ee_steps,
        "ee_last_step_xyz": ee_steps[-1] if ee_steps else None,
        "plan_final_ee_xyz_in_audit": audit.get("plan_final_ee_xyz"),
    }

    if args.compare_stored and "fk_debug" in audit:
        stored = audit["fk_debug"].get("ee_per_step_xyz")
        if stored is not None:
            a = np.asarray(stored, dtype=np.float64)
            b = np.asarray(ee_steps, dtype=np.float64)
            out["max_abs_diff_vs_stored_fk_debug"] = float(np.max(np.abs(a - b)))

    print(f"ee_last_step_xyz: {out['ee_last_step_xyz']}")
    # print(json.dumps(out, indent=2 if args.pretty else None))


if __name__ == "__main__":
    main()
