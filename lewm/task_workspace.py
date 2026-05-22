"""
Fixed task-space safety polytope for GR-1 pickup (world frame, meters).

Corners are baked in below (convex hull + Bezier arc samples). Same geometry is
built independently in ``lewm_server`` (CEM final-step gate) and clients
(visualization only) — not sent over ZMQ.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gr1_config import COMPACT_WIRE_JOINTS, SCENE_PATH
from gr1_protocol import StandardScaler
from dataset.polytope_utils import EE_BODY
from gr1_scene_sync import (
    apply_sim_scene_to_qpos,
    build_qpos_from_wire32,
    ee_body_xyz,
    scene_snapshot,
    table_footprint_check,
)

# Squared slack outside {x | Hx <= d}; feasible if violation <= eps
DEFAULT_FEASIBILITY_EPS = 1e-4
INFEASIBLE_COST = 1e12

# Fixed task workspace corners (world frame) — edit here to retune the hull
TASK_WORKSPACE_CORNERS = np.array(
    [
        [0.0, 0.0, 0.82],
        [0.8, 0.0, 0.82],
        [0.45, -0.5, 0.75],
        [0.45, 0.5, 0.75],
        [0.45, -0.5, 1.25],
        [0.45, 0.5, 1.25],
        [0.45, 0.0, 1.5],
        [-0.1, -0.5, 1.5],
        [0.9, 0.0, 1.25],
        [0.27, -0.5, 0.75],
        [0.27, 0.5, 0.75],
        [0.63, -0.5, 1.25],
        [0.63, 0.5, 1.25],
    ],
    dtype=np.float64,
)

# Bezier arcs (0-based corner indices) for a smoother convex hull
_ARC_TRIPLETS = [
    (7, 6, 8),
    (2, 0, 3),
    (4, 6, 5),
]

ARC_SAMPLES_PER_TRIPLET = 6


def wire32_to_qpos(
    model: mujoco.MjModel, qpos: np.ndarray, wire32_rad: np.ndarray
) -> None:
    for i, name in enumerate(COMPACT_WIRE_JOINTS):
        j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j_id != -1:
            qpos[model.jnt_qposadr[j_id]] = float(wire32_rad[i])


def ee_halfspace_violation(ee: np.ndarray, H: np.ndarray, d: np.ndarray) -> float:
    slack = H @ np.asarray(ee, dtype=np.float64).reshape(3) - d
    return float(np.sum(np.maximum(0.0, slack) ** 2))


def _bezier_arc(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, n: int) -> np.ndarray:
    if n <= 0:
        return np.empty((0, 3), dtype=np.float64)
    t = np.linspace(0.0, 1.0, n + 2)[1:-1]
    p0, p1, p2 = np.asarray(p0), np.asarray(p1), np.asarray(p2)
    return np.array(
        [(1 - s) ** 2 * p0 + 2 * (1 - s) * s * p1 + s**2 * p2 for s in t],
        dtype=np.float64,
    )


def expand_corners_for_hull(
    corners: np.ndarray | None = None,
    samples_per_arc: int = ARC_SAMPLES_PER_TRIPLET,
) -> np.ndarray:
    corners = np.asarray(
        corners if corners is not None else TASK_WORKSPACE_CORNERS,
        dtype=np.float64,
    ).reshape(-1, 3)
    extra = [corners]
    for tri in _ARC_TRIPLETS:
        if max(tri) >= len(corners):
            continue
        extra.append(
            _bezier_arc(
                corners[tri[0]], corners[tri[1]], corners[tri[2]], samples_per_arc
            )
        )
    return np.vstack(extra)


@dataclass
class TaskWorkspacePolytope:
    vertices: np.ndarray  # (3, N)
    H: np.ndarray
    d: np.ndarray
    face_indices: np.ndarray
    corner_points: np.ndarray

    def find_faces(self):
        return None

    def find_halfplanes(self):
        return None


_DRAW_POLYTOPE_CACHE: TaskWorkspacePolytope | None = None


def get_task_workspace_draw_polytope(
    samples_per_arc: int = ARC_SAMPLES_PER_TRIPLET,
) -> TaskWorkspacePolytope:
    """Cached fixed hull for viz only (no extra MuJoCo model)."""
    global _DRAW_POLYTOPE_CACHE
    if _DRAW_POLYTOPE_CACHE is None:
        _DRAW_POLYTOPE_CACHE = build_task_workspace_polytope(
            samples_per_arc=samples_per_arc
        )
    return _DRAW_POLYTOPE_CACHE


def build_task_workspace_polytope(
    samples_per_arc: int = ARC_SAMPLES_PER_TRIPLET,
) -> TaskWorkspacePolytope:
    corners = TASK_WORKSPACE_CORNERS
    pts = expand_corners_for_hull(corners, samples_per_arc=samples_per_arc)

    hull = ConvexHull(pts)
    H = np.asarray(hull.equations[:, :3], dtype=np.float64)
    d = np.asarray(-hull.equations[:, 3], dtype=np.float64)

    return TaskWorkspacePolytope(
        vertices=pts.T,
        H=H,
        d=d,
        face_indices=np.asarray(hull.simplices, dtype=np.int64),
        corner_points=corners.copy(),
    )


class TaskWorkspaceMPCConstraint:
    """Fixed hull; CEM gate checks **final plan step** fingertip position only."""

    def __init__(
        self,
        scene_path: str = SCENE_PATH,
        feasibility_eps: float = DEFAULT_FEASIBILITY_EPS,
        samples_per_arc: int = ARC_SAMPLES_PER_TRIPLET,
    ):
        self.feasibility_eps = feasibility_eps
        self.poly = build_task_workspace_polytope(samples_per_arc=samples_per_arc)
        self._H = self.poly.H
        self._d = self.poly.d

        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        self.scaler = StandardScaler()
        self._baseline_qpos = self.model.qpos0.copy()
        self._baseline_sim_sync = True
        self.ee_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY
        )

    def get_draw_polytope(self) -> TaskWorkspacePolytope:
        return self.poly

    def get_halfspaces(self) -> tuple[np.ndarray, np.ndarray]:
        return self._H, self._d

    def set_baseline_from_wire32(
        self,
        wire32_rad: np.ndarray,
        *,
        sim_scene_sync: bool = True,
        cube_xyz: np.ndarray | None = None,
    ) -> None:
        self._baseline_qpos = build_qpos_from_wire32(
            self.model,
            wire32_rad,
            sim_scene_sync=sim_scene_sync,
            cube_xyz=cube_xyz,
        )
        self._baseline_sim_sync = bool(sim_scene_sync)

    def _fk_ee_after_plan_prefix(
        self,
        plan_norm: np.ndarray,
        end_exclusive: int,
        *,
        sim_scene_sync: bool | None = None,
        cube_xyz: np.ndarray | None = None,
    ) -> np.ndarray:
        """FK after applying plan[0:end_exclusive] sequentially (matches CEM horizon semantics)."""
        if sim_scene_sync is None:
            sim_scene_sync = getattr(self, "_baseline_sim_sync", True)
        q = np.array(self._baseline_qpos, dtype=np.float64, copy=True)
        for t in range(end_exclusive):
            wire_rad = self.scaler.unscale_action(plan_norm[t])
            wire32_to_qpos(self.model, q, wire_rad)
            if sim_scene_sync:
                apply_sim_scene_to_qpos(self.model, q, cube_xyz=cube_xyz)
        self.data.qpos[:] = q
        mujoco.mj_forward(self.model, self.data)
        return ee_body_xyz(self.model, self.data)

    def final_plan_step_ee(
        self,
        wire32_rad: np.ndarray,
        plan_norm: np.ndarray,
        *,
        cube_xyz: np.ndarray | None = None,
    ) -> np.ndarray:
        """Fingertip EE after applying the full planned horizon from the request-time baseline."""
        self.set_baseline_from_wire32(
            np.asarray(wire32_rad, dtype=np.float64), cube_xyz=cube_xyz
        )
        plan_norm = np.asarray(plan_norm, dtype=np.float64)
        if plan_norm.ndim == 1:
            plan_norm = plan_norm.reshape(1, -1)
        return self._fk_ee_after_plan_prefix(
            plan_norm, plan_norm.shape[0], cube_xyz=cube_xyz
        )

    def plan_violation(
        self,
        wire32_rad: np.ndarray,
        plan_norm: np.ndarray,
        *,
        check_all_steps: bool = False,
        cube_xyz: np.ndarray | None = None,
    ) -> float:
        self.set_baseline_from_wire32(
            np.asarray(wire32_rad, dtype=np.float64), cube_xyz=cube_xyz
        )
        plan_norm = np.asarray(plan_norm, dtype=np.float64)
        if plan_norm.ndim == 1:
            plan_norm = plan_norm.reshape(1, -1)
        n = plan_norm.shape[0]
        steps = range(n) if check_all_steps else [n - 1]
        max_v = 0.0
        for t in steps:
            ee = self._fk_ee_after_plan_prefix(plan_norm, t + 1, cube_xyz=cube_xyz)
            max_v = max(max_v, ee_halfspace_violation(ee, self._H, self._d))
        return max_v

    def is_feasible(
        self,
        wire32_rad: np.ndarray,
        plan_norm: np.ndarray,
        eps: Optional[float] = None,
        **kwargs,
    ) -> bool:
        eps = self.feasibility_eps if eps is None else eps
        return self.plan_violation(wire32_rad, plan_norm, **kwargs) <= eps

    def plan_violation_batch(
        self,
        wire32_rad: np.ndarray,
        plan_norm_bs: np.ndarray,
        *,
        check_all_steps: bool = False,
        cube_xyz: np.ndarray | None = None,
    ) -> np.ndarray:
        plan_norm_bs = np.asarray(plan_norm_bs, dtype=np.float64)
        if plan_norm_bs.ndim == 2:
            plan_norm_bs = plan_norm_bs[:, np.newaxis, :]
        n = plan_norm_bs.shape[0]
        out = np.zeros(n, dtype=np.float64)
        for s in range(n):
            out[s] = self.plan_violation(
                wire32_rad,
                plan_norm_bs[s],
                check_all_steps=check_all_steps,
                cube_xyz=cube_xyz,
            )
        return out

    def feasible_mask_batch(
        self,
        wire32_rad: np.ndarray,
        plan_norm_bs: np.ndarray,
        eps: Optional[float] = None,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        eps = self.feasibility_eps if eps is None else eps
        violations = self.plan_violation_batch(wire32_rad, plan_norm_bs, **kwargs)
        return violations <= eps, violations

    def fk_debug_report(
        self,
        wire32_rad: np.ndarray,
        plan_norm: np.ndarray,
        *,
        check_final_only: bool = True,
        cube_xyz: np.ndarray | None = None,
    ) -> dict:
        """
        Structured FK audit for blue-dot vs plan debugging.
        Compares full-chain final EE, last-row-only FK, and per-step EE.
        """
        wire32_rad = np.asarray(wire32_rad, dtype=np.float64).reshape(-1)
        plan_norm = np.asarray(plan_norm, dtype=np.float64)
        if plan_norm.ndim == 1:
            plan_norm = plan_norm.reshape(1, -1)
        n = int(plan_norm.shape[0])

        # --- Sim-aligned FK (root z=0.95, cube from request or default) ---
        self.set_baseline_from_wire32(
            wire32_rad, sim_scene_sync=True, cube_xyz=cube_xyz
        )
        ee_baseline = self._fk_ee_after_plan_prefix(plan_norm, 0)
        ee_per_step = [
            self._fk_ee_after_plan_prefix(plan_norm, t + 1).tolist() for t in range(n)
        ]
        ee_full = np.asarray(ee_per_step[-1], dtype=np.float64)
        ee_last_only = self._fk_ee_after_plan_prefix(plan_norm[-1:], 1)

        self.data.qpos[:] = self._baseline_qpos
        mujoco.mj_forward(self.model, self.data)
        snap_baseline = scene_snapshot(self.model, self.data, label="fk_baseline")

        q_final = np.array(self._baseline_qpos, dtype=np.float64, copy=True)
        wire_last = self.scaler.unscale_action(plan_norm[-1])
        wire32_to_qpos(self.model, q_final, wire_last)
        apply_sim_scene_to_qpos(self.model, q_final)
        self.data.qpos[:] = q_final
        mujoco.mj_forward(self.model, self.data)
        snap_final = scene_snapshot(self.model, self.data, label="fk_final")

        # --- Legacy FK (qpos0 root z=0) for regression ---
        q_leg = build_qpos_from_wire32(
            self.model, wire32_rad, sim_scene_sync=False, legacy_qpos0_only=True
        )
        self.data.qpos[:] = q_leg
        mujoco.mj_forward(self.model, self.data)
        ee_baseline_legacy = ee_body_xyz(self.model, self.data).tolist()
        wire32_to_qpos(self.model, q_leg, wire_last)
        self.data.qpos[:] = q_leg
        mujoco.mj_forward(self.model, self.data)
        ee_final_legacy = ee_body_xyz(self.model, self.data).tolist()

        wire_delta = float(np.linalg.norm(wire_last - wire32_rad))
        chain_matches_last = bool(
            np.allclose(ee_full, ee_last_only, atol=1e-5, rtol=1e-5)
        )
        legacy_final_delta = float(
            np.linalg.norm(np.asarray(ee_final_legacy) - ee_full)
        )

        viol_full = self.plan_violation(
            wire32_rad,
            plan_norm,
            check_all_steps=not check_final_only,
            cube_xyz=cube_xyz,
        )
        viol_last_only = float(ee_halfspace_violation(ee_full, self._H, self._d))

        per_step_table = [table_footprint_check(ee) for ee in ee_per_step]

        report = {
            "ee_body": EE_BODY,
            "n_plan_steps": n,
            "sim_scene_sync": True,
            "scene_cube_xyz": (cube_xyz.tolist() if cube_xyz is not None else None),
            "ee_baseline_xyz": ee_baseline.tolist(),
            "ee_per_step_xyz": ee_per_step,
            "ee_full_chain_final_xyz": ee_full.tolist(),
            "ee_last_row_only_xyz": ee_last_only.tolist(),
            "ee_baseline_legacy_root_z0_xyz": ee_baseline_legacy,
            "ee_final_legacy_root_z0_xyz": ee_final_legacy,
            "legacy_final_vs_sim_sync_delta_m": legacy_final_delta,
            "chain_final_equals_last_row_fk": chain_matches_last,
            "wire32_baseline_rad": wire32_rad.tolist(),
            "wire32_last_row_rad": wire_last.tolist(),
            "wire32_delta_norm_rad": wire_delta,
            "plan_last_row_norm": plan_norm[-1].tolist(),
            "table_footprint_per_step": per_step_table,
            "scene_snapshot_baseline": snap_baseline,
            "scene_snapshot_final": snap_final,
            "violation_full_chain": float(viol_full),
            "violation_last_row_only_ee": viol_last_only,
            "feasible_full_chain": bool(viol_full <= self.feasibility_eps),
            "feasible_last_row_only_ee": bool(viol_last_only <= self.feasibility_eps),
        }
        return report


__all__ = [
    "INFEASIBLE_COST",
    "TASK_WORKSPACE_CORNERS",
    "TaskWorkspaceMPCConstraint",
    "TaskWorkspacePolytope",
    "build_task_workspace_polytope",
    "get_task_workspace_draw_polytope",
    "ee_halfspace_violation",
    "fk_debug_report",
]
