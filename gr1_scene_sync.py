"""
Shared MuJoCo scene alignment + debug snapshots for GR-1 sim vs task-workspace FK.

Sim / teleop always run with floating root at z=0.95. Legacy task_workspace FK used
qpos0 only (root z=0), which shifts EE world coordinates (~0.65 m in z, wrong xy context).
"""

from __future__ import annotations

import os
from typing import Any

import mujoco
import numpy as np

from gr1_config import COMPACT_WIRE_JOINTS, SCENE_PATH
from dataset.polytope_utils import EE_BODY

# Match simulation_base.reset_env / lewm_server.render_skeleton_mask
SIM_ROOT_XYZ = np.array([0.0, 0.0, 0.95], dtype=np.float64)
DEFAULT_CUBE_XYZ = np.array([0.5, 0.0, 0.82], dtype=np.float64)

# Table / cube sampling (simulation_base.reset_env margins)
TABLE_TOP_Z = 0.82
CUBE_X_RANGE = (0.27, 0.63)
CUBE_Y_RANGE = (-0.23, 0.23)

SCENE_DEBUG = os.environ.get("LEWM_SCENE_DEBUG", "0").strip().lower() in (
    "1",
    "true",
    "yes",
)


def _joint_qpos_slice(model: mujoco.MjModel, joint_name: str) -> slice | None:
    j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if j_id < 0:
        return None
    adr = int(model.jnt_qposadr[j_id])
    jtype = int(model.jnt_type[j_id])
    width = {0: 7, 1: 4, 2: 1, 3: 1}.get(jtype, 1)
    return slice(adr, adr + width)


def apply_sim_scene_to_qpos(
    model: mujoco.MjModel,
    q: np.ndarray,
    *,
    cube_xyz: np.ndarray | None = None,
    sync_root: bool = True,
) -> None:
    """Apply the same floating-base / default cube pose the teleop sim uses."""
    if sync_root:
        sl = _joint_qpos_slice(model, "root")
        if sl is not None:
            # root is a free joint (7 qpos); sim only sets translation like simulation_base
            q[sl.start : sl.start + 3] = np.asarray(SIM_ROOT_XYZ, dtype=np.float64)
    sl = _joint_qpos_slice(model, "cube_joint")
    if sl is not None:
        pos = (
            DEFAULT_CUBE_XYZ
            if cube_xyz is None
            else np.asarray(cube_xyz, dtype=np.float64)
        )
        q[sl.start : sl.start + 3] = pos[:3]
        if sl.stop - sl.start >= 7:
            q[sl.start + 3 : sl.start + 7] = [1.0, 0.0, 0.0, 0.0]


def wire32_to_qpos_on_q(
    model: mujoco.MjModel, q: np.ndarray, wire32_rad: np.ndarray
) -> None:
    wire32_rad = np.asarray(wire32_rad, dtype=np.float64).reshape(-1)
    for i, name in enumerate(COMPACT_WIRE_JOINTS):
        j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j_id != -1:
            q[model.jnt_qposadr[j_id]] = float(wire32_rad[i])


def ee_body_xyz(
    model: mujoco.MjModel, data: mujoco.MjData, body: str = EE_BODY
) -> np.ndarray:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    if bid < 0:
        return np.zeros(3, dtype=np.float64)
    return np.asarray(data.xpos[bid], dtype=np.float64).copy()


def table_footprint_check(ee: np.ndarray) -> dict[str, Any]:
    """Rough axis-aligned check vs cube randomization / table band (not the full hull)."""
    ee = np.asarray(ee, dtype=np.float64).reshape(3)
    return {
        "ee_xyz": ee.tolist(),
        "x_in_cube_band": bool(CUBE_X_RANGE[0] <= ee[0] <= CUBE_X_RANGE[1]),
        "y_in_cube_band": bool(CUBE_Y_RANGE[0] <= ee[1] <= CUBE_Y_RANGE[1]),
        "z_above_table_top": bool(ee[2] >= TABLE_TOP_Z - 0.05),
        "cube_x_range": list(CUBE_X_RANGE),
        "cube_y_range": list(CUBE_Y_RANGE),
        "table_top_z": TABLE_TOP_Z,
    }


def scene_snapshot(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    label: str = "",
    extra_bodies: tuple[str, ...] = ("R_thumb_tip_link", "right_hand_pitch_link"),
) -> dict[str, Any]:
    root_sl = _joint_qpos_slice(model, "root")
    cube_sl = _joint_qpos_slice(model, "cube_joint")
    root_q = data.qpos[root_sl].tolist() if root_sl is not None else None
    cube_q = data.qpos[cube_sl].tolist() if cube_sl is not None else None

    bodies = {EE_BODY: ee_body_xyz(model, data, EE_BODY).tolist()}
    for b in extra_bodies:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
        if bid >= 0:
            bodies[b] = np.asarray(data.xpos[bid], dtype=np.float64).tolist()

    snap = {
        "label": label,
        "root_qpos_xyz": root_q[:3] if root_q else None,
        "cube_qpos_xyz": cube_q[:3] if cube_q else None,
        "bodies": bodies,
        "table_footprint_index_tip": table_footprint_check(bodies[EE_BODY]),
    }
    if cube_q and EE_BODY in bodies:
        cube_xyz = np.asarray(cube_q[:3], dtype=np.float64)
        tip = np.asarray(bodies[EE_BODY], dtype=np.float64)
        snap["index_to_cube_xyz_dist"] = float(np.linalg.norm(tip - cube_xyz))
    return snap


def log_scene_snapshot(
    snapshot: dict[str, Any], *, prefix: str = "[SCENE_DEBUG]"
) -> None:
    if not SCENE_DEBUG:
        return
    label = snapshot.get("label", "")
    root = snapshot.get("root_qpos_xyz")
    cube = snapshot.get("cube_qpos_xyz")
    tip = snapshot.get("bodies", {}).get(EE_BODY)
    fp = snapshot.get("table_footprint_index_tip", {})
    print(f"{prefix} [{label}] root_xyz={root} cube_xyz={cube}")
    if tip is not None:
        print(
            f"{prefix} [{label}] {EE_BODY}={tuple(round(v, 4) for v in tip)} "
            f"x_band={fp.get('x_in_cube_band')} y_band={fp.get('y_in_cube_band')} "
            f"z_ok={fp.get('z_above_table_top')}"
        )
    if "index_to_cube_xyz_dist" in snapshot:
        print(
            f"{prefix} [{label}] |index-cube|={snapshot['index_to_cube_xyz_dist']:.4f} m"
        )


def build_qpos_from_wire32(
    model: mujoco.MjModel,
    wire32_rad: np.ndarray,
    *,
    sim_scene_sync: bool = True,
    cube_xyz: np.ndarray | None = None,
    legacy_qpos0_only: bool = False,
) -> np.ndarray:
    """Build full qpos for FK: wire32 joints + optional sim root/cube."""
    q = model.qpos0.copy()
    if legacy_qpos0_only:
        wire32_to_qpos_on_q(model, q, wire32_rad)
        return q
    wire32_to_qpos_on_q(model, q, wire32_rad)
    if sim_scene_sync:
        apply_sim_scene_to_qpos(model, q, cube_xyz=cube_xyz, sync_root=True)
    return q
