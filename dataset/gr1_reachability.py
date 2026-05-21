"""
MuJoCo + pycapacity reachable workspace for Fourier GR-1.
Mirrors: https://auctus-team.github.io/pycapacity/examples/reachable_workspace.html
but uses the project's MuJoCo scene instead of Pinocchio/Panda.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable, Literal, Optional
import mujoco
import numpy as np
import rerun as rr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gr1_config import (
    COMPACT_WIRE_JOINTS,
    JOINT_LIMITS_MIN,
    JOINT_LIMITS_MAX,
    SCENE_PATH,
)

try:
    import pycapacity.robot as pycap
except ImportError as e:
    raise ImportError("pip install pycapacity") from e

# Matches goal_mapper / lewm_server precision box (indices 17-20 in 32-d protocol)
PRECISION_ARM_WIRE_IDX = [17, 18, 19, 20]
ARM_MIN = np.array([-0.312, -0.098, -0.156, -0.098], dtype=np.float64)
ARM_MAX = np.array([1.172, 0.098, 0.236, 0.098], dtype=np.float64)

# Right arm (16-22) + right hand proximal joints (23-28)
DEFAULT_ACTIVE_WIRE_IDX = list(range(16, 29))
# Faster subset for live teleop overlay (7-DoF arm only)
TELEOP_ACTIVE_WIRE_IDX = list(range(16, 23))

EE_BODY = "R_index_tip_link"

RIGHT_ARM_BODY_CHAIN = [
    "right_upper_arm_pitch_link",
    "right_upper_arm_roll_link",
    "right_upper_arm_yaw_link",
    "right_lower_arm_pitch_link",
    "right_hand_yaw_link",
    "right_hand_roll_link",
    "right_hand_pitch_link",
    "R_index_tip_link",
    "R_thumb_tip_link",
]


@dataclass
class ReachabilityConfig:
    time_horizon: float = 0.25  # ~2-3 control steps at 10 Hz
    dq_max_rad_s: float = 1.5  # tune to match max_delta in MPC
    limit_mode: Literal["xml", "mpc_box", "hybrid"] = "hybrid"
    convex_hull: bool = True
    n_samples: int = 3
    facet_dim: int = 2
    calculate_faces: bool = True


class GR1ReachabilityEngine:
    def __init__(self, scene_path: str = SCENE_PATH, active_wire_idx=None):
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        self.active_wire_idx = list(active_wire_idx or DEFAULT_ACTIVE_WIRE_IDX)
        self._wire_to_qpos = []
        for i in self.active_wire_idx:
            name = COMPACT_WIRE_JOINTS[i]
            j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if j_id == -1:
                raise ValueError(f"Joint not in model: {name}")
            self._wire_to_qpos.append(self.model.jnt_qposadr[j_id])

        self.ee_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY
        )
        self._baseline_qpos = self.model.qpos0.copy()
        root_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        if root_id != -1:
            q_idx = self.model.jnt_qposadr[root_id]
            self._baseline_qpos[q_idx : q_idx + 3] = [0.0, 0.0, 0.95]

    def set_baseline_from_qpos(self, full_qpos: np.ndarray):
        self._baseline_qpos = np.array(full_qpos, dtype=np.float64, copy=True)

    def set_baseline_from_wire32(self, wire32_rad: np.ndarray):
        q = self._baseline_qpos.copy()
        for i, name in enumerate(COMPACT_WIRE_JOINTS):
            j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if j_id != -1:
                q[self.model.jnt_qposadr[j_id]] = wire32_rad[i]
        self.set_baseline_from_qpos(q)

    def _q_limits(self, mode: str) -> tuple[np.ndarray, np.ndarray]:
        n = len(self.active_wire_idx)
        q_min = np.zeros(n, dtype=np.float64)
        q_max = np.zeros(n, dtype=np.float64)
        for k, wire_i in enumerate(self.active_wire_idx):
            if mode in ("mpc_box", "hybrid") and wire_i in PRECISION_ARM_WIRE_IDX:
                j = PRECISION_ARM_WIRE_IDX.index(wire_i)
                q_min[k], q_max[k] = ARM_MIN[j], ARM_MAX[j]
            else:
                q_min[k] = JOINT_LIMITS_MIN[wire_i]
                q_max[k] = JOINT_LIMITS_MAX[wire_i]
        return q_min, q_max

    def make_forward_fn(self) -> Callable[[np.ndarray], np.ndarray]:
        def fk(q_active: np.ndarray) -> np.ndarray:
            q = self._baseline_qpos.copy()
            for val, q_idx in zip(np.asarray(q_active).flatten(), self._wire_to_qpos):
                q[q_idx] = val
            self.data.qpos[:] = q
            mujoco.mj_forward(self.model, self.data)
            return self.data.xpos[self.ee_body_id].copy()

        return fk

    def compute(
        self,
        q0_active: Optional[np.ndarray] = None,
        cfg: Optional[ReachabilityConfig] = None,
    ):
        cfg = cfg or ReachabilityConfig()
        q0 = (
            np.array(q0_active, dtype=np.float64)
            if q0_active is not None
            else np.array(
                [self._baseline_qpos[q_idx] for q_idx in self._wire_to_qpos],
                dtype=np.float64,
            )
        )
        q_min, q_max = self._q_limits(cfg.limit_mode)
        n = len(q0)
        dq_max = np.full(n, cfg.dq_max_rad_s)
        dq_min = -dq_max

        opt = {
            "calculate_faces": cfg.calculate_faces,
            "convex_hull": cfg.convex_hull,
            "n_samples": cfg.n_samples,
            "facet_dim": min(cfg.facet_dim, n - 1) if n > 1 else 0,
        }
        return pycap.reachable_space_nonlinear(
            forward_func=self.make_forward_fn(),
            q0=q0,
            q_max=q_max,
            q_min=q_min,
            dq_max=dq_max,
            dq_min=dq_min,
            time_horizon=cfg.time_horizon,
            options=opt,
        )


def _camera_world_to_image(
    model: mujoco.MjModel, data: mujoco.MjData, cam_id: int
) -> tuple:
    """World→camera rotation used by MuJoCo renderer / skeleton priors."""
    R_cam = data.cam_xmat[cam_id].reshape(3, 3) @ np.array(
        [[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64
    )
    t_cam = data.cam_xpos[cam_id].copy()
    return R_cam, t_cam


def draw_polytope_on_rgb(
    rgb: np.ndarray,
    poly,
    cam_name: str,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    ee_body: str = EE_BODY,
    wire_color: tuple[int, int, int] = (0, 0, 255),
    ee_color: tuple[int, int, int] = (0, 0, 255),
    fill_alpha: float = 0.12,
) -> np.ndarray:
    """
    Project reachable polytope wireframe (+ current EE dot) onto a MuJoCo camera RGB frame.

    Uses the same projection path as ``dataset/skeleton/projection_utils`` and ``lewm_server``.
    """
    from dataset.skeleton.projection_utils import get_projection_matrix, project_point

    try:
        import cv2
    except ImportError as e:
        raise ImportError("opencv-python required for 2D polytope overlay") from e

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cam_id == -1:
        return rgb

    h, w = rgb.shape[:2]
    K = get_projection_matrix(cam_id, model, w, h)
    R_cam, t_cam = _camera_world_to_image(model, data, cam_id)

    out = np.asarray(rgb, dtype=np.uint8)
    if not out.flags.writeable:
        out = out.copy()

    verts = np.asarray(poly.vertices.T, dtype=np.float64)
    face_indices = getattr(poly, "face_indices", None)
    if face_indices is None or len(face_indices) == 0:
        if getattr(poly, "faces", None) is not None:
            poly.find_faces()
            face_indices = getattr(poly, "face_indices", None)

    projected: list[np.ndarray] = []
    for v in verts:
        p2d, _ = project_point(v, K, R_cam, t_cam)
        if p2d is not None:
            projected.append(p2d)

    if projected and fill_alpha > 0:
        hull_pts = np.array(projected, dtype=np.float32)
        if len(hull_pts) >= 3:
            hull = cv2.convexHull(hull_pts).astype(np.int32)
            overlay = out.copy()
            cv2.fillConvexPoly(overlay, hull, wire_color, lineType=cv2.LINE_AA)
            cv2.addWeighted(overlay, fill_alpha, out, 1.0 - fill_alpha, 0, out)

    if face_indices is not None and len(face_indices) > 0:
        faces = np.asarray(face_indices, dtype=np.int64)
        for tri in faces:
            pts = []
            for vi in (tri[0], tri[1], tri[2]):
                p2d, _ = project_point(verts[vi], K, R_cam, t_cam)
                if p2d is not None:
                    pts.append((int(round(p2d[0])), int(round(p2d[1]))))
            if len(pts) == 3:
                tri_np = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(out, [tri_np], True, wire_color, 1, cv2.LINE_AA)
    elif projected:
        for p in projected:
            cv2.circle(
                out,
                (int(round(p[0])), int(round(p[1]))),
                2,
                wire_color,
                -1,
                cv2.LINE_AA,
            )

    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_body)
    if ee_id != -1:
        p2d, _ = project_point(data.xpos[ee_id], K, R_cam, t_cam)
        if p2d is not None:
            cv2.circle(
                out,
                (int(round(p2d[0])), int(round(p2d[1]))),
                5,
                ee_color,
                -1,
                cv2.LINE_AA,
            )

    return out


def log_polytope_rerun(
    poly,
    entity_path: str = "world/reachable_workspace",
    wireframe_path: str = "world/reachable_wireframe",
    wireframe: bool = True,
):
    """Log reachable polytope mesh (+ optional wireframe) to Rerun."""
    verts = poly.vertices.T  # (N, 3)
    face_indices = getattr(poly, "face_indices", None)
    if face_indices is not None and len(face_indices) > 0:
        faces = np.asarray(face_indices, dtype=np.int64)
        rr.log(
            entity_path,
            rr.Mesh3D(
                vertex_positions=verts,
                triangle_indices=faces.astype(np.uint32),
                albedo_factor=[0.1, 0.4, 1.0, 0.2],
            ),
        )
        if wireframe:
            strips = []
            for tri in faces:
                p0, p1, p2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
                strips.append(np.array([p0, p1, p2, p0]))
            rr.log(
                wireframe_path,
                rr.LineStrips3D(strips, radii=0.0015, colors=[0, 80, 255]),
            )
    else:
        rr.log(
            entity_path,
            rr.Points3D(verts, radii=0.008, colors=[50, 120, 255]),
        )


def apply_active_joint_motion(
    engine: "GR1ReachabilityEngine",
    q_active: np.ndarray,
) -> None:
    """Write reduced joint vector into the engine baseline qpos."""
    for val, q_idx in zip(np.asarray(q_active).flatten(), engine._wire_to_qpos):
        engine._baseline_qpos[q_idx] = float(val)


def teleop_reachability_config(horizon: float = 0.25) -> ReachabilityConfig:
    """Low-cost settings for live Rerun overlay during teleop."""
    return ReachabilityConfig(
        time_horizon=horizon,
        limit_mode="hybrid",
        n_samples=2,
        facet_dim=1,
        calculate_faces=True,
    )


def plot_polytope_matplotlib(poly, ax=None):
    from pycapacity.visual import plot_polytope

    plot_polytope(poly, plot=ax, wireframe=True, alpha=0.15, show_vertices=False)


def log_right_arm_skeleton_rerun(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    entity_path: str = "world/right_arm_skeleton",
):
    """Log right arm + hand chain as a 3D polyline for Rerun spatial context."""
    points = []
    for name in RIGHT_ARM_BODY_CHAIN:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id != -1:
            points.append(data.xpos[body_id].copy())
    if len(points) >= 2:
        rr.log(
            entity_path,
            rr.LineStrips3D([np.array(points)], radii=0.006, colors=[200, 200, 200]),
        )


def log_ee_anchor_rerun(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    entity_path: str = "world/ee_index_tip",
):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY)
    if body_id != -1:
        rr.log(
            entity_path,
            rr.Points3D([data.xpos[body_id]], radii=0.015, colors=[255, 80, 80]),
        )


def _mesh_geom_groups_to_log(model: mujoco.MjModel) -> set[int]:
    """Pick visual mesh groups; skip MuJoCo Menagerie collision hulls (group 2)."""
    mesh_groups = {
        int(model.geom_group[g])
        for g in range(model.ngeom)
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
    }
    if 3 in mesh_groups:
        return {3}
    if 0 in mesh_groups:
        return {0}
    return mesh_groups - {2}


def _geom_quat_xyzw(xmat: np.ndarray) -> np.ndarray:
    """MuJoCo geom rotation matrix → quaternion (xyzw for Rerun)."""
    quat_wxyz = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat_wxyz, xmat.reshape(9).astype(np.float64))
    return quat_wxyz[[1, 2, 3, 0]]


def log_mujoco_scene_rerun(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    prefix: str = "world/robot",
    geom_groups: set[int] | None = None,
    rgba: tuple[float, float, float, float] = (0.82, 0.82, 0.85, 1.0),
):
    """
    Log MuJoCo scene geoms (mesh + capsule + box) in world frame for Rerun 3D.

    MuJoCo's native renderer draws all geom types; logging meshes alone leaves GR-1
    looking like disconnected shards (capsule/box geoms were missing).
    Menagerie Panda: skip collision mesh group 2, prefer visual group 3.
    """
    allowed = (
        geom_groups if geom_groups is not None else _mesh_geom_groups_to_log(model)
    )
    color = np.array(rgba, dtype=np.float32)

    cap_lengths, cap_radii, cap_centers, cap_quats = [], [], [], []
    box_sizes, box_centers, box_quats = [], [], []

    for gid in range(model.ngeom):
        gtype = int(model.geom_type[gid])
        if gtype == int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue
        if (
            gtype == int(mujoco.mjtGeom.mjGEOM_MESH)
            and int(model.geom_group[gid]) not in allowed
        ):
            continue

        R = data.geom_xmat[gid].reshape(3, 3)
        p = data.geom_xpos[gid].copy()
        quat = _geom_quat_xyzw(R)

        if gtype == int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh_id = model.geom_dataid[gid]
            vertadr = model.mesh_vertadr[mesh_id]
            vertnum = model.mesh_vertnum[mesh_id]
            faceadr = model.mesh_faceadr[mesh_id]
            facenum = model.mesh_facenum[mesh_id]
            if vertnum == 0 or facenum == 0:
                continue

            verts_local = (
                model.mesh_vert[vertadr : vertadr + vertnum].reshape(-1, 3).copy()
            )
            verts_local *= model.geom_size[gid]
            verts_world = (R @ verts_local.T).T + p
            faces = model.mesh_face[faceadr : faceadr + facenum].reshape(-1, 3)

            body_id = model.geom_bodyid[gid]
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if not body_name:
                body_name = f"body_{body_id}"

            rr.log(
                f"{prefix}/meshes/{body_name}/geom_{gid}",
                rr.Mesh3D(
                    vertex_positions=verts_world.astype(np.float32),
                    triangle_indices=faces.astype(np.uint32),
                    albedo_factor=list(rgba),
                ),
            )

        elif gtype == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            cap_radii.append(float(model.geom_size[gid, 0]))
            cap_lengths.append(float(2.0 * model.geom_size[gid, 1]))
            cap_centers.append(p)
            cap_quats.append(quat)

        elif gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
            box_sizes.append((2.0 * model.geom_size[gid]).astype(np.float32))
            box_centers.append(p)
            box_quats.append(quat)

    if cap_centers:
        rr.log(
            f"{prefix}/capsules",
            rr.Capsules3D(
                lengths=np.array(cap_lengths, dtype=np.float32),
                radii=np.array(cap_radii, dtype=np.float32),
                translations=np.array(cap_centers, dtype=np.float32),
                quaternions=np.array(cap_quats, dtype=np.float32),
                colors=color,
            ),
        )
    if box_centers:
        rr.log(
            f"{prefix}/boxes",
            rr.Boxes3D(
                sizes=np.array(box_sizes, dtype=np.float32),
                centers=np.array(box_centers, dtype=np.float32),
                quaternions=np.array(box_quats, dtype=np.float32),
                colors=color,
            ),
        )


def log_mujoco_robot_meshes_rerun(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    prefix: str = "world/robot",
    geom_groups: set[int] | None = None,
):
    """Backward-compatible alias — logs full scene (meshes + capsules + boxes)."""
    log_mujoco_scene_rerun(model, data, prefix=prefix, geom_groups=geom_groups)
