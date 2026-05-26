"""Lightweight MuJoCo helper for workspace probe IK and rendering (no LeRobot recording)."""

from __future__ import annotations

import os
import sys

import mujoco
import numpy as np
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dataset.polytope_utils import EE_BODY
from dataset.skeleton.projection_utils import (
    get_projection_matrix,
    is_allowed_action_chain,
    project_point,
)
from gr1_config import SCENE_PATH
from gr1_protocol import StandardScaler
from gr1_scene_sync import DEFAULT_CUBE_XYZ, build_qpos_from_wire32, ee_body_xyz
from lewm.task_workspace import build_task_workspace_polytope, ee_halfspace_violation
from simulation_base import GR1MuJoCoBase

CAM_NAMES = [
    "world_top",
    "world_left",
    "world_right",
    "world_center",
    "world_wrist",
]
RENDER_SIZE = 224
IK_QUAT_DOWN = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)

# Wrist / thumb offsets relative to index target (teleop-style).
IK_OFFSET_PRESETS = [
    {"wrist": [0.0, 0.0, 0.06], "thumb": [-0.02, 0.0, 0.0]},
    {"wrist": [0.0, 0.0, 0.08], "thumb": [-0.02, 0.02, 0.0]},
    {"wrist": [0.02, 0.0, 0.05], "thumb": [-0.04, 0.0, 0.0]},
]

# Open-hand finger qpos indices in full MuJoCo vector (teleop phase 2).
_OPEN_FINGER_QIDX = [50, 51, 52, 53, 54, 55, 56]


class ProbeSimulator(GR1MuJoCoBase):
    """GR-1 sim for static probe poses; disables episode recording side effects."""

    def __init__(self):
        super().__init__(scene_path=SCENE_PATH, restrict_ik=True)
        self.recorder = None  # type: ignore[assignment]
        self.poly = build_task_workspace_polytope()
        self._H = self.poly.H
        self._d = self.poly.d
        self._feas_eps = 1e-4
        self._cube_xyz = np.array(DEFAULT_CUBE_XYZ, dtype=np.float64)
        self._renderer_224 = mujoco.Renderer(
            self.model, height=RENDER_SIZE, width=RENDER_SIZE
        )
        self._idx_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "R_index_tip_link"
        )
        self._thm_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "R_thumb_tip_link"
        )

    def reset_probe_scene(self, *, lock_posture: bool = True) -> None:
        self.reset_env(lock_posture=lock_posture, randomize_cube=False)

    def set_pose_from_wire32_rad(
        self,
        wire32_rad: np.ndarray,
        *,
        cube_xyz: np.ndarray | None = None,
    ) -> None:
        cube = (
            self._cube_xyz
            if cube_xyz is None
            else np.asarray(cube_xyz, dtype=np.float64)
        )
        q = build_qpos_from_wire32(
            self.model,
            wire32_rad,
            sim_scene_sync=True,
            cube_xyz=cube,
        )
        self.data.qpos[:] = q
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def fingertip_xyz(self) -> np.ndarray:
        return ee_body_xyz(self.model, self.data, EE_BODY)

    def hull_violation(self, ee: np.ndarray | None = None) -> float:
        if ee is None:
            ee = self.fingertip_xyz()
        return ee_halfspace_violation(ee, self._H, self._d)

    def apply_open_hand_qpos(self, q: np.ndarray) -> np.ndarray:
        out = q.copy()
        for idx in _OPEN_FINGER_QIDX:
            if idx < len(out):
                out[idx] = 0.0
        return out

    def solve_probe_ik(
        self,
        ee_target: np.ndarray,
        *,
        max_presets: int = 3,
    ) -> tuple[np.ndarray | None, dict]:
        """Return full qpos on success, else (None, diagnostics)."""
        ee_target = np.asarray(ee_target, dtype=np.float64).reshape(3)
        best_q = None
        best_err = np.inf
        best_diag: dict = {}

        # Fresh neutral posture per target (avoids drift from prior failed solves).
        self.reset_probe_scene(lock_posture=True)

        for preset_id, preset in enumerate(IK_OFFSET_PRESETS[:max_presets]):
            pos_index = ee_target
            pos_wrist = ee_target + np.asarray(preset["wrist"], dtype=np.float64)
            pos_thumb = ee_target + np.asarray(preset["thumb"], dtype=np.float64)
            try:
                q = self.solve_ik(
                    pos_wrist,
                    IK_QUAT_DOWN,
                    pos_index=pos_index,
                    pos_thumb=pos_thumb,
                    posture_cost=1e-6,
                )
            except Exception as exc:
                best_diag = {"preset_id": preset_id, "error": str(exc)}
                continue

            q = self.apply_open_hand_qpos(q)
            self.data.qpos[:] = q
            mujoco.mj_forward(self.model, self.data)
            achieved = self.fingertip_xyz()
            err = float(np.linalg.norm(achieved - ee_target))
            viol = self.hull_violation(achieved)

            if err < best_err:
                best_err = err
                best_q = q.copy()
                best_diag = {
                    "preset_id": preset_id,
                    "ik_error_m": err,
                    "hull_violation": viol,
                    "ee_achieved_xyz": achieved.tolist(),
                }

        # 5 cm tolerance — hull targets are often overhead reaches; 3 cm was too tight.
        if best_q is None or best_err > 0.05:
            return None, {**best_diag, "status": "ik_fail", "ik_error_m": best_err}

        wire32 = self.qpos_to_action_32(best_q)
        viol = self.hull_violation()
        if viol > self._feas_eps:
            return None, {**best_diag, "status": "hull_fail", "hull_violation": viol}

        scaler = StandardScaler()
        state_norm = scaler.scale_state(wire32)
        return best_q, {
            **best_diag,
            "status": "ok",
            "wire32_rad": wire32.astype(np.float64).tolist(),
            "state_norm": state_norm.astype(np.float64).tolist(),
            "ee_target_xyz": ee_target.tolist(),
        }

    def render_rgb_views(self) -> dict[str, np.ndarray]:
        """Return uint8 HWC 224×224 per camera."""
        views = {}
        for name in CAM_NAMES:
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            self._renderer_224.update_scene(self.data, camera=name)
            rgb = self._renderer_224.render()
            views[name] = np.asarray(rgb, dtype=np.uint8)
        return views

    def render_skeleton_mask(self, view_name: str) -> np.ndarray:
        """1-channel uint8 skeleton mask at 224×224 (server parity)."""
        from PIL import ImageDraw

        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, view_name)
        K = get_projection_matrix(cam_id, self.model, RENDER_SIZE, RENDER_SIZE)
        t_cam = self.data.cam_xpos[cam_id]
        R_cam = self.data.cam_xmat[cam_id].reshape(3, 3) @ np.array(
            [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
            dtype=np.float64,
        )

        mask = Image.new("L", (RENDER_SIZE, RENDER_SIZE), 0)
        draw = ImageDraw.Draw(mask)
        for b_id in range(1, self.model.nbody):
            p_id = self.model.body_parentid[b_id]
            if is_allowed_action_chain(b_id, self.model) and is_allowed_action_chain(
                p_id, self.model
            ):
                ps, _ = project_point(self.data.xpos[b_id], K, R_cam, t_cam)
                pp, _ = project_point(self.data.xpos[p_id], K, R_cam, t_cam)
                if ps is not None and pp is not None:
                    draw.line([tuple(ps), tuple(pp)], fill=255, width=1)

        cube = self._cube_xyz
        size = 0.02
        corners = (
            np.array(
                [
                    [-1, -1, -1],
                    [1, -1, -1],
                    [1, 1, -1],
                    [-1, 1, -1],
                    [-1, -1, 1],
                    [1, -1, 1],
                    [1, 1, 1],
                    [-1, 1, 1],
                ],
                dtype=np.float64,
            )
            * size
            + cube
        )
        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]
        for i, j in edges:
            pi, _ = project_point(corners[i], K, R_cam, t_cam)
            pj, _ = project_point(corners[j], K, R_cam, t_cam)
            if pi is not None and pj is not None:
                draw.line([tuple(pi), tuple(pj)], fill=255, width=1)

        return np.array(mask, dtype=np.uint8)
