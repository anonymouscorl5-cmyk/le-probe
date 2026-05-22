"""
MuJoCo + pycapacity reachable workspace for Fourier GR-1 (legacy / exploration).

Production MPC and teleop use the **fixed** hull in ``lewm/task_workspace.py``.
This module remains for ``draw_polytope_on_rgb``, ``log_polytope_rerun``, and optional
kinematic pycapacity experiments — not wired into ``lewm_server`` by default.
"""

from __future__ import annotations

import os
import sys
import mujoco
import numpy as np
import rerun as rr
import cv2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dataset.skeleton.projection_utils import get_projection_matrix, project_point

EE_BODY = "R_index_tip_link"


def _camera_world_to_image(
    model: mujoco.MjModel, data: mujoco.MjData, cam_id: int
) -> tuple:
    """World→camera rotation used by MuJoCo renderer / skeleton priors."""
    R_cam = data.cam_xmat[cam_id].reshape(3, 3) @ np.array(
        [[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64
    )
    t_cam = data.cam_xpos[cam_id].copy()
    return R_cam, t_cam


def _depth_visible(
    depth_buffer: np.ndarray | None,
    col: int,
    row: int,
    z_cam: float,
    *,
    depth_eps: float = 0.003,
) -> bool:
    """True if point is in front of the MuJoCo-rendered surface at (col, row)."""
    if depth_buffer is None:
        return True
    h, w = depth_buffer.shape
    if col < 0 or row < 0 or col >= w or row >= h:
        return False
    scene_z = float(depth_buffer[row, col])
    if not np.isfinite(scene_z) or scene_z <= 0:
        return True
    return z_cam <= scene_z + depth_eps


def _draw_depth_tested_line(
    out: np.ndarray,
    depth_buffer: np.ndarray | None,
    p0: np.ndarray,
    z0: float,
    p1: np.ndarray,
    z1: float,
    color: tuple[int, int, int],
    *,
    depth_eps: float = 0.003,
) -> None:
    """Rasterize a segment; skip pixels where the polytope lies behind the scene."""
    import cv2

    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    if n <= 1:
        c, r = int(round(x0)), int(round(y0))
        if _depth_visible(depth_buffer, c, r, z0, depth_eps=depth_eps):
            cv2.circle(out, (c, r), 1, color, -1, cv2.LINE_AA)
        return

    for i in range(n):
        t = i / (n - 1)
        x = x0 + t * (x1 - x0)
        y = y0 + t * (y1 - y0)
        z = z0 + t * (z1 - z0)
        c, r = int(round(x)), int(round(y))
        if _depth_visible(depth_buffer, c, r, z, depth_eps=depth_eps):
            out[r, c] = color


def _fill_triangle_depth_tested(
    out: np.ndarray,
    depth_buffer: np.ndarray | None,
    pts2d: np.ndarray,
    zs: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
    *,
    depth_eps: float = 0.003,
) -> None:
    """Fill a triangle only where it is closer than the scene depth buffer."""
    tri = np.round(pts2d).astype(np.int32)
    if tri.shape != (3, 2):
        return

    h, w = out.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, tri, 255)
    rows, cols = np.where(mask > 0)
    if rows.size == 0:
        return

    v0 = pts2d[2] - pts2d[0]
    v1 = pts2d[1] - pts2d[0]
    denom = v0[0] * v1[1] - v1[0] * v0[1]
    if abs(denom) < 1e-12:
        return

    p = np.stack([cols, rows], axis=1).astype(np.float64) - pts2d[0]
    w2 = (p[:, 0] * v1[1] - v1[0] * p[:, 1]) / denom
    w1 = (v0[0] * p[:, 1] - p[:, 0] * v0[1]) / denom
    w0 = 1.0 - w1 - w2
    z_pix = w0 * zs[0] + w1 * zs[1] + w2 * zs[2]

    if depth_buffer is not None:
        scene_z = depth_buffer[rows, cols]
        visible = (
            (~np.isfinite(scene_z)) | (scene_z <= 0) | (z_pix <= scene_z + depth_eps)
        )
    else:
        visible = np.ones(rows.shape[0], dtype=bool)

    if not np.any(visible):
        return
    rows, cols, z_pix = rows[visible], cols[visible], z_pix[visible]
    del z_pix  # unused after visibility; kept for API symmetry

    blended = out.astype(np.float32)
    c = np.array(color, dtype=np.float32)
    blended[rows, cols] = (1.0 - alpha) * blended[rows, cols] + alpha * c
    out[:] = blended.astype(np.uint8)


def draw_polytope_on_rgb(
    rgb: np.ndarray,
    poly,
    cam_name: str,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    depth_buffer: np.ndarray | None = None,
    ee_body: str = EE_BODY,
    wire_color: tuple[int, int, int] = (0, 255, 0),
    ee_color: tuple[int, int, int] = (0, 255, 0),
    fill_alpha: float = 0.12,
    depth_eps: float = 0.003,
) -> np.ndarray:
    """
    Project reachable polytope wireframe (+ current EE dot) onto a MuJoCo camera RGB frame.

    Uses the same projection path as ``dataset/skeleton/projection_utils`` and ``lewm_server``.
    If ``depth_buffer`` is provided (metric depth from ``mujoco.Renderer`` with depth rendering
    enabled), hull pixels behind the robot / table are not drawn.
    """
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

    if face_indices is not None and len(face_indices) > 0:
        faces = np.asarray(face_indices, dtype=np.int64)
        for tri in faces:
            pts2d, zs = [], []
            for vi in (tri[0], tri[1], tri[2]):
                p2d, z = project_point(verts[vi], K, R_cam, t_cam)
                if p2d is not None:
                    pts2d.append(p2d)
                    zs.append(z)
            if len(pts2d) != 3:
                continue
            pts2d_arr = np.array(pts2d, dtype=np.float64)
            zs_arr = np.array(zs, dtype=np.float64)

            if fill_alpha > 0:
                _fill_triangle_depth_tested(
                    out,
                    depth_buffer,
                    pts2d_arr,
                    zs_arr,
                    wire_color,
                    fill_alpha,
                    depth_eps=depth_eps,
                )

            for i, j in ((0, 1), (1, 2), (2, 0)):
                _draw_depth_tested_line(
                    out,
                    depth_buffer,
                    pts2d_arr[i],
                    zs_arr[i],
                    pts2d_arr[j],
                    zs_arr[j],
                    wire_color,
                    depth_eps=depth_eps,
                )
    else:
        for v in verts:
            p2d, z = project_point(v, K, R_cam, t_cam)
            if p2d is None:
                continue
            c, r = int(round(p2d[0])), int(round(p2d[1]))
            if _depth_visible(depth_buffer, c, r, z, depth_eps=depth_eps):
                cv2.circle(out, (c, r), 2, wire_color, -1, cv2.LINE_AA)

    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_body)
    if ee_id != -1:
        p2d, z = project_point(data.xpos[ee_id], K, R_cam, t_cam)
        if p2d is not None:
            c, r = int(round(p2d[0])), int(round(p2d[1]))
            if _depth_visible(depth_buffer, c, r, z, depth_eps=depth_eps):
                cv2.circle(out, (c, r), 5, ee_color, -1, cv2.LINE_AA)

    return out


def draw_world_points_on_rgb(
    rgb: np.ndarray,
    points: np.ndarray,
    cam_name: str,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    depth_buffer: np.ndarray | None = None,
    color: tuple[int, int, int] = (0, 255, 0),
    radius: int = 7,
    label_points: bool = True,
    edges: list[tuple[int, int]] | None = None,
    depth_eps: float = 0.003,
) -> np.ndarray:
    """Project world-frame 3D points onto a camera RGB frame (same blue as polytope overlay)."""
    from dataset.skeleton.projection_utils import get_projection_matrix, project_point

    try:
        import cv2
    except ImportError as e:
        raise ImportError("opencv-python required for 2D point overlay") from e

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cam_id == -1:
        return rgb

    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    h, w = rgb.shape[:2]
    K = get_projection_matrix(cam_id, model, w, h)
    R_cam, t_cam = _camera_world_to_image(model, data, cam_id)

    out = np.asarray(rgb, dtype=np.uint8)
    if not out.flags.writeable:
        out = out.copy()

    projected: list[tuple[np.ndarray, float] | None] = [None] * len(pts)
    for i, p in enumerate(pts):
        p2d, z = project_point(p, K, R_cam, t_cam)
        if p2d is not None:
            projected[i] = (p2d, z)

    if edges:
        for i, j in edges:
            if i >= len(projected) or j >= len(projected):
                continue
            if projected[i] is None or projected[j] is None:
                continue
            p0, z0 = projected[i]
            p1, z1 = projected[j]
            _draw_depth_tested_line(
                out,
                depth_buffer,
                p0,
                z0,
                p1,
                z1,
                color,
                depth_eps=depth_eps,
            )

    for idx, pr in enumerate(projected):
        if pr is None:
            continue
        p2d, z = pr
        c, r = int(round(p2d[0])), int(round(p2d[1]))
        if not _depth_visible(depth_buffer, c, r, z, depth_eps=depth_eps):
            continue
        cv2.circle(out, (c, r), radius, color, -1, cv2.LINE_AA)
        if label_points:
            cv2.putText(
                out,
                str(idx + 1),
                (c + 6, r - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
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
