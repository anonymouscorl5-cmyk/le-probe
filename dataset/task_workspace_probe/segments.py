"""Spatial grid segment labels for workspace probes (XY relative to cube on table)."""

from __future__ import annotations

import os
import sys

import numpy as np

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from gr1_scene_sync import DEFAULT_CUBE_XYZ

# Half-width of "center" band in X/Y (meters, world frame, relative to cube center).
# Front = lower X (robot side); back = higher X; left = −Y; right = +Y.
_CENTER_BAND_M = 0.12


def segment_hint(
    ee_xyz: np.ndarray,
    cube_xyz: np.ndarray | None = None,
) -> str:
    """
    Assign one of six table-centric regions in the horizontal plane (Z ignored):

    ``left_front``, ``right_front``, ``left_back``, ``right_back``,
    ``center_front``, ``center_right``.
    """
    ee = np.asarray(ee_xyz, dtype=np.float64).reshape(3)
    cube = (
        np.asarray(DEFAULT_CUBE_XYZ, dtype=np.float64)
        if cube_xyz is None
        else np.asarray(cube_xyz, dtype=np.float64).reshape(3)
    )
    dx = float(ee[0] - cube[0])
    dy = float(ee[1] - cube[1])
    t = _CENTER_BAND_M

    front = dx < -t
    back = dx > t
    left = dy < -t
    right = dy > t
    center_x = not front and not back
    center_y = not left and not right

    if center_y and front:
        return "center_front"
    if center_x and right:
        return "center_right"
    if front and left:
        return "left_front"
    if front and right:
        return "right_front"
    if back and left:
        return "left_back"
    if back and right:
        return "right_back"
    if center_x and left:
        return "left_front"
    if center_x and back:
        return "left_back" if dy <= 0.0 else "right_back"
    if back and center_y:
        return "left_back" if dy <= 0.0 else "right_back"
    return "center_front"


SEGMENT_COLORS = {
    "left_front": "#4C78A8",
    "right_front": "#F58518",
    "left_back": "#54A24B",
    "right_back": "#E45756",
    "center_front": "#B279A2",
    "center_right": "#72B7B2",
    "unknown": "#888888",
}

SEGMENT_ORDER = (
    "left_front",
    "center_front",
    "right_front",
    "left_back",
    "center_right",
    "right_back",
    "unknown",
)
