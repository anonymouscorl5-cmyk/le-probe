"""Spatial segment labels for workspace probes (EE vs cube + table footprint)."""

from __future__ import annotations

import os
import sys

import numpy as np

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from gr1_scene_sync import (
    CUBE_X_RANGE,
    CUBE_Y_RANGE,
    DEFAULT_CUBE_XYZ,
    TABLE_TOP_Z,
)

# Meters (world frame)
_THRESH_AT_CUBE = 0.20
_THRESH_NEAR_TABLE = 0.20
_THRESH_APPROACH = 0.50


def _dist_to_interval(val: float, lo: float, hi: float) -> float:
    if val < lo:
        return lo - val
    if val > hi:
        return val - hi
    return 0.0


def is_near_table(
    ee_xyz: np.ndarray,
    *,
    margin: float = _THRESH_NEAR_TABLE,
    table_z: float = TABLE_TOP_Z,
    x_range: tuple[float, float] = CUBE_X_RANGE,
    y_range: tuple[float, float] = CUBE_Y_RANGE,
) -> bool:
    """
    True if the fingertip lies within ``margin`` of the table-top rectangle
    (X in cube sampling span, Y in cube sampling span, Z at ``table_z``).
    """
    ee = np.asarray(ee_xyz, dtype=np.float64).reshape(3)
    dx = _dist_to_interval(float(ee[0]), x_range[0], x_range[1])
    dy = _dist_to_interval(float(ee[1]), y_range[0], y_range[1])
    dz = abs(float(ee[2]) - table_z)
    return dx <= margin and dy <= margin and dz <= margin


def segment_hint(
    ee_xyz: np.ndarray,
    cube_xyz: np.ndarray | None = None,
) -> str:
    """
    Label priority:
    1. ``at_cube`` — within 0.2 m of cube center (3D).
    2. ``near_table`` — within 0.2 m of table surface patch (per axis).
    3. ``approach`` — not above, but within 0.5 m of cube.
    4. ``far`` — everything else.
    """
    ee = np.asarray(ee_xyz, dtype=np.float64).reshape(3)
    cube = (
        np.asarray(DEFAULT_CUBE_XYZ, dtype=np.float64)
        if cube_xyz is None
        else np.asarray(cube_xyz, dtype=np.float64).reshape(3)
    )
    dist = float(np.linalg.norm(ee - cube))

    if dist <= _THRESH_AT_CUBE:
        return "at_cube"
    if is_near_table(ee):
        return "near_table"
    if dist <= _THRESH_APPROACH:
        return "approach"
    return "far"


SEGMENT_COLORS = {
    "far": "#4C78A8",
    "approach": "#F58518",
    "near_table": "#E45756",
    "at_cube": "#72B7B2",
    "unknown": "#B279A2",
}
