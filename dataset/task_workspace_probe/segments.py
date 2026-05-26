"""Provisional spatial segment labels for workspace probes (review before manifold overlay)."""

from __future__ import annotations

import os
import sys

import numpy as np

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from gr1_scene_sync import DEFAULT_CUBE_XYZ

# Distances in meters (world frame); tune after B4 review grid.
_THRESH_FAR = 0.30
_THRESH_APPROACH = 0.12
_THRESH_AT_CUBE = 0.08
_THRESH_ABOVE_Z = 0.12


def segment_hint(
    ee_xyz: np.ndarray,
    cube_xyz: np.ndarray | None = None,
) -> str:
    """Heuristic label from index-tip position relative to cube."""
    ee = np.asarray(ee_xyz, dtype=np.float64).reshape(3)
    cube = (
        np.asarray(DEFAULT_CUBE_XYZ, dtype=np.float64)
        if cube_xyz is None
        else np.asarray(cube_xyz, dtype=np.float64).reshape(3)
    )
    dist = float(np.linalg.norm(ee - cube))
    dz = float(ee[2] - cube[2])

    if dist > _THRESH_FAR:
        return "far"
    if dist > _THRESH_APPROACH:
        return "approach"
    if dz > _THRESH_ABOVE_Z:
        return "above"
    if dist <= _THRESH_AT_CUBE:
        return "at_cube"
    return "near_table"


SEGMENT_COLORS = {
    "far": "#4C78A8",
    "approach": "#F58518",
    "near_table": "#E45756",
    "at_cube": "#72B7B2",
    "above": "#54A24B",
    "unknown": "#B279A2",
}
