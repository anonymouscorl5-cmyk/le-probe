"""Segment labels for workspace probes (scheme: lateral table region or cube distance)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal

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

LabelScheme = Literal["lateral", "distance", "pose"]

POSE_CLUSTERS_FILE = "workspace_probe_pose_clusters.json"

# --- Lateral (Y vs table edges) ---
TABLE_CENTER_Y = 0.0
TABLE_HALF_Y = 0.25

# --- Distance / phase (EE vs cube + table) ---
_THRESH_AT_CUBE_M = 0.20
_THRESH_NEAR_TABLE_M = 0.20
_THRESH_APPROACH_M = 0.50


def _dist_to_interval(val: float, lo: float, hi: float) -> float:
    if val < lo:
        return lo - val
    if val > hi:
        return val - hi
    return 0.0


def is_near_table(
    ee_xyz: np.ndarray,
    *,
    margin: float = _THRESH_NEAR_TABLE_M,
    table_z: float = TABLE_TOP_Z,
    x_range: tuple[float, float] = CUBE_X_RANGE,
    y_range: tuple[float, float] = CUBE_Y_RANGE,
) -> bool:
    ee = np.asarray(ee_xyz, dtype=np.float64).reshape(3)
    dx = _dist_to_interval(float(ee[0]), x_range[0], x_range[1])
    dy = _dist_to_interval(float(ee[1]), y_range[0], y_range[1])
    dz = abs(float(ee[2]) - table_z)
    return dx <= margin and dy <= margin and dz <= margin


def segment_hint_lateral(ee_xyz: np.ndarray, cube_xyz: np.ndarray | None = None) -> str:
    """Left / center / right by table Y edges (±0.25 m)."""
    ee = np.asarray(ee_xyz, dtype=np.float64).reshape(3)
    y = float(ee[1])
    y_lo = TABLE_CENTER_Y - TABLE_HALF_Y
    y_hi = TABLE_CENTER_Y + TABLE_HALF_Y
    if y < y_lo:
        return "left"
    if y > y_hi:
        return "right"
    return "center"


def segment_hint_distance(
    ee_xyz: np.ndarray, cube_xyz: np.ndarray | None = None
) -> str:
    """
    Distance / table phase (priority order):

    ``at_cube`` → ``near_table`` → ``approach`` → ``far``.
    """
    ee = np.asarray(ee_xyz, dtype=np.float64).reshape(3)
    cube = (
        np.asarray(DEFAULT_CUBE_XYZ, dtype=np.float64)
        if cube_xyz is None
        else np.asarray(cube_xyz, dtype=np.float64).reshape(3)
    )
    dist = float(np.linalg.norm(ee - cube))

    if dist <= _THRESH_AT_CUBE_M:
        return "at_cube"
    if is_near_table(ee):
        return "near_table"
    if dist <= _THRESH_APPROACH_M:
        return "approach"
    return "far"


def load_pose_cluster_labels(
    probe_dir: Path | None = None,
) -> tuple[dict[int, str], dict[str, str]]:
    """Return ``probe_id -> label`` and ``label -> color`` from discovery JSON."""
    base = (
        probe_dir
        or Path(__file__).resolve().parents[2] / "datasets/workspace_probe_grasp"
    )
    path = base / POSE_CLUSTERS_FILE
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run discover_pose_clusters.py first.")
    import json

    doc = json.loads(path.read_text())
    by_id = {
        int(pid): str(lab) for pid, lab in zip(doc["probe_ids"], doc["segment_hint"])
    }
    colors = doc.get("colors") or {}
    return by_id, colors


def segment_hint(
    ee_xyz: np.ndarray,
    cube_xyz: np.ndarray | None = None,
    *,
    scheme: LabelScheme = "distance",
    probe_id: int | None = None,
    pose_labels: dict[int, str] | None = None,
) -> str:
    if scheme == "lateral":
        return segment_hint_lateral(ee_xyz, cube_xyz=cube_xyz)
    if scheme == "distance":
        return segment_hint_distance(ee_xyz, cube_xyz=cube_xyz)
    if scheme == "pose":
        if probe_id is None or pose_labels is None:
            raise ValueError("scheme=pose requires probe_id and pose_labels")
        return pose_labels.get(int(probe_id), "unknown")
    raise ValueError(f"Unknown scheme: {scheme!r}")


SEGMENT_COLORS = {
    # distance / phase
    "far": "#4C78A8",
    "approach": "#F58518",
    "near_table": "#E45756",
    "at_cube": "#72B7B2",
    # lateral
    "left": "#4C78A8",
    "center": "#72B7B2",
    "right": "#F58518",
    "unknown": "#888888",
}

SEGMENT_ORDER_BY_SCHEME: dict[LabelScheme, tuple[str, ...]] = {
    "distance": ("far", "approach", "near_table", "at_cube", "unknown"),
    "lateral": ("left", "center", "right", "unknown"),
    "pose": tuple(),  # filled from discovery JSON at runtime
}


def segment_order(
    scheme: LabelScheme = "distance",
    *,
    present_labels: set[str] | list[str] | None = None,
) -> tuple[str, ...]:
    if scheme == "pose" and present_labels is not None:
        pose_sorted = sorted(
            {s for s in present_labels if str(s).startswith("pose_")},
            key=lambda s: int(str(s).split("_")[1]),
        )
        rest = sorted(set(present_labels) - set(pose_sorted))
        return tuple(pose_sorted) + tuple(rest)
    return SEGMENT_ORDER_BY_SCHEME.get(scheme) or segment_order("distance")


def infer_scheme_from_labels(labels: set[str] | list[str]) -> LabelScheme:
    present = set(labels)
    if any(str(s).startswith("pose_") for s in present):
        return "pose"
    if present & {"left", "center", "right"}:
        return "lateral"
    return "distance"


def segment_colors_for_labels(labels: set[str] | list[str]) -> dict[str, str]:
    """Merge static palette with pose-cluster colors when present."""
    out = dict(SEGMENT_COLORS)
    if any(str(s).startswith("pose_") for s in labels):
        try:
            _, pose_colors = load_pose_cluster_labels()
            out.update(pose_colors)
        except FileNotFoundError:
            for i, lab in enumerate(
                sorted({s for s in labels if str(s).startswith("pose_")})
            ):
                out[lab] = POSE_CLUSTER_COLORS[i % len(POSE_CLUSTER_COLORS)]
    return out


# Used when discovery JSON is missing from palette
POSE_CLUSTER_COLORS = [
    "#4C78A8",
    "#F58518",
    "#E45756",
    "#72B7B2",
    "#B279A2",
    "#FF9DA6",
    "#9D7660",
    "#59A14F",
    "#EDC948",
    "#AF7AA1",
    "#499894",
    "#86BCB6",
]


# Default import for distance-labeled plots
SEGMENT_ORDER = segment_order("distance")
