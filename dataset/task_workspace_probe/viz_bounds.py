"""Axis limits from point clouds (not decorative geometry)."""

from __future__ import annotations

import numpy as np


def axis_limits(
    points: np.ndarray,
    *,
    margin_frac: float = 0.08,
    min_pad: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-axis [lo, hi] with proportional padding."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        raise ValueError("axis_limits: empty point set")
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    span = np.maximum(hi - lo, min_pad)
    pad = span * margin_frac
    return lo - pad, hi + pad


def plotly_axis_dict(lo: np.ndarray, hi: np.ndarray, axis: int, *, title: str) -> dict:
    return dict(title=title, range=[float(lo[axis]), float(hi[axis])])


def plotly_scene_from_limits(
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    x_title: str = "dim 0",
    y_title: str = "dim 1",
    z_title: str = "dim 2",
    aspectmode: str = "cube",
    backgroundcolor: str | None = None,
    camera: dict | None = None,
) -> dict:
    """Plotly 3D scene with explicit ranges (avoids decorative meshes setting bounds)."""

    def _ax(i: int, title: str) -> dict:
        d = plotly_axis_dict(lo, hi, i, title=title)
        if backgroundcolor is not None:
            d["backgroundcolor"] = backgroundcolor
        return d

    scene = dict(
        xaxis=_ax(0, x_title),
        yaxis=_ax(1, y_title),
        zaxis=_ax(2, z_title),
        aspectmode=aspectmode,
    )
    if camera is not None:
        scene["camera"] = camera
    return scene
