#!/usr/bin/env python3
"""B4: 3D scatter of achieved EE positions (static PNG + interactive HTML scene)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from dataset.task_workspace_probe.segments import SEGMENT_COLORS
from gr1_scene_sync import DEFAULT_CUBE_XYZ, TABLE_TOP_Z
from lewm.task_workspace import build_task_workspace_polytope

# scene_gr1_pickup.xml — table body + geom (world frame)
TABLE_CENTER = np.array([0.45, 0.0, 0.4], dtype=np.float64)
TABLE_HALF = np.array([0.2, 0.25, 0.4], dtype=np.float64)
CUBE_HALF = np.array([0.02, 0.02, 0.02], dtype=np.float64)
FLOOR_Z = 0.0
FLOOR_HALF = np.array([1.2, 1.2, 0.01], dtype=np.float64)
FLOOR_CENTER = np.array([0.45, 0.0, FLOOR_Z - 0.01], dtype=np.float64)


def _as_numpy(x) -> np.ndarray:
    if hasattr(x, "numpy"):
        return np.asarray(x.numpy())
    return np.asarray(x)


def _box_mesh(
    center: np.ndarray,
    half: np.ndarray,
    *,
    color: str,
    opacity: float,
    name: str,
    showlegend: bool = True,
) -> go.Mesh3d:
    """Axis-aligned box as Plotly Mesh3d."""
    cx, cy, cz = center
    hx, hy, hz = half
    verts = np.array(
        [
            [cx - hx, cy - hy, cz - hz],
            [cx + hx, cy - hy, cz - hz],
            [cx + hx, cy + hy, cz - hz],
            [cx - hx, cy + hy, cz - hz],
            [cx - hx, cy - hy, cz + hz],
            [cx + hx, cy - hy, cz + hz],
            [cx + hx, cy + hy, cz + hz],
            [cx - hx, cy + hy, cz + hz],
        ],
        dtype=np.float64,
    )
    # 12 triangles (two per face)
    i, j, k = [], [], []
    faces = [
        (0, 1, 2),
        (0, 2, 3),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (2, 3, 7),
        (2, 7, 6),
        (0, 3, 7),
        (0, 7, 4),
        (1, 2, 6),
        (1, 6, 5),
    ]
    for a, b, c in faces:
        i.append(a)
        j.append(b)
        k.append(c)
    return go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=i,
        j=j,
        k=k,
        color=color,
        opacity=opacity,
        name=name,
        showlegend=showlegend,
        hoverinfo="name",
        flatshading=True,
    )


def _hull_mesh(poly, *, opacity: float = 0.12) -> go.Mesh3d:
    pts = np.asarray(poly.vertices, dtype=np.float64).T
    faces = np.asarray(poly.face_indices, dtype=np.int64)
    return go.Mesh3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color="rgba(0,180,80,0.35)",
        opacity=opacity,
        name="task hull",
        showlegend=True,
        hoverinfo="name",
    )


def _add_sim_scene(fig: go.Figure, cube: np.ndarray) -> None:
    """Table + cube + floor matching scene_gr1_pickup.xml (no robot)."""
    fig.add_trace(
        _box_mesh(
            FLOOR_CENTER,
            FLOOR_HALF,
            color="#3a3a3a",
            opacity=0.35,
            name="floor",
        )
    )
    fig.add_trace(
        _box_mesh(
            TABLE_CENTER,
            TABLE_HALF,
            color="#808080",
            opacity=0.85,
            name="table",
        )
    )
    fig.add_trace(
        _box_mesh(
            cube,
            CUBE_HALF,
            color="#cc2222",
            opacity=1.0,
            name="cube",
        )
    )
    # Table top reference (thin slab at TABLE_TOP_Z)
    top = np.array([TABLE_CENTER[0], TABLE_CENTER[1], TABLE_TOP_Z], dtype=np.float64)
    fig.add_trace(
        _box_mesh(
            top,
            np.array([TABLE_HALF[0], TABLE_HALF[1], 0.002], dtype=np.float64),
            color="#999999",
            opacity=0.5,
            name="table top (z=0.82)",
            showlegend=False,
        )
    )


def _load_bundle(bundle_path: str) -> dict:
    data = torch.load(bundle_path, weights_only=False)
    ee = _as_numpy(data["ee_achieved_xyz"]).astype(np.float64)
    cube = _as_numpy(data["cube_xyz"]).reshape(3).astype(np.float64)
    segments = data.get("segment_hint", ["unknown"] * len(ee))
    if hasattr(segments, "tolist"):
        segments = segments.tolist()
    probe_ids = data.get("probe_ids")
    if probe_ids is not None:
        probe_ids = _as_numpy(probe_ids).astype(int)
    else:
        probe_ids = np.arange(len(ee), dtype=int)
    dist = np.linalg.norm(ee - cube.reshape(1, 3), axis=1)
    return {
        "ee": ee,
        "cube": cube,
        "segments": segments,
        "probe_ids": probe_ids,
        "dist": dist,
    }


def build_interactive_figure(payload: dict) -> go.Figure:
    ee = payload["ee"]
    cube = payload["cube"]
    segments = payload["segments"]
    probe_ids = payload["probe_ids"]
    dist = payload["dist"]

    fig = go.Figure()
    _add_sim_scene(fig, cube)

    poly = build_task_workspace_polytope()
    fig.add_trace(_hull_mesh(poly))

    for seg in sorted(set(segments)):
        mask = np.array([s == seg for s in segments])
        if not mask.any():
            continue
        hover = [
            f"probe {probe_ids[i]}<br>{seg}<br>d_cube={dist[i]:.3f} m<br>"
            f"({ee[i,0]:.3f}, {ee[i,1]:.3f}, {ee[i,2]:.3f})"
            for i in np.where(mask)[0]
        ]
        fig.add_trace(
            go.Scatter3d(
                x=ee[mask, 0],
                y=ee[mask, 1],
                z=ee[mask, 2],
                mode="markers",
                name=seg,
                marker=dict(
                    size=4,
                    color=SEGMENT_COLORS.get(seg, "#888888"),
                    opacity=0.85,
                    line=dict(width=0),
                ),
                text=hover,
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title="Workspace probe fingertips (world frame) — drag to orbit, scroll to zoom",
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.7)"),
        scene=dict(
            xaxis=dict(title="X (m)", backgroundcolor="#1a1a1a"),
            yaxis=dict(title="Y (m)", backgroundcolor="#1a1a1a"),
            zaxis=dict(title="Z (m)", backgroundcolor="#1a1a1a"),
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.35, y=-1.1, z=0.75),
                center=dict(x=0.45, y=0.0, z=0.95),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        paper_bgcolor="#111111",
        font=dict(color="#eeeeee"),
    )
    return fig


def save_matplotlib_png(payload: dict, out: Path) -> None:
    ee = payload["ee"]
    cube = payload["cube"]
    segments = payload["segments"]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for seg in sorted(set(segments)):
        mask = np.array([s == seg for s in segments])
        if not mask.any():
            continue
        ax.scatter(
            ee[mask, 0],
            ee[mask, 1],
            ee[mask, 2],
            c=SEGMENT_COLORS.get(seg, "#888"),
            label=seg,
            s=12,
            alpha=0.7,
        )

    ax.scatter(
        [cube[0]],
        [cube[1]],
        [cube[2]],
        c="red",
        marker="*",
        s=120,
        label="cube",
    )

    poly = build_task_workspace_polytope()
    corners = poly.corner_points
    ax.plot(
        np.append(corners[:, 0], corners[0, 0]),
        np.append(corners[:, 1], corners[0, 1]),
        np.append(corners[:, 2], corners[0, 2]),
        "g--",
        alpha=0.4,
        linewidth=1,
        label="hull corners",
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="upper left", fontsize=8)
    plt.title("Workspace probe fingertip positions")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle",
        type=str,
        default=str(
            REPO_DIR / "datasets/workspace_probe_grasp/workspace_probe_bundle.pt"
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(REPO_DIR / "assets/workspace_probe_ee_scatter.png"),
    )
    parser.add_argument(
        "--html",
        type=str,
        default=str(REPO_DIR / "assets/workspace_probe_ee_scatter.html"),
        help="Interactive Plotly scene (table + cube + hull + points)",
    )
    parser.add_argument("--no-png", action="store_true")
    parser.add_argument("--no-html", action="store_true")
    args = parser.parse_args()

    payload = _load_bundle(args.bundle)
    if np.linalg.norm(payload["cube"] - DEFAULT_CUBE_XYZ) > 1e-4:
        print(
            f"ℹ️  bundle cube_xyz={payload['cube'].tolist()} (default {DEFAULT_CUBE_XYZ.tolist()})"
        )

    if not args.no_html:
        html_path = Path(args.html)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        fig = build_interactive_figure(payload)
        fig.write_html(
            str(html_path),
            include_plotlyjs="cdn",
            config=dict(displayModeBar=True, scrollZoom=True),
        )
        print(f"✅ Interactive EE scatter → {html_path}")
        print(f"   Open in browser: file://{html_path.resolve()}")

    if not args.no_png:
        png_path = Path(args.out)
        save_matplotlib_png(payload, png_path)
        print(f"✅ EE scatter PNG → {png_path}")


if __name__ == "__main__":
    main()
