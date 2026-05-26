#!/usr/bin/env python3
"""B4: Review grid of probe snapshots (world_center view)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from dataset.task_workspace_probe.segments import SEGMENT_COLORS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle",
        type=str,
        default=str(
            REPO_DIR / "datasets/workspace_probe_grasp/workspace_probe_bundle.pt"
        ),
    )
    parser.add_argument("--num", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=str,
        default=str(REPO_DIR / "assets/workspace_probe_review_50.png"),
    )
    parser.add_argument(
        "--view",
        type=str,
        default="world_center",
        choices=[
            "world_top",
            "world_left",
            "world_right",
            "world_center",
            "world_wrist",
        ],
    )
    args = parser.parse_args()

    data = torch.load(args.bundle, weights_only=False)
    cam_names = list(data["cam_names"])
    v_idx = cam_names.index(args.view)
    rgb = data["rgb"].numpy()  # N,V,H,W,3
    n_total = rgb.shape[0]
    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(n_total, size=min(args.num, n_total), replace=False)
    idxs.sort()

    cols = 10
    rows = (len(idxs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    axes = np.atleast_1d(axes).flatten()
    segments = data.get("segment_hint", ["unknown"] * n_total)
    if hasattr(segments, "tolist"):
        segments = segments.tolist()
    dists = data.get("dist_to_cube_m")
    if dists is not None and hasattr(dists, "numpy"):
        dists = dists.numpy()

    if n_total < args.num:
        print(
            f"⚠️ Bundle has only {n_total} probes (requested {args.num} for grid). "
            "Showing all available — check B2 stats in workspace_probe_poses.json."
        )

    for ax_i, i in enumerate(idxs):
        ax = axes[ax_i]
        ax.imshow(rgb[i, v_idx])
        seg = segments[i] if i < len(segments) else "unknown"
        color = SEGMENT_COLORS.get(seg, "#888")
        dist_s = ""
        if dists is not None:
            dist_s = f" d={float(dists[i]):.2f}m"
        ax.set_title(
            f"id={int(data['probe_ids'][i])} {seg}{dist_s}",
            fontsize=7,
            color=color,
        )
        ax.axis("off")

    for j in range(len(idxs), len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"Workspace probes ({args.view})", fontsize=12)
    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"✅ Review grid → {out}")


if __name__ == "__main__":
    main()
