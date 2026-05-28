# Task Workspace Probe Pipeline

Static probe pipeline for Appendix-style workspace audits.

## Purpose

- Sample valid end-effector targets inside the task workspace.
- Encode probe snapshots with trained checkpoints.
- Visualize latent organization by workspace labels (lateral / distance / pose).

## Quick Start (from repo root)

```bash
# 1) Sample and bundle probes
uv run le-probe/dataset/task_workspace_probe/sample_ee_targets.py --n 500
uv run le-probe/dataset/task_workspace_probe/solve_probe_poses.py
uv run le-probe/dataset/task_workspace_probe/record_probe_snapshots.py --with_skeleton

# 2) World-frame scatter
uv run le-probe/dataset/task_workspace_probe/visualize_probe_ee_scatter.py \
  --html le-probe/workspace_visualization/distance_to_cube/workspace_probe_ee_scatter.html \
  --out le-probe/workspace_visualization/distance_to_cube/workspace_probe_ee_scatter.png

# 3) Latent overlays (all variants, probes only)
uv run le-probe/interpretability/manifold/run_all_probe_overlays.py \
  --out-dir le-probe/workspace_visualization/distance_to_cube
```

## Outputs

- Workspace coordinates and labels
- Probe latent dumps per checkpoint
- PCA/t-SNE/UMAP overlays under `le-probe/workspace_visualization/`

## Notes

- Probe overlays are generated from probe snapshots only (not training manifolds).
- Label schemes: `lateral`, `distance`, `pose`.
