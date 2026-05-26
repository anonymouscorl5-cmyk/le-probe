# Task-workspace probe pipeline (Tier B)

500 workspace probes: sample → encode → visualize.

## Visualization (two scripts, do not mix)

| Script | Points | Space |
|--------|-------:|--------|
| `visualize_probe_ee_scatter.py` | **500** | World frame (m) — table/cube for sampling context |
| `visualize_workspace_probe_latents.py` | **500** | Latent PCA/UMAP/t-SNE — **probes only**, no training manifold |

## Quick start

From **repo root** (`cortex-os/`, `.venv` here):

```bash
# B1–B3 — sample, bundle (500 probes)
uv run le-probe/dataset/task_workspace_probe/sample_ee_targets.py --n 500
uv run le-probe/dataset/task_workspace_probe/solve_probe_poses.py
uv run le-probe/dataset/task_workspace_probe/record_probe_snapshots.py --with_skeleton

# B4 — world-frame review
uv run le-probe/dataset/task_workspace_probe/visualize_probe_ee_scatter.py \
  --html le-probe/workspace_visualization/distance_to_cube/workspace_probe_ee_scatter.html \
  --out le-probe/workspace_visualization/distance_to_cube/workspace_probe_ee_scatter.png

# B5 — encode (4 checkpoints → workspace_probe_latents_*.pt)
# … harvest_workspace_probes.py per variant …

# B6 — latent viz (500 points only, all variants × umap/tsne/pca)
uv run le-probe/interpretability/manifold/run_all_probe_overlays.py \
  --out-dir le-probe/workspace_visualization/distance_to_cube
```

Outputs → `le-probe/workspace_visualization/` (or pass `--out-dir` under repo root).

## Pose clusters (skeleton-only)

```bash
# Sweep k — BIC vs silhouette curve → workspace_visualization/pose_clusters/k_sweep.png
uv run le-probe/dataset/task_workspace_probe/discover_pose_clusters.py \
  --feature skeleton --all-views --sweep-k --k-min 3 --k-max 12

# Fit chosen k (e.g. from max silhouette on sweep)
uv run le-probe/dataset/task_workspace_probe/discover_pose_clusters.py \
  --feature skeleton --all-views --k 8
uv run le-probe/dataset/task_workspace_probe/relabel_probe_segments.py --scheme pose
```

## Notes

- Segment schemes (`segments.py`): `lateral` | `distance` | `pose` (from `discover_pose_clusters.py`).  
  `relabel_probe_segments.py --scheme …` → re-run B4/B6 (no re-encode).
- B6 does **not** use `manifold_data.pt` or any training frames.
