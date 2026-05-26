# Task-workspace probe pipeline (Tier B)

500 workspace probes: sample → encode → visualize.

## Visualization (two scripts, do not mix)

| Script | Points | Space |
|--------|-------:|--------|
| `visualize_probe_ee_scatter.py` | **500** | World frame (m) — table/cube for sampling context |
| `visualize_workspace_probe_latents.py` | **500** | Latent PCA/UMAP/t-SNE — **probes only**, no training manifold |

## Quick start

```bash
cd le-probe

# B1–B3 — sample, bundle (500 probes)
python dataset/task_workspace_probe/sample_ee_targets.py --n 500
python dataset/task_workspace_probe/solve_probe_poses.py
python dataset/task_workspace_probe/record_probe_snapshots.py --with_skeleton

# B4 — world-frame review
python dataset/task_workspace_probe/visualize_probe_ee_scatter.py

# B5 — encode (4 checkpoints → workspace_probe_latents_*.pt)
# … harvest_workspace_probes.py per variant …

# B6 — latent viz (500 points only, all variants × umap/tsne/pca)
python interpretability/manifold/run_all_probe_latent_viz.py
```

Outputs → `workspace_visualization/`

## Notes

- Segment labels: spatial grid in `segments.py` (left/right × front/back + center_front, center_right). Refresh: `relabel_probe_segments.py` then re-run B4/B6 (**no re-encode**).
- B6 does **not** use `manifold_data.pt` or any training frames.
