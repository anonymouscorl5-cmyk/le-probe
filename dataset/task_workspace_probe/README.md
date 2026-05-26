# Task-workspace probe pipeline (Tier B)

Sample feasible fingertip poses inside the fixed MPC hull, IK, record snapshots, encode, and overlay on training manifolds.

## Quick start

```bash
cd le-probe
source .venv/bin/activate

# B1 — 500 reachable configs (joint uniform on 16–31, FK → EE; hull filter on)
.venv/bin/python dataset/task_workspace_probe/sample_ee_targets.py --n 500

# B2 — pass-through joint poses (100% ok when B1 used joint sampling)
.venv/bin/python dataset/task_workspace_probe/solve_probe_poses.py

# B3 — RGB bundle (+ skeleton masks for skel/dino encode)
.venv/bin/python dataset/task_workspace_probe/record_probe_snapshots.py --with_skeleton

# B4 — Review before locking segments
.venv/bin/python dataset/task_workspace_probe/visualize_probe_grid.py --num 50
.venv/bin/python dataset/task_workspace_probe/visualize_probe_ee_scatter.py
# Interactive 3D (table + cube + hull, orbit/zoom in browser):
# open assets/workspace_probe_ee_scatter.html

# B5 — Encode (3 checkpoints)
.venv/bin/python interpretability/manifold/harvest_workspace_probes.py \
  --bundle datasets/workspace_probe_grasp/workspace_probe_bundle.pt \
  --model gr1_reward_tuned_v2.ckpt --multi_view --tag mv

.venv/bin/python interpretability/manifold/harvest_workspace_probes.py \
  --bundle datasets/workspace_probe_grasp/workspace_probe_bundle.pt \
  --model gr1_reward_tuned_v6.ckpt --multi_view --use_skeleton --tag skel

.venv/bin/python interpretability/manifold/harvest_workspace_probes.py \
  --bundle datasets/workspace_probe_grasp/workspace_probe_bundle.pt \
  --model gr1_reward_tuned_v1.ckpt --multi_view --use_skeleton --use_dino --tag dino2

# B6 — Overlay (requires training manifold_data.pt per experiment)
.venv/bin/python interpretability/manifold/overlay_workspace_probes.py \
  --training interpretability/manifold/manifold_data.pt \
  --probes datasets/workspace_probe_grasp/workspace_probe_latents_mv.pt \
  --method pca \
  --out assets/workspace_probe_overlay_mv_pca.png
```

## Outputs

| File | Phase |
|------|-------|
| `datasets/workspace_probe_grasp/workspace_probe_targets.json` | B1 |
| `datasets/workspace_probe_grasp/workspace_probe_poses.json` | B2 |
| `datasets/workspace_probe_grasp/workspace_probe_bundle.pt` | B3 |
| `datasets/workspace_probe_grasp/workspace_probe_latents_{mv,skel,dino2}.pt` | B5 |

## Sampling modes (B1)

- **`--joint-mode wild`** (default): uniform on wire32 indices **16–31** (right arm + hand + waist), same as dataset `wild_reset`.
- **`--joint-mode ik`**: only joints in `ik_joints.txt` (teleop default sliders).
- **`--no-hull-filter`**: keep all joint samples even if FK fingertip is outside the MPC hull.

Legacy hull-only EE targets (no `wire32_rad`) still run IK in B2 unless `--force-ik` is unused and targets lack `wire32_rad`.

## Notes

- Does **not** modify `task_workspace.py`, `harvest_manifold.py`, or MPC paths.
- Probes are **single static poses** (one encoder step), same as training manifold harvest.
- Segment labels are **hints** from `segments.py` (at_cube / near_table / approach / far). After threshold changes, run `relabel_probe_segments.py` then re-run B4 viz (no re-record).
