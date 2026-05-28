# Le-Probe: Representation Audits for LeWM

<div align="center">
  <img src="assets/banner.png" width="100%" style="border-radius: 12px; margin-bottom: 20px;">
</div>

Le-Probe is the experiment and analysis stack used in our CoRL 2026 submission to diagnose why latent MPC succeeds or fails across LeRobot World Model variants on a high-DoF cube-pick task.

## Paper-Aligned Scope

- **Core question:** how representation quality changes planning behavior as structured priors are added.
- **Variant ladder:** `Single-View RGB -> Multi-View RGB -> Multi-View RGB + Skeletal Priors -> Multi-View RGB + Skeletal Priors + DINOv3 Waypoints`.
- **Protocol:** rollout behavior, training-manifold audits, static workspace probes, and mechanistic CLT analysis.

## Repository Map

- [`dataset/`](./dataset): teleoperation, dataset curation, priors, and workspace probe generation.
- [`vla/`](./vla): GR00T-N1 baseline training and simulation inference.
- [`lewm/`](./lewm): LeWM training, reward tuning, goal-gallery harvest, and latent MPC serving.
- [`interpretability/`](./interpretability): manifold audits, static probe audits, CLT training, and Neuronpedia-backed inspection.
- [`scripts/`](./scripts): maintenance and reproducibility utilities.

## Setup

```bash
git clone --recursive https://github.com/anonymouscorl5-cmyk/le-probe.git
cd le-probe
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduction

```bash
# 1) Harvest goal gallery for MPC
.venv/bin/python lewm/harvest_goals.py

# 2) Start planner server (full variant example)
.venv/bin/python lewm/lewm_server.py \
  --model <ckpt> \
  --gallery goal_gallery.pth \
  --multi_view --use_skeleton --use_dino
```

## Representation Variants

| Variant | Added Signal | Goal |
| :--- | :--- | :--- |
| Single-View RGB | `world_center` only | Baseline JEPA + MPC |
| Multi-View RGB | 5 camera views | Improve state coverage |
| Multi-View RGB + Skeletal Priors | 4th kinematic channel | Anchor task-relevant structure |
| Multi-View RGB + Skeletal Priors + DINOv3 Waypoints | phase waypoints | Improve long-horizon subgoal alignment |

## Key Artifacts

### Behavior Progression

<div align="center">
  <img src="assets/lewm_grasp.gif" width="220" alt="Single-View RGB rollout">
  <img src="assets/lewm_grasp_multiview.gif" width="220" alt="Multi-View RGB rollout">
  <img src="assets/lewm_grasp_multiview_skeleton.gif" width="220" alt="Multi-View RGB plus Skeletal Priors rollout">
  <img src="assets/lewm_grasp_multiview_skeleton_dino.gif" width="220" alt="Multi-View RGB plus Skeletal Priors plus DINOv3 Waypoints rollout">
</div>

### Training-Manifold Audit (PCA / t-SNE / UMAP)

| Variant | 3D PCA | 3D t-SNE | 3D UMAP |
| :--- | :---: | :---: | :---: |
| **Single-View RGB** | ![PCA](assets/manifold_3d_pca.png) | ![t-SNE](assets/manifold_3d_tsne.png) | ![UMAP](assets/manifold_3d_umap.png) |
| **Multi-View RGB** | ![PCA](assets/manifold_3d_multiview_pca.png) | ![t-SNE](assets/manifold_3d_multiview_tsne.png) | ![UMAP](assets/manifold_3d_multiview_umap.png) |
| **Multi-View RGB + Skeletal Priors** | ![PCA](assets/manifold_3d_multiview_skeleton_pca.png) | ![t-SNE](assets/manifold_3d_multiview_skeleton_tsne.png) | ![UMAP](assets/manifold_3d_multiview_skeleton_umap.png) |
| **Multi-View RGB + Skeletal Priors + DINOv3 Waypoints** | ![PCA](assets/manifold_3d_multiview_skeleton_dino_pca.png) | ![t-SNE](assets/manifold_3d_multiview_skeleton_dino_tsne.png) | ![UMAP](assets/manifold_3d_multiview_skeleton_dino_umap.png) |

### Static Workspace Probes

<div align="center">
  <img src="assets/task_workspace.png" width="23%" alt="Task workspace">
  <img src="assets/lateral_table_region.png" width="23%" alt="Lateral regions">
  <img src="assets/distance_to_cube.png" width="23%" alt="Distance bins">
  <img src="assets/pose_clusters.png" width="23%" alt="Pose clusters">
</div>

- **Observed trend:** separability and cluster continuity improve across variants, with strongest structure in `Multi-View RGB + Skeletal Priors` and `Multi-View RGB + Skeletal Priors + DINOv3 Waypoints`.
- **Full outputs:** `workspace_visualization/lateral_table_region/`, `workspace_visualization/distance_to_cube/`, `workspace_visualization/pose_clusters/`.

## Mechanistic Interpretability Snapshot

<div align="center">
  <img src="assets/neuronpedia_dashboard.png" width="720" style="border-radius: 12px; margin-bottom: 20px;">
</div>

## Dataset and Checkpoint Sources

- Datasets and pretrained artifacts are distributed through supplementary storage links documented in module READMEs.
- Dataset IDs used in scripts/configs are anonymized (`gr1_pickup_grasp`, `gr1_reward_pred`, `gr1_reward_pred_v2`).

### Supplementary Storage Links

#### Datasets

- `gr1_pickup_grasp`: [Google Drive folder](https://drive.google.com/drive/folders/18wbnfFm3c51hM97ZnRBc5H2hkrH0RVyy?usp=sharing)
- `gr1_reward_pred`: [Google Drive folder](https://drive.google.com/drive/folders/1Rb9KgvoqdevNvlJ540QD2sy_ZVDP0f8k?usp=sharing)
- `gr1_reward_pred_v2`: [Google Drive folder](https://drive.google.com/drive/folders/1fg36JJ4RlNYOWc5y6fXR4Ji1VwGZuAU4?usp=sharing)

#### LeWM Checkpoints and Goal Galleries

| Variant | Model Checkpoint | Goal Gallery |
| :--- | :--- | :--- |
| Single-View RGB | [gr1_reward_tuned_v2.ckpt](https://drive.google.com/file/d/1dPp-yuSEKMywKPH1mzKT4m7f7Rq5ak7A/view?usp=sharing) | [goal_gallery.pth](https://drive.google.com/file/d/1KDxrZVbrlB2wDDPJAQfHIZxZi48ZhN8U/view?usp=sharing) |
| Multi-View RGB | [gr1_reward_tuned_v2.ckpt](https://drive.google.com/file/d/1pGMMicqYL_Z8GCS1TOe2A_kAAJQLV3qd/view?usp=sharing) | [goal_gallery.pth](https://drive.google.com/file/d/1gYk_P9Godif20boD64M8epR5xSSSxugn/view?usp=sharing) |
| Multi-View RGB + Skeletal Priors | [gr1_reward_tuned_v2.ckpt](https://drive.google.com/file/d/1tiN-awjiMl0oUy8uLE9JT0850QQOPCUI/view?usp=sharing) | [goal_gallery.pth](https://drive.google.com/file/d/1R9uuqpd1yb7t7-NwuvEq7VrOuI6wI152/view?usp=sharing) |
| Multi-View RGB + Skeletal Priors + DINOv3 Waypoints | [gr1_reward_tuned_v1.ckpt](https://drive.google.com/file/d/18xFB2lbxY5Q7EFs-18V9tkmED7NSQelR/view?usp=sharing) | [goal_gallery.pth](https://drive.google.com/file/d/1nFW8J_6PQhFaB1agzd8vaEZ1yIy8cCPA/view?usp=sharing) |

#### Interpretability Artifacts

- Activations harvests: [Google Drive folder](https://drive.google.com/drive/folders/1wAUUsT88b458OUQ6qdTsIe8hCzuinNc4?usp=sharing)
- Transcoder weights: [Google Drive folder](https://drive.google.com/drive/folders/1LRxPy4A02ZTanGnQmsosvC_oxq-8AHM6?usp=sharing)
- Manifold harvest (Single-View RGB): [manifold_data.pt](https://drive.google.com/file/d/17f2l3ebzrX0chu5Zy0GiWEYqGZ-M0CyK/view?usp=sharing)
- Manifold harvest (Multi-View RGB): [manifold_data.pt](https://drive.google.com/file/d/1ix3_ISc80CX91RWKafP0pV8ZA9RlO49f/view?usp=sharing)
- Manifold harvest (Multi-View RGB + Skeletal Priors): [manifold_data.pt](https://drive.google.com/file/d/1XG1Bt6jfV7uTy5wSd9INDIY-g0hu5U1i/view?usp=sharing)
- Manifold harvest (Multi-View RGB + Skeletal Priors + DINOv3 Waypoints): [manifold_data.pt](https://drive.google.com/file/d/1nnAQZNHOSeIb_dLfYZCy-MjN9BIKtRji/view?usp=sharing)

## Limitations

- This repository provides reproducibility artifacts for a bounded submission window; analyses are intentionally scoped to one task family.
- Static probes diagnose geometry and separability but are not equivalent to full in-distribution policy evaluation.
