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

The canonical reference workflow for reproducing LeWM experiments is:

- [`lewm/LeWM_Training.ipynb`](./lewm/LeWM_Training.ipynb) for training and checkpoint generation.
- [`lewm/LeWM_E2E.ipynb`](./lewm/LeWM_E2E.ipynb) for end-to-end planning/inference evaluation.

Notebook-aligned CLI equivalents (including priors, fused cache, trainer/tuner flow) are documented in [`lewm/README.md`](./lewm/README.md).

```bash
# 1) Start planner server (full variant example)
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
  <table>
    <tr>
      <th>Single-View RGB</th>
      <th>Multi-View RGB</th>
      <th>Skeletal Priors</th>
      <th>DINOv3 Waypoints</th>
    </tr>
    <tr>
      <td><img src="assets/lewm_grasp.gif" width="180" alt="Single-View RGB rollout"></td>
      <td><img src="assets/lewm_grasp_multiview.gif" width="180" alt="Multi-View RGB rollout"></td>
      <td><img src="assets/lewm_grasp_multiview_skeleton.gif" width="180" alt="Multi-View RGB plus Skeletal Priors rollout"></td>
      <td><img src="assets/lewm_grasp_multiview_skeleton_dino.gif" width="180" alt="Multi-View RGB plus Skeletal Priors plus DINOv3 Waypoints rollout"></td>
    </tr>
  </table>
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
  <table>
    <tr>
      <th>Task Workspace</th>
      <th>Lateral Table Region</th>
      <th>Distance to Cube</th>
      <th>Pose Clusters</th>
    </tr>
    <tr>
      <td><img src="assets/task_workspace.png" width="180" alt="Task workspace"></td>
      <td><img src="assets/lateral_table_region.png" width="180" alt="Lateral regions"></td>
      <td><img src="assets/distance_to_cube.png" width="180" alt="Distance bins"></td>
      <td><img src="assets/pose_clusters.png" width="180" alt="Pose clusters"></td>
    </tr>
  </table>
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

- `gr1_pickup_grasp`: [Google Drive folder](https://drive.google.com/drive/folders/1yYMT7J_eRkQmXDq3tcisNd4kRSWeTI40?usp=sharing)
- `gr1_reward_pred`: [Google Drive folder](https://drive.google.com/drive/folders/1QWra9dRJ9aceUqOpmj56OG8SaVUCVr-g?usp=sharing)
- `gr1_reward_pred_v2`: [Google Drive folder](https://drive.google.com/drive/folders/1iwz_1LeEi4vbMWDeIXU_Pb6tVxDqcbNE?usp=sharing)

#### LeWM Checkpoints and Goal Galleries

| Variant | Model Checkpoint | Goal Gallery |
| :--- | :--- | :--- |
| Single-View RGB | [gr1_reward_tuned_v2.ckpt](https://drive.google.com/file/d/1L0RE9V647-JduSCJ40y1TEI-N8MIO62D/view?usp=sharing) | [goal_gallery.pth](https://drive.google.com/file/d/1CA9KxgnvHeJjslUOKoaxvmPV4TnhzWeS/view?usp=sharing) |
| Multi-View RGB | [gr1_reward_tuned_v2.ckpt](https://drive.google.com/file/d/1VEEAa4vWcnqQN1PMK5422FK_1QJ0Hu74/view?usp=sharing) | [goal_gallery.pth](https://drive.google.com/file/d/1ntMBODRRDP-bZDFUrbxli-3WxT4zveAv/view?usp=sharing) |
| Multi-View RGB + Skeletal Priors | [gr1_reward_tuned_v6.ckpt](https://drive.google.com/file/d/1W2UUco30AJE1ygjeGjRK1jFWB7PvGXEx/view?usp=sharing) | [goal_gallery.pth](https://drive.google.com/file/d/1YEsGDwT1AvWetxS7vbLGL94xTOEDJtyP/view?usp=sharing) |
| Multi-View RGB + Skeletal Priors + DINOv3 Waypoints | [gr1_reward_tuned_v1.ckpt](https://drive.google.com/file/d/1Yt1Q60yvvDPPFE3JjICq48ocOycUALGT/view?usp=sharing) | [goal_gallery.pth](https://drive.google.com/file/d/1jpApbuPUHIAb3Ae87VzFAvFBVhVZr3X6/view?usp=sharing) |

#### Interpretability Artifacts

- Manifold harvest (Single-View RGB): [manifold_data.pt](https://drive.google.com/file/d/18us_mOIVa2QgIP2VoISC-wpVzI7moCyV/view?usp=sharing)
- Manifold harvest (Multi-View RGB): [manifold_data.pt](https://drive.google.com/file/d/1lqcmNQGiiECSPG4CM1h2c1S3JxwUQ_mP/view?usp=sharing)
- Manifold harvest (Multi-View RGB + Skeletal Priors): [manifold_data.pt](https://drive.google.com/file/d/19lxR0rJ-Oo7drudU_NyXQL3_cvlOGIcO/view?usp=sharing)
- Manifold harvest (Multi-View RGB + Skeletal Priors + DINOv3 Waypoints): [manifold_data.pt](https://drive.google.com/file/d/1Xhc9kMDilG3TpBA8GdDFLF4l7oe4j3Wz/view?usp=sharing)
- Transcoder weights (Single-View RGB): [Google Drive folder](https://drive.google.com/drive/folders/13Aw6iF1PfWqBR2CRh3A-wjqub6DP_Ty2?usp=sharing)
- Transcoder weights (Multi-View RGB): [Google Drive folder](https://drive.google.com/drive/folders/12vq8hnySCqt6Z6rYGioz-ghjoFIdvcCv?usp=sharing)
- Transcoder weights (Multi-View RGB + Skeletal Priors): [Google Drive folder](https://drive.google.com/drive/folders/1TXS4sObpbvBxI-GUrdoicY1hNXPh_c1Q?usp=sharing)
- Transcoder weights (Multi-View RGB + Skeletal Priors + DINOv3 Waypoints): [Google Drive folder](https://drive.google.com/drive/folders/1Kak0qNzLPJr_jmDWCJMLu5ss4eg1Dsvb?usp=sharing)

## Limitations

- This repository provides reproducibility artifacts for a bounded submission window; analyses are intentionally scoped to one task family.
- Static probes diagnose geometry and separability but are not equivalent to full in-distribution policy evaluation.
