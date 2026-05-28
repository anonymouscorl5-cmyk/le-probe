# Interpretability: Topology, Static Probes, and Mechanistic Audits

This module covers the analysis side of Le-Probe: why planning behavior changes across representation variants.

## Protocol Coverage

- **Training-topology audits:** PCA/t-SNE/UMAP on trajectory latents.
- **Static probes:** workspace-labeled latent projections on fixed probe snapshots.
- **Mechanistic audits:** CLTs and Neuronpedia-based attribution traces.

## Submodules

- [`manifold/`](./manifold): latent harvesting and dimensionality reduction.
- [`transcoders/`](./transcoders): activation harvest, audits, and CLT training.
- [`dashboard/`](./dashboard): bridge code for Neuronpedia visualization.
- [`LeWM_Interpretability.ipynb`](./LeWM_Interpretability.ipynb): notebook pipeline.

## Training-Manifold Snapshot

| Variant | 3D PCA | 3D t-SNE | 3D UMAP |
| :--- | :---: | :---: | :---: |
| **Single-View RGB** | ![PCA](../assets/manifold_3d_pca.png) | ![t-SNE](../assets/manifold_3d_tsne.png) | ![UMAP](../assets/manifold_3d_umap.png) |
| **Multi-View RGB** | ![PCA](../assets/manifold_3d_multiview_pca.png) | ![t-SNE](../assets/manifold_3d_multiview_tsne.png) | ![UMAP](../assets/manifold_3d_multiview_umap.png) |
| **Multi-View RGB + Skeletal Priors** | ![PCA](../assets/manifold_3d_multiview_skeleton_pca.png) | ![t-SNE](../assets/manifold_3d_multiview_skeleton_tsne.png) | ![UMAP](../assets/manifold_3d_multiview_skeleton_umap.png) |
| **Multi-View RGB + Skeletal Priors + DINOv3 Waypoints** | ![PCA](../assets/manifold_3d_multiview_skeleton_dino_pca.png) | ![t-SNE](../assets/manifold_3d_multiview_skeleton_dino_tsne.png) | ![UMAP](../assets/manifold_3d_multiview_skeleton_dino_umap.png) |

## Static Probe Results

- **Primary takeaway:** static latent organization improves consistently across variants, with best separation in `Multi-View RGB + Skeletal Priors` and `Multi-View RGB + Skeletal Priors + DINOv3 Waypoints`.
- **Output locations:** `workspace_visualization/lateral_table_region/`, `workspace_visualization/distance_to_cube/`, `workspace_visualization/pose_clusters/`.

## Setup

```bash
cd le-probe
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## CLT Workflow (Minimal)

```bash
# 1) Harvest activations
.venv/bin/python interpretability/transcoders/harvest_activations.py \
  --model <ckpt> \
  --output_dir activations_granular_multiview_skeleton_dino \
  --multi_view --use_skeleton --use_dino --workers 4

# 2) Audit
.venv/bin/python interpretability/transcoders/audit_harvest.py \
  --model <ckpt> \
  --dir activations_granular_multiview_skeleton_dino \
  --multi_view --use_skeleton --use_dino
```

## Neuronpedia Visualization

```bash
cd interpretability/neuronpedia
make webapp-localhost-dev

.venv/bin/python interpretability/dashboard/engine.py \
  --repo gr1_pickup_grasp \
  --meta activations_granular_multiview_skeleton/encoder_L0.json \
  --model <ckpt> \
  --transcoders <transcoder_dir> \
  --multi_view --use_skeleton --min-k 10

.venv/bin/python interpretability/dashboard/neuronpedia_server.py
```

## Current Mechanistic Artifact

<div align="center">
  <img src="../assets/neuronpedia_dashboard.png" width="720" style="border-radius: 12px; margin-bottom: 20px;">
</div>

## Supplementary Artifacts

- Activations harvests: [Google Drive folder](https://drive.google.com/drive/folders/1wAUUsT88b458OUQ6qdTsIe8hCzuinNc4?usp=sharing)
- Transcoder weights: [Google Drive folder](https://drive.google.com/drive/folders/1LRxPy4A02ZTanGnQmsosvC_oxq-8AHM6?usp=sharing)

### Manifold Harvest Dumps

- Single-View RGB: [manifold_data.pt](https://drive.google.com/file/d/17f2l3ebzrX0chu5Zy0GiWEYqGZ-M0CyK/view?usp=sharing)
- Multi-View RGB: [manifold_data.pt](https://drive.google.com/file/d/1ix3_ISc80CX91RWKafP0pV8ZA9RlO49f/view?usp=sharing)
- Multi-View RGB + Skeletal Priors: [manifold_data.pt](https://drive.google.com/file/d/1XG1Bt6jfV7uTy5wSd9INDIY-g0hu5U1i/view?usp=sharing)
- Multi-View RGB + Skeletal Priors + DINOv3 Waypoints: [manifold_data.pt](https://drive.google.com/file/d/1nnAQZNHOSeIb_dLfYZCy-MjN9BIKtRji/view?usp=sharing)
