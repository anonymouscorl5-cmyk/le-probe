# Interpretability: Probing the Latent Mystery

As we saw in the [**`lewm/README.md`**](../lewm/README.md), LeWM struggles with **Latent Discriminability**—the inability to find a clear path to the goal in the latent manifold for our dataset.

This motivated the need for a deeper understanding of the latent space for our model before attempting further improvements to the model and the training pipeline as a whole.

## 📐 Methodology

Following is the architecture used for experimenting with the trained model for interpretability,

<div align="center">
  <img src="../assets/interpretability_architecture.png" width="70%" style="border-radius: 12px; margin-top: 20px;">
  <p><i>LeWM Interpretability: Mechanistic Analysis & Causal Intervention Stack</i></p>
</div>

### 🛠 Key Components

- [`dashboard`](./dashboard): contains the scripts used to visualize the neuronpedia dashboard based on the trained transcoders.
- [`manifold`](./manifold): 3D PCA/t-SNE visualization of the latent manifold across trajectories.
- [`transcoders`](./transcoders): Unified module for both SAE (Identity) and CLT (Transition) probes for harvesting as well as training.
- [`steering`](./steering): Latent Steering used for interpretability (hasn't been tried yet).
- [`teleop_ui_interpret.py`](./teleop_ui_interpret.py): Dashboard to view the top 15 features triggered the most by a given state-action configuration in a bar plot.
- [`simulation_teleop_interpret.py`](./simulation_teleop_interpret.py): A simplified version of [`dataset/simulation_teleop.py`] for the features that are needed with static brain snapshots to complement the activation plot.
- [`latent_server.py`](./latent_server.py): Server used to compute features with the most activation, used by the dashboard to contruct the bar plot.

### 🏗 Process

We have implemented a high-fidelity mechanistic interpretability stack that operates with **Zero-Impact Modularity** (using PyTorch hooks to avoid modifying the core `lewm` code).

#### ⚡ Layer-Wise Transcoder Stack (`/transcoders`)
Instead of single-layer probes, we now employ a full-stack attribution strategy:
- **Comprehensive Grounding**: Sparse Autoencoders (SAE) are trained on Layer 0 to isolate physical primitives (edges, colors, spatial anchors).
- **Causal Chaining**: Cross-Layer Transcoders (CLT) are trained for **every single layer** of the Encoder (L0-L11) and Predictor (L12-L17), mapping the "Chain of Custody" from raw pixels to future state predictions.
- **JEPA Alignment**: The attribution engine reflects the 5-stage JEPA flow: `Inputs` $\rightarrow$ `Encoder` $\rightarrow$ `Joints` $\rightarrow$ `Predictor` $\rightarrow$ `Reward Head`.

### 👀 Neuronpedia Adaptation

While [neuronpedia](https://github.com/hijohnnylin/neuronpedia) is mainly used for interpretability in LLMs, I have tried adapting it to work with our data by creating a [fork](https://github.com/vedpatwardhan/neuronpedia).

The major changes are:
1. Graphs are supposed to be created using the [`dashboard`](./dashboard) scripts rather than the default buttons on neuronpedia.
2. Additional pane at the bottom right for visualizing the patches in the image visualized on the graph.
3. A proxy server that serves data to the frontend for the interpretability pipeline.

### 📊 Latent Topology Audit: The "Discriminability Gap"

Separate from the mechanistic circuit analysis, we analyzed the topological structure of the latent space to diagnose why the MPC solver "stalls" during search. By projecting 200 episodes into 3D space, we identified "manifold fragmentation."

## 🔬 Results: The "Residual Highway"

The attribution graph was computed using integrated gradients,

<div align="center">
  <img src="../assets/neuronpedia_dashboard.png" width="100%" style="border-radius: 12px; margin-bottom: 20px;">
  <p><i>The Le-Probe Dashboard: Hierarchical circuit tracing from pixels to reward probability.</i></p>
</div>

*   **Observation**: High-level decision hubs in the late encoder (L11) draw raw spatial data directly from early sensory layers (L0/L1) via 10+ layer skip connections.
*   **Verification**: Feature **`F848`** (L11) was identified as a critical causal junction for grasp success, receiving direct injections from perceptual features like **`F5000`** (L0).
*   **Connectivity Filtering**: We implemented a **Direction-Aware Union Min-K Filter** to maintain graph clarity while preserving these vital long-range causal links.

Here's the results of the manifold visualization,
| Tool | **3D PCA** | **3D t-SNE** | **3D UMAP** |
| :--- | :---: | :---: | :---: |
| **Methodology** | **Global Variance Audit**: Measures representational diversity and collapse. | **Local Neighborhood Audit**: Measures phase-wise temporal consistency. | **Manifold Topology Audit**: Maps global task continuity and goal reachability. |
| **Single-View Result** | ![PCA](../assets/manifold_3d_pca.png) | ![t-SNE](../assets/manifold_3d_tsne.png) | ![UMAP](../assets/manifold_3d_umap.png) |
| **Finding** | High entropy cloud; saturated by environmental noise. | High entanglement; phase-transition "jitter." | **Failure**: Disconnected islands (stalled planning). |
| **Multi-View Result** | ![PCA](../assets/manifold_3d_multiview_pca.png) | ![t-SNE](../assets/manifold_3d_multiview_tsne.png) | ![UMAP](../assets/manifold_3d_multiview_umap.png) |
| **Finding** | Low entropy threads; linearized physics. | Distinct episode "hairs"; temporal smoothness. | **Success**: Threaded manifold (global continuity). |

## 🚀 Research Roadmap: Next Steps

The dashboard now allows us to audit the effects of key architectural changes:
1. **Kinematic Polytopes**: Using reachability analysis to avoid out-of-distribution failure modes like arm folding.
2. **Latent Steering**: Closing the causal loop by using discovered features as reward boosters during real-time inference.

## 🚀 Workflows

### 1. Training the CLT

The activations and weights are available here:

| Type | Google Drive Link |
| --- | --- |
| Activations | [activations_granular](https://drive.google.com/drive/folders/1wAUUsT88b458OUQ6qdTsIe8hCzuinNc4?usp=sharing) |
| Weights | [transcoder_weights_residual](https://drive.google.com/drive/folders/1LRxPy4A02ZTanGnQmsosvC_oxq-8AHM6?usp=sharing) |

Optionally, the activations can be harvested with the following scripts, also covered in [**`LeWM_Interpretability.ipynb`**](./LeWM_Interpretability.ipynb)

```bash
# 1. Harvest the activations
.venv/bin/python interpretability/transcoders/harvest_activations.py \
    --model gr1_reward_tuned_v2.ckpt
    --output activations_granular \
    --workers 4

# 2. Audit the harvest
.venv/bin/python interpretability/transcoders/audit_harvest.py \
    --model gr1_reward_tuned_v2.ckpt \
    --dir activations_granular

# 3. Train CLT
bash interpretability/transcoders/batch_train.sh
```


### 2. Neuronpedia Visualization

The dashboard requires a local Dockerized Neuronpedia instance and a proxy bridge to the attribution engine.

```bash
# 1. Start the Neuronpedia Dashboard (Docker)
cd interpretability/neuronpedia
make webapp-localhost-dev

# 2. Start the engine (Colab)
.venv/bin/python interpretability/dashboard/engine.py \
    --repo vedpatwardhan/gr1_pickup_grasp \
    --meta activations_granular/encoder_L0.json \
    --model gr1_reward_tuned_v2.ckpt \
    --transcoders transcoder_weights_residual \
    --min-k 10

# 3. Start the Dashboard Proxy (Local)
# This tunnels requests from the dashboard to the GPU engine using the COLAB_URL
.venv/bin/python interpretability/dashboard/neuronpedia_server.py

# 4. Generate Graphs
.venv/bin/python interpretability/dashboard/regenerate_graphs.py
```

Once the graphs are generated, the interactive HTML can be viewed at http://localhost:3000.


### 3. Latent Manifold Visualization

Analyze the topological structure of the latent space to diagnose discriminability.

**Latest Checkpoints & Manifolds:**
| Version | Interactive Manifold (UMAP) |
| :--- | :--- |
| **Single-View** | [manifold_data.pt](https://drive.google.com/file/d/17f2l3ebzrX0chu5Zy0GiWEYqGZ-M0CyK/view?usp=sharing) |
| **Multi-View** | [manifold_data.pt](https://drive.google.com/file/d/1ix3_ISc80CX91RWKafP0pV8ZA9RlO49f/view?usp=sharing) |

```bash
# 1. Harvest latents for the entire trajectory
.venv/bin/python interpretability/manifold/harvest_manifold.py --episodes 100

# 2. Generate 3D Visualization
.venv/bin/python interpretability/manifold/visualize_manifold.py --method umap
```
