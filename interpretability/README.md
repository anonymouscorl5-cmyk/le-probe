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
  <img src="../assets/neuronpedia_dashboard.png" width="720" style="border-radius: 12px; margin-bottom: 20px;">
  <p><i>The Le-Probe Dashboard: Hierarchical circuit tracing from pixels to reward probability.</i></p>
</div>

*   **Observation**: High-level decision hubs in the late encoder (L11) draw raw spatial data directly from early sensory layers (L0/L1) via 10+ layer skip connections.
*   **Verification**: Feature **`F848`** (L11) was identified as a critical causal junction for grasp success, receiving direct injections from perceptual features like **`F5000`** (L0).
*   **Connectivity Filtering**: We implemented a **Direction-Aware Union Min-K Filter** to maintain graph clarity while preserving these vital long-range causal links.

Here's the results of the manifold visualization,
| Tool | **3D PCA** | **3D t-SNE** | **3D UMAP** |
| :--- | :---: | :---: | :---: |
| **Methodology** | **Global Variance Audit**: Measures representational diversity and collapse. | **Local Neighborhood Audit**: Measures phase-wise temporal consistency. | **Manifold Topology Audit**: Maps global task continuity and goal reachability. |
| **Single-View** | ![PCA](../assets/manifold_3d_pca.png) | ![t-SNE](../assets/manifold_3d_tsne.png) | ![UMAP](../assets/manifold_3d_umap.png) |
| **Finding** | **Diffuse High-Entropy Cloud**: Extremely scattered spatial representation showing zero coherent coordinate trajectories or macro-structure. High sensitivity to background pixels and peripheral noise prevents representation alignment. | **Jittered Local Clustered Fields**: Demonstrates fragmented temporal sequences with severe phase-transition jitter and micro-loops, disrupting planning continuity. | **Fragmented Archipelago**: Separates into highly disconnected topological islands with extremely low semantic grouping, causing pathfinding failure. |
| **Multi-View** | ![PCA](../assets/manifold_3d_multiview_pca.png) | ![t-SNE](../assets/manifold_3d_multiview_tsne.png) | ![UMAP](../assets/manifold_3d_multiview_umap.png) |
| **Finding** | **Dispersed Trajectory Threads**: Exhibits broad temporal flow vectors from start (yellow) to finish (brown), but fails to establish a structured planning corridor. Early-stage (yellow) embeddings are widely scattered, indicating a lack of unified task-space coordination. | **Overfitted Trajectory Splitting**: Offers cleaner separation but is highly prone to memorization (overfitting). Distinct, isolated thread "hairs" run parallel on the peripheries with a dense interior cluster, failing to merge into shared execution directions. | **Isolated Trajectory Islands**: Exhibits extreme topological collapse and overfitting, with isolated single-episode islands (rarely grouped by more than two). This confirms that without physical constraints, the model struggles to generalize. |
| **Multi-View + Skeletal Priors** | ![PCA](../assets/manifold_3d_multiview_skeleton_pca.png) | ![t-SNE](../assets/manifold_3d_multiview_skeleton_tsne.png) | ![UMAP](../assets/manifold_3d_multiview_skeleton_umap.png) |
| **Finding** | **Skeletal Directional Highway**: Establishes a highly structured, low-entropy directional corridor. Trajectories originate from a clustered interior (start/yellow), transition systematically through the task-space base (approach/orange), and terminate consistently at the target zone (success/brown). | **Coherent Structural Bundles**: Highly interpretable, clustered trajectories. Large groups of distinct episodes align into unified directional bundles, indicating generalized movement policies rather than overfitting or memorization. | **Unified Task Continua**: Exceptional topological recovery. Large, continuous semantic islands emerge with smooth, convergent trajectory arcs, demonstrating the critical role of skeletal priors in pruning noise. |
| **Multi-View + Skeletal Priors + DINOv3 Waypoints** | ![PCA](../assets/manifold_3d_multiview_skeleton_dino_pca.png) | ![t-SNE](../assets/manifold_3d_multiview_skeleton_dino_tsne.png) | ![UMAP](../assets/manifold_3d_multiview_skeleton_dino_umap.png) |
| **Finding** | **Clear Directional Highway**: The corridor visible in the previous experiment becomes even more defined with all the start points at the top and all the end points at the bottom, courtesy of learning the global objectives with the waypoints. | **Clear Manifold Structure**: The manifold structure is even better than just relying on the skeletons, similar to PCA there's even more separation between start and end states, solving the discriminability problem between goal and non-goal states | **More Cohesion**: The UMAP plot with the waypoints included shows even more cohesion than with just the skeletons, there's about 4 major islands connected with each other through narrow links. |

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
# 1. Harvest activations (pick ONE experiment per run; flags match lewm_server.py)
# Single-View
.venv/bin/python interpretability/transcoders/harvest_activations.py \
    --model gr1_reward_tuned_v2.ckpt \
    --output_dir activations_granular_single_view \
    --workers 4

# Multi-View
.venv/bin/python interpretability/transcoders/harvest_activations.py \
    --model gr1_reward_tuned_v2.ckpt \
    --output_dir activations_granular_multiview \
    --multi_view --workers 4

# Multi-View + Skeletal Priors
.venv/bin/python interpretability/transcoders/harvest_activations.py \
    --model gr1_reward_tuned_v6.ckpt \
    --output_dir activations_granular_multiview_skeleton \
    --multi_view --use_skeleton --workers 4

# Multi-View + Skeletal + DINO Waypoints
.venv/bin/python interpretability/transcoders/harvest_activations.py \
    --model gr1_reward_tuned_v1.ckpt \
    --output_dir activations_granular_multiview_skeleton_dino \
    --multi_view --use_skeleton --use_dino --workers 4

# 2. Audit (use the same flags as harvest)
.venv/bin/python interpretability/transcoders/audit_harvest.py \
    --model gr1_reward_tuned_v2.ckpt \
    --dir activations_granular_multiview \
    --multi_view

# 3. Train CLT for that experiment only (set dirs, then run once)
ACTIVATIONS_DIR=activations_granular_multiview \
OUTPUT_DIR=transcoder_weights_multiview \
bash interpretability/transcoders/batch_train.sh
```


### 2. Neuronpedia Visualization

The dashboard requires a local Dockerized Neuronpedia instance and a proxy bridge to the attribution engine.

```bash
# 1. Start the Neuronpedia Dashboard (Docker)
cd interpretability/neuronpedia
make webapp-localhost-dev

# 2. Start the engine (Colab) — flags must match the harvested experiment
.venv/bin/python interpretability/dashboard/engine.py \
    --repo vedpatwardhan/gr1_pickup_grasp \
    --meta activations_granular_multiview_skeleton/encoder_L0.json \
    --model gr1_reward_tuned_v6.ckpt \
    --transcoders transcoder_weights_multiview_skeleton \
    --multi_view --use_skeleton \
    --min-k 10

# DINO experiment: add --use_dino and --attribution_target subgoal (optional)

# 3. Start the Dashboard Proxy (Local)
# This tunnels requests from the dashboard to the GPU engine using the COLAB_URL
.venv/bin/python interpretability/dashboard/neuronpedia_server.py

# 4. Generate Graphs
.venv/bin/python interpretability/dashboard/regenerate_graphs.py
```

Once the graphs are generated, the interactive HTML can be viewed at http://localhost:3000.


### 3. Latent Manifold Visualization

Analyze the topological structure of the latent space to diagnose discriminability.

**Latest Manifolds:**
| Version | Manifold Harvest |
| :--- | :--- |
| **Single-View** | [manifold_data.pt](https://drive.google.com/file/d/17f2l3ebzrX0chu5Zy0GiWEYqGZ-M0CyK/view?usp=sharing) |
| **Multi-View** | [manifold_data.pt](https://drive.google.com/file/d/1ix3_ISc80CX91RWKafP0pV8ZA9RlO49f/view?usp=sharing) |
| **Multi-View + Skeletal Priors** | [manifold_data.pt](https://drive.google.com/file/d/1XG1Bt6jfV7uTy5wSd9INDIY-g0hu5U1i/view?usp=sharing) |
| **Multi-View + Skeletal Priors + DINOv3 Waypoints** | [manifold_data.pt](https://drive.google.com/file/d/1nnAQZNHOSeIb_dLfYZCy-MjN9BIKtRji/view?usp=sharing) |

```bash
# Single-View
.venv/bin/python interpretability/manifold/harvest_manifold.py --episodes 200

# Multi-View
.venv/bin/python interpretability/manifold/harvest_manifold.py --episodes 200 --multi_view

# Multi-View + Skeleton Priors
.venv/bin/python interpretability/manifold/harvest_manifold.py --episodes 200 --multi_view --use_skeleton

# Visualize the manifold
.venv/bin/python interpretability/manifold/visualize_manifold.py --method umap --output manifold_3d_umap.html [--highlight <ID1> <ID2>, ...]

```