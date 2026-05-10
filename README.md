# Le-Probe: Probing LeWM

<div align="center">
  <img src="assets/banner.png" width="100%" style="border-radius: 12px; margin-bottom: 20px;">
</div>

Le-Probe is a project meant to analyze and compare **LeWM + MPC** against traditional **Vision-Language-Action (VLA)** policies like GR00T-N1.

My investigation focuses on a high-DoF (32+) manipulation task that require multi-phase coordination, specifically comparing two distinct behavioral strategies: **Grasp** and **Cup**.

## 🚀 Repository Structure

- [**`dataset/`**](./dataset): Teleoperation and high-fidelity data collection (32-frame episodes).
- [**`vla/`**](./vla): GR00T-N1 baselines. Successfully demonstrates both Grasp and Cup behaviors.
- [**`lewm/`**](./lewm): World model training and Oracle MPC. Currently struggles with latent discriminability.
- [**`interpretability/`**](./interpretability): The "Search for the Why"—mechanistic analysis of LeWM failure modes.
- [**`scripts/`**](./scripts): Maintenance, dataset compression, and reward calibration tools.

## 📚 Contents

- **Core Mission:** Explains the work done so far and results.
- **Getting Started:** Installation and setup instructions to reproduce the results.
- **Details:** Each of the sub-folders mentioned above have their own README files providing more details about the process and the results.

## 🔬 Core Mission: VLA vs. LeWM

The project was born from a comparative study of GR00T N1 with LeWM for **picking up a red cube** from the table, but eventually turned into a mechanistic interpretability project for LeWM to understand the latent space in more detail.

### 1. Target Behaviors (Ground Truth)

I've created two datasets aimed at picking up the cube with different behavioural strategies:

<div align="center">
  <table>
    <tr>
      <th>Dataset: Grasp Pattern</th>
      <th>Dataset: Cup Pattern</th>
    </tr>
    <tr>
      <td><img src="assets/dataset_grasp.gif" width="320"></td>
      <td><img src="assets/dataset_cup.gif" width="320"></td>
    </tr>
  </table>
</div>

More details are available in [**`dataset/README.md`**](./dataset/README.md).

### 2. VLA Baseline Success (GR00T-N1)

I trained GR00T-N1 to imitate both styles using BC. While the robot isn't able to actually pick up the cube, the behaviour of the model trained with the grasp movement as opposed to the cup movement is clearly visible.

More details available in [**`vla/README.md`**](./vla/README.md).

<div align="center">
  <table>
    <tr>
      <th>VLA: Grasp Execution</th>
      <th>VLA: Cup Execution</th>
    </tr>
    <tr>
      <td><img src="assets/vla_grasp.gif" width="320"></td>
      <td><img src="assets/vla_cup.gif" width="320"></td>
    </tr>
  </table>
</div>

### 3. LeWM Challenges (The Discriminability Gap)
LeWM, despite training with a large softrank, failed to sufficiently discriminate the goal state from non-goal states in the latent space.

More details available in [**`lewm/README.md`**](./lewm/README.md).

#### Reward Head Intervention

To try and still get some sort of idea of the quality of training, I trained an auxiliary reward head on snapshot data with a broader range of trajectories predict the reward from the latent space. While reward prediction was much better, the MPC solver still didn't manage to actually pick up the cube and instead just got close to it and moved away as you can see in the video below.

<div align="center">
  <b>LeWM: Grasp Execution</b>
  <hr width="320">
  <img src="assets/lewm_grasp.gif" width="320">
</div>

#### Next Steps

Given the behaviour somewhat works but nowhere near good enough, the next step is to try and probe into the model to identify the sparse features driving these latent representations.

### 4. Interpretability: The "Residual Highway"

To understand why LeWM struggles with goal discrimination, I am working with a fork of [neuronpedia](https://github.com/hijohnnylin/neuronpedia) [here](https://github.com/vedpatwardhan/neuronpedia).

#### Architecture

We use a full-stack attribution engine that probes every layer of the Encoder and Predictor.

<div align="center">
  <img src="assets/interpretability_architecture.png" width="70%" style="border-radius: 12px; margin-top: 20px;">
  <p><i>LeWM Interpretability: Global Causal Tracing from Pixels to Reward.</i></p>
</div>

#### Results

High-level decision hubs (L11) draw raw spatial anchors directly from early sensory layers (L0/L1) via 10+ layer skip connections.

<div align="center">
  <img src="assets/neuronpedia_dashboard.png" width="100%" style="border-radius: 12px; margin-bottom: 20px;">
  <p><i>The Le-Probe Dashboard: Mapping the L0 $\rightarrow$ L11 skip connections.</i></p>
</div>

*   **Neuronpedia Dashboard**: Hierarchical circuit tracing from pixels to reward probability (L0 $\rightarrow$ L11 skip connections).
    *   **Visual Patch Audit**: Mapping feature activations back to specific image patches with green-box highlighting.
    *   **Integrated Gradients**: Tracing the exact causal path from pixels to Success Probability.
    *   **Directional Filtering**: Using a **Min-K Union** filter to isolate the most critical causal circuits.
*   **Latent Topology Audit**: Dimensionality reduction (PCA, t-SNE, UMAP) to diagnose manifold fragmentation and MPC search failures.

| 3D PCA | 3D t-SNE | 3D UMAP |
| :---: | :---: | :---: |
| ![PCA](assets/manifold_3d_pca.png) | ![t-SNE](assets/manifold_3d_tsne.png) | ![UMAP](assets/manifold_3d_umap.png) |

More details are available in [**`interpretability/README.md`**](./interpretability/README.md).

#### Next Steps
With the diagnostic infrastructure stabilized, we are investigating:
1. **Multi-View Data**: Training with 5 camera views to match VLA input density.
2. **Kinematic Polytopes**: Using reachability analysis to prevent out-of-distribution arm folding.
3. **Latent Steering**: Using discovered features as reward boosters for real-time MPC.

## 🛠 Getting Started

### 1. Installation
```bash
# Clone with submodules (includes the custom Neuronpedia fork)
git clone --recursive https://github.com/vedpatwardhan/le-probe.git
cd le-probe && python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Infrastructure Setup
The mechanistic dashboard requires a local Dockerized Neuronpedia instance.

```bash
# Start the Dashboard (Docker)
cd interpretability/neuronpedia
make up

# Start the Attribution Proxy
# Tunnels requests from the dashboard to the model engine
.venv/bin/python interpretability/dashboard/neuronpedia_server.py
```

### 1. Data Collection & Datasets

I have published three core datasets used for the above results:
- [**`gr1_pickup_grasp`**](https://huggingface.co/datasets/vedpatwardhan/gr1_pickup_grasp): Precision "pinch" grasp trajectories.
- [**`gr1_pickup_cup`**](https://huggingface.co/datasets/vedpatwardhan/gr1_pickup_cup): Robust "surrounding" containment trajectories.
- [**`gr1_reward_pred`**](https://huggingface.co/datasets/vedpatwardhan/gr1_reward_pred): Multi-behavioral data used to train the Reward Head.

Optionally, if you'd like to record new datasets you can use the following:

#### Data Collection
```bash
# Start the Rerun server
rerun

# Start Sim Server
.venv/bin/python dataset/simulation_teleop.py

# Start Dashboard
streamlit run dataset/teleop_ui.py
```

#### Dataset Upload
```bash
.venv/bin/python dataset/upload_dataset.py --repo_id <>
```

### 2. VLA (GR00T-N1)

#### Training

The model was trained using [**`vla/GR00T_N1_BC.ipynb`**](vla/GR00T_N1_BC.ipynb)

To run the stabilized VLA policy in simulation, the model weights/configs are available at the following folders:

| Type of Movement | Google Drive Link |
| --- | --- |
| **Grasp** | [pretrained_model](https://drive.google.com/drive/folders/1077_msVzs_8AQPaEbDm6XPiq8T_hxirp?usp=sharing) |
| **Cup** | [pretrained_model](https://drive.google.com/drive/folders/1f5p6-5p6_20PpfbONcq-n5T1P7DhHfBw?usp=sharing) |


#### Inference

1. **Inference Server**: Was run using [**`vla/GR00T_N1_E2E.ipynb`**](vla/GR00T_N1_E2E.ipynb) using a Pinggy tunnel.
   ```bash
   .venv/bin/python vla/gr00t_server.py --weights <path to pretrained_model folder>
   ```

2. **Simulation Host**:
   ```bash
   .venv/bin/python vla/simulation_vla.py --host <host> --port <port> --chunks <num_chunks>
   ```

### 3. LeWM + CEM/MPC

#### Training

The model was trained using [**`lewm/LeWM_Training.ipynb`**](lewm/LeWM_Training.ipynb). The original model was trained under the `GR-1 Pickup Grasp` section and the reward head was separately trained under the `GR-1 Reward Pred` section.

Following the training, all goal states in the dataset were harvested in the latent space using [**`lewm/harvest_goals.py`**](lewm/harvest_goals.py) to save inference time.

The weights of the reward-tuned model can be found at [`gr1_reward_tuned_v2.ckpt`](https://drive.google.com/file/d/1dPp-yuSEKMywKPH1mzKT4m7f7Rq5ak7A/view?usp=sharing) and the harvested goals can be found at [`goal_gallery.pth](https://drive.google.com/file/d/1KDxrZVbrlB2wDDPJAQfHIZxZi48ZhN8U/view?usp=sharing).

#### Inference

1. **Inference Server**: Was run using [**`lewm/LEWM_E2E.ipynb`**](lewm/LEWM_E2E.ipynb) using a Pinggy tunnel.
   ```bash
   .venv/bin/python lewm/lewm_server.py --model gr1_reward_tuned_v2.ckpt --gallery goal_gallery.pth
   ```

2. **Simulation Host**:
   ```bash
   .venv/bin/python lewm/simulation_lewm.py --host <host> --port <port>
   ```

### 4. Interpretability

#### Train CLT

The activations and weights are available here:

| Type | Google Drive Link |
| --- | --- |
| Activations | [activations_granular](https://drive.google.com/drive/folders/1wAUUsT88b458OUQ6qdTsIe8hCzuinNc4?usp=sharing) |
| Weights | [transcoder_weights_residual](https://drive.google.com/drive/folders/1LRxPy4A02ZTanGnQmsosvC_oxq-8AHM6?usp=sharing) |

Optionally, the activations can be harvested with the following steps, also covered in [**`LeWM_Interpretability.ipynb`**](./LeWM_Interpretability.ipynb)

1. **Harvest the Activations**: Stores activations for all layers of the LeWM with all the data
```bash
.venv/bin/python interpretability/transcoders/harvest_activations.py \
    --model gr1_reward_tuned_v2.ckpt \
    --output activations_granular \
    --workers 4
```

2. **Audit the Harvest**: Just to check if the harvest worked
```bash
.venv/bin/python interpretability/transcoders/audit_harvest.py \
    --model gr1_reward_tuned_v2.ckpt \
    --dir activations_granular
```

3. **Train the CLT**: The [`batch_train.sh`](interpretability/transcoders/batch_train.sh) script trains the CLT for all layers using the harvested activations.
```bash
bash interpretability/transcoders/batch_train.sh
```

#### Neuronpedia Visualization

1. **Start the Neuronpedia Dashboard (Docker)**: This spins up our fork of neuronpedia.
```bash
cd interpretability/neuronpedia
make webapp-localhost-dev
```

2. **Start the Engine (Colab)**: The engine runs on colab using a Pinggy tunnel
```bash
.venv/bin/python interpretability/dashboard/engine.py \
    --repo vedpatwardhan/gr1_pickup_grasp \
    --meta activations_granular/encoder_L0.json \
    --model gr1_reward_tuned_v2.ckpt \
    --transcoders transcoder_weights_residual \
    --min-k 10
```

3. **Start the Neuronpedia Dashboard Proxy (Local)**: The proxy runs locally and tunnels requests from the dashboard to the engine
```bash
.venv/bin/python interpretability/dashboard/neuronpedia_server.py
```

4. **Generate Graphs**: Uses the server to pre-compute the graphs for certain states in the dataset
```bash
.venv/bin/python interpretability/dashboard/regenerate_graphs.py
```

#### Latent Manifold Topology
Analyze the internal "map" of the latent space to diagnose planning failures.
| ![PCA](assets/manifold_3d_pca.png) | ![t-SNE](assets/manifold_3d_tsne.png) | ![UMAP](assets/manifold_3d_umap.png) |


1. **Harvest Latents**: Uses the server to pre-compute the graphs for certain states in the dataset
```bash
.venv/bin/python interpretability/manifold/harvest_manifold.py --episodes 200
```

2. **Visualize Latent Topology**:
```bash
.venv/bin/python interpretability/manifold/visualize_manifold.py --method umap --output manifold_3d_umap.html
```

---
*Developed by Ved Patwardhan.*
