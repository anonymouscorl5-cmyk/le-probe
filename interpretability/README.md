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

- [`transcoders`]: Unified module for both SAE (Identity) and CLT (Transition) probes.
- [`steering`]: Latent Steering used for interpretability (hasn't been tried yet).
- [`teleop_ui_interpret.py`]: Dashboard to view the top 15 features triggered the most by a given state-action configuration in a bar plot.
- [`simulation_teleop_interpret.py`]: A simplified version of [`dataset/simulation_teleop.py`] for the features that are needed with static brain snapshots to complement the activation plot.
- [`latent_server.py`]: Server used to compute features with the most activation, used by the dashboard to contruct the bar plot.

### 🏗 Process

We have implemented a high-fidelity mechanistic interpretability stack that operates with **Zero-Impact Modularity** (using PyTorch hooks to avoid modifying the core `lewm` code).

#### ⚡ Layer-Wise Transcoder Stack (`/transcoders`)
Instead of single-layer probes, we now employ a full-stack attribution strategy:
- **Comprehensive Grounding**: Sparse Autoencoders (SAE) are trained on Layer 0 to isolate physical primitives (edges, colors, spatial anchors).
- **Causal Chaining**: Cross-Layer Transcoders (CLT) are trained for **every single layer** of the Encoder (L0-L11) and Predictor (L12-L17), mapping the "Chain of Custody" from raw pixels to future state predictions.
- **JEPA Alignment**: The attribution engine reflects the 5-stage JEPA flow: `Inputs` $\rightarrow$ `Encoder` $\rightarrow$ `Joints` $\rightarrow$ `Predictor` $\rightarrow$ `Reward Head`.

## 🔬 Results: The "Residual Highway"

Using the new dashboard, we discovered that LeWM v8 does not reason in a strictly sequential manner. Instead, it utilizes a massive **Residual Highway**:

*   **Discovery**: High-level decision hubs in the late encoder (L11) draw raw spatial data directly from early sensory layers (L0/L1) via 10+ layer skip connections.
*   **Verification**: Feature **`F848`** (L11) was identified as a critical causal junction for grasp success, receiving direct injections from perceptual features like **`F5000`** (L0).
*   **Connectivity Filtering**: We implemented a **Direction-Aware Union Min-K Filter** to maintain graph clarity while preserving these vital long-range causal links.

<div align="center">
  <img src="../assets/neuronpedia_dashboard.png" width="100%" style="border-radius: 12px; margin-bottom: 20px;">
  <p><i>The Le-Probe Dashboard: Hierarchical circuit tracing from pixels to reward probability.</i></p>
</div>

## 👀 Visualization: Neuronpedia Integration

We have forked and adapted the **Neuronpedia** webapp to handle robotic multi-modal inputs:

*   **Visual Patch Audit**: The dashboard now includes a persistent gallery that maps activations back to physical image patches.
*   **Saliency Grounding**: Features are highlighted with **green boxes** on the original robotic frames to identify their spatial focus.
*   **Interactive Attribution**: Users can click any node to see its most influential causal precursors (inputs) and downstream targets (outputs).

## 🚀 Research Roadmap: Next Steps

The dashboard now allows us to audit the effects of key architectural changes:
1. **Multi-View Data**: Training LeWM with 5 views to match the VLA input density and observing if it resolves the "Latent confusion" in the Predictor stack.
2. **Kinematic Polytopes**: Using reachability analysis to avoid out-of-distribution failure modes like arm folding.
3. **Latent Steering**: Closing the causal loop by using discovered features as reward boosters during real-time inference.

## 🚀 Workflows

### 0. Infrastructure Setup

The dashboard requires a local Dockerized Neuronpedia instance and a proxy bridge to the attribution engine.

```bash
# 1. Start the Neuronpedia Dashboard (Docker)
cd interpretability/neuronpedia
make webapp-localhost-dev

# 2. Start the Attribution Proxy (Local)
# This tunnels requests from the dashboard to the GPU engine
.venv/bin/python interpretability/dashboard/neuronpedia_server.py
```

### 1. Feature Training
Decompose the latent space across the entire transformer stack:
```bash
# Train transcoders for a specific layer pair (e.g., L10 to L11)
.venv/bin/python interpretability/transcoders/train_transcoder.py --source L10.pt --target L11.pt --output clt_L10_L11.pt
```

### 2. Mechanistic Audit
Generate the causal graphs for specific robotic scenarios:
```bash
# Regenerate all canonical graphs (Success, Failure, Approach)
.venv/bin/python interpretability/dashboard/regenerate_graphs.py
```