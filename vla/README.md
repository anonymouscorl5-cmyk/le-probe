# VLA Baseline: GR00T-N1

This module contains the Vision-Language-Action baseline used as behavioral context for Le-Probe.

## Why It Exists

- Establishes that behavior-style imitation (grasp vs cup) is learnable with a policy baseline.
- Serves as a reference point against world-model latent planning behavior.

## Files

- [`GR00T_N1_BC.ipynb`](./GR00T_N1_BC.ipynb): behavioral cloning training.
- [`GR00T_N1_E2E.ipynb`](./GR00T_N1_E2E.ipynb): end-to-end evaluation.
- [`gr00t_server.py`](./gr00t_server.py): inference server.
- [`simulation_vla.py`](./simulation_vla.py): MuJoCo client.

## Setup

```bash
cd le-probe
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training Signal

<div align="center">
  <img src="../assets/gr00t_grasp_loss.png" width="240" alt="grasp loss">
  <img src="../assets/gr00t_cup_loss.png" width="240" alt="cup loss">
</div>

## Rollout Examples

<div align="center">
  <img src="../assets/vla_grasp.gif" width="240" alt="grasp execution">
  <img src="../assets/vla_cup.gif" width="240" alt="cup execution">
</div>

## Minimal Inference Workflow

```bash
# Start inference host
.venv/bin/python vla/gr00t_server.py --weights <path_to_pretrained_model> -p 5555

# Run simulation
.venv/bin/python vla/simulation_vla.py --base_url https://<id>.ngrok-free.app --chunks <num_chunks>
```
