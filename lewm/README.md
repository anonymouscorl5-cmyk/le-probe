# LeWM: World Model Training and Latent MPC

This module contains the LeRobot World Model training and planning stack used for:

`Single-View RGB -> Multi-View RGB -> Multi-View RGB + Skeletal Priors -> Multi-View RGB + Skeletal Priors + DINOv3 Waypoints`

## Variant Flags

| Variant | Runtime Flags |
| :--- | :--- |
| Single-View RGB | *(none)* |
| Multi-View RGB | `--multi_view` |
| Multi-View RGB + Skeletal Priors | `--multi_view --use_skeleton` |
| Multi-View RGB + Skeletal Priors + DINOv3 Waypoints | `--multi_view --use_skeleton --use_dino` |

## What This Module Covers

- JEPA-based training for world-model dynamics.
- Reward-head tuning for planning signal calibration.
- Goal-latent gallery harvesting (`z_g` bank for MPC).
- Latent CEM/MPC serving and simulation rollouts.

## Core Files

- [`LeWM_Training.ipynb`](./LeWM_Training.ipynb): experiment notebook for model + reward training.
- [`train_lewm.py`](./train_lewm.py): training entrypoint.
- [`tune_reward_head.py`](./tune_reward_head.py): reward refinement with snapshot data.
- [`harvest_goals.py`](./harvest_goals.py): saves latent goal gallery.
- [`lewm_server.py`](./lewm_server.py): planner server (`POST /plan`).
- [`simulation_lewm.py`](./simulation_lewm.py): MuJoCo rollout host.
- [`diagnose_mpc.py`](./diagnose_mpc.py): CEM trajectory diagnostics.
- [`skeleton/`](./skeleton): 4th-channel skeletal training/tuning path.

## Variant Deltas

| Variant | Runtime Flags | Extra Data Signal |
| :--- | :--- | :--- |
| Single-View RGB | *(none)* | `world_center` |
| Multi-View RGB | `--multi_view` | 5 camera streams |
| Multi-View RGB + Skeletal Priors | `--multi_view --use_skeleton` | tiled kinematic channel |
| Multi-View RGB + Skeletal Priors + DINOv3 Waypoints | `--multi_view --use_skeleton --use_dino` | phase waypoint targets |

## Setup

```bash
cd le-probe
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Workflow

```bash
# 1) Train
.venv/bin/python lewm/train_lewm.py data.dataset.repo_id="gr1_pickup_grasp"

# 2) Harvest goal gallery
.venv/bin/python lewm/harvest_goals.py

# 3) Start planner server (example: Multi-View RGB + Skeletal Priors + DINOv3 Waypoints)
.venv/bin/python lewm/lewm_server.py \
  --model <ckpt> \
  --gallery goal_gallery.pth \
  --multi_view --use_skeleton --use_dino

# 4) Run simulation client
.venv/bin/python lewm/simulation_lewm.py \
  --base_url https://<id>.ngrok-free.app \
  --multi_view --use_skeleton --use_dino
```

## Rollout Artifacts

<div align="center">
  <img src="../assets/lewm_grasp.gif" width="220" alt="Single-View RGB rollout">
  <img src="../assets/lewm_grasp_multiview.gif" width="220" alt="Multi-View RGB rollout">
  <img src="../assets/lewm_grasp_multiview_skeleton.gif" width="220" alt="Multi-View RGB plus Skeletal Priors rollout">
  <img src="../assets/lewm_grasp_multiview_skeleton_dino.gif" width="220" alt="Multi-View RGB plus Skeletal Priors plus DINOv3 Waypoints rollout">
</div>

## Notes

- Default dataset IDs are anonymized in configs/scripts.
- This module focuses on representation and planning behavior, not end-task success guarantees.

## Pretrained Artifacts (Supplementary Storage)

| Variant | Model Checkpoint | Goal Gallery |
| :--- | :--- | :--- |
| Single-View RGB | [gr1_reward_tuned_v2.ckpt](https://drive.google.com/file/d/1dPp-yuSEKMywKPH1mzKT4m7f7Rq5ak7A/view?usp=sharing) | [goal_gallery.pth](https://drive.google.com/file/d/1KDxrZVbrlB2wDDPJAQfHIZxZi48ZhN8U/view?usp=sharing) |
| Multi-View RGB | [gr1_reward_tuned_v2.ckpt](https://drive.google.com/file/d/1pGMMicqYL_Z8GCS1TOe2A_kAAJQLV3qd/view?usp=sharing) | [goal_gallery.pth](https://drive.google.com/file/d/1gYk_P9Godif20boD64M8epR5xSSSxugn/view?usp=sharing) |
| Multi-View RGB + Skeletal Priors | [gr1_reward_tuned_v2.ckpt](https://drive.google.com/file/d/1tiN-awjiMl0oUy8uLE9JT0850QQOPCUI/view?usp=sharing) | [goal_gallery.pth](https://drive.google.com/file/d/1R9uuqpd1yb7t7-NwuvEq7VrOuI6wI152/view?usp=sharing) |
| Multi-View RGB + Skeletal Priors + DINOv3 Waypoints | [gr1_reward_tuned_v1.ckpt](https://drive.google.com/file/d/18xFB2lbxY5Q7EFs-18V9tkmED7NSQelR/view?usp=sharing) | [goal_gallery.pth](https://drive.google.com/file/d/1nFW8J_6PQhFaB1agzd8vaEZ1yIy8cCPA/view?usp=sharing) |
