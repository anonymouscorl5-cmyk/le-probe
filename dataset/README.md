# Dataset Management & Teleoperation

This module handles the lifecycle of robotic data: from real-time MuJoCo teleoperation to LeRobot-formatted cloud datasets.

## 📊 Dataset Standards

After experimenting with 12, 32 and 52 frame episodes, I have standardized on **32-frame episodes** (recorded at 10Hz) to capture the full reach-to-grasp trajectory for our task. I maintain two primary behavioral variants:

<div align="center">
  <table>
    <tr>
      <th>Dataset: Grasp Pattern</th>
      <th>Dataset: Cup Pattern</th>
    </tr>
    <tr>
      <td><img src="../assets/dataset_grasp.gif" width="240"></td>
      <td><img src="../assets/dataset_cup.gif" width="240"></td>
    </tr>
  </table>
</div>

## 📐 Methodology

### 🖥 Teleoperation Interface

I use a custom-built Streamlit dashboard for real-time control, IK requests, and dataset auditing using the `teleop_ui.py` which is eventually visualized via Rerun.

<div align="center">
  <img src="../assets/teleop_dashboard.png" width="720" style="border-radius: 8px;">
</div>

### 🛠 Key Components

- [`teleop_ui.py`](teleop_ui.py):
  - Streamlit dashboard for 32-DoF joint control and IK-assisted manipulation.
  - Contains options for controlling the robot manually through the sliders, using the IK solver for consistent movement, the recording of episodes into the dataset, monitoring of distance between the right hand fingers and the cube, etc.

- [`simulation_teleop.py`](simulation_teleop.py):
  - ZMQ server driving the MuJoCo simulation and handling 32-DoF IK requests and supports all the features listed in the `teleop_ui.py` dashboard.
  - Currently the IK solver operates in a grasp move, but essentially a few tweaks to the target coordinates can allow operations in a cup movement as well.

- [`lerobot_manager.py`](lerobot_manager.py):
  - Core recording logic in the LeRobot format.
  - Implements the 32-dim identity protocol and "Smart Reward" injection.
  - Rewards are currently assigned as an inverse of the distance between the right hand fingers and the cube (capped at `10`) using the `lerobot_manager.py`.

- [`simulation_replay.py`](simulation_replay.py):
  - Visual audit tool for replaying recorded episodes for verification.

- [`upload_dataset.py`](upload_dataset.py):
  - A script to upload the dataset to the Hugging Face Hub.

- [`skeleton`](skeleton/):
  - contains scripts for generating the priors for both the main dataset and the reward tuning dataset.
  - other scripts to audit and verify the generation of the priors.

## 📊 Current Datasets

The following datasets have been curated and uploaded to the Hugging Face Hub:

- [**`gr1_pickup_grasp`**](https://huggingface.co/datasets/vedpatwardhan/gr1_pickup_grasp): Precision "pinch" grasp trajectories.
- [**`gr1_pickup_cup`**](https://huggingface.co/datasets/vedpatwardhan/gr1_pickup_cup): Robust "surrounding" containment trajectories.
- [**`gr1_reward_pred`**](https://huggingface.co/datasets/vedpatwardhan/gr1_reward_pred): Multi-behavioral data used to train the Reward Head. Wasn't curated using the IK Solver, but instead using Wild Randomization with the Snapshot button on the teleoperator for having a significant proportion of failing states.

## 🦴 Skeleton Priors

- In order to improve the performance of the LeWM model, a 4th channel was added to the data alongside RGB which would basically just be a bunch of lines that represent the motion from the waist joints to the right shoulder and then to the right fingers.
- Left arm, head, neck, etc. were all ignored while generating the skeleton given the dataset doesn't even use those joints for picking up the cube, and our goal is to ensure that the world model understands what's important in the cube pick up.
- Here's an example for what the skeleton looks like for one of the episodes in `gr1_pickup_grasp`:

<div align="center">
  <img src="../assets/skeletal_priors.gif" width="480" style="border-radius: 8px;">
</div>

## 🦖 DINOv3 Waypoints

- In order to improve the model's understanding of the global trajectory during the episode beyond the 3-frame default window, we use DINOv3 to generate waypoints.
- The goal is to have a separate waypoint for the target position of each of the 4 sub-phases during an episode.
- Here's an example for the DINOv3 representation of the full episode, although we only rely on 4 waypoints out of a 32-frame episode.

<div align="center">
  <b>DINOv3 Representation of Episode</b>
  <hr width="720">
  <img src="../assets/dino_skeletal_priors.gif" width="720" alt="DINOv3 Representation of Episode">
</div>

## 🚀 Workflows

### 1. Data Collection

```bash
# Start the Rerun server
rerun

# Start Sim Server
.venv/bin/python dataset/simulation_teleop.py

# Start Dashboard
streamlit run dataset/teleop_ui.py
```

### 2. Dataset Upload
```bash
.venv/bin/python dataset/upload_dataset.py --repo_id <>
```

### 3. Skeletal Priors
```bash
# Pulls the video dataset used to train the LeWM and makes in-place changes
.venv/bin/python dataset/skeleton/generate_priors.py vedpatwardhan/gr1_pickup_grasp

# Verify that the tiling worked
.venv/bin/python le-probe/dataset/skeleton/verify_tiling.py /root/.cache/huggingface/lerobot/vedpatwardhan/gr1_pickup_grasp/videos/observation.images.world_center_tiled/chunk-000/file-000.mp4

# Pulls the reward prediction dataset used to fine-tune the reward head
.venv/bin/python dataset/skeleton/generate_reward_priors.py vedpatwardhan/gr1_reward_pred

# Audit the reward priors
.venv/bin/python dataset/skeleton/audit_priors.py --repo_id vedpatwardhan/gr1_reward_pred_v2 --frames dataset_skel_frames
```

### 4. DINOv3 Waypoints
```bash
# Generate Waypoints for Main Dataset
.venv/bin/python dataset/skeleton/generate_dino_priors.py
```
