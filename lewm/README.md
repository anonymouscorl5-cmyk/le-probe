# LeWM: LeRobot World Model & Oracle MPC

This module implements the **LeWM** (LeRobot World Model) training and inference stack, using a JEPA-based architecture for latent imagination and Oracle MPC for planning.

## 📐 Methodology

The model was initially trained on the `gr1_pickup_grasp` dataset using a single-view baseline. Following the discovery of the **Discriminability Gap**, we upgraded to a **Multi-View JEPA** architecture which utilizes 5 camera streams to stabilize the latent manifold.

Reward model tuning is performed using `tune_reward_head.py` to calibrate the MPC cost logic against a broad spectrum of successful and failing trajectories.

Finally, the performance was evaluated using [**`LEWM_E2E.ipynb`**](LEWM_E2E.ipynb) and [**`simulation_lewm.py`**](simulation_lewm.py).

### 🛠 Key Components

- [`LeWM_Training.ipynb`](LeWM_Training.ipynb): Notebook used for training the LeWM and reward head.
- [`train_lewm.py`](train_lewm.py): Core training logic for the world model and reward head.
- [`gr1_modules.py`](gr1_modules.py): Additional modules for adapting the LeWM for Fourier GR-1.
- [`lewm_data_plugin.py`](lewm_data_plugin.py): Custom data plugin for LeWM.
- [`metrics.py`](metrics.py): Training metrics observed including softrank, signal ratio, participation ratio, etc.
- [`tune_reward_head.py`](tune_reward_head.py): A utility to tune the reward head of the LeWM with additional snapshots.
- [`harvest_goals.py`](harvest_goals.py): Utility to pre-compute goal embeddings for testing.
- [`diagnose_mpc.py`](diagnose_mpc.py): A utility to visualize the CEM planner's latent trajectory (the planning algorithm). This is not an online server but just a sanity check after training.
- [`LeWM_E2E.ipynb`](LeWM_E2E.ipynb): Notebook to run the server for the model (tunneled through Pinggy).
- [`lewm_server.py`](lewm_server.py): HTTP inference host (`POST /plan`, msgpack) used in the notebook.
- [`goal_mapper.py`](goal_mapper.py): Manages latent goal memory and manifold traversal.
- [`goal_utils.py`](goal_utils.py): Utilities for handling goal embeddings.
- [`simulation_lewm.py`](simulation_lewm.py): MuJoCo simulation environment for LeWM testing.
- [`skeleton`](skeleton/):
  - [`trainer`](skeleton/trainer.py): trainer to train the model with the trainer
  - [`tuner`](skeleton/tuner.py): to tune the reward head after training with broader randomization
  - [`data.py`](skeleton/data.py): skeletal data plugin to include the skeletal prior
  - [`encoder`](skeleton/encoder.py): the patched encoder for processing the 4th channel

## 📊 Results

Here are some of the single-view training metrics:

<div align="center">
  <table>
    <tr>
      <td><img src="../assets/lewm_softrank.png" width="300"><br><b>Softrank</b></td>
      <td><img src="../assets/lewm_signal_ratio.png" width="300"><br><b>Signal Ratio</b></td>
      <td><img src="../assets/lewm_participation_ratio.png" width="300"><br><b>Participation Ratio</b></td>
    </tr>
    <tr>
      <td><img src="../assets/lewm_pred_loss.png" width="300"><br><b>Prediction Loss</b></td>
      <td><img src="../assets/lewm_sigreg_loss.png" width="300"><br><b>SigReg Loss</b></td>
      <td><img src="../assets/lewm_reward_loss.png" width="300"><br><b>Reward Loss</b></td>
    </tr>
  </table>
</div>

As can be seen, the softrank ends up close to a 60-90 range which was initially expected based on the dataset without colapsing lower than 45 at the start of training and the sigreg loss also drops significantly indicating that we're able to capture the dynamics without representation collapse.

Plots for other training regimes haven't been added but they follow the same pattern apart from minor variations.

### 📈 Metric Glossary: Understanding Manifold Health

To monitor the stability of the latent world, we track several topological and structural metrics:

*   **SoftRank**: Measures latent "expressivity"; a higher rank indicates a rich, feature-dense manifold, while a low rank flags "representation collapse."
*   **Participation Ratio**: The "Latent Vocabulary Size"; it counts the number of effective independent dimensions used by the model to represent a scene.
*   **Signal Ratio**: Tracks the energy balance between predicted and target embeddings; values near 1.0 indicate a stable, non-exploding temporal flow.
*   **SigReg Loss**: Signal Regularization; a critical loss term that prevents latent variance from vanishing, forcing the model to keep the manifold "breathing."
*   **Path Straightening**: Measures how "predictable" the latent trajectories are; higher values indicate smoother transitions and more physically consistent dynamics.
*   **Skeletal Relative Importance**: Tracks the L1-norm ratio of the 4th-channel (kinematics) vs RGB weights; monitors how aggressively the model is anchoring in structural reality.


## 🏆 Current Performance

### Single-View RGB

The results with solely relying on the goal state embeddings weren't useful, but after training with an auxillary reward head instead, the robot atleast managed to get close to the table but not able to close in on the cube.

<div align="center">
  <b>LeWM: Grasp Execution</b>
  <hr width="240">
  <img src="../assets/lewm_grasp.gif" width="240" alt="LeWM: Grasp Execution">
</div>

### Multi-View RGB

Previously, we had only trained the LeWM model with single-view images (`world_center`). Completed another training run while including all the views with late fusion at the encoder side, and got this result where the robot does smash the cube off the table but the only challenge is for the hand to get on top of the table.

<div align="center">
  <b>LeWM: Grasp Execution (Multi-View)</b>
  <hr width="240">
  <img src="../assets/lewm_grasp_multiview.gif" width="240" alt="LeWM: Grasp Execution (Multi-View)">
</div>

### Multi-View + Skeletal Priors

- As can be seen with the Multi-View RGB example, once the robot hand is on top of the table it does show a clear intent approaching the cube, but it experiences a fair bit of resistance getting the hand on top of the cube in the first place.
- An intuitive explanation could be that the model still ends up trying to learn about the position of joints that aren't really that important for the motion.
- To improve the behaviour, skeletal priors were added to the training data that solely focused on the joints that are actually important for picking up the cube as the 4th channel after RGB.

<div align="center">
  <b>Skeletal Priors</b>
  <hr width="480">
  <img src="../assets/skeletal_priors.gif" width="480" alt="Skeletal Priors">
</div>

The model trained doesn't experience the same kind of resistance faced when we were relying on the skeletal and it also somewhat attempted the pickup movement albeit a bit too rapidly and smashed the cube off the table after 2 failed attempts.

<div align="center">
  <b>LeWM: Grasp Execution (Multi-View + Skeletal Priors)</b>
  <hr width="240">
  <img src="../assets/lewm_grasp_multiview_skeleton.gif" width="240" alt="LeWM: Grasp Execution (Multi-View + Skeletal Priors)">
</div>

It still doesn't actually pick up the cube, we need to find further ways of learning all 4 sub-phases of movement needed for the task separately.

### Multi-View + Skeletal Priors + DINOv3 Waypoints

The previous attempt did show some signs of learning the broader motion of grasping the cube as shown in the dataset but not as precisely as before. One argument is that the training window only contains about 3 frames which prevents that from happening. As a result, in this experiment we've included DINOv3 Waypoints for the target position of all 4 sub-phases in the training data,

<div align="center">
  <b>DINOv3 Representation of Episode</b>
  <hr width="720">
  <img src="assets/dino_skeletal_priors.gif" width="720" alt="DINOv3 Representation of Episode">
</div>

While the above GIF shows DINOv3 representations for all frames in an episode, we only rely on 4 frames in every 32-frame episode. That representation is passed in to an additional reward head on top of the predictor to ensure that every transition tries to close in to the upcoming sub-goal. This does demonstrate that now we are able to follow the grasp trajectory of the training data more closely than any previous experiments, especially the first phase of approaching the cube.

<div align="center">
  <b>LeWM: Grasp Execution (Multi-View + Skeletal Priors + DINOv3 Waypoints)</b>
  <hr width="240">
  <img src="assets/lewm_grasp_multiview_skeleton_dino.gif" width="240" alt="LeWM: Grasp Execution (Multi-View + Skeletal Priors + DINOv3 Waypoints)">
</div>

## 🎨 Motivation for Interpretability

Given how the behaviour differs between different kinds of training data, it makes sense to try and get a better idea of what the model has ended up learning, that motivated the need for interpretability, covered in [**`interpretability/README.md`**](../interpretability/README.md).

## 🚀 Workflows

### 1. Training
The model is trained using [**`LeWM_Training.ipynb`**](LeWM_Training.ipynb).
- **General Training**: Performed under the `GR-1 Pickup Grasp` section.
- **Reward Head Tuning**: Performed under the `GR-1 Reward Pred` section.

After training, use [**`harvest_goals.py`**](harvest_goals.py) to harvest latent goal embeddings into a gallery.

**Pre-trained Artifacts:**
| Version | Description | Checkpoint (G-Drive) | Goal Gallery |
| :--- | :--- | :--- | :--- |
| **Single-View** | Standard Baseline | [`gr1_reward_tuned_v2.ckpt`](https://drive.google.com/file/d/1dPp-yuSEKMywKPH1mzKT4m7f7Rq5ak7A/view?usp=sharing) | [`goal_gallery.pth`](https://drive.google.com/file/d/1KDxrZVbrlB2wDDPJAQfHIZxZi48ZhN8U/view?usp=sharing) |
| **Multi-View** | Multi-Camera Oracle | [`gr1_reward_tuned_v2.ckpt](https://drive.google.com/file/d/1pGMMicqYL_Z8GCS1TOe2A_kAAJQLV3qd/view?usp=drive_link) | [`goal_gallery.pth`](https://drive.google.com/file/d/1gYk_P9Godif20boD64M8epR5xSSSxugn/view?usp=drive_link) |
| **Multi-View + Skeletal Priors** | 4th Channel with Skeletal Priors | [`gr1_reward_tuned_v6.ckpt`](https://drive.google.com/file/d/1tiN-awjiMl0oUy8uLE9JT0850QQOPCUI/view?usp=sharing) | [`goal_gallery.pth`](https://drive.google.com/file/d/1R9uuqpd1yb7t7-NwuvEq7VrOuI6wI152/view?usp=sharing) |
| **Multi-View + Skeletal Priors + DINOv3 Waypoints** | 4th Channel with Skeletal Priors + DINOv3 Waypoints | [`gr1_reward_tuned_v1.ckpt`](https://drive.google.com/file/d/18xFB2lbxY5Q7EFs-18V9tkmED7NSQelR/view?usp=sharing) | [`goal_gallery.pth`](https://drive.google.com/file/d/1nFW8J_6PQhFaB1agzd8vaEZ1yIy8cCPA/view?usp=sharing) |

### 2. Inference
To test the World Model and MPC planner:

#### LeWM MPC Server
```bash
# For Single-View
.venv/bin/python lewm/lewm_server.py --model gr1_reward_tuned_v2.ckpt --gallery goal_gallery.pth

# For Multi-View
.venv/bin/python lewm/lewm_server.py --model gr1_reward_tuned_v2.ckpt --gallery goal_gallery.pth --multi_view

# For Multi-View + Skeletal Priors
.venv/bin/python lewm/lewm_server.py --model gr1_reward_tuned_v6.ckpt --gallery goal_gallery.pth --multi_view --use_skeleton

# For Multi-View + Skeletal Priors + DINOv3 Waypoints
.venv/bin/python lewm/lewm_server.py --model gr1_reward_tuned_v1.ckpt --gallery goal_gallery.pth --multi_view --use_skeleton --use_dino
```

#### Simulation Host
```bash
# Local (server on same machine)
.venv/bin/python lewm/simulation_lewm.py --base_url http://127.0.0.1:5555

# Remote via ngrok: ngrok http 5555, then pass the https URL
.venv/bin/python lewm/simulation_lewm.py --base_url https://<id>.ngrok-free.app --multi_view --use_skeleton
```
