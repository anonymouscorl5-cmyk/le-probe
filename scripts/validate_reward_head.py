# --- Path Stabilization ---
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# --------------------------

import os
import torch
import numpy as np
import mujoco
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

# Import local project modules
from simulation_base import GR1MuJoCoBase
from lewm.goal_mapper import GoalMapper
from gr1_config import COMPACT_WIRE_JOINTS, JOINT_LIMITS_MIN, JOINT_LIMITS_MAX
from gr1_protocol import StandardScaler


class RewardValidator:
    def __init__(self, checkpoint_path, dataset_root):
        print(
            f"🏗️  Initializing Reward Validation (Checkpoint: {os.path.basename(checkpoint_path)})"
        )

        # 1. Initialize Model FIRST (to get device and transforms)
        # GoalMapper handles loading weights into self.model.model
        self.mapper = GoalMapper(model_path=checkpoint_path, dataset_root=dataset_root)
        self.mapper.model.eval()
        self.device = self.mapper.device

        # 2. Initialize Simulation
        self.sim = GR1MuJoCoBase(restrict_ik=False)  # Full protocol freedom
        self.sim.reset_env(lock_posture=True)

        self.results = []

    def get_ground_truth_reward(self):
        """Calculates the canonical training reward: (1.0 - target_dist) * 10.0"""
        physics = self.sim.get_physics_state()
        target_dist = physics["target_dist"]
        # Proximity reward (scaled to 10.0)
        proximity = max(0, 1.0 - target_dist)
        return proximity * 10.0, target_dist

    def run_audit(self, num_samples=200):
        print(f"🚀 Starting Audit for {num_samples} random poses...")

        # Capture Initial Pose for joints 0-15
        initial_pose = self.sim.get_state_32()

        for i in tqdm(range(num_samples)):
            # 1. Generate a Random Pose
            # We only randomize the Right Arm (16-28) and Waist (29-31)
            # Left Side (0-15) remains at the initial pose
            random_action = initial_pose.copy()

            # Randomize active joints within physical limits
            for j in range(16, 32):
                random_action[j] = np.random.uniform(-1.0, 1.0)

            # 2. Teleport Robot to that pose
            self.sim.process_target_32(random_action)
            # Use mj_forward to update positions without physics stepping
            self.sim.data.qpos[:] = self.sim.last_target_q.copy()
            mujoco.mj_forward(self.sim.model, self.sim.data)

            # 3. Get Ground Truth
            gt_reward, dist = self.get_ground_truth_reward()

            # 4. Get Model Prediction
            # Render world_center
            self.sim.renderer.update_scene(self.sim.data, camera="world_center")
            rgb = self.sim.renderer.render()

            # Prepare image for model using GoalMapper's canonical transform
            # Mapper transform expects a dict with "pixels" as a PIL image or numpy array
            batch = self.mapper.transform({"pixels": rgb})
            processed_pixels = batch["pixels"].to(self.device)
            # Add batch and sequence dims: (1, 1, 3, 224, 224)
            img_tensor = processed_pixels.unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                # Encode Image -> Latent
                # Accessing model.model.encoder directly (JEPA structure)
                latent = self.mapper.model.encoder(img_tensor)
                # Predict Reward using the reward_head
                # latent is [1, 1, 192], head expects [B, S, D]
                pred_reward = self.mapper.model.reward_head(latent).item()

            self.results.append(
                {
                    "sample": i,
                    "gt_reward": float(gt_reward),
                    "pred_reward": float(pred_reward),
                    "dist_cm": float(dist * 100.0),
                }
            )

    def visualize(self):
        if not self.results:
            print("No results to visualize.")
            return

        gt = [r["gt_reward"] for r in self.results]
        pred = [r["pred_reward"] for r in self.results]
        dist = [r["dist_cm"] for r in self.results]

        plt.figure(figsize=(12, 5))

        # Plot 1: Correlation
        plt.subplot(1, 2, 1)
        plt.scatter(gt, pred, alpha=0.5, c=dist, cmap="viridis")
        plt.plot([0, 10], [0, 10], "r--", label="Perfect Calibration")
        plt.xlabel("Ground Truth Reward (Proximity)")
        plt.ylabel("Predicted Reward")
        plt.title("Reward Head Calibration Audit")
        plt.colorbar(label="Distance to Cube (cm)")
        plt.grid(True)
        plt.legend()

        # Plot 2: Error over Distance
        plt.subplot(1, 2, 2)
        errors = [abs(g - p) for g, p in zip(gt, pred)]
        plt.scatter(dist, errors, alpha=0.5, color="orange")
        plt.xlabel("Distance to Cube (cm)")
        plt.ylabel("Prediction Error (Absolute)")
        plt.title("Error vs. Physical Distance")
        plt.grid(True)

        plt.tight_layout()
        plot_path = "reward_audit_results.png"
        plt.savefig(plot_path)
        print(f"📊 Visualization saved to {plot_path}")

        # Save JSON data
        with open("reward_audit_data.json", "w") as f:
            json.dump(self.results, f, indent=4)


if __name__ == "__main__":
    # Paths
    CKPT = "/workspace/cortex-os/le-probe/research/outputs/gr1_grasp_v3/checkpoints/gr1-epoch=99-step=004400.ckpt"
    DATASET = "/workspace/cortex-os/le-probe/datasets"

    validator = RewardValidator(CKPT, DATASET)
    validator.run_audit(num_samples=200)  # Quick audit
    validator.visualize()
