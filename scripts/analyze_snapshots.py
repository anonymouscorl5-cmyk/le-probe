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
import json
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Project-specific imports
from lewm.goal_mapper import GoalMapper


class SnapshotAnalyzer:
    def __init__(self, checkpoint_path, snapshots_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🏗️  Loading GoalMapper from {Path(checkpoint_path).name}")
        self.mapper = GoalMapper(model_path=checkpoint_path, dataset_root=".")
        self.snapshots_dir = Path(snapshots_dir)
        self.results = []

    def run_analysis(self):
        json_files = sorted([f for f in self.snapshots_dir.glob("*.json")])
        if not json_files:
            print(f"❌ No snapshots found in {self.snapshots_dir}")
            return

        print(f"🧐 Analyzing {len(json_files)} snapshots...")

        for json_path in tqdm(json_files):
            with open(json_path, "r") as f:
                data = json.load(f)

            # 1. Extract Ground Truth
            gt_reward = data["progress"]

            # 2. Extract and Prepare Image
            # Format in JSON: (C, H, W) list
            img_list = data["observation.images.world_center"]
            img_tensor = torch.tensor(img_list).float().to(self.device)

            # GoalMapper expects (Batch, T, C, H, W)
            # Since the JSON image is already 224x224 and in (C, H, W)
            # We just need to add the Batch and T dimensions
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)

            # 3. Model Prediction
            with torch.no_grad():
                # Encode -> Latent
                info = self.mapper.model.encode({"pixels": img_tensor})
                latent = info["emb"]  # (1, 1, 192)

                # Predict Reward
                pred_reward = self.mapper.model.reward_head(latent).item()

            self.results.append(
                {
                    "id": json_path.stem,
                    "gt_reward": float(gt_reward),
                    "pred_reward": float(pred_reward),
                    "error": abs(gt_reward - pred_reward),
                }
            )

    def report(self):
        if not self.results:
            return

        print("\n--- 📊 SNAPSHOT AUDIT REPORT ---")
        print(f"{'ID':<6} | {'GT Reward':<10} | {'Pred Reward':<11} | {'Error':<6}")
        print("-" * 45)
        for r in self.results:
            print(
                f"{r['id']:<6} | {r['gt_reward']:<10.4f} | {r['pred_reward']:<11.4f} | {r['error']:<6.4f}"
            )

        # Visualization
        gt = [r["gt_reward"] for r in self.results]
        pred = [r["pred_reward"] for r in self.results]

        plt.figure(figsize=(8, 6))
        plt.scatter(gt, pred, alpha=0.7, color="blue", edgecolors="k")
        plt.plot([0, 10], [0, 10], "r--", label="Perfect Calibration")
        plt.xlabel("Ground Truth Reward")
        plt.ylabel("Model Predicted Reward")
        plt.title("Reward Head: Reality vs. Neural Intent")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()

        plot_path = "snapshot_audit_results.png"
        plt.savefig(plot_path)
        print(f"\n📈 Audit plot saved to {plot_path}")


if __name__ == "__main__":
    CKPT = "/workspace/cortex-os/le-probe/research/outputs/gr1_grasp_v3/checkpoints/gr1-epoch=99-step=004400.ckpt"
    SNAPSHOTS = (
        "/workspace/cortex-os/le-probe/datasets/anonymouscorl5-cmyk/gr1_reward_pred"
    )

    analyzer = SnapshotAnalyzer(CKPT, SNAPSHOTS)
    analyzer.run_analysis()
    analyzer.report()
