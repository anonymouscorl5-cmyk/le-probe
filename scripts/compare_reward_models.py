# --- Path Stabilization ---
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# --------------------------

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
import pandas as pd

# Project Imports
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
lewm_root = os.path.join(project_root, "le_wm")

if project_root not in sys.path:
    sys.path.append(project_root)
if script_dir not in sys.path:
    sys.path.append(script_dir)
if lewm_root not in sys.path:
    sys.path.append(lewm_root)

from goal_mapper import GoalMapper


def compare_models(naive_ckpt, v1_ckpt, v2_ckpt, df):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"🚀 Comparing on device: {device}")

    # 1. Load All Models
    def load_model(ckpt, name):
        print(f"🏗️  Loading {name}: {ckpt}")
        mapper = GoalMapper(model_path=ckpt, dataset_root=".")
        return mapper.model.to(device).eval(), mapper.transform

    model_naive, transform = load_model(naive_ckpt, "Naive Model")
    model_v1, _ = load_model(v1_ckpt, "Tuned V1")
    model_v2, _ = load_model(v2_ckpt, "Tuned V2")

    # 2. Dataset size
    print(f"📦 Auditing {len(df)} snapshots from Parquet.")

    results = []

    # 3. Process Dataset Rows
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Auditing Snapshots"):
        gt_reward = float(row.get("progress", 0.0))
        raw_img = row["observation.images.world_center"]

        # Manual reconstruction by channel to break object-array nesting
        img_np = np.stack(
            [
                np.array(raw_img[0].tolist(), dtype=np.uint8),
                np.array(raw_img[1].tolist(), dtype=np.uint8),
                np.array(raw_img[2].tolist(), dtype=np.uint8),
            ],
            axis=-1,
        )

        # Transform (same for all)
        batch = transform({"pixels": img_np})
        pixel_tensor = (
            batch["pixels"].to(device).unsqueeze(0).unsqueeze(0)
        )  # (1, 1, C, H, W)

        with torch.no_grad():
            # Naive Pred
            info_n = model_naive.encode({"pixels": pixel_tensor})
            pred_n = model_naive.reward_head(info_n["emb"]).squeeze(-1).item()

            # V1 Pred
            info_v1 = model_v1.encode({"pixels": pixel_tensor})
            pred_v1 = model_v1.reward_head(info_v1["emb"]).squeeze(-1).item()

            # V2 Pred
            info_v2 = model_v2.encode({"pixels": pixel_tensor})
            pred_v2 = model_v2.reward_head(info_v2["emb"]).squeeze(-1).item()

        results.append({"gt": gt_reward, "naive": pred_n, "v1": pred_v1, "v2": pred_v2})

    # 4. Calculate Stats
    gt_arr = np.array([r["gt"] for r in results])
    naive_arr = np.array([r["naive"] for r in results])
    v1_arr = np.array([r["v1"] for r in results])
    v2_arr = np.array([r["v2"] for r in results])

    def get_metrics(pred):
        mse = np.mean((gt_arr - pred) ** 2)
        mae = np.mean(np.abs(gt_arr - pred))
        return mse, mae

    n_mse, n_mae = get_metrics(naive_arr)
    v1_mse, v1_mae = get_metrics(v1_arr)
    v2_mse, v2_mae = get_metrics(v2_arr)

    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE THREE-WAY AUDIT")
    print("=" * 60)
    print(f"{'Model':<25} | {'MSE':<10} | {'MAE':<10}")
    print("-" * 60)
    print(f"{'NAIVE (Original)':<25} | {n_mse:<10.4f} | {n_mae:<10.4f}")
    print(f"{'TUNED V1 (Buggy)':<25} | {v1_mse:<10.4f} | {v1_mae:<10.4f}")
    print(f"{'TUNED V2 (Calibrated)':<25} | {v2_mse:<10.4f} | {v2_mae:<10.4f}")
    print("=" * 60)
    print(f"🔥 V2 Improvement over Naive: {((n_mse - v2_mse)/n_mse)*100:.2f}%")
    print(f"📈 V2 Improvement over V1: {((v1_mse - v2_mse)/v1_mse)*100:.2f}%")
    print("=" * 60)

    # 5. Generate Comparison Plots
    plt.figure(figsize=(20, 12))

    # Scatter Plots
    def plot_scatter(idx, pred, title, color):
        plt.subplot(2, 3, idx)
        plt.scatter(gt_arr, pred, alpha=0.3, color=color)
        plt.plot([0, 10], [0, 10], "k--", alpha=0.5)
        plt.title(title)
        plt.xlabel("Ground Truth Reward")
        plt.ylabel("Predicted Reward")
        plt.grid(True)

    plot_scatter(1, naive_arr, "Naive Model", "red")
    plot_scatter(2, v1_arr, "Tuned V1 (Buggy)", "orange")
    plot_scatter(3, v2_arr, "Tuned V2 (Calibrated)", "green")

    # Distribution Plots
    plt.subplot(2, 1, 2)
    bins = np.linspace(-1, 11, 60)
    plt.hist(gt_arr, bins=bins, alpha=0.3, label="Ground Truth", color="blue")
    plt.hist(naive_arr, bins=bins, alpha=0.2, label="Naive Pred", color="red")
    plt.hist(v1_arr, bins=bins, alpha=0.2, label="V1 Pred", color="orange")
    plt.hist(v2_arr, bins=bins, alpha=0.4, label="V2 Pred (Final)", color="green")
    plt.title("Reward Distributions (GT vs Naive vs V1 vs V2)")
    plt.xlabel("Reward Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("reward_comparison_audit_v2.png")
    print("\n📈 Comprehensive V2 audit plot saved to reward_comparison_audit_v2.png")


if __name__ == "__main__":
    parquet_path = Path(
        "le-probe/datasets/anonymouscorl5-cmyk/gr1_reward_pred/dataset.parquet"
    )

    if not parquet_path.exists():
        print(f"❌ Error: Parquet dataset not found at {parquet_path}")
        sys.exit(1)

    print(f"📊 Loading consolidated dataset: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    compare_models(
        naive_ckpt="gr1-epoch=99-step=004400.ckpt",
        v1_ckpt="gr1_reward_tuned.ckpt",
        v2_ckpt="gr1_reward_tuned_v2.ckpt",
        df=df,
    )
