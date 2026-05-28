import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


def get_avg_rewards(sidecar_path):
    if not os.path.exists(sidecar_path):
        print(f"Error: {sidecar_path} not found")
        return None

    df = pd.read_parquet(sidecar_path)

    # Each episode has exactly 32 frames according to the user
    # We need to calculate the step index (0-31) for each frame
    # We can group by episode_index and assign a sequential rank
    df["step"] = df.groupby("episode_index").cumcount()

    # Filter for step < 32 just in case
    df = df[df["step"] < 32]

    # Average reward per step
    avg_rewards = df.groupby("step")["progress_sparse"].mean()
    return avg_rewards


def plot_comparison():
    cup_path = "/workspace/cortex-os/le-probe/datasets/anonymouscorl5-cmyk/gr1_pickup_cup/progress_sparse.parquet"
    grasp_path = "/workspace/cortex-os/le-probe/datasets/anonymouscorl5-cmyk/gr1_pickup_grasp/progress_sparse.parquet"

    avg_cup = get_avg_rewards(cup_path)
    avg_grasp = get_avg_rewards(grasp_path)

    if avg_cup is None or avg_grasp is None:
        return

    plt.figure(figsize=(12, 6))
    plt.plot(
        avg_cup.index,
        avg_cup.values,
        label="Cup Movement (Robust)",
        color="#00D1FF",
        linewidth=2.5,
        marker="o",
        markersize=4,
    )
    plt.plot(
        avg_grasp.index,
        avg_grasp.values,
        label="Grasp Movement (Unstable)",
        color="#FF3D00",
        linewidth=2.5,
        marker="s",
        markersize=4,
    )

    # Formatting
    plt.title(
        "Average Reward per Step: Cup vs. Grasp",
        fontsize=16,
        fontweight="bold",
        color="white",
        pad=20,
    )
    plt.xlabel("Step Index (32 Frames per Episode)", fontsize=12, color="#CCCCCC")
    plt.ylabel("Average Reward (Progress)", fontsize=12, color="#CCCCCC")

    # Styling for Dark Mode look (Rich Aesthetics)
    plt.gca().set_facecolor("#1E1E1E")
    plt.gcf().set_facecolor("#121212")
    plt.grid(True, linestyle="--", alpha=0.3, color="#444444")
    plt.tick_params(colors="white")

    plt.legend(facecolor="#2D2D2D", edgecolor="none", labelcolor="white")

    # Highlight the 4 phases
    plt.axvspan(0, 7.5, color="#444444", alpha=0.2)
    plt.text(
        3.5, plt.ylim()[1] * 0.05, "Approach", color="#888888", ha="center", fontsize=10
    )

    plt.axvspan(7.5, 15.5, color="#555555", alpha=0.2)
    plt.text(
        11.5, plt.ylim()[1] * 0.05, "Place", color="#888888", ha="center", fontsize=10
    )

    plt.axvspan(15.5, 23.5, color="#444444", alpha=0.2)
    plt.text(
        19.5, plt.ylim()[1] * 0.05, "Capture", color="#888888", ha="center", fontsize=10
    )

    plt.axvspan(23.5, 31.5, color="#555555", alpha=0.2)
    plt.text(
        27.5, plt.ylim()[1] * 0.05, "Lift", color="#888888", ha="center", fontsize=10
    )

    output_path = "/workspace/cortex-os/reward_comparison_cup_vs_grasp.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Comparison plot saved to {output_path}")


if __name__ == "__main__":
    plot_comparison()
