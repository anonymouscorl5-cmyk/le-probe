# --- Path Stabilization ---
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# --------------------------

import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def sample_snapshots(snapshots_dir, num_samples=10):
    snap_path = Path(snapshots_dir)
    json_files = [f for f in snap_path.glob("*.json")]

    if not json_files:
        print(f"No snapshots found in {snapshots_dir}")
        return

    # Sample random files
    samples = random.sample(json_files, min(num_samples, len(json_files)))
    samples.sort()  # Keep them in numerical order for easier reading

    cols = 5
    rows = (len(samples) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = np.array(axes).flatten()

    for i, json_file in enumerate(samples):
        with open(json_file, "r") as f:
            data = json.load(f)

        # Extract Image (C, H, W) -> (H, W, C)
        img_data = np.array(data["observation.images.world_center"], dtype=np.uint8)
        img_data = img_data.transpose(1, 2, 0)

        # Extract Reward
        reward = data.get("progress", 0.0)

        ax = axes[i]
        ax.imshow(img_data)
        ax.set_title(f"ID: {json_file.stem}\nReward: {reward:.2f}", fontsize=12, pad=10)
        ax.axis("off")

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    output_path = "snapshot_sample.png"
    plt.savefig(output_path)
    print(f"✅ Sample grid saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=str,
        default="le-probe/datasets/anonymouscorl5-cmyk/gr1_reward_pred",
    )
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--output", type=str, default="snapshot_sample.png")
    args = parser.parse_args()

    sample_snapshots(args.dir, args.num_samples)
