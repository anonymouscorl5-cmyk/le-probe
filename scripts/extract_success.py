# --- Path Stabilization ---
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# --------------------------

import pandas as pd
import json
import numpy as np
import os
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def extract_high_reward_frame(parquet_path, output_json, target_step=20):
    df = pd.read_parquet(parquet_path)

    # Target frame at target_step of Episode 0
    # Assuming frames are sequential in the first parquet
    frame_idx = target_step

    # Verify reward from sidecar
    dataset_dir = os.path.dirname(os.path.dirname(os.path.dirname(parquet_path)))
    sidecar_path = os.path.join(dataset_dir, "progress_sparse.parquet")

    if os.path.exists(sidecar_path):
        sdf = pd.read_parquet(sidecar_path)
        reward = float(sdf.iloc[frame_idx]["progress_sparse"])
    else:
        reward = 10.0  # Fallback

    print(f"Targeting Step {target_step} (Reward: {reward})")

    # Extract the image using LeRobotDataset API

    dataset_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(parquet_path)))
    )
    repo_id = os.path.basename(dataset_path)

    print(f"Loading dataset from {dataset_path}...")
    ds = LeRobotDataset(repo_id=repo_id, root=dataset_path)

    frame = ds[frame_idx]
    # In LeRobot, the frame dictionary contains PIL images for keys starting with 'observation.images'
    img = frame["observation.images.world_center"]
    img_np = np.array(img)

    # Ensure (C, H, W)
    if img_np.shape[-1] == 3:  # (H, W, C)
        img_np = img_np.transpose(2, 0, 1)

    print(f"Final Image Shape: {img_np.shape}")

    snapshot = {"observation.images.world_center": img_np.tolist(), "progress": reward}

    with open(output_json, "w") as f:
        json.dump(snapshot, f)

    print(f"✅ Success snapshot saved to {output_json}")


if __name__ == "__main__":
    extract_high_reward_frame(
        "le-probe/datasets/anonymouscorl5-cmyk/gr1_pickup_grasp/data/chunk-000/file-000.parquet",
        "success_test.json",
    )
