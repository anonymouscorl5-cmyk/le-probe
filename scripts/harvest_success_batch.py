# --- Path Stabilization ---
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# --------------------------

import os
import json
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def harvest_spectrum_batch(dataset_repo, steps=[8, 16, 24, 31], num_episodes=200):
    dataset_path = os.path.abspath(dataset_repo)
    repo_id = os.path.basename(dataset_path)

    print(f"🚜 Harvesting reward spectrum ({len(steps)} frames/ep) from {repo_id}...")
    ds = LeRobotDataset(repo_id=repo_id, root=dataset_path)

    # Target Folder
    output_dir = Path("le-probe/datasets/anonymouscorl5-cmyk/gr1_success_gold")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load sidecar for rewards
    sidecar_path = os.path.join(dataset_path, "progress_sparse.parquet")
    sdf = pd.read_parquet(sidecar_path)

    harvested = 0
    for ep_idx in tqdm(range(min(num_episodes, ds.num_episodes))):
        for step in steps:
            global_idx = (ep_idx * 32) + step

            if global_idx >= ds.num_frames:
                break

            frame = ds[global_idx]
            reward = float(sdf.iloc[global_idx]["progress_sparse"])

            img = frame["observation.images.world_center"]

            # img is a Torch Tensor (C, H, W). Use torchvision resize.
            img_resized = F.resize(img, (224, 224), antialias=True)

            img_np = (
                (img_resized * 255).byte().numpy()
                if img_resized.dtype == torch.float32
                else img_resized.byte().numpy()
            )

            # Ensure (C, H, W)
            if img_np.shape[-1] == 3:
                img_np = img_np.transpose(2, 0, 1)

            snapshot = {
                "observation.images.world_center": img_np.tolist(),
                "progress": reward,
                "episode_index": int(ep_idx),
                "step": int(step),
            }

            snap_path = output_dir / f"spec_{ep_idx:03d}_s{step:02d}.json"
            with open(snap_path, "w") as f:
                json.dump(snapshot, f)

            harvested += 1

    print(f"✅ Spectrum harvest complete! {harvested} snapshots stored in {output_dir}")


if __name__ == "__main__":
    # Steps: 8, 16 (Mediocre), 20 (Peak), 24, 31 (Post-Grasp)
    harvest_spectrum_batch(
        "le-probe/datasets/anonymouscorl5-cmyk/gr1_pickup_grasp",
        steps=[8, 16, 20, 24, 31],
    )
