# --- Path Stabilization ---
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# --------------------------

import os
import numpy as np
import shutil
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image
from tqdm import tqdm


def compress_dataset():
    # Final folder names
    source_repo = "gr1_pickup_final"
    target_repo = "gr1_pickup_compressed"

    # Absolute paths for local loading
    parent_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "datasets",
        "anonymouscorl5-cmyk",
    )
    source_path = os.path.join(parent_dir, source_repo)
    target_path = os.path.join(parent_dir, target_repo)

    if not os.path.isdir(source_path):
        print(f"❌ Could not find local dataset folder at: {source_path}")
        return

    print(f"📂 Loading source dataset from: {source_path}")
    # Use pyav to avoid torchcodec library errors
    source_ds = LeRobotDataset(
        repo_id=source_repo, root=source_path, video_backend="pyav"
    )
    features = source_ds.features

    print(f"✨ Creating target dataset: {target_repo}")
    if os.path.exists(target_path):
        shutil.rmtree(target_path)

    # Use official create then patch for strict 1:1 mapping
    target_ds = LeRobotDataset.create(
        repo_id=target_repo,
        fps=10,
        root=target_path,
        features=features,
        use_videos=True,
        vcodec="h264",
    )

    # Patch metadata settings for strict 1:1 file-to-episode mapping
    target_ds.meta.metadata_buffer_size = 1
    target_ds.meta.update_chunk_settings(
        data_files_size_in_mb=0.0001, video_files_size_in_mb=0.0001
    )

    sampling_factor = 4
    total_episodes = int(source_ds.num_episodes)

    print(f"🚀 Compressing {total_episodes} episodes to 13 frames each...")

    for ep_idx in tqdm(range(total_episodes)):
        ep = source_ds.meta.episodes[ep_idx]
        from_idx = int(ep["dataset_from_index"])
        to_idx = int(ep["dataset_to_index"])

        # Goal-Oriented Slicing: Captures reaching (12), grasping (25, 28), and success (51)
        rel_indices = [7, 12, 16, 20, 25, 28, 32, 36, 40, 44, 48, 51]
        indices = [from_idx + i for i in rel_indices]

        for idx in indices:
            frame_data = source_ds[int(idx)]

            new_frame = {
                "observation.state": frame_data["observation.state"],
                "action": frame_data["action"],
                "task": (
                    frame_data["task"]
                    if "task" in frame_data
                    else "Pick up the red cube"
                ),
            }

            # Convert tensors to PIL for target_ds.add_frame
            for key in features:
                if key.startswith("observation.images."):
                    img_t = frame_data[key]
                    img_np = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    new_frame[key] = Image.fromarray(img_np)

            target_ds.add_frame(new_frame)

        # Parallel encoding disabled for stability in narrow thresholds
        target_ds.save_episode(parallel_encoding=False)

    print(f"✅ SUCCESS: Dataset compressed to {target_ds.num_frames} frames.")
    print(
        f"📊 Each episode is {target_ds.num_frames / target_ds.num_episodes:.1f} frames long."
    )
    print(f"📍 Location: {target_path}")


if __name__ == "__main__":
    compress_dataset()
