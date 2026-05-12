# --- Path Stabilization ---
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# --------------------------

import os
import json
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path


def harvest_spectrum_batch_v2_cv2(
    dataset_root, steps=[8, 16, 20, 24, 31], num_episodes=200
):
    root = Path(dataset_root)
    print(
        f"🚜 Harvesting MULTI-VIEW reward spectrum ({len(steps)} frames/ep) using CV2..."
    )

    # Target Folder
    output_dir = Path("le-probe/datasets/vedpatwardhan/gr1_reward_pred_v2")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load sidecar for rewards
    sidecar_path = root / "progress_sparse.parquet"
    if not sidecar_path.exists():
        print(f"❌ Error: Reward sidecar not found at {sidecar_path}")
        return

    sdf = pd.read_parquet(sidecar_path)

    cam_keys = [
        "observation.images.world_center",
        "observation.images.world_left",
        "observation.images.world_right",
        "observation.images.world_top",
        "observation.images.world_wrist",
    ]

    harvested = 0
    for ep_idx in tqdm(range(num_episodes)):
        # Parquet path
        parquet_path = root / "data" / "chunk-000" / f"file-{ep_idx:03d}.parquet"
        if not parquet_path.exists():
            print(f"⚠️ Warning: Parquet not found for episode {ep_idx}. Skipping.")
            continue

        df = pd.read_parquet(parquet_path)

        # Video paths
        cam_videos = {}
        for cam in cam_keys:
            v_path = root / "videos" / cam / "chunk-000" / f"file-{ep_idx:03d}.mp4"
            if v_path.exists():
                cam_videos[cam] = str(v_path)

        if len(cam_videos) < 5:
            print(
                f"⚠️ Warning: Missing cameras for episode {ep_idx}. Found {len(cam_videos)}/5."
            )

        for step in steps:
            # Global index in the entire dataset (32 frames per episode)
            global_idx = (ep_idx * 32) + step

            if global_idx >= len(sdf):
                break

            # Get reward from sidecar
            reward = float(sdf.iloc[global_idx]["progress_sparse"])

            # Get state and action from episode parquet
            # The index in the episode parquet is the step index (0-31)
            if step >= len(df):
                print(f"⚠️ Warning: Step {step} out of bounds for episode {ep_idx}.")
                continue

            row = df.iloc[step]

            snapshot = {
                "progress": reward,
                "episode_index": int(ep_idx),
                "step": int(step),
                "observation.state": row["observation.state"].tolist(),
                "action": row["action"].tolist(),
            }

            # Extract frames from videos
            for cam, v_path in cam_videos.items():
                img = extract_frame_cv2(v_path, step)
                if img is not None:
                    # Resize to 224x224
                    img_resized = cv2.resize(
                        img, (224, 224), interpolation=cv2.INTER_AREA
                    )
                    # Convert to (C, H, W)
                    img_chw = img_resized.transpose(2, 0, 1)
                    snapshot[cam] = img_chw.tolist()
                else:
                    print(f"❌ Error: Could not extract frame {step} from {v_path}")

            snap_path = output_dir / f"spec_{ep_idx:03d}_s{step:02d}.json"
            with open(snap_path, "w") as f:
                json.dump(snapshot, f)

            harvested += 1

    print(
        f"✅ Multi-View Spectrum harvest complete! {harvested} snapshots stored in {output_dir}"
    )


def extract_frame_cv2(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    # BGR -> RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


if __name__ == "__main__":
    # Steps: 8, 16, 20, 24, 31
    harvest_spectrum_batch_v2_cv2(
        "le-probe/datasets/vedpatwardhan/gr1_pickup_grasp", steps=[8, 16, 20, 24, 31]
    )
