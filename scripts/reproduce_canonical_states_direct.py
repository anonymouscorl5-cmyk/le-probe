import os
import json
import cv2
import pandas as pd
import numpy as np
from PIL import Image

# Base Path
DATASET_ROOT = (
    "/workspace/cortex-os/le-probe/datasets/anonymouscorl5-cmyk/gr1_pickup_grasp"
)
OUTPUT_DIR = "/workspace/cortex-os/le-probe/temp_repro"
JSON_PATH = "/workspace/cortex-os/le-probe/canonical_reproduction.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Feature Trigger Definitions
TARGETS = {
    "feature_358": {"episode": 111, "index": 27, "label": "Spatial Lockdown"},
    "feature_90": {"episode": 115, "index": 23, "label": "Tactile Engagement"},
    "feature_743": {"episode": 19, "index": 25, "label": "Alignment Precision"},
}


def extract_frame(video_path, frame_idx):
    """Helper to extract a specific frame and resize it."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb


def reproduce():
    results = {}

    print("🚀 Starting Direct Canonical Extraction...")

    for key, info in TARGETS.items():
        ep_num = info["episode"]
        idx = info["index"]
        label = info["label"]

        # Paths (Padding with zeros as per LeRobot file-NNN convention)
        parquet_path = os.path.join(
            DATASET_ROOT, "data", "chunk-000", f"file-{ep_num:03d}.parquet"
        )
        video_path = os.path.join(
            DATASET_ROOT,
            "videos",
            "observation.images.world_center",
            "chunk-000",
            f"file-{ep_num:03d}.mp4",
        )

        print(f"📦 Processing {label} (Ep {ep_num}, Frame {idx})...")

        if not os.path.exists(parquet_path) or not os.path.exists(video_path):
            print(f"  ❌ Missing files for Episode {ep_num}. Skipping.")
            continue

        # 1. Get Action and State from Parquet
        df = pd.read_parquet(parquet_path)
        # In LeRobot parquets, the index in the dataframe corresponds to the frame index
        action_row = df.iloc[idx]["action"]
        action = [float(x) for x in action_row]

        state_row = df.iloc[idx]["observation.state"]
        state = [float(x) for x in state_row]

        # 2. Get Images from MP4
        # Current Frame
        img_after = extract_frame(video_path, idx)
        path_after = os.path.join(OUTPUT_DIR, f"{key}_after.png")
        Image.fromarray(img_after).save(path_after)

        # Previous Frame (8 frames behind)
        img_before = extract_frame(video_path, idx - 8)
        path_before = os.path.join(OUTPUT_DIR, f"{key}_before.png")
        Image.fromarray(img_before).save(path_before)

        results[key] = {
            "label": label,
            "episode": ep_num,
            "index": idx,
            "action_vector": action,
            "observation_state": state,
            "image_after": path_after,
            "image_before": path_before,
        }

    # Final JSON save
    with open(JSON_PATH, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n✅ Direct Reproduction data saved: {JSON_PATH}")
    print(f"📸 Reference images saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    reproduce()
