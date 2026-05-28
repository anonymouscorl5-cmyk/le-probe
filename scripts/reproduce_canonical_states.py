from datasets import load_from_disk
import os
import json
import numpy as np
from PIL import Image

# Absolute paths - UPDATED to the correct local path
DATASET_DIR = (
    "/workspace/cortex-os/le-probe/datasets/anonymouscorl5-cmyk/gr1_pickup_grasp"
)
OUTPUT_DIR = "/workspace/cortex-os/le-probe/temp_repro"
JSON_PATH = "/workspace/cortex-os/le-probe/canonical_reproduction.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# The 3 target indices (Global Activation indices)
TARGETS = {
    "feature_90": {"idx": 12117, "label": "Tactile Engagement"},
    "feature_358": {"idx": 11993, "label": "Spatial Lockdown"},
    "feature_743": {"idx": 9047, "label": "Alignment Precision"},
}


def reproduce():
    print(f"📦 Loading dataset from disk: {DATASET_DIR}")
    # This directory should contain 'train' and 'test' subdirs or just the dataset files
    # Try loading the main dir, or check for 'train' subdir
    try:
        ds = load_from_disk(DATASET_DIR)
    except:
        train_dir = os.path.join(DATASET_DIR, "train")
        print(f"🔄 Retrying with {train_dir}...")
        ds = load_from_disk(train_dir)

    results = {}

    for key, info in TARGETS.items():
        # Shift to dataset index
        ds_idx = info["idx"] - 2002
        label = info["label"]

        print(f"🔍 Extracting {label} (Index {ds_idx})...")

        # 1. Action Vector (Current Frame)
        row_curr = ds[ds_idx]
        action = [float(x) for x in row_curr["action"]]

        # 2. Images (Current and Previous)
        # Current Frame
        img_curr_raw = row_curr["observation.images.world_center"]
        if isinstance(img_curr_raw, Image.Image):
            img_curr = img_curr_raw
        else:
            img_curr = Image.fromarray(np.array(img_curr_raw, dtype=np.uint8))

        path_after = os.path.join(OUTPUT_DIR, f"{key}_after.png")
        img_curr.save(path_after)

        # Previous Frame
        row_prev = ds[ds_idx - 1]
        img_prev_raw = row_prev["observation.images.world_center"]
        if isinstance(img_prev_raw, Image.Image):
            img_prev = img_prev_raw
        else:
            img_prev = Image.fromarray(np.array(img_prev_raw, dtype=np.uint8))

        path_before = os.path.join(OUTPUT_DIR, f"{key}_before.png")
        img_prev.save(path_before)

        results[key] = {
            "label": label,
            "global_index": info["idx"],
            "dataset_index": ds_idx,
            "action_vector": action,
            "image_after": path_after,
            "image_before": path_before,
        }

    # Save to JSON
    with open(JSON_PATH, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n✅ Reproduction data saved to: {JSON_PATH}")
    print(f"📸 Reference images saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    if os.path.exists(DATASET_DIR):
        reproduce()
    else:
        print(f"❌ Error: Dataset directory not found at: {DATASET_DIR}")
