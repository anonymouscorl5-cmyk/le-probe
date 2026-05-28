from datasets import load_from_disk
import os
import numpy as np

# Absolute path to the dataset directory
DATASET_DIR = "/workspace/user/.cache/huggingface/datasets/parquet/default-7a73cfdf8450bb36/0.0.0/9c460aabd2aa27d1496e5e38d2060760561f0ac2cd6a110134eefa5b3f153b8d"

# The 3 specific frames we want to audit
TARGETS = {
    "Tactile Engagement (90)": {
        "index": 12117,
        "desc": "Episode 115, Frame 23 (Peak Grasp)",
    },
    "Spatial Lockdown (358)": {
        "index": 10983,
        "desc": "Episode 80, Frame 9 (Descent Start)",
    },
    "Alignment Precision (743)": {
        "index": 9047,
        "desc": "Episode 19, Frame 25 (Perfect Align)",
    },
}


def extract_joints():
    # Load the dataset using the standard HF method
    ds = load_from_disk(DATASET_DIR)

    print("\n🏗️ CANONICAL JOINT POSITIONS (32-dim Normalized Actions):")
    print("-" * 80)

    for name, info in TARGETS.items():
        idx = info["index"]
        # Shift index to match episode-only dataset (Activations: 0-2002 are snapshots)
        ds_idx = idx - 2002

        if ds_idx < 0:
            print(f"Skipping {name} - index {idx} is a static snapshot.")
            continue

        row = ds[ds_idx]
        action = np.array(row["action"], dtype=np.float32)

        print(f"\n🌟 {name}")
        print(f"Context: {info['desc']}")

        # Right Arm: Shoulder Pitch(16), Roll(17), Yaw(18), Elbow(19), Wrist R(20), P(21), Y(22)
        r_arm = action[16:23]
        print(f"  Right Arm (16-22): {r_arm.tolist()}")
        print(f"  Full 32-dim Vector:")
        print(f"  {action.tolist()}")


if __name__ == "__main__":
    if os.path.exists(DATASET_DIR):
        extract_joints()
    else:
        print(f"❌ Dataset directory not found at: {DATASET_DIR}")
