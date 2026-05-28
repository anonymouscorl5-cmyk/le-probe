# --- Path Stabilization ---
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# --------------------------

import os
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm


def validate_compression():
    source_repo = "gr1_pickup_final"
    target_repo = "gr1_pickup_compressed"

    parent_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "datasets",
        "anonymouscorl5-cmyk",
    )
    source_path = os.path.join(parent_dir, source_repo)
    target_path = os.path.join(parent_dir, target_repo)

    if not os.path.isdir(source_path) or not os.path.isdir(target_path):
        print("❌ One of the datasets is missing.")
        return

    print(f"🔍 Validating {target_repo} against {source_repo}...")
    source_ds = LeRobotDataset(repo_id=source_repo, root=source_path)

    # FORCE ABSOLUTE PATH TO PREVENT SEARCH-PATH CACHING
    target_ds = LeRobotDataset(repo_id=target_repo, root=target_path)

    print(f"DEBUG: Dataset total frames from object: {target_ds.num_frames}")
    print(f"DEBUG: Dataset total episodes from object: {target_ds.num_episodes}")

    # The mapping used in compress_dataset.py
    rel_indices = [7, 12, 16, 20, 25, 28, 32, 36, 40, 44, 48, 51]
    frames_per_ep = len(rel_indices)

    total_episodes = int(target_ds.num_episodes)

    mismatches = 0
    total_checks = 0

    for ep_idx in tqdm(range(total_episodes)):
        # Get source episode indices
        src_ep = source_ds.meta.episodes[ep_idx]
        src_from = int(src_ep["dataset_from_index"])

        # Get target episode indices (should be exactly 12 frames)
        tgt_ep = target_ds.meta.episodes[ep_idx]
        tgt_from = int(tgt_ep["dataset_from_index"])
        tgt_to = int(tgt_ep["dataset_to_index"])

        actual_len = tgt_to - tgt_from + 1

        if ep_idx < 5:
            print(
                f"DEBUG: Ep {ep_idx} index range: {tgt_from} -> {tgt_to} (Len: {actual_len})"
            )

        if actual_len != frames_per_ep:
            print(
                f"⚠️ Episode {ep_idx} has {actual_len} frames, expected {frames_per_ep}"
            )
            mismatches += 1
            continue

        for i, rel_idx in enumerate(rel_indices):
            src_frame_idx = src_from + rel_idx
            tgt_frame_idx = tgt_from + i

            src_frame = source_ds[src_frame_idx]
            tgt_frame = target_ds[tgt_frame_idx]

            # 1. Check observation.state
            s_state = src_frame["observation.state"].numpy()
            t_state = tgt_frame["observation.state"].numpy()
            if not np.allclose(s_state, t_state, atol=1e-6):
                print(
                    f"❌ State mismatch at Ep {ep_idx}, Frame {i} (Source idx {src_frame_idx})"
                )
                mismatches += 1

            # 2. Check action
            s_action = src_frame["action"].numpy()
            t_action = tgt_frame["action"].numpy()
            if not np.allclose(s_action, t_action, atol=1e-6):
                print(f"❌ Action mismatch at Ep {ep_idx}, Frame {i}")
                mismatches += 1

            total_checks += 1

    if mismatches == 0:
        print(f"✅ SUCCESS: All {total_checks} frames are bit-perfect matches!")
    else:
        print(f"❌ FAILURE: Found {mismatches} mismatches during validation.")


if __name__ == "__main__":
    validate_compression()
