"""
FULL SPECTRUM HARVESTER
Role: Distills the entire dataset (150+ episodes) into a single Diagnostic Gallery.
Output: goal_gallery.pth (~350MB)
"""

# --- Path Stabilization ---
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# --------------------------


import torch
import argparse
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import snapshot_download

# Resolve project paths dynamically
RESEARCH_DIR = Path(__file__).parent.absolute()
CORTEX_GR1 = RESEARCH_DIR.parent
if str(CORTEX_GR1) not in sys.path:
    sys.path.append(str(CORTEX_GR1))

from lewm.goal_mapper import GoalMapper
from lewm.goal_utils import get_episode_video_path, extract_frame_at_index

REPO_ID = "vedpatwardhan/gr1_pickup_grasp"


def harvest(model_path, dataset_root, output_path, use_multi_view=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 0. Sync Dataset if path is missing or invalid
    if dataset_root is None or not Path(dataset_root).exists():
        print(f"☁️ Syncing dataset from Hugging Face Hub: {REPO_ID}...")
        dataset_root = snapshot_download(repo_id=REPO_ID, repo_type="dataset")

    print(
        f"🎬 Starting Full Spectrum Harvest (All Episodes) on {device} (Multi-View: {use_multi_view})..."
    )

    mapper = GoalMapper(
        model_path,
        dataset_root,
        use_multi_view=use_multi_view,
        num_views=5 if use_multi_view else 1,
    )
    gallery = {
        "goals": {},  # {id: goal_latent}
        "diagnostics": {},  # {id: {pixels: 3,3,224,224, action: 4,64}}
    }

    # Iterate through every episode in the dataset
    for i in tqdm(range(2000), desc="Harvesting Dataset Context"):
        try:
            # 1. Capture the Goal State (Latent)
            success = mapper.set_goal(episode_idx=i)
            if not success:
                print(f"\n⏹️ End of dataset reached at index {i}")
                break
            gallery["goals"][i] = mapper.goal_latent.cpu()

            # 2. Capture the Start State (Center Frame only for diagnostics)
            video_path = get_episode_video_path(
                dataset_root, i, camera_key="observation.images.world_center"
            )
            start_frames = []
            for frame_idx in range(3):
                frame_np = extract_frame_at_index(video_path, frame_idx)
                if frame_np is None:
                    continue
                # Transform expects (C, H, W) in [0, 1]
                transformed = mapper.transform({"pixels": frame_np})["pixels"]
                start_frames.append(transformed)

            # 3. Store full context
            if start_frames:
                gallery["diagnostics"][i] = {
                    "pixels": torch.stack(start_frames).cpu(),
                    "action": torch.zeros(4, 32),
                }

        except Exception as e:
            print(f"⚠️ Error harvesting episode {i}: {e}")
            break

    # 4. Save the Final Artifact
    if gallery["goals"]:
        torch.save(gallery, output_path)
        print(
            f"✅ Full Spectrum Gallery saved: {output_path} ({len(gallery['goals'])} episodes)"
        )
    else:
        print("❌ Dataset harvest failed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Local dataset path (optional, will sync from Hub if missing)",
    )
    parser.add_argument("--output", type=str, default="goal_gallery.pth")
    parser.add_argument("--multi_view", action="store_true", default=False)
    args = parser.parse_args()
    harvest(args.model, args.dataset, args.output, use_multi_view=args.multi_view)
