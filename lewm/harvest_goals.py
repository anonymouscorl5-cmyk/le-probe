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
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Resolve project paths dynamically
RESEARCH_DIR = Path(__file__).parent.absolute()
CORTEX_GR1 = RESEARCH_DIR.parent
if str(CORTEX_GR1) not in sys.path:
    sys.path.append(str(CORTEX_GR1))

from lewm.goal_mapper import GoalMapper
from lewm.lewm_data_plugin import LEWMDataPlugin
from lewm.skeleton.data import SkeletonDataPlugin

REPO_ID = "vedpatwardhan/gr1_pickup_grasp"


def harvest(
    model_path,
    repo_id,
    output_path,
    use_multi_view=False,
    use_skeleton=False,
):
    """
    Sweeps the dataset and encodes success states using the DataPlugin engine.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚜 Harvesting Success States to {output_path}...")

    # 1. Initialize Mapper
    mapper = GoalMapper(
        model_path,
        dataset_root=None,
        use_multi_view=use_multi_view,
        num_views=5 if use_multi_view else 1,
        use_skeleton=use_skeleton,
    )

    # 2. Initialize Data Engine (Plugin)
    PluginClass = SkeletonDataPlugin if use_skeleton else LEWMDataPlugin
    plugin = PluginClass(
        repo_id=repo_id,
        keys_to_load=["observation.images.world_center", "action"],
        num_steps=1,
        use_multi_view=use_multi_view,
    )
    dataset = plugin.dataset
    num_episodes = dataset.num_episodes

    gallery = {"goals": {}, "diagnostics": {}}

    # 3. Sweep Episodes
    for i in tqdm(range(num_episodes), desc="Harvesting Episodes"):
        try:
            # A. Identify Success Frame
            last_global_idx = dataset.episode_data_index["to"][i] - 1

            # B. Fetch via Plugin
            goal_batch = plugin[last_global_idx]
            goal_pixels = goal_batch["pixels"]

            # C. Encode
            mapper.encode_goal_from_pixels(goal_pixels)
            gallery["goals"][i] = mapper.goal_latent.cpu()

            # D. Capture Diagnostics (First 3 frames)
            diag_pixels = []
            diag_actions = []
            start_global_idx = dataset.episode_data_index["from"][i]

            for f_offset in range(3):
                idx = start_global_idx + f_offset
                if idx > last_global_idx:
                    break
                batch = plugin[idx]
                diag_pixels.append(batch["pixels"])
                diag_actions.append(batch["action"][0])

            if diag_pixels:
                gallery["diagnostics"][i] = {
                    "pixels": torch.stack(diag_pixels).cpu(),
                    "action": torch.stack(diag_actions).cpu(),
                }

        except Exception as e:
            print(f"⚠️ Error harvesting episode {i}: {e}")
            continue

    # 4. Save
    if gallery["goals"]:
        torch.save(gallery, output_path)
        print(f"✅ Gallery saved: {output_path} ({len(gallery['goals'])} episodes)")
    else:
        print("❌ Dataset harvest failed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default=REPO_ID)
    parser.add_argument("--output", type=str, default="goal_gallery.pth")
    parser.add_argument("--multi_view", action="store_true", default=False)
    parser.add_argument("--use_skeleton", action="store_true", default=False)
    args = parser.parse_args()

    harvest(
        args.model,
        args.dataset,
        args.output,
        use_multi_view=args.multi_view,
        use_skeleton=args.use_skeleton,
    )
