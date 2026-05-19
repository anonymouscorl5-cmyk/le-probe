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
from tqdm import tqdm
import argparse
import numpy as np
from pathlib import Path

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
    use_dino=False,
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
        use_dino=use_dino,
    )

    # 2. Initialize Data Engine (Plugin)
    PluginClass = SkeletonDataPlugin if use_skeleton else LEWMDataPlugin
    keys = ["world_center", "action"]
    if use_multi_view:
        keys = [
            "observation.state",
            "action",
            "world_center",
            "world_left",
            "world_right",
            "world_top",
            "world_wrist",
        ]

    plugin = PluginClass(
        repo_id=repo_id,
        keys_to_load=keys,
        num_steps=1,
        use_multi_view=use_multi_view,
    )
    dataset = plugin.lerobot_dataset
    num_episodes = plugin.lerobot_dataset.num_episodes

    gallery = {"goals": {}, "diagnostics": {}}

    # 3. Sweep Episodes
    for i in tqdm(range(num_episodes), desc="Harvesting Episodes"):
        try:
            # A. Identify Success Frame (Manual boundary calculation for robustness)
            indices = (plugin.episode_indices == i).nonzero(as_tuple=True)[0]
            if len(indices) == 0:
                raise ValueError(
                    f"🚨 Integrity Error: Episode {i} exists in metadata but has 0 frames in the dataset!"
                )

            start_global_idx = indices[0].item()
            last_global_idx = indices[-1].item()

            # B. Fetch via Plugin
            goal_batch = plugin[last_global_idx]
            goal_pixels = goal_batch["pixels"].squeeze(0)  # (V, C, H, W) or (C, H, W)
            goal_skeleton = None
            if use_skeleton:
                # If cached tensor is used, it already has 4 channels (RGB + Skeleton).
                # Only look for skeletons_raw if the pixels do not already contain the 4th channel.
                channels = (
                    goal_pixels.shape[1]
                    if goal_pixels.ndim == 4
                    else goal_pixels.shape[0]
                )
                if channels < 4:
                    goal_skeleton = goal_batch.get("skeletons_raw")
                    if goal_skeleton is None:
                        raise ValueError(
                            f"🚨 Missing 'skeletons_raw' in episode {i} despite use_skeleton=True!"
                        )
                    goal_skeleton = goal_skeleton.squeeze(
                        0
                    )  # (V, 1, H, W) or (1, H, W)
                    goal_skeleton = (
                        goal_skeleton.float().mean(dim=-3, keepdim=True).byte()
                    )

            # C. Encode
            mapper.encode_goal_from_pixels(goal_pixels, skeleton=goal_skeleton)
            gallery["goals"][i] = mapper.goal_latent.cpu()

            # D. Capture Diagnostics (First 3 frames)
            diag_pixels = []
            diag_actions = []

            for f_offset in range(3):
                idx = start_global_idx + f_offset
                if idx > last_global_idx:
                    break
                batch = plugin[idx]

                # Fuse for diagnostics if in skeletal mode
                diag_p = batch["pixels"].squeeze(0)  # (V, C, H, W) or (C, H, W)
                if use_skeleton:
                    diag_channels = (
                        diag_p.shape[1] if diag_p.ndim == 4 else diag_p.shape[0]
                    )
                    if diag_channels < 4:
                        diag_skel = batch.get("skeletons_raw")
                        if diag_skel is not None:
                            diag_skel = diag_skel.squeeze(0)
                            # Force single-channel (C is always -3)
                            diag_skel = (
                                diag_skel.float().mean(dim=-3, keepdim=True).byte()
                            )

                            # Fuse: (V, 3, H, W) + (V, 1, H, W) -> (V, 4, H, W)
                            if diag_p.ndim == 4:
                                diag_p = torch.cat([diag_p, diag_skel], dim=1)
                            else:
                                # Single view: (3, H, W) + (1, H, W) -> (4, H, W)
                                diag_p = torch.cat([diag_p, diag_skel], dim=0)

                diag_pixels.append(diag_p)
                diag_actions.append(batch["action"].squeeze(0))

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
    parser.add_argument("--use_dino", action="store_true", default=False)
    args = parser.parse_args()

    harvest(
        args.model,
        args.dataset,
        args.output,
        use_multi_view=args.multi_view,
        use_skeleton=args.use_skeleton,
        use_dino=args.use_dino,
    )
