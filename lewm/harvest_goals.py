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
from tqdm import tqdm
from huggingface_hub import snapshot_download

# Resolve project paths dynamically
RESEARCH_DIR = Path(__file__).parent.absolute()
CORTEX_GR1 = RESEARCH_DIR.parent
if str(CORTEX_GR1) not in sys.path:
    sys.path.append(str(CORTEX_GR1))

from lewm.goal_mapper import GoalMapper
from lewm.goal_utils import get_episode_video_path, extract_frame_at_index
from lewm.skeleton.skeletal_utils import get_skeletal_diagnostic_frames

REPO_ID = "vedpatwardhan/gr1_pickup_grasp"


def harvest(
    model_path,
    dataset_root,
    output_path,
    use_multi_view=False,
    use_skeleton=False,
    skel_frames_dir=None,
):
    """
    Sweeps the dataset and encodes success states.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚜 Harvesting Success States to {output_path}...")

    # 1. Initialize Mapper
    mapper = GoalMapper(
        model_path,
        dataset_root,
        use_multi_view=use_multi_view,
        num_views=5 if use_multi_view else 1,
        use_skeleton=use_skeleton,
        skel_frames_dir=skel_frames_dir,
    )

    gallery = {"goals": {}, "diagnostics": {}}

    # 2. Sweep Episodes
    for i in tqdm(range(2000), desc="Harvesting Episodes"):
        try:
            # 1. Encode the Goal State
            success = mapper.set_goal(episode_idx=i)
            if not success:
                print(f"\n⏹️ End of dataset reached at index {i}")
                break
            gallery["goals"][i] = mapper.goal_latent.cpu()

            # 2. Capture the Start State (Diagnostic Previews)
            start_frames = get_skeletal_diagnostic_frames(
                episode_idx=i,
                dataset_root=dataset_root,
                skel_frames_dir=skel_frames_dir,
                use_skeleton=use_skeleton,
                mapper=mapper,
            )

            # 3. Store full context
            if start_frames:
                gallery["diagnostics"][i] = {
                    "pixels": torch.stack(start_frames).cpu(),
                    "action": torch.zeros(len(start_frames), 32),
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
    parser.add_argument("--use_skeleton", action="store_true", default=False)
    parser.add_argument("--skel_frames", type=str, default=None)
    args = parser.parse_args()

    dataset_root = args.dataset
    if dataset_root is None:
        # Default to local clone or Hub sync
        dataset_root = f"le-probe/datasets/{REPO_ID}"
        if not Path(dataset_root).exists():
            print(f"📥 Syncing dataset from Hub: {REPO_ID}")
            dataset_root = snapshot_download(repo_id=REPO_ID, repo_type="dataset")

    harvest(
        args.model,
        dataset_root,
        args.output,
        use_multi_view=args.multi_view,
        use_skeleton=args.use_skeleton,
        skel_frames_dir=args.skel_frames,
    )
