import sys
import torch
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader

# --- Path Stabilization ---
CURRENT_FILE = Path(__file__).resolve()
ROOT_DIR = CURRENT_FILE.parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

LEWM_DIR = ROOT_DIR / "lewm"
if str(LEWM_DIR) not in sys.path:
    sys.path.append(str(LEWM_DIR))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lewm.goal_mapper import GoalMapper
from lewm.lewm_data_plugin import LEWMDataPlugin
from lewm.skeleton.data import SkeletonDataPlugin


def harvest_manifold(
    model_path,
    dataset_repo,
    output_file,
    num_episodes=0,
    num_workers=4,
    use_multi_view=True,
    fusion_type="linear",
    use_skeleton=False,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_path = Path(output_file).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Initializing Manifold Harvest | Device: {device}")

    # 1. Resolve Dataset Root
    try:
        ds = LeRobotDataset(dataset_repo)
        resolved_root = ds.root
        print(f"📦 Local Dataset detected: {resolved_root}")
    except Exception:
        resolved_root = "."

    # 2. Load Model
    mapper = GoalMapper(
        model_path=model_path,
        dataset_root=resolved_root,
        use_multi_view=use_multi_view,
        num_views=5 if use_multi_view else 1,
        use_skeleton=use_skeleton,
    )
    model = mapper.model.to(device).eval()

    # 2. Initialize Data Plugin (num_steps=1 for frame-level granularity)
    # Use exact canonical keys from train_lewm.py for strict parity
    keys_to_load = ["action"]
    if use_multi_view:
        keys_to_load += [
            "observation.images.world_center",
            "observation.images.world_left",
            "observation.images.world_right",
            "observation.images.world_top",
            "observation.images.world_wrist",
        ]
    else:
        keys_to_load += ["pixels"]

    if use_skeleton:
        data_plugin = SkeletonDataPlugin(
            repo_id=dataset_repo,
            keys_to_load=keys_to_load,
            num_steps=1,
            use_multi_view=use_multi_view,
        )
    else:
        data_plugin = LEWMDataPlugin(
            repo_id=dataset_repo,
            keys_to_load=keys_to_load,
            num_steps=1,
            use_multi_view=use_multi_view,
        )
    data_plugin.clear_cache()

    dataloader = DataLoader(
        data_plugin,
        batch_size=64,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    all_latents = []
    all_indices = []
    all_episode_indices = []

    # 3. Calculate processing scope
    total_frames = len(data_plugin)
    if num_episodes > 0:
        total_frames = min(total_frames, num_episodes * 32)

    batch_size = 64
    num_batches = (total_frames + batch_size - 1) // batch_size

    print(f"📊 Target: {num_episodes if num_episodes > 0 else 'Full Dataset'} episodes")
    print(f"📊 Processing {total_frames} total frames (~{num_batches} batches)...")

    try:
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Harvesting", total=num_batches)
            for i, batch in enumerate(pbar):
                if i >= num_batches:
                    break

                raw_pixels = batch["pixels"].to(device)
                actions = batch["action"].to(device)

                if use_skeleton:
                    raw_skel = batch["skeletons_raw"].to(device)
                    # Convert 3-channel skeleton mask to 1-channel by taking the mean across the color dimension
                    raw_skel_1ch = raw_skel.float().mean(dim=-3, keepdim=True).byte()

                # --- 🎯 Unified 6D Protocol (B, T, V, C, H, W) ---
                if i == 0:
                    print(f"\n🔍 [BATCH 0] SHAPE TRACE:")
                    print(f"  - raw_pixels:    {raw_pixels.shape}")
                    if use_skeleton:
                        print(f"  - raw_skel_1ch:  {raw_skel_1ch.shape}")

                if raw_pixels.ndim == 5:
                    if not use_multi_view:
                        # (B, T, C, H, W) -> (B, T, 1, C, H, W)
                        pixels_6d = raw_pixels.unsqueeze(2)
                        if use_skeleton:
                            skel_6d = raw_skel_1ch.unsqueeze(2)
                    else:
                        # (B, V, C, H, W) -> (B, 1, V, C, H, W)
                        pixels_6d = raw_pixels.unsqueeze(1)
                        if use_skeleton:
                            skel_6d = raw_skel_1ch.unsqueeze(1)
                else:
                    pixels_6d = raw_pixels
                    if use_skeleton:
                        skel_6d = raw_skel_1ch

                if i == 0:
                    print(f"  - pixels_6d:     {pixels_6d.shape}")
                    if use_skeleton:
                        print(f"  - skel_6d:       {skel_6d.shape}")

                B, T, V, C, H, W = pixels_6d.shape
                raw_pixels_flat = pixels_6d.reshape(B * T * V, C, H, W)
                processed_pixels = mapper.transform({"pixels": raw_pixels_flat})[
                    "pixels"
                ]
                pixels = processed_pixels.view(B, T, V, C, 224, 224)

                if use_skeleton:
                    # Resize/transform skeleton to 224x224
                    skel_flat = skel_6d.reshape(B * T * V, 1, H, W)
                    if skel_flat.shape[-2:] != (224, 224):
                        skel_flat_resized = torch.nn.functional.interpolate(
                            skel_flat.float(), size=(224, 224), mode="nearest"
                        ).byte()
                    else:
                        skel_flat_resized = skel_flat

                    skel_final = skel_flat_resized.view(B, T, V, 1, 224, 224)

                    # Concat RGB and Skeleton along channel dimension (index -3): (B, T, V, 4, 224, 224)
                    pixels = torch.cat([pixels, skel_final.to(pixels.dtype)], dim=-3)

                if i == 0:
                    print(f"  - mapper_out:    {processed_pixels.shape}")
                    print(f"  - pixels_final:  {pixels.shape}")
                    print(f"  - actions:       {actions.shape}\n")
                # -----------------------------------------------

                if torch.isnan(actions).any():
                    actions = torch.nan_to_num(actions, 0.0)

                with torch.amp.autocast("cuda"):
                    # Extract the joint embedding (Encoder output)
                    info = model.encode({"pixels": pixels, "action": actions})
                    emb = info["emb"]  # (B, T, D)

                all_latents.append(emb.cpu().numpy().astype(np.float32))

                # Fetch frame indices for this batch
                start_idx = i * B
                end_idx = min((i + 1) * B, len(data_plugin.frame_indices))

                batch_indices = data_plugin.frame_indices[start_idx:end_idx]
                all_indices.append(batch_indices.numpy())

                batch_ep_indices = data_plugin.episode_indices[start_idx:end_idx]
                all_episode_indices.append(batch_ep_indices.numpy())

    finally:
        data_plugin.clear_cache()

    # Concatenate results
    latents = np.concatenate(all_latents, axis=0)  # (N, T, D)
    latents = latents.reshape(-1, latents.shape[-1])  # (N*T, D)
    indices = np.concatenate(all_indices, axis=0)  # (N*T,)
    episode_indices = np.concatenate(all_episode_indices, axis=0)  # (N*T,)

    print(f"💾 Saving manifold data...")
    data = {
        "latents": latents,
        "frame_indices": indices,
        "episode_indices": episode_indices,
    }
    torch.save(data, output_path)
    print(f"✨ Manifold data saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gr1_reward_tuned_v2.ckpt")
    parser.add_argument("--dataset", type=str, default="vedpatwardhan/gr1_pickup_grasp")
    parser.add_argument(
        "--output",
        type=str,
        default=str(CURRENT_FILE.parent / "manifold_data.pt"),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="Number of episodes to harvest (0 for all)",
    )
    parser.add_argument("--multi_view", action="store_true", default=False)
    parser.add_argument("--fusion", type=str, default="linear")
    parser.add_argument("--use_skeleton", action="store_true", default=False)
    args = parser.parse_args()

    harvest_manifold(
        model_path=args.model,
        dataset_repo=args.dataset,
        output_file=args.output,
        num_episodes=args.episodes,
        use_multi_view=args.multi_view,
        fusion_type=args.fusion,
        use_skeleton=args.use_skeleton,
    )
