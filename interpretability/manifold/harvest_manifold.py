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

from lewm.goal_mapper import GoalMapper
from lewm.lewm_data_plugin import LEWMDataPlugin

def harvest_manifold(
    model_path, dataset_repo, output_file, num_episodes=0, num_workers=4
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_path = Path(output_file).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Initializing Manifold Harvest | Device: {device}")

    # 1. Load Model
    mapper = GoalMapper(model_path=model_path, dataset_root=".")
    model = mapper.model.to(device).eval()

    # 2. Initialize Data Plugin
    # We use num_steps=1 to process every frame individually
    data_plugin = LEWMDataPlugin(
        repo_id=dataset_repo, keys_to_load=["pixels", "action"], num_steps=1
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

    actual_total = len(dataloader)
    if num_episodes > 0:
        actual_total = min(actual_total, num_episodes * 32)

    print(f"📊 Processing {actual_total} frames...")

    try:
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Harvesting", total=actual_total)
            for i, batch in enumerate(pbar):
                if num_episodes > 0 and i >= num_episodes * 32:
                    break

                raw_pixels = batch["pixels"].to(device)
                actions = batch["action"].to(device)

                # --- 🎯 Model-Specific Transform ---
                B, T, C, H, W = raw_pixels.shape
                raw_pixels_flat = raw_pixels.view(B * T, C, H, W)
                processed_pixels = mapper.transform({"pixels": raw_pixels_flat})["pixels"]
                pixels = processed_pixels.view(B, T, C, 224, 224)
                # -----------------------------------

                if torch.isnan(actions).any():
                    actions = torch.nan_to_num(actions, 0.0)

                with torch.amp.autocast("cuda"):
                    # Extract the joint embedding (Encoder output)
                    info = model.encode({"pixels": pixels, "action": actions})
                    emb = info["emb"] # (B, T, D)
                
                all_latents.append(emb.cpu().numpy().astype(np.float32))
                
                # Fetch frame indices for this batch
                start_idx = i * B
                end_idx = min((i + 1) * B, len(data_plugin.frame_indices))
                batch_indices = data_plugin.frame_indices[start_idx:end_idx]
                all_indices.append(batch_indices.numpy())

    finally:
        data_plugin.clear_cache()

    # Concatenate results
    latents = np.concatenate(all_latents, axis=0) # (N, T, D)
    latents = latents.reshape(-1, latents.shape[-1]) # (N*T, D)
    indices = np.concatenate(all_indices, axis=0) # (N*T,)

    print(f"💾 Saving manifold data...")
    data = {
        "latents": latents,
        "frame_indices": indices
    }
    torch.save(data, output_path)
    print(f"✨ Manifold data saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gr1_reward_tuned_v2.ckpt")
    parser.add_argument("--dataset", type=str, default="vedpatwardhan/gr1_pickup_grasp")
    parser.add_argument("--output", type=str, default="le-probe/interpretability/manifold/manifold_data.pt")
    parser.add_argument("--episodes", type=int, default=0, help="Number of episodes to harvest (0 for all)")
    args = parser.parse_args()

    harvest_manifold(
        model_path=args.model,
        dataset_repo=args.dataset,
        output_file=args.output,
        num_episodes=args.episodes
    )
