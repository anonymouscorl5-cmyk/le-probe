import sys
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# --- Path Stabilization ---
CURRENT_FILE = Path(__file__).resolve()
ROOT_DIR = CURRENT_FILE.parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

LEWM_DIR = ROOT_DIR / "lewm"
if str(LEWM_DIR) not in sys.path:
    sys.path.append(str(LEWM_DIR))

from lewm.goal_mapper import GoalMapper
from lewm.skeleton.encoder import patch_vit_for_skeleton


class SkeletalHarvestDataset(Dataset):
    """
    Dataset that loads 4-channel [RGB + Skeleton] images from Parquet.
    """

    def __init__(self, df, transform, use_multi_view=True):
        self.df = df
        self.transform = transform
        self.use_multi_view = use_multi_view
        self.cam_keys = (
            [
                "world_center",
                "world_left",
                "world_right",
                "world_top",
                "world_wrist",
            ]
            if use_multi_view
            else ["world_center"]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        views = []
        for vn in self.cam_keys:
            raw_img = row[f"observation.images.{vn}"]
            # Reconstruct (H, W, 4) from list of lists
            img_np = np.stack(
                [np.array(c.tolist(), dtype=np.uint8) for c in raw_img], axis=-1
            )
            # Transform (handles 4ch if pixels is a tensor)
            transformed = self.transform({"pixels": img_np})["pixels"]
            views.append(transformed)

        # Stack into (V, C, H, W) then add T=1 dimension: (1, V, C, H, W)
        img_pixels = torch.stack(views, dim=0).unsqueeze(0)

        # Actions (if available)
        action = torch.zeros(1, 6)  # Default zero action if not in parquet
        if "action" in row:
            action = torch.tensor(row["action"]).float().unsqueeze(0)

        return {
            "pixels": img_pixels,
            "action": action,
            "frame_index": row.get("frame_index", idx),
            "episode_index": row.get("episode_index", 0),
        }


def harvest_manifold_skel(
    model_path,
    parquet_path,
    output_file,
    num_episodes=0,
    batch_size=32,
    use_multi_view=True,
    fusion_type="linear",
):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"🚀 [SKELETAL MANIFOLD HARVEST] Device: {device}")

    # 1. Load Parquet
    print(f"📊 Loading Skeletal Parquet: {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    if num_episodes > 0:
        # Filter for top N episodes
        unique_eps = sorted(df["episode_index"].unique())[:num_episodes]
        df = df[df["episode_index"].isin(unique_eps)]
        print(f"✂️ Truncated to {num_episodes} episodes ({len(df)} frames)")

    # 2. Load Model & Patch
    mapper = GoalMapper(
        model_path=model_path,
        dataset_root=".",
        use_multi_view=use_multi_view,
        fusion_type=fusion_type,
        num_views=5 if use_multi_view else 1,
    )

    print("🩹 Patching model backbone for 4-channel skeletal input...")
    mapper.model.encoder.backbone = patch_vit_for_skeleton(
        mapper.model.encoder.backbone
    )

    # Reload weights to populate patched layer
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    new_sd = {k.replace("model.", ""): v for k, v in state_dict.items()}
    mapper.model.load_state_dict(new_sd, strict=False)

    model = mapper.model.to(device).eval()

    # 3. Data Loader
    dataset = SkeletalHarvestDataset(df, mapper.transform, use_multi_view)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_latents = []
    all_indices = []
    all_episode_indices = []

    print(f"🖼️ Harvesting latents for {len(df)} frames...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Harvesting"):
            pixels = batch["pixels"].to(device)
            actions = batch["action"].to(device)

            with (
                torch.amp.autocast("cuda") if "cuda" in str(device) else torch.no_grad()
            ):
                info = model.encode({"pixels": pixels, "action": actions})
                emb = info["emb"]  # (B, T, D)

            # Flatten T=1: (B, D)
            latents_flat = emb.squeeze(1).cpu().numpy().astype(np.float32)
            all_latents.append(latents_flat)

            all_indices.append(batch["frame_index"].numpy())
            all_episode_indices.append(batch["episode_index"].numpy())

    # 4. Save
    latents = np.concatenate(all_latents, axis=0)
    indices = np.concatenate(all_indices, axis=0)
    episode_indices = np.concatenate(all_episode_indices, axis=0)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "latents": latents,
        "frame_indices": indices,
        "episode_indices": episode_indices,
    }
    torch.save(data, output_path)
    print(f"✨ Skeletal Manifold saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, help="Path to v31 checkpoint"
    )
    parser.add_argument(
        "--parquet", type=str, required=True, help="Path to dataset_skel.parquet"
    )
    parser.add_argument("--output", type=str, default="manifold_skel_v31.pt")
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--fusion", type=str, default="linear")
    args = parser.parse_args()

    harvest_manifold_skel(
        model_path=args.model,
        parquet_path=args.parquet,
        output_file=args.output,
        num_episodes=args.episodes,
        batch_size=args.batch_size,
        fusion_type=args.fusion,
    )
