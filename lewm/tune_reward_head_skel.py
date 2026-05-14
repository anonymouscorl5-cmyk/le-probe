import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from pathlib import Path
from huggingface_hub import snapshot_download
import pandas as pd

# --- Path Stabilization ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# --------------------------

from lewm.goal_mapper import GoalMapper
from lewm.skeleton.encoder import patch_vit_for_skeleton


def train_reward_head_skel(
    checkpoint_path,
    repo_id,
    epochs=20,
    lr=1e-4,
    batch_size=32,
    use_multi_view=True,
    fusion_type="linear",
):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"🚀 [SKELETAL REWARD TUNING] Training on {device}")

    # 1. Load Dataset
    local_dir = "gr1_reward_pred_data"
    # We expect the upgraded parquet file here
    parquet_file = Path(local_dir) / "dataset_skel.parquet"

    if not parquet_file.exists():
        print(f"📂 Upgraded dataset missing. Fetching base from HF: {repo_id}...")
        snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=local_dir)
        # Note: User must run generate_reward_priors.py if dataset_skel.parquet is still missing
        if not parquet_file.exists():
            # Check if it was just downloaded as dataset.parquet
            base_parquet = Path(local_dir) / "dataset.parquet"
            if base_parquet.exists():
                print(
                    "⚠️ Found base dataset.parquet. Please run generate_reward_priors.py first!"
                )
                sys.exit(1)

    print(f"📊 Loading Skeletal Parquet: {parquet_file}...")
    df = pd.read_parquet(parquet_file)

    # 2. Initialize Mapper & Model
    mapper = GoalMapper(
        model_path=checkpoint_path,
        dataset_root=local_dir,
        use_multi_view=use_multi_view,
        fusion_type=fusion_type,
        num_views=5 if use_multi_view else 1,
    )

    # 🩹 SKELETAL PATCH: Expand the model to 4 channels
    print("🩹 Patching model backbone for 4-channel skeletal input...")
    mapper.model.encoder.backbone = patch_vit_for_skeleton(
        mapper.model.encoder.backbone
    )

    # Re-load weights to ensure the patched layer gets the trained values
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    new_sd = {k.replace("model.", ""): v for k, v in state_dict.items()}
    mapper.model.load_state_dict(new_sd, strict=False)

    model = mapper.model.to(device)

    # Freeze everything except the Reward Head
    for param in model.parameters():
        param.requires_grad = False
    for param in model.reward_head.parameters():
        param.requires_grad = True

    # 3. Flexible N-Channel Dataset
    class SkeletalParquetDataset(Dataset):
        def __init__(self, dataframe, transform):
            self.df = dataframe
            self.transform = transform

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            cam_keys = (
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

            views = []
            for vn in cam_keys:
                raw_img = row[f"observation.images.{vn}"]
                # Stack all available channels (Expects 4: R, G, B, S)
                img_np = np.stack(
                    [np.array(c.tolist(), dtype=np.uint8) for c in raw_img], axis=-1
                )

                # Apply transform (Note: Transform must be channel-agnostic or handle 4ch)
                # Our get_img_preprocessor handles this by treating 'pixels' as a tensor
                transformed = self.transform({"pixels": img_np})
                views.append(transformed["pixels"])

            img_pixels = torch.stack(views, dim=0).unsqueeze(0)  # (1, V, C, H, W)
            return img_pixels, torch.tensor([row["progress"]], dtype=torch.float32)

    dataset = SkeletalParquetDataset(df, mapper.transform)
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [int(0.9 * len(dataset)), len(dataset) - int(0.9 * len(dataset))]
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    optimizer = optim.AdamW(model.reward_head.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # 4. Training Loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for imgs, rewards in pbar:
            imgs, rewards = imgs.to(device), rewards.to(device)
            optimizer.zero_grad()
            info = model.encode({"pixels": imgs})
            pred_reward = model.reward_head(info["emb"]).squeeze(-1)
            loss = criterion(pred_reward, rewards)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, rewards in val_loader:
                imgs, rewards = imgs.to(device), rewards.to(device)
                info = model.encode({"pixels": imgs})
                pred_reward = model.reward_head(info["emb"]).squeeze(-1)
                val_loss += criterion(pred_reward, rewards).item()
        print(
            f"✅ Epoch {epoch+1} Complete. Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}"
        )

    # 5. Save Final Checkpoint
    output_path = "gr1_reward_tuned_v31_skel.ckpt"
    full_ckpt = torch.load(checkpoint_path, map_location="cpu")
    full_ckpt["state_dict"] = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(full_ckpt, output_path)
    print(f"💾 Skeletal Reward Model saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--snapshots", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    train_reward_head_skel(args.ckpt, args.snapshots, args.epochs)
