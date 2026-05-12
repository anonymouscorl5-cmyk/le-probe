# --- Path Stabilization ---
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# --------------------------

import os
import argparse
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from pathlib import Path
from huggingface_hub import snapshot_download
import sys
import pandas as pd

# Add project root to sys.path for absolute imports on Colab
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from goal_mapper import GoalMapper


def train_reward_head(
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
    print(f"🚀 Training on {device}")

    # 1. Load Dataset (Local first, then HF)
    local_dir = "gr1_reward_pred_data"
    parquet_file = Path(local_dir) / "dataset.parquet"

    if not parquet_file.exists():
        print(f"📂 Dataset not found locally. Fetching from HF: {repo_id}...")
        snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=local_dir)

    print(f"📊 Loading Parquet Dataset: {parquet_file}...")
    df = pd.read_parquet(parquet_file)
    print(f"📦 Dataset Loaded: {len(df)} samples.")

    # 2. Initialize Mapper & Model
    mapper = GoalMapper(
        model_path=checkpoint_path,
        dataset_root=local_dir,
        use_multi_view=use_multi_view,
        fusion_type=fusion_type,
        num_views=5 if use_multi_view else 1,
    )
    model = mapper.model.to(device)

    # Freeze everything except the Reward Head
    for param in model.parameters():
        param.requires_grad = False
    for param in model.reward_head.parameters():
        param.requires_grad = True

    # 3. Create PyTorch Dataset & DataLoader for Batching
    class ParquetDataset(Dataset):
        def __init__(self, dataframe, transform):
            self.df = dataframe
            self.transform = transform

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]

            if use_multi_view:
                # Load all 5 cameras
                cam_keys = [
                    "observation.images.world_center",
                    "observation.images.world_left",
                    "observation.images.world_right",
                    "observation.images.world_top",
                    "observation.images.world_wrist",
                ]
                views = []
                for k in cam_keys:
                    raw_img = row[k]
                    img_np = np.stack(
                        [
                            np.array(raw_img[0].tolist(), dtype=np.uint8),
                            np.array(raw_img[1].tolist(), dtype=np.uint8),
                            np.array(raw_img[2].tolist(), dtype=np.uint8),
                        ],
                        axis=-1,
                    )
                    # Apply transform to EACH view individually to match ToImage expectations
                    transformed = self.transform({"pixels": img_np})
                    views.append(transformed["pixels"])

                # Stack to (V, C, H, W)
                img_pixels = torch.stack(views, dim=0)
                # Add T=1 dimension: (1, V, C, H, W)
                img_pixels = img_pixels.unsqueeze(0)
            else:
                raw_img = row["observation.images.world_center"]
                img_np = np.stack(
                    [
                        np.array(raw_img[0].tolist(), dtype=np.uint8),
                        np.array(raw_img[1].tolist(), dtype=np.uint8),
                        np.array(raw_img[2].tolist(), dtype=np.uint8),
                    ],
                    axis=-1,
                )
                transformed = self.transform({"pixels": img_np})
                # Add T=1 and V=1 dimensions: (1, 1, C, H, W)
                img_pixels = transformed["pixels"].unsqueeze(0).unsqueeze(0)

            return img_pixels, torch.tensor([row["progress"]], dtype=torch.float32)

    dataset = ParquetDataset(df, mapper.transform)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

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
            # imgs: (B, T=1, V, C, H, W), rewards: (B, 1)
            imgs, rewards = imgs.to(device), rewards.to(device)

            optimizer.zero_grad()
            info = model.encode({"pixels": imgs})
            pred_reward = model.reward_head(info["emb"]).squeeze(-1)

            loss = criterion(pred_reward, rewards)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Validation
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
    output_path = "gr1_reward_tuned_v2.ckpt"
    full_ckpt = torch.load(checkpoint_path, map_location="cpu")
    full_ckpt["state_dict"] = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(full_ckpt, output_path)
    print(f"💾 Spectrum-Calibrated Reward Head (V2) saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--snapshots", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--multi_view", action="store_true", default=True)
    parser.add_argument("--fusion", type=str, default="linear")
    args = parser.parse_args()

    train_reward_head(
        args.ckpt,
        args.snapshots,
        args.epochs,
        args.lr,
        args.batch_size,
        use_multi_view=args.multi_view,
        fusion_type=args.fusion,
    )
