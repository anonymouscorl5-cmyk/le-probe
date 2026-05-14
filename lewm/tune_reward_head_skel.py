import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import numpy as np
from tqdm import tqdm
import argparse
import sys
import os

# --- Path Stabilization ---
CURRENT_FILE = Path(__file__).resolve()
ROOT_DIR = CURRENT_FILE.parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from lewm.goal_mapper import GoalMapper
from lewm.skeleton.encoder import patch_vit_for_skeleton


class SkeletalFrameDataset(Dataset):
    """
    Dataset that loads pre-computed [RGB+S] 4-channel frames from individual .pt files.
    """

    def __init__(self, frames_dir, transform, use_multi_view=True):
        self.frames_dir = Path(frames_dir)
        self.transform = transform
        self.use_multi_view = use_multi_view

        # Load metadata
        self.metadata = torch.load(self.frames_dir / "metadata.pt")
        self.num_frames = len(self.metadata["progress"])

        self.cam_keys = (
            ["world_center", "world_left", "world_right", "world_top", "world_wrist"]
            if use_multi_view
            else ["world_center"]
        )

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        # Load the pre-computed 4-channel tensors for this frame
        frame_path = self.frames_dir / f"frame_{idx:06d}.pt"
        frame_tensors = torch.load(frame_path)  # Dict of {view_name: Tensor[4, H, W]}

        views = []
        for vn in self.cam_keys:
            # frame_tensors[vn] is already [4, H, W]
            # Convert back to (H, W, 4) for the standard transform if needed,
            # or pass directly if transform supports it.
            # Most transforms expect PIL or numpy (H, W, C).
            img_np = frame_tensors[vn].permute(1, 2, 0).numpy()

            # Apply standard ViT preprocessing
            transformed = self.transform({"pixels": img_np})["pixels"]
            views.append(transformed)

        # Stack into (V, C, H, W) then add T=1: (1, V, C, H, W)
        img_pixels = torch.stack(views, dim=0).unsqueeze(0)
        reward = torch.tensor([self.metadata["progress"][idx]], dtype=torch.float32)

        return img_pixels, reward


def train_reward_head_skel(
    checkpoint_path, frames_dir, epochs=20, lr=1e-4, batch_size=32
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 [SKELETAL REWARD TUNING] Device: {device}")

    # 1. Initialize Mapper & Model
    mapper = GoalMapper(
        model_path=checkpoint_path,
        dataset_root=".",
        use_multi_view=True,
        fusion_type="linear",
        num_views=5,
    )

    print("🩹 Patching model backbone for 4-channel skeletal input...")
    mapper.model.encoder.backbone = patch_vit_for_skeleton(
        mapper.model.encoder.backbone
    )

    # Reload weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    new_sd = {k.replace("model.", ""): v for k, v in state_dict.items()}
    mapper.model.load_state_dict(new_sd, strict=False)

    model = mapper.model.to(device)

    # Freeze everything except reward head
    for param in model.parameters():
        param.requires_grad = False
    for param in model.reward_head.parameters():
        param.requires_grad = True

    # 2. Dataset & Loader
    dataset = SkeletalFrameDataset(frames_dir, mapper.transform)
    print(f"📊 Dataset initialized with {len(dataset)} frames from {frames_dir}")

    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [int(0.9 * len(dataset)), len(dataset) - int(0.9 * len(dataset))]
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=2)

    optimizer = optim.AdamW(model.reward_head.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # 3. Training Loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for imgs, rewards in pbar:
            imgs, rewards = imgs.to(device), rewards.to(device)

            optimizer.zero_grad()
            # Actions are not used for reward head tuning in this mode, pass dummy
            dummy_actions = torch.zeros(imgs.shape[0], 1, 6).to(device)

            with torch.amp.autocast("cuda"):
                pred_reward = model.predict_reward(
                    {"pixels": imgs, "action": dummy_actions}
                )
                loss = criterion(pred_reward, rewards)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, rewards in val_loader:
                imgs, rewards = imgs.to(device), rewards.to(device)
                dummy_actions = torch.zeros(imgs.shape[0], 1, 6).to(device)
                pred = model.predict_reward({"pixels": imgs, "action": dummy_actions})
                val_loss += criterion(pred, rewards).item()

        print(
            f"✅ Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.6f} | Val Loss: {val_loss/len(val_loader):.6f}"
        )

    # 4. Save
    output_path = "gr1_reward_tuned_v31_skel.ckpt"
    full_ckpt = torch.load(checkpoint_path, map_location="cpu")
    full_ckpt["state_dict"] = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(full_ckpt, output_path)
    print(f"✨ Skeletal Reward Model saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to v31 backbone")
    parser.add_argument(
        "--frames", type=str, required=True, help="Path to dataset_skel_frames dir"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()

    train_reward_head_skel(args.ckpt, args.frames, args.epochs, args.lr)
