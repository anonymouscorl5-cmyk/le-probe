import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms
import argparse
import stable_pretraining as spt
from stable_pretraining import data as dt
from einops import rearrange

# --- Path Stabilization ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

LEWM_DIR = os.path.join(REPO_ROOT, "lewm")
LEWM_ROOT = os.path.join(LEWM_DIR, "le_wm")
for p in [LEWM_DIR, LEWM_ROOT]:
    if p not in sys.path:
        sys.path.append(p)

from lewm.gr1_modules import MultiViewJEPA, GR1Embedder, GR1MLP
from lewm.train_lewm import RewardPredictor
from lewm.skeleton.encoder import patch_vit_for_skeleton
from lewm.skeleton.skeletal_utils import load_skeletal_state_dict, reconstruct_4ch_frame
from lewm.multi_view_encoder import get_multi_view_encoder
from omegaconf import OmegaConf


class SkeletalFrameDataset(Dataset):
    def __init__(self, frames_dir, transform=None):
        self.frames_dir = Path(frames_dir)
        self.transform = transform
        self.metadata = torch.load(self.frames_dir / "metadata.pt", weights_only=False)
        self.num_frames = len(self.metadata["progress"])
        self.cam_keys = (
            "world_center",
            "world_left",
            "world_right",
            "world_top",
            "world_wrist",
        )

    def __len__(self):
        return self.num_frames

    def _transform_adapter(self, batch):
        if self.transform:
            return self.transform(batch)
        return batch

    def __getitem__(self, idx):
        frame_path = self.frames_dir / f"frame_{idx:06d}.pt"
        frame_tensors = torch.load(frame_path, weights_only=False)
        views = []
        for vn in self.cam_keys:
            pixel_4ch = frame_tensors[vn]
            # Use centralized reconstruct_4ch_frame
            views.append(
                reconstruct_4ch_frame(pixel_4ch, transform_fn=self._transform_adapter)
            )

        img_pixels = torch.stack(views, dim=0).unsqueeze(0)
        reward = torch.tensor([self.metadata["progress"][idx]], dtype=torch.float32)
        return img_pixels, reward


def train_reward_head(
    model_path,
    frames_dir,
    epochs=10,
    lr=1e-4,
    batch_size=32,
    use_multi_view=True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 [SKELETAL TUNER] Device: {device}")

    # 1. Setup Model
    # Load base JEPA and patch it

    # Simple config for initialization
    cfg = OmegaConf.create(
        {
            "use_multi_view": use_multi_view,
            "fusion": "linear",
            "num_views": 5 if use_multi_view else 1,
            "img_size": 224,
            "backbone": "vit_tiny_patch16_224",
        }
    )

    # Define standard ImageNet transform directly to avoid GoalMapper dependency crash
    imagenet_stats = dt.dataset_stats.ImageNet
    transform = dt.transforms.Compose(
        dt.transforms.ToImage(**imagenet_stats, source="pixels", target="pixels"),
        dt.transforms.Resize(224, source="pixels", target="pixels"),
    )

    encoder = get_multi_view_encoder(cfg)
    patch_vit_for_skeleton(encoder.backbone)

    world_model = MultiViewJEPA(
        encoder=encoder,
        predictor=None,  # Not needed for reward tuning
        action_encoder=None,
        projector=None,
        pred_proj=None,
    )
    world_model.reward_head = RewardPredictor(
        input_dim=encoder.config.hidden_size, hidden_dim=512
    )

    # Load weights (handles model. prefix)
    sd = load_skeletal_state_dict(model_path, device=device)
    world_model.load_state_dict(sd, strict=False)
    world_model.to(device)

    # 2. Load Dataset
    dataset = SkeletalFrameDataset(frames_dir, transform=transform)
    num_train = int(len(dataset) * 0.9)
    num_val = len(dataset) - num_train
    train_set, val_set = torch.utils.data.random_split(
        dataset, [num_train, num_val], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # 3. Tuning Loop
    optimizer = torch.optim.Adam(world_model.reward_head.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print(f"🔥 Starting Skeletal Tuning | Epochs: {epochs}")
    for epoch in range(epochs):
        world_model.train()
        train_loss = 0
        for pixels, progress in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            pixels = pixels.to(device)  # (B, 1, V, C, H, W)
            progress = progress.to(device).float()  # (B, 1)

            optimizer.zero_grad()
            with torch.no_grad():
                # JEPA.encode expects (B, T, V, C, H, W)
                emb = world_model.encode({"pixels": pixels})["emb"]  # (B, 1, D)
                emb = emb.squeeze(1)

            pred = world_model.reward_head(emb)
            loss = criterion(pred, progress)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        world_model.eval()
        val_loss = 0
        with torch.no_grad():
            for pixels, progress in val_loader:
                pixels = pixels.to(device)
                progress = progress.to(device).float()
                emb = world_model.encode({"pixels": pixels})["emb"].squeeze(1)
                pred = world_model.reward_head(emb)
                val_loss += criterion(pred, progress).item()

        print(
            f"📈 Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.6f} | Val Loss: {val_loss/len(val_loader):.6f}"
        )

    # 4. Save Artifact
    out_name = Path(model_path).stem + "_reward_calibrated.ckpt"
    torch.save(world_model.state_dict(), out_name)
    print(f"✨ Calibrated Skeletal Model saved: {out_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--frames", type=str, required=True)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    train_reward_head(args.model, args.frames, epochs=args.epochs, lr=args.lr)
