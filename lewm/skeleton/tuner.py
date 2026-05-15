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
from lewm.multi_view_encoder import LateFusionEncoder
from lewm.skeleton.encoder import patch_vit_for_skeleton
from module import ARPredictor, SIGReg


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

    def __getitem__(self, idx):
        frame_path = self.frames_dir / f"frame_{idx:06d}.pt"
        frame_tensors = torch.load(frame_path, weights_only=False)
        views = []
        for vn in self.cam_keys:
            pixel_4ch = frame_tensors[vn]
            rgb = pixel_4ch[:3]
            skel = pixel_4ch[3:]
            if self.transform:
                transformed_rgb = self.transform(rgb)
            else:
                transformed_rgb = rgb.float() / 255.0
            skel_float = skel.float() / 255.0
            views.append(torch.cat([transformed_rgb, skel_float], dim=0))
        img_pixels = torch.stack(views, dim=0).unsqueeze(0)
        reward = torch.tensor([self.metadata["progress"][idx]], dtype=torch.float32)
        return img_pixels, reward


def train_reward_head_skel(
    checkpoint_path, frames_dir, epochs=20, lr=1e-4, batch_size=32
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 [SKELETAL REWARD TUNING] Device: {device}")

    # 1. Architecture Assembly
    print("🏗️  Building Multi-View Architecture...")
    backbone = spt.backbone.utils.vit_hf(
        "tiny", patch_size=14, image_size=224, pretrained=False, use_mask_token=False
    )
    backbone = patch_vit_for_skeleton(backbone)
    encoder = LateFusionEncoder(backbone, embed_dim=192, fusion="linear", num_views=5)

    predictor = ARPredictor(
        num_frames=3,
        depth=6,
        heads=16,
        mlp_dim=2048,
        input_dim=192,
        hidden_dim=192,
        output_dim=192,
        dim_head=64,
        dropout=0.1,
    )

    action_encoder = GR1Embedder(input_dim=32, smoothed_dim=256, emb_dim=192)
    projector = GR1MLP(input_dim=192, output_dim=192, hidden_dim=2048)
    pred_proj = GR1MLP(input_dim=192, output_dim=192, hidden_dim=2048)

    model = MultiViewJEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj,
    )
    model.reward_head = RewardPredictor(input_dim=192, hidden_dim=512)

    # 2. Load Weights
    cp = Path(checkpoint_path).absolute()
    if not cp.exists():
        print(f"❌ ERROR: Checkpoint not found at: {cp}")
        sys.exit(1)

    print(f"🧠 Loading Weights: {cp.name}")
    checkpoint = torch.load(str(cp), map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    new_sd = {k.replace("model.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(new_sd, strict=False)
    print(
        f"✅ Loaded weights. Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}"
    )

    model = model.to(device)

    # 3. Freeze & Setup
    for param in model.parameters():
        param.requires_grad = False
    for param in model.reward_head.parameters():
        param.requires_grad = True

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = SkeletalFrameDataset(frames_dir, transform)
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [int(0.9 * len(dataset)), len(dataset) - int(0.9 * len(dataset))]
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    optimizer = torch.optim.AdamW(model.reward_head.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print(f"📈 Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            pixels, target = [x.to(device) for x in batch]

            optimizer.zero_grad()
            # Forward: Encoder -> Predictor Projector -> Reward Head
            enc_out = model.encoder(pixels).last_hidden_state  # (B, T, D)
            b, t, d = enc_out.shape
            features = rearrange(enc_out, "b t d -> (b t) d")
            features = model.pred_proj(features)
            features = rearrange(features, "(b t) d -> b t d", b=b, t=t)

            pred = model.reward_head(features).squeeze(-1)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                pixels, target = [x.to(device) for x in batch]
                enc_out = model.encoder(pixels).last_hidden_state
                b, t, d = enc_out.shape
                features = rearrange(enc_out, "b t d -> (b t) d")
                features = model.pred_proj(features)
                features = rearrange(features, "(b t) d -> b t d", b=b, t=t)

                pred = model.reward_head(features).squeeze(-1)
                val_loss += criterion(pred, target).item()
        print(
            f"✅ Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.6f} | Val Loss: {val_loss/len(val_loader):.6f}"
        )

    # 4. Save
    output_path = "gr1_reward_tuned_v31_skel.ckpt"
    full_ckpt = torch.load(str(cp), map_location="cpu", weights_only=False)
    full_ckpt["state_dict"] = {
        f"model.{k}": v.cpu() for k, v in model.state_dict().items()
    }
    torch.save(full_ckpt, output_path)
    print(f"✨ Skeletal Reward Model saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--frames", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    train_reward_head_skel(
        args.ckpt, args.frames, args.epochs, args.lr, args.batch_size
    )
