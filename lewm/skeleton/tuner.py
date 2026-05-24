"""
Retune only ``reward_head`` on harvested multi-view skeleton frames.

Forward path: 4ch pixels -> encoder -> reward_head (no DINO anchors in the loop).
The checkpoint must still instantiate the same module tree as training (strict load).

Typical pipeline:
  1. generate_reward_priors.py --repo_id vedpatwardhan/gr1_reward_pred_v2
  2. audit_priors.py --repo_id vedpatwardhan/gr1_reward_pred_v2 --frames dataset_skel_frames
  3. tuner.py --model <gr1_grasp_skeleton_dino_v7>/epoch=99-step=17900.ckpt --frames dataset_skel_frames
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import argparse
from stable_pretraining import data as dt

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
from module import ARPredictor
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


def _detect_ckpt_arch(state_dict: dict) -> tuple[bool, str]:
    """Infer use_dino / fusion_type from checkpoint keys (for strict load parity)."""
    keys = list(state_dict.keys())
    use_dino = any(
        marker in k
        for k in keys
        for marker in ("dino_projector", "dino_fusion_layer", "high_level_predictor")
    )
    fusion_type = "linear" if any("dino_fusion_layer" in k for k in keys) else "mean"
    return use_dino, fusion_type


def resolve_arch(
    state_dict: dict,
    use_dino: bool | None,
    fusion_type: str | None,
) -> tuple[bool, str]:
    detected_dino, detected_fusion = _detect_ckpt_arch(state_dict)
    if use_dino is None:
        use_dino = detected_dino
    elif use_dino != detected_dino:
        raise ValueError(
            f"use_dino={use_dino} but checkpoint "
            f"{'contains' if detected_dino else 'does not contain'} DINO modules. "
            "Omit --use_dino / --no_use_dino to auto-detect."
        )
    if fusion_type is None:
        fusion_type = detected_fusion if use_dino else "mean"
    return use_dino, fusion_type


def train_reward_head(
    model_path,
    frames_dir,
    epochs=10,
    lr=1e-4,
    batch_size=32,
    use_multi_view=True,
    use_dino: bool | None = None,
    fusion_type: str | None = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 [SKELETAL TUNER] Device: {device}")

    # 1. Load checkpoint keys first so the module tree matches (strict=True).
    sd = load_skeletal_state_dict(model_path, device=device)
    use_dino, fusion_type = resolve_arch(sd, use_dino, fusion_type)
    num_views = 5 if use_multi_view else 1
    print(
        f"📦 Checkpoint arch: use_dino={use_dino}, fusion_type={fusion_type}, "
        f"num_views={num_views} (reward_head-only training; DINO not used in forward)"
    )

    cfg = OmegaConf.create(
        {
            "use_multi_view": use_multi_view,
            "fusion_type": fusion_type,
            "num_views": num_views,
            "img_size": 224,
            "patch_size": 14,
            "encoder_scale": "tiny",
            "backbone": "vit_tiny_patch14_224",
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
        predictor=ARPredictor(
            num_frames=3,
            input_dim=192,
            hidden_dim=encoder.config.hidden_size,
            output_dim=encoder.config.hidden_size,
            depth=6,
            heads=16,
            mlp_dim=2048,
        ),
        action_encoder=GR1Embedder(input_dim=32, emb_dim=192),
        projector=GR1MLP(
            input_dim=encoder.config.hidden_size, output_dim=192, hidden_dim=2048
        ),
        pred_proj=GR1MLP(
            input_dim=encoder.config.hidden_size, output_dim=192, hidden_dim=2048
        ),
        use_dino=use_dino,
        fusion_type=fusion_type,
        num_views=num_views,
    )
    world_model.reward_head = RewardPredictor(input_dim=192, hidden_dim=512)

    world_model.load_state_dict(sd, strict=True)
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

    # 4. Save Artifact (Preserve full Lightning Checkpoint structure for 213MB parity)
    raw_ckpt = torch.load(model_path, map_location=device)
    current_sd = world_model.state_dict()

    # Surgical injection into the correct key ('state_dict' or root)
    target_sd = raw_ckpt.get("state_dict", raw_ckpt)

    # Sync keys (handle model. prefix)
    reward_keys = [k for k in current_sd if "reward_head" in k]
    updated_keys = 0
    for k in reward_keys:
        target_k = k if k in target_sd else f"model.{k}"
        if target_k in target_sd:
            target_sd[target_k] = current_sd[k]
            updated_keys += 1

    out_name = Path(model_path).stem + "_reward_calibrated.ckpt"
    torch.save(raw_ckpt, out_name)
    print(f"✨ Calibrated checkpoint saved: {out_name}")
    print(
        f"✅ Injection audit: updated {updated_keys}/{len(reward_keys)} reward_head tensors "
        "(all other weights, including DINO if present, preserved from input ckpt)."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retune reward_head on harvested skeleton frames; auto-detects DINO ckpt layout."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Lightning ckpt from skeleton trainer (e.g. gr1_grasp_skeleton_dino_v7 epoch=99-step=17900.ckpt)",
    )
    parser.add_argument(
        "--frames",
        type=str,
        required=True,
        help="Directory from generate_reward_priors.py (e.g. dataset_skel_frames)",
    )
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    arch = parser.add_mutually_exclusive_group()
    arch.add_argument(
        "--use_dino",
        action="store_true",
        help="Force DINO module tree (normally auto-detected from ckpt)",
    )
    arch.add_argument(
        "--no_use_dino",
        action="store_true",
        help="Force skeleton-only module tree (normally auto-detected)",
    )
    args = parser.parse_args()
    use_dino = True if args.use_dino else False if args.no_use_dino else None

    train_reward_head(
        args.model,
        args.frames,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        use_dino=use_dino,
    )
