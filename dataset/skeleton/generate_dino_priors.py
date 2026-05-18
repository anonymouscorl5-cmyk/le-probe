"""
generate_dino_priors.py

Pre-computes and caches high-fidelity zero-shot DINOv3 visual waypoint anchors
for the Hierarchical World Model (HWM). By pre-calculating embeddings at static
checkpoints (Frames 8, 16, 24, 32), we completely eliminate visual forward pass
overhead and frozen model VRAM consumption during training.
"""

import os
import sys
import torch
import cv2
import numpy as np
import timm
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# --- Path Stabilization ---
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
# --------------------------


def process_episode(ep_idx, dataset_path, model, transform, device):
    """
    Extracts landmark frames from the world_center view, passes them through
    the frozen DINOv3 visual backbone, and caches the flat 384-dimensional features.
    """
    rgb_v_path = (
        dataset_path
        / f"videos/observation.images.world_center/chunk-000/file-{ep_idx:03d}.mp4"
    )
    out_dir = dataset_path / "cache_dino/chunk-000"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"file-{ep_idx:03d}_dino.pt"

    if not rgb_v_path.exists():
        print(f"⚠️ Video missing for Episode {ep_idx}: {rgb_v_path}")
        return

    # Checkpoint frame indices (0-based) for Frames 8, 16, 24, 32
    checkpoints = [7, 15, 23, 31]
    cap = cv2.VideoCapture(str(rgb_v_path))

    embeddings = []
    frame_idx = 0

    while True:
        ret, rgb_frame = cap.read()
        if not ret:
            break

        if frame_idx in checkpoints:
            # Prepare image
            pil_img = Image.fromarray(cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB))
            img_tensor = transform(pil_img).unsqueeze(0).to(device)

            # Forward pass through frozen DINOv3
            with torch.no_grad():
                # vit_small outputs shape [1, 384] for pooled output (num_classes=0)
                embedding = model(img_tensor).squeeze(0).cpu()
                embeddings.append(embedding)

        frame_idx += 1

    cap.release()

    # Integrity check: Ensure all 4 checkpoints were extracted
    if len(embeddings) < 4:
        # Fallback padding if video ended prematurely
        while len(embeddings) < 4:
            embeddings.append(torch.zeros(384))
        print(
            f"⚠️ Episode {ep_idx} had truncated frames ({frame_idx}). Padded to 4 waypoints."
        )

    # Stack into a [4, 384] float32 tensor
    stacked_embeddings = torch.stack(embeddings, dim=0).float()
    torch.save(stacked_embeddings, out_path)


def main(repo_id="vedpatwardhan/gr1_pickup_grasp"):
    print(f"📦 [DINO CACHE GENERATOR] Initializing: {repo_id}")
    dataset = LeRobotDataset(repo_id)
    dataset_path = Path(dataset.root)

    print("🚀 Loading tiny DINOv3 model (vit_small_patch16_dinov3)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(
        "vit_small_patch16_dinov3", pretrained=True, num_classes=0
    )
    model = model.to(device)
    model.eval()

    # Freeze parameters completely
    for p in model.parameters():
        p.requires_grad = False
    print("✅ DINOv3 model loaded and frozen successfully!")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    print(
        f"🎬 Pre-computing DINO visual waypoints for {dataset.num_episodes} episodes on {device}..."
    )
    for ep_idx in tqdm(range(dataset.num_episodes), desc="DINO Caching"):
        process_episode(ep_idx, dataset_path, model, transform, device)

    print(f"🎉 Success! Cached DINO anchors saved to: {dataset_path / 'cache_dino'}")


if __name__ == "__main__":
    repo = sys.argv[1] if len(sys.argv) > 1 else "vedpatwardhan/gr1_pickup_grasp"
    main(repo)
