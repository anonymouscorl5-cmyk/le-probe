"""
generate_dino_priors.py

Pre-computes and caches high-fidelity zero-shot DINOv3 visual waypoint anchors
for the Hierarchical World Model (HWM). For each episode, extracts frozen DINO
embeddings at phase checkpoints (frames 8, 16, 24, 32) from all five camera views.

Output per episode: float32 tensor of shape [4, 5, 384] (phases × views × dim).
"""

import os
import sys
import torch
import cv2
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

from lewm.skeleton.dino_constants import (  # noqa: E402
    DINO_DIM,
    DINO_PHASE_CHECKPOINT_FRAMES,
    DINO_VIEW_KEYS,
    NUM_DINO_PHASES,
    dino_waypoints_shape,
)

# --------------------------


def extract_view_checkpoints(
    rgb_v_path: Path,
    model,
    transform,
    device,
    checkpoints: tuple[int, ...],
) -> list[torch.Tensor]:
    """Run frozen DINO on checkpoint frames for one camera view."""
    cap = cv2.VideoCapture(str(rgb_v_path))
    embeddings: list[torch.Tensor] = []
    frame_idx = 0
    needed = set(checkpoints)

    while True:
        ret, rgb_frame = cap.read()
        if not ret:
            break
        if frame_idx in needed:
            pil_img = Image.fromarray(cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB))
            img_tensor = transform(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model(img_tensor).squeeze(0).cpu()
                embeddings.append(embedding)
        frame_idx += 1

    cap.release()

    while len(embeddings) < NUM_DINO_PHASES:
        embeddings.append(torch.zeros(DINO_DIM))
    return embeddings[:NUM_DINO_PHASES]


def process_episode(ep_idx, dataset_path, model, transform, device):
    """
    Extract landmark frames from every training view, pass through frozen DINOv3,
    and cache [4, num_views, 384] float32 features.
    """
    out_dir = dataset_path / "cache_dino/chunk-000"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"file-{ep_idx:03d}_dino.pt"

    view_tensors = []
    for view in DINO_VIEW_KEYS:
        rgb_v_path = (
            dataset_path
            / f"videos/observation.images.{view}/chunk-000/file-{ep_idx:03d}.mp4"
        )
        if not rgb_v_path.exists():
            print(f"⚠️ Video missing for Episode {ep_idx} view {view}: {rgb_v_path}")
            view_tensors.append(torch.zeros(NUM_DINO_PHASES, DINO_DIM))
            continue

        per_phase = extract_view_checkpoints(
            rgb_v_path,
            model,
            transform,
            device,
            DINO_PHASE_CHECKPOINT_FRAMES,
        )
        if len(per_phase) < NUM_DINO_PHASES:
            print(
                f"⚠️ Episode {ep_idx} view {view}: truncated frames. "
                f"Padded to {NUM_DINO_PHASES} waypoints."
            )
        view_tensors.append(torch.stack(per_phase, dim=0))  # [4, 384]

    # [4, V, 384]
    stacked_embeddings = torch.stack(view_tensors, dim=1).float()
    assert stacked_embeddings.shape == dino_waypoints_shape(), stacked_embeddings.shape
    torch.save(stacked_embeddings, out_path)


def main(repo_id="vedpatwardhan/gr1_pickup_grasp"):
    print(f"📦 [DINO CACHE GENERATOR] Initializing: {repo_id}")
    print(f"   Views: {list(DINO_VIEW_KEYS)}")
    print(f"   Output shape per episode: {dino_waypoints_shape()}")
    dataset = LeRobotDataset(repo_id)
    dataset_path = Path(dataset.root)

    print("🚀 Loading tiny DINOv3 model (vit_small_patch16_dinov3)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(
        "vit_small_patch16_dinov3", pretrained=True, num_classes=0
    )
    model = model.to(device)
    model.eval()

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
        f"🎬 Pre-computing multi-view DINO waypoints for {dataset.num_episodes} "
        f"episodes on {device}..."
    )
    for ep_idx in tqdm(range(dataset.num_episodes), desc="DINO Caching"):
        process_episode(ep_idx, dataset_path, model, transform, device)

    print(f"🎉 Success! Cached DINO anchors saved to: {dataset_path / 'cache_dino'}")


if __name__ == "__main__":
    repo = sys.argv[1] if len(sys.argv) > 1 else "vedpatwardhan/gr1_pickup_grasp"
    main(repo)
