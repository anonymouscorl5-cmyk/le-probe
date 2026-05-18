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


def process_episode(ep_idx, dataset_path, views, model, transform, device):
    for view in views:
        rgb_v_path = (
            dataset_path
            / f"videos/observation.images.{view}/chunk-000/file-{ep_idx:03d}.mp4"
        )
        out_v_path = (
            dataset_path
            / f"videos/observation.images.{view}_dino_tiled/chunk-000/file-{ep_idx:03d}.mp4"
        )

        if not rgb_v_path.exists():
            continue

        cap = cv2.VideoCapture(str(rgb_v_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        tmp_raw = out_v_path.with_suffix(".raw.mp4")
        video = cv2.VideoWriter(
            str(tmp_raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (960, 480), isColor=True
        )

        # Hook to capture attention weights
        attn_weights = []

        def get_attn_hook(module, input, output):
            x = input[0]
            B, N, C = x.shape
            qkv = (
                module.qkv(x)
                .reshape(B, N, 3, module.num_heads, module.head_dim)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv.unbind(0)
            attn = (q @ k.transpose(-2, -1)) * module.scale
            attn = attn.softmax(dim=-1)
            attn_weights.append(attn.detach().cpu())

        hook = model.blocks[-1].attn.register_forward_hook(get_attn_hook)

        for _ in range(total_frames):
            ret, rgb_frame = cap.read()
            if not ret:
                break

            # 1. Prepare input for DINOv3
            pil_img = Image.fromarray(cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB))
            img_tensor = transform(pil_img).unsqueeze(0).to(device)

            # 2. Forward pass & extract attention maps
            attn_weights.clear()
            with torch.no_grad():
                _ = model(img_tensor)

            # Get attention weights of shape (B, num_heads, N, N)
            attn = attn_weights[0][0]  # Batch index 0

            # Self-attention of the CLS token to all patch tokens (excluding CLS and 4 registers)
            cls_attn = attn[:, 0, 5:]

            # Average attention map across all heads
            mean_attn = cls_attn.mean(dim=0).reshape(14, 14).numpy()

            # Normalize to [0, 1]
            mean_attn = (mean_attn - mean_attn.min()) / (
                mean_attn.max() - mean_attn.min() + 1e-8
            )

            # Scale to uint8 and resize to match standard 480x480 panel size
            mean_attn_u8 = (mean_attn * 255).astype(np.uint8)
            mean_attn_resized = cv2.resize(mean_attn_u8, (480, 480))

            # Apply Inferno Colormap for gorgeous consistent visuals
            dino_colormap = cv2.applyColorMap(mean_attn_resized, cv2.COLORMAP_INFERNO)

            # 3. Stack original and DINOv3 side by side (960x480)
            if rgb_frame.shape[:2] != (480, 480):
                rgb_frame = cv2.resize(rgb_frame, (480, 480))

            video.write(np.hstack([rgb_frame, dino_colormap]))

        cap.release()
        video.release()
        hook.remove()

        # Compress to final web-native web MP4
        os.system(
            f"ffmpeg -y -i {tmp_raw} -vcodec libx264 -crf 28 -preset ultrafast -pix_fmt yuv420p {out_v_path} > /dev/null 2>&1"
        )
        if tmp_raw.exists():
            tmp_raw.unlink()


def main(repo_id="vedpatwardhan/gr1_pickup_grasp"):
    print(f"📦 [DINO PRIOR GENERATOR] Initializing: {repo_id}")
    dataset = LeRobotDataset(repo_id)
    dataset_path = Path(dataset.root)
    views = ["world_center", "world_left", "world_right", "world_top", "world_wrist"]

    # Setup output directories
    for view in views:
        (dataset_path / f"videos/observation.images.{view}_dino_tiled/chunk-000").mkdir(
            parents=True, exist_ok=True
        )

    print("🚀 Loading tiny DINOv3 model (vit_small_patch16_dinov3)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(
        "vit_small_patch16_dinov3", pretrained=True, num_classes=0
    )
    model = model.to(device)
    model.eval()
    print("✅ DINOv3 model loaded successfully!")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    print(f"🎬 Processing {dataset.num_episodes} episodes sequentially on {device}...")
    for ep_idx in tqdm(range(dataset.num_episodes), desc="Episodes"):
        process_episode(ep_idx, dataset_path, views, model, transform, device)

    print("🎉 Success! All DINO priors generated successfully.")


if __name__ == "__main__":
    repo = sys.argv[1] if len(sys.argv) > 1 else "vedpatwardhan/gr1_pickup_grasp"
    main(repo)
