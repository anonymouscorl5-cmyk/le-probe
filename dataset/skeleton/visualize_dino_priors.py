"""
visualize_dino_priors.py

Extracts and visualizes spatial-semantic attention maps from the tiny DINOv3 model (vit_small_patch16_dinov3)
across the 4 key static phase checkpoints of the first episode in gr1_pickup_grasp.
"""

import os
import sys
import torch
import cv2
import numpy as np
import timm
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# --- Setup Paths ---
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

VIDEO_PATH = "/workspace/cortex-os/le-probe/datasets/anonymouscorl5-cmyk/gr1_pickup_grasp/videos/observation.images.world_center/chunk-000/file-000.mp4"
OUTPUT_DIR = "/workspace/cortex-os/le-probe/dataset/skeleton/dino_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# 1. Attention Hook to capture the attention maps of the final transformer block
class AttentionHook:
    def __init__(self):
        self.attention_map = None

    def __call__(self, module, input, output):
        # TIMM Attention module: output is (output_features, attn_weights) or just features.
        # To get the raw attention weights, we hook the softmax output or recreate it.
        # Let's extract q and k to compute the attention weights robustly.
        pass


def hook_fn(module, input, output):
    # In timm, the output of the attention forward pass is the updated representation.
    # To get the attention weights, we can hook the softmax operation or compute it ourselves.
    pass


def main():
    print("🚀 Loading tiny DINOv3 model (vit_small_patch16_dinov3)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(
        "vit_small_patch16_dinov3", pretrained=True, num_classes=0
    )
    model = model.to(device)
    model.eval()
    print("✅ DINOv3 model loaded successfully!")

    # 2. Extract the specific checkpoint frames from the video (Frames 8, 16, 24, 32)
    checkpoint_frames = [8, 16, 24, 32]
    checkpoint_indices = [f - 1 for f in checkpoint_frames]

    cap = cv2.VideoCapture(VIDEO_PATH)
    frames = []
    f_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if f_idx in checkpoint_indices:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((f_idx + 1, rgb_frame))
        f_idx += 1
    cap.release()
    print(
        f"🎬 Successfully extracted {len(frames)} key frames at checkpoints: {[f[0] for f in frames]}"
    )

    # 3. Define standard image transformations
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Hook to capture self-attention matrix in the last attention layer
    # For timm ViT models, the last block is model.blocks[-1].attn
    attn_weights = []

    def get_attn_hook(module, input, output):
        # We compute attention weights directly from input to the Attention layer
        x = input[0]
        B, N, C = x.shape
        # Replicate QKV projection to get query and key
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

    # 4. Run inference and extract attention maps
    fig, axes = plt.subplots(len(frames), 2, figsize=(10, 16))

    for i, (frame_num, frame_rgb) in enumerate(frames):
        pil_img = Image.fromarray(frame_rgb)
        img_tensor = transform(pil_img).unsqueeze(0).to(device)

        # Reset hook weights list
        attn_weights.clear()

        with torch.no_grad():
            _ = model(img_tensor)

        # Get attention weights of shape (B, num_heads, N, N)
        # N = 1 + number of patches = 1 + (224/16)*(224/16) = 197 tokens
        attn = attn_weights[0][0]  # Batch index 0

        # Self-attention of the CLS token to all other patch tokens (excluding CLS and 4 registers)
        # CLS token is at index 0. Register tokens are at indices 1 to 4.
        # Patch tokens are at indices 5 to 200 (total of 196 patches).
        # Shape becomes (num_heads, 196)
        cls_attn = attn[:, 0, 5:]

        # Average attention map across all heads
        mean_attn = cls_attn.mean(dim=0).reshape(14, 14).numpy()

        # Resize attention map back to original frame size (224x224)
        mean_attn = cv2.resize(mean_attn, (224, 224))

        # Normalize for visualization
        mean_attn = (mean_attn - mean_attn.min()) / (mean_attn.max() - mean_attn.min())

        # Plot original frame
        axes[i, 0].imshow(pil_img.resize((224, 224)))
        axes[i, 0].set_title(f"Frame {frame_num} - Phase Checkpoint")
        axes[i, 0].axis("off")

        # Plot DINOv3 spatial attention heatmaps
        axes[i, 1].imshow(mean_attn, cmap="inferno")
        axes[i, 1].set_title(f"DINOv3 Tiny Spatial Attention Map")
        axes[i, 1].axis("off")

    hook.remove()

    output_path = os.path.join(OUTPUT_DIR, "episode_01_dinov3_priors.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(
        f"🎉 Spatial-semantic attention map audit successfully saved to: {output_path}!"
    )


if __name__ == "__main__":
    main()
