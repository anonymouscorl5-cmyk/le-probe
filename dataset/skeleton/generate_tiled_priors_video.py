"""
generate_tiled_priors_video.py

Generates a beautiful side-by-side composite video (and high-quality GIF) showing:
1. Raw RGB (world_center)
2. Projected Skeleton Prior
3. Tiny DINOv3 Spatial-Semantic Attention Map (Colored with Inferno)
stacked horizontally side-by-side for the first episode (file-000.mp4).
"""

import os
import sys
import torch
import cv2
import numpy as np
import timm
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

# --- Setup Paths ---
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

RGB_VIDEO_PATH = "/Users/vedpatwardhan/Desktop/cortex-os/le-probe/datasets/vedpatwardhan/gr1_pickup_grasp/videos/observation.images.world_center/chunk-000/file-000.mp4"
SKEL_VIDEO_PATH = "/Users/vedpatwardhan/Desktop/cortex-os/le-probe/datasets/vedpatwardhan/gr1_pickup_grasp/videos/observation.images.world_center_skeleton/chunk-000/file-000.mp4"

OUT_MP4_PATH = (
    "/Users/vedpatwardhan/Desktop/cortex-os/le-probe/assets/dino_skeletal_priors.mp4"
)
OUT_GIF_PATH = (
    "/Users/vedpatwardhan/Desktop/cortex-os/le-probe/assets/dino_skeletal_priors.gif"
)


def main():
    print("🚀 Loading tiny DINOv3 model (vit_small_patch16_dinov3)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(
        "vit_small_patch16_dinov3", pretrained=True, num_classes=0
    )
    model = model.to(device)
    model.eval()
    print("✅ DINOv3 model loaded successfully!")

    # Open both videos
    cap_rgb = cv2.VideoCapture(RGB_VIDEO_PATH)
    cap_skel = cv2.VideoCapture(SKEL_VIDEO_PATH)

    fps = cap_rgb.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap_rgb.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🎬 Processing video: {total_frames} frames @ {fps} FPS")

    # Hook to capture attention weights from final transformer block
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

    # Standard image transforms for DINOv3
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    tmp_raw = (
        "/Users/vedpatwardhan/Desktop/cortex-os/le-probe/assets/temp_tiled_priors.mp4"
    )

    # 3 frames side-by-side: each 480x480 -> Total width = 1440, Total height = 480
    video_writer = cv2.VideoWriter(
        tmp_raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (1440, 480), isColor=True
    )

    for f_idx in tqdm(range(total_frames), desc="Tiling Frames"):
        ret_rgb, frame_rgb = cap_rgb.read()
        ret_skel, frame_skel = cap_skel.read()

        if not ret_rgb or not ret_skel:
            break

        # 1. Standardize RGB frame shape (BGR color format)
        frame_rgb_resized = cv2.resize(frame_rgb, (480, 480))

        # 2. Standardize Skeleton frame shape (BGR color format)
        frame_skel_resized = cv2.resize(frame_skel, (480, 480))

        # 3. Compute DINOv3 Attention Map on RGB frame
        # Convert BGR to RGB PIL image for PyTorch transforms
        pil_img = Image.fromarray(cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB))
        img_tensor = transform(pil_img).unsqueeze(0).to(device)

        attn_weights.clear()
        with torch.no_grad():
            _ = model(img_tensor)

        # Retrieve attention map for the CLS token, skipping CLS (index 0) and the 4 register tokens (indices 1-4)
        attn = attn_weights[0][0]  # Shape: (num_heads, N, N)
        cls_attn = attn[:, 0, 5:]  # Shape: (num_heads, 196)

        # Mean across all self-attention heads
        mean_attn = cls_attn.mean(dim=0).reshape(14, 14).numpy()

        # Min-max normalization to [0, 1]
        mean_attn = (mean_attn - mean_attn.min()) / (
            mean_attn.max() - mean_attn.min() + 1e-8
        )

        # Scale to uint8 [0, 255]
        mean_attn_u8 = (mean_attn * 255).astype(np.uint8)

        # Resize to match standard 480x480 panel size
        mean_attn_resized = cv2.resize(mean_attn_u8, (480, 480))

        # Apply Inferno Colormap for gorgeous visual consistency
        dino_colormap = cv2.applyColorMap(mean_attn_resized, cv2.COLORMAP_INFERNO)

        # 4. Horizontally stack all three columns
        stacked_frame = np.hstack(
            [frame_rgb_resized, frame_skel_resized, dino_colormap]
        )

        # Write to final video
        video_writer.write(stacked_frame)

    # Cleanup resources
    hook.remove()
    cap_rgb.release()
    cap_skel.release()
    video_writer.release()

    # 5. Compress using ffmpeg for maximum network and web rendering efficiency
    print("🎬 Rendering highly compressed web-native MP4 using ffmpeg...")
    os.system(
        f"ffmpeg -y -i {tmp_raw} -vcodec libx264 -crf 18 -pix_fmt yuv420p {OUT_MP4_PATH} > /dev/null 2>&1"
    )

    print("🖼️ Generating high-fidelity visual GIF for GitHub rendering...")
    os.system(
        f'ffmpeg -y -i {OUT_MP4_PATH} -vf "fps=10,scale=1440:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 {OUT_GIF_PATH} > /dev/null 2>&1'
    )

    # Remove temporary raw file
    if os.path.exists(tmp_raw):
        os.remove(tmp_raw)

    print(f"🎉 Tiled priors video successfully generated at: {OUT_MP4_PATH}")
    print(f"🎉 Tiled priors GIF successfully generated at: {OUT_GIF_PATH}")


if __name__ == "__main__":
    main()
