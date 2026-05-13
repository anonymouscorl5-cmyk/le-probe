import cv2
import numpy as np
import sys
from pathlib import Path


def verify_tiled_video(video_path, output_img="verification_tiled.png"):
    print(f"🧐 Verifying tiled video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("❌ Failed to read video frame.")
        return

    h, w, c = frame.shape
    print(f"📏 Frame Dimensions: {w}x{h} (Channels: {c})")

    if w != 960 or h != 480:
        print(f"⚠️ Warning: Expected 960x480, got {w}x{h}")

    mid = w // 2
    rgb = frame[:, :mid, :]
    skel = frame[:, mid:, :]

    # Create a diagnostic view: [RGB | Skel | Overlaid]
    # To overlay, we'll make the skeleton green
    skel_gray = cv2.cvtColor(skel, cv2.COLOR_BGR2GRAY)
    overlay = rgb.copy()
    overlay[skel_gray > 50] = [0, 255, 0]  # Bright green for skeleton lines

    diagnostic = np.hstack([rgb, skel, overlay])
    cv2.imwrite(output_img, diagnostic)
    print(f"✅ Verification image saved to: {output_img}")
    print(f"   [Left: RGB | Middle: Skeleton | Right: Green Overlay Alignment]")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_tiling.py <path_to_tiled_video>")
    else:
        verify_tiled_video(sys.argv[1])
