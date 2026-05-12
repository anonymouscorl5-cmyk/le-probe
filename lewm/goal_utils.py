"""
GOAL PERCEPTION UTILS
Role: Dataset management and success-state visual extraction.

This script handles the 'Visual Memory' of the agent. It:
1. Locates the success frame (last frame) for any given episode in the dataset.
2. Decodes the video using torchcodec (high-performance AV1 support).
3. Prepares the goal image for latent encoding by the JEPA model.
"""

# --- Path Stabilization ---
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# --------------------------


from pathlib import Path

import cv2
from pathlib import Path

# Try to import torchcodec for high-performance AV1 decoding
try:
    import torchcodec

    HAS_TORCHCODEC = True
except ImportError:
    HAS_TORCHCODEC = False


def extract_frame_at_index(video_path, frame_idx):
    """
    Decodes a specific frame index from the video.
    Returns: torch.Tensor (3, H, W) in [0, 1]
    """
    if not Path(video_path).exists():
        return None

    # Try TorchCodec first (if available and not erroring)
    if HAS_TORCHCODEC:
        try:
            decoder = torchcodec.decoders.VideoDecoder(str(video_path))
            frame_batch = decoder.get_frames_at(indices=[frame_idx])
            import torch

            return frame_batch.data[0].float() / 255.0
        except Exception:
            pass  # Fallback to OpenCV

    # OpenCV Fallback
    try:
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None

        # BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        import torch

        # (H, W, C) -> (C, H, W)
        return torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
    except Exception as e:
        print(f"❌ Failed to extract frame {frame_idx} with OpenCV: {e}")
        return None


def get_goal_pixels(video_path):
    """
    Extracts the last frame (success state) from an episode video.
    """
    if not Path(video_path).exists():
        return None

    try:
        # Use OpenCV for metadata if torchcodec is unavailable
        cap = cv2.VideoCapture(str(video_path))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        last_frame_idx = num_frames - 1
        return extract_frame_at_index(video_path, last_frame_idx)
    except Exception:
        return None


def get_episode_video_path(
    dataset_root, episode_idx, camera_key="observation.images.world_center"
):
    """
    Finds the MP4 file for a given episode index.
    Matches the LeRobot/HF directory structure.
    """
    root = Path(dataset_root)
    video_path = (
        root / "videos" / camera_key / "chunk-000" / f"file-{episode_idx:03d}.mp4"
    )
    return video_path
