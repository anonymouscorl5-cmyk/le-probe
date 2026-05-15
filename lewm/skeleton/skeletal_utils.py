import os
import torch
import torch.nn.functional as F
from pathlib import Path


def load_skeletal_state_dict(checkpoint_path, device="cpu"):
    """
    Standardized weight surgery for skeletal models.
    Removes 'model.' prefix and handles partial state dicts.
    """
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"❌ Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)

    # Strip 'model.' prefix if present
    new_sd = {k.replace("model.", ""): v for k, v in state_dict.items()}
    return new_sd


def reconstruct_4ch_frame(pixels, transform_fn=None):
    """
    UNIFIED SKELETAL RECONSTRUCTION
    Handles 3 cases:
    1. Tiled Frames (960x480): Splits into RGB + Skeleton and stacks.
    2. 4-Channel Tensors: Returns as-is (but resizes if needed).
    3. 3-Channel RGB: Stacks with an empty skeletal mask.
    """
    # pixels is (C, H, W)
    c, h, w = pixels.shape

    # CASE 1: Tiled Format (Side-by-Side RGB | Skeleton)
    if w == 2 * h:
        rgb = pixels[:, :, :h]
        skel = pixels[:, 0:1, h:]  # Extract first skeletal channel

        if transform_fn:
            transformed = transform_fn({"pixels": rgb})["pixels"]
        else:
            transformed = rgb.float() / 255.0 if rgb.dtype == torch.uint8 else rgb

        if skel.shape[-2:] != (224, 224):
            skel = F.interpolate(
                skel.unsqueeze(0), size=(224, 224), mode="nearest"
            ).squeeze(0)

        if skel.dtype == torch.uint8:
            skel = skel.float() / 255.0

        return torch.cat([transformed, skel], dim=0)

    # CASE 2: Already 4-Channel
    elif c == 4:
        if transform_fn:
            rgb = pixels[:3]
            skel = pixels[3:]
            transformed = transform_fn({"pixels": rgb})["pixels"]
            if skel.shape[-2:] != (224, 224):
                skel = F.interpolate(
                    skel.unsqueeze(0), size=(224, 224), mode="nearest"
                ).squeeze(0)
            if skel.dtype == torch.uint8:
                skel = skel.float() / 255.0
            return torch.cat([transformed, skel], dim=0)
        return pixels

    # CASE 3: Fallback (RGB + Empty Skeleton)
    else:
        if transform_fn:
            transformed = transform_fn({"pixels": pixels})["pixels"]
        else:
            transformed = (
                pixels.float() / 255.0 if pixels.dtype == torch.uint8 else pixels
            )

        skel_empty = torch.zeros((1, 224, 224), device=transformed.device)
        return torch.cat([transformed, skel_empty], dim=0)


def get_skeletal_diagnostic_frames(
    episode_idx, dataset_root, skel_frames_dir=None, use_skeleton=False, mapper=None
):
    """
    Centralized logic for fetching the first 3 frames of an episode with 4-channel support.
    Used by harvest_goals.py and diagnostics.
    """
    from lewm.goal_utils import get_episode_video_path, extract_frame_at_index

    start_frames = []

    # Logic: Try tiled video first, then snapshots, then RGB fallback
    cam_key = (
        "observation.images.world_center_tiled"
        if use_skeleton
        else "observation.images.world_center"
    )
    video_path = get_episode_video_path(dataset_root, episode_idx, camera_key=cam_key)

    if not video_path.exists() and "_tiled" in cam_key:
        video_path = get_episode_video_path(
            dataset_root, episode_idx, camera_key="observation.images.world_center"
        )

    if video_path.exists():
        for f_idx in range(3):
            pixels = extract_frame_at_index(video_path, f_idx)
            if pixels is None:
                continue

            if use_skeleton:
                start_frames.append(
                    reconstruct_4ch_frame(
                        pixels, transform_fn=mapper.transform if mapper else None
                    )
                )
            else:
                transformed = (
                    mapper.transform({"pixels": pixels})["pixels"] if mapper else pixels
                )
                start_frames.append(transformed)

    # Snapshot Fallback
    if not start_frames and use_skeleton and skel_frames_dir:
        skel_frames_dir = Path(skel_frames_dir)
        if not hasattr(mapper, "_skel_meta"):
            mapper._skel_meta = torch.load(
                skel_frames_dir / "metadata.pt", weights_only=False
            )

        indices = [
            idx
            for idx, eid in enumerate(mapper._skel_meta["episode_index"])
            if eid == episode_idx
        ]
        for f_idx in range(min(3, len(indices))):
            frame_data = torch.load(
                skel_frames_dir / f"frame_{indices[f_idx]:06d}.pt", weights_only=False
            )
            pixels_4ch = frame_data["world_center"]
            start_frames.append(
                reconstruct_4ch_frame(
                    pixels_4ch, transform_fn=mapper.transform if mapper else None
                )
            )

    return start_frames
