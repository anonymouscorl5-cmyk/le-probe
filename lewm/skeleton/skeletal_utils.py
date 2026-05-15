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

    # Strip 'model.' prefix and filter out training-only regularizers (sigreg)
    new_sd = {}
    for k, v in state_dict.items():
        clean_key = k.replace("model.", "") if k.startswith("model.") else k
        if not clean_key.startswith("sigreg."):
            new_sd[clean_key] = v
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
