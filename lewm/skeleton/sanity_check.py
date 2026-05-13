import os
import sys
import torch
import numpy as np
from omegaconf import OmegaConf

# --- Path Stabilization ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# --------------------------

from lewm.skeleton.encoder import get_skeleton_encoder
from lewm.skeleton.data import SkeletonDataPlugin


def test_encoder_handshake():
    print("\n🧪 [TEST 1] Testing Encoder Handshake & Zero-Init...")
    cfg = OmegaConf.create(
        {
            "encoder_scale": "tiny",
            "patch_size": 14,
            "img_size": 224,
            "num_views": 5,
            "fusion_type": "linear",
        }
    )

    encoder = get_skeleton_encoder(cfg)

    # Create mock input: (B=1, T=1, V=5, C=4, H=224, W=224)
    mock_input = torch.randn(1, 1, 5, 4, 224, 224)

    # 1. Baseline: Skeleton channel is zero
    input_zero_skel = mock_input.clone()
    input_zero_skel[:, :, :, 3, :, :] = 0.0

    # 2. Augmented: Skeleton channel is random
    input_with_skel = mock_input.clone()
    input_with_skel[:, :, :, 3, :, :] = torch.randn_like(
        input_with_skel[:, :, :, 3, :, :]
    )

    with torch.no_grad():
        out_baseline = encoder(input_zero_skel).last_hidden_state
        out_augmented = encoder(input_with_skel).last_hidden_state

    diff = (out_baseline - out_augmented).abs().max().item()
    print(f"  - Output Shape: {out_baseline.shape}")
    print(f"  - Max Diff (Skeleton vs No-Skeleton): {diff:.12f}")

    if diff < 1e-10:
        print("  ✅ SUCCESS: Zero-Initialization confirmed. 4th channel is dormant.")
    else:
        print(
            f"  ❌ FAILURE: 4th channel is contributing to output (Diff: {diff:.12f})"
        )


def test_data_plugin():
    print("\n🧪 [TEST 2] Testing SkeletonDataPlugin Fusion...")
    try:
        cfg = {
            "repo_id": "vedpatwardhan/gr1_pickup_grasp",
            "keys_to_load": ["world_center"],
            "num_steps": 1,
            "use_multi_view": True,
            "img_size": 224,
        }

        plugin = SkeletonDataPlugin(
            repo_id=cfg["repo_id"],
            keys_to_load=cfg["keys_to_load"],
            num_steps=cfg["num_steps"],
            use_multi_view=cfg["use_multi_view"],
            img_size=cfg["img_size"],
        )

        print(f"  - Plugin Root: {plugin.root}")
        sample_skel_path = plugin._get_video_path(
            0, "observation.images.world_center_skeleton"
        )
        print(f"  - Sample Skeleton Path: {sample_skel_path}")
        if sample_skel_path.exists():
            print("  ✅ SUCCESS: Skeleton video found.")
        else:
            print("  ⚠️ NOTE: Skeleton videos not found at path.")

    except Exception as e:
        print(f"  ❌ PLUGIN FAILED: {e}")


if __name__ == "__main__":
    print("🚀 [STARTING SANITY CHECK]")
    test_encoder_handshake()
    test_data_plugin()
