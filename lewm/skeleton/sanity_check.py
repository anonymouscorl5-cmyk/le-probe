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

from lewm.multi_view_encoder import get_multi_view_encoder
from lewm.skeleton.encoder import patch_vit_for_skeleton
from lewm.skeleton.data import SkeletonDataPlugin


def test_encoder_patching():
    print("\n🧪 [TEST 1] Testing Encoder Patching & Zero-Init...")
    cfg = OmegaConf.create(
        {
            "encoder_scale": "tiny",
            "patch_size": 14,
            "img_size": 224,
            "num_views": 5,
            "fusion_type": "linear",
            "wm": {"embed_dim": 192},
        }
    )

    # 1. Start with 3-channel
    encoder = get_multi_view_encoder(cfg)

    # 2. Patch to 4-channel
    patch_vit_for_skeleton(encoder.backbone)

    # Create mock input: (B=1, T=1, V=5, C=4, H=224, W=224)
    mock_input = torch.randn(1, 1, 5, 4, 224, 224)

    # A. Baseline: Skeleton channel is zero
    input_zero_skel = mock_input.clone()
    input_zero_skel[:, :, :, 3, :, :] = 0.0

    # B. Augmented: Skeleton channel is random
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


def test_data_plugin_tiled_config():
    print("\n🧪 [TEST 2] Testing Tiled Data Configuration...")
    try:
        cfg = {
            "repo_id": "vedpatwardhan/gr1_pickup_grasp",
            "keys_to_load": ["world_center", "world_left"],
            "num_steps": 1,
            "use_multi_view": True,
            "img_size": 224,
        }

        # Initialize (This should no longer crash)
        plugin = SkeletonDataPlugin(
            repo_id=cfg["repo_id"],
            keys_to_load=cfg["keys_to_load"],
            num_steps=cfg["num_steps"],
            use_multi_view=cfg["use_multi_view"],
            img_size=cfg["img_size"],
        )

        # Check key mapping
        target = "world_center"
        mapped = plugin.key_map.get(target)
        expected = "observation.images.world_center_tiled"

        print(f"  - Map Check: {target} -> {mapped}")
        if mapped == expected:
            print(f"  ✅ SUCCESS: Key mapping correctly redirects to {expected}")
        else:
            print(
                f"  ❌ FAILURE: Key mapping incorrect. Got {mapped}, expected {expected}"
            )

        # Check transform wrapping
        if (
            hasattr(plugin, "orig_transform")
            and plugin.transform.__name__ == "tiled_transform_wrapper"
        ):
            print("  ✅ SUCCESS: Transform is correctly wrapped.")
        else:
            print("  ❌ FAILURE: Transform not wrapped correctly.")

    except Exception as e:
        print(f"  ❌ PLUGIN FAILED: {e}")
        import traceback

        traceback.print_exc()


def test_fusion_logic():
    print("\n🧪 [TEST 3] Testing 4-Channel Fusion Logic...")
    try:
        # Create a mock plugin
        plugin = SkeletonDataPlugin(
            repo_id="vedpatwardhan/gr1_pickup_grasp",
            keys_to_load=["world_center"],
            num_steps=1,
            use_multi_view=True,
            img_size=224,
        )

        # Create mock nested batch with tiled image [1, 3, 960, 480]
        # (RGB on top 480x480, Skel on bottom 480x480)
        # Wait, the bypass logic uses 960x480 (stacked vertically)
        mock_tiled = torch.randn(1, 3, 960, 480)
        mock_batch = {"observation": {"images": {"world_center": mock_tiled}}}

        # Run wrapper
        transformed = plugin.tiled_transform_wrapper(mock_batch)

        # Check output
        fused = transformed["observation"]["images"]["world_center"]
        print(f"  - Fused Shape: {fused.shape}")

        if fused.shape[1] == 4:
            print("  ✅ SUCCESS: Fused tensor has 4 channels.")
        else:
            print(f"  ❌ FAILURE: Expected 4 channels, got {fused.shape[1]}")

    except Exception as e:
        print(f"  ❌ FUSION TEST FAILED: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    print("🚀 [STARTING SKELETON SANITY CHECK]")
    test_encoder_patching()
    test_data_plugin_tiled_config()
    test_fusion_logic()
