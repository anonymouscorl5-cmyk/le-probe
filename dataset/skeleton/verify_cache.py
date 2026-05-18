"""
verify_cache.py

A high-fidelity diagnostic utility to verify the integrity, shapes, data types,
and values of pre-computed DINOv3 priors and compiled fused high-speed dataset
caches before launching training.
"""

import sys
import torch
from pathlib import Path
from tqdm import tqdm


def main():
    repo_dir = Path("/Users/vedpatwardhan/Desktop/cortex-os/le-probe")
    dataset_path = repo_dir / "datasets/vedpatwardhan/gr1_pickup_grasp"
    dino_cache_dir = dataset_path / "cache_dino/chunk-000"
    fused_cache_dir = dataset_path / "cache"

    print("🔎 [CACHE & PRIOR VERIFIER] Starting high-fidelity diagnostic sweep...")
    print(f"📂 Dataset Root: {dataset_path}")

    # --- Step 1: Verify DINO Priors ---
    print("\n--- Phase 1: Auditing DINOv3 Prior Tensors ---")
    if not dino_cache_dir.exists():
        print(f"❌ Error: DINO cache directory does not exist: {dino_cache_dir}")
        print(
            "💡 Please run: .venv/bin/python le-probe/dataset/skeleton/generate_dino_priors.py first."
        )
        sys.exit(1)

    dino_files = list(dino_cache_dir.glob("file-*_dino.pt"))
    print(f"Found {len(dino_files)} pre-computed DINO prior files.")

    dino_failures = 0
    for dino_path in tqdm(dino_files, desc="Verifying DINO Priors"):
        try:
            tensor = torch.load(dino_path, map_location="cpu")

            # 1. Shape Audit
            if tensor.shape != (4, 384):
                print(
                    f"❌ Shape mismatch in {dino_path.name}: Expected (4, 384), got {tensor.shape}"
                )
                dino_failures += 1
                continue

            # 2. Data Type (Dtype) Audit
            if tensor.dtype != torch.float32:
                print(
                    f"❌ Dtype mismatch in {dino_path.name}: Expected torch.float32, got {tensor.dtype}"
                )
                dino_failures += 1
                continue

            # 3. Numeric Health Audit (NaNs, Infs, dead zeros)
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                print(f"❌ NaNs or Infs detected in DINO prior: {dino_path.name}")
                dino_failures += 1
            elif torch.all(tensor == 0.0):
                print(f"⚠️ Warning: Prior contains all zeros in {dino_path.name}")

        except Exception as e:
            print(f"❌ Failed to load DINO prior {dino_path.name}: {e}")
            dino_failures += 1

    if dino_failures == 0:
        print("✅ Phase 1: ALL DINOv3 Prior Tensors Passed Integrity Checks!")
    else:
        print(
            f"❌ Phase 1 Failed with {dino_failures} errors. Please re-run prior generation."
        )
        sys.exit(1)

    # --- Step 2: Verify Compiled Fused Caches ---
    print("\n--- Phase 2: Auditing High-Speed Fused Dataset Caches ---")
    if not fused_cache_dir.exists():
        print(f"❌ Error: Fused cache directory does not exist: {fused_cache_dir}")
        print(
            "💡 Please run: .venv/bin/python le-probe/dataset/skeleton/cache_fused_dataset.py first."
        )
        sys.exit(1)

    fused_files = list(fused_cache_dir.glob("episode_*_fused.pt"))
    print(f"Found {len(fused_files)} pre-compiled fused cache files.")

    fused_failures = 0
    expected_state_dim = None
    expected_action_dim = None

    for fused_path in tqdm(fused_files, desc="Verifying Fused Caches"):
        try:
            data = torch.load(fused_path, map_location="cpu")
            required_keys = {"pixels", "state", "action", "dino_waypoints"}
            missing_keys = required_keys - data.keys()

            if missing_keys:
                print(
                    f"❌ Missing keys in cache file {fused_path.name}: {missing_keys}"
                )
                fused_failures += 1
                continue

            pixels = data["pixels"]
            state = data["state"]
            action = data["action"]
            dino_waypoints = data["dino_waypoints"]

            # 1. Data Type (Dtype) Audits
            if pixels.dtype != torch.uint8:
                print(
                    f"❌ Pixels Dtype mismatch in {fused_path.name}: Expected torch.uint8, got {pixels.dtype}"
                )
                fused_failures += 1
            if state.dtype != torch.float32:
                print(
                    f"❌ State Dtype mismatch in {fused_path.name}: Expected torch.float32, got {state.dtype}"
                )
                fused_failures += 1
            if action.dtype != torch.float32:
                print(
                    f"❌ Action Dtype mismatch in {fused_path.name}: Expected torch.float32, got {action.dtype}"
                )
                fused_failures += 1
            if dino_waypoints.dtype != torch.float32:
                print(
                    f"❌ DINO Waypoints Dtype mismatch in {fused_path.name}: Expected torch.float32, got {dino_waypoints.dtype}"
                )
                fused_failures += 1

            # 2. Shape Audits
            # pixels: [32 steps, 5 views, 4 channels (RGB+Skel), 224, 224]
            if (
                len(pixels.shape) != 5
                or pixels.shape[0] != 32
                or pixels.shape[1] != 5
                or pixels.shape[2] != 4
                or pixels.shape[3] != 224
                or pixels.shape[4] != 224
            ):
                print(
                    f"❌ Pixels shape mismatch in {fused_path.name}: Expected (32, 5, 4, 224, 224), got {pixels.shape}"
                )
                fused_failures += 1

            # state step length must be exactly 32
            if len(state.shape) != 2 or state.shape[0] != 32:
                print(
                    f"❌ State shape mismatch in {fused_path.name}: Expected (32, D_state), got {state.shape}"
                )
                fused_failures += 1
            else:
                # Track and assert consistent state dimensions across all files
                if expected_state_dim is None:
                    expected_state_dim = state.shape[1]
                elif state.shape[1] != expected_state_dim:
                    print(
                        f"❌ State dimension inconsistency in {fused_path.name}: Expected {expected_state_dim}, got {state.shape[1]}"
                    )
                    fused_failures += 1

            # action step length must be exactly 32
            if len(action.shape) != 2 or action.shape[0] != 32:
                print(
                    f"❌ Action shape mismatch in {fused_path.name}: Expected (32, D_action), got {action.shape}"
                )
                fused_failures += 1
            else:
                # Track and assert consistent action dimensions across all files
                if expected_action_dim is None:
                    expected_action_dim = action.shape[1]
                elif action.shape[1] != expected_action_dim:
                    print(
                        f"❌ Action dimension inconsistency in {fused_path.name}: Expected {expected_action_dim}, got {action.shape[1]}"
                    )
                    fused_failures += 1

            # dino_waypoints shape must be exactly [4, 384]
            if dino_waypoints.shape != (4, 384):
                print(
                    f"❌ DINO waypoints shape mismatch in {fused_path.name}: Expected (4, 384), got {dino_waypoints.shape}"
                )
                fused_failures += 1

            # 3. Numeric Health and Range Audits
            if torch.isnan(state).any() or torch.isinf(state).any():
                print(f"❌ NaNs or Infs in state: {fused_path.name}")
                fused_failures += 1
            if torch.isnan(action).any() or torch.isinf(action).any():
                print(f"❌ NaNs or Infs in action: {fused_path.name}")
                fused_failures += 1
            if torch.isnan(dino_waypoints).any() or torch.isinf(dino_waypoints).any():
                print(f"❌ NaNs or Infs in cached dino_waypoints: {fused_path.name}")
                fused_failures += 1

            # Pixel value bounds audit [0..255]
            if pixels.min() < 0 or pixels.max() > 255:
                print(
                    f"❌ Pixel value bounds violation in {fused_path.name}: Values must be in [0, 255], got range [{pixels.min()}, {pixels.max()}]"
                )
                fused_failures += 1

        except Exception as e:
            print(f"❌ Failed to load fused cache file {fused_path.name}: {e}")
            fused_failures += 1

    if fused_failures == 0:
        print(
            "\n🎉 [SUCCESS] ALL PRE-COMPUTED DATASETS AND HIGH-SPEED CACHES PASSED INTEGRITY SWEEPS!"
        )
        print(
            f"📈 Verified State Dim: {expected_state_dim} | Action Dim: {expected_action_dim}"
        )
        print(
            "🚀 Your training environment is fully verified and ready for lightning-fast training execution!"
        )
    else:
        print(
            f"\n❌ Phase 2 Failed with {fused_failures} errors. Please re-run fused cache generation."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
