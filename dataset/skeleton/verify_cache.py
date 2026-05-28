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


from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Repo root for shared DINO layout constants
REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
from lewm.skeleton.dino_constants import dino_waypoints_shape  # noqa: E402


def main(repo_id="gr1_pickup_grasp"):
    # Dynamically resolve dataset path using LeRobot's own dataset engine or bust
    dataset = LeRobotDataset(repo_id)
    dataset_path = Path(dataset.root)

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

            expected = dino_waypoints_shape()
            if tensor.shape != expected:
                print(
                    f"❌ Shape mismatch in {dino_path.name}: "
                    f"Expected {expected}, got {tensor.shape}. "
                    "Re-run generate_dino_priors.py."
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

            if dino_waypoints.shape != dino_waypoints_shape():
                print(
                    f"❌ DINO waypoints shape mismatch in {fused_path.name}: "
                    f"Expected {dino_waypoints_shape()}, got {dino_waypoints.shape}. "
                    "Re-run generate_dino_priors.py and cache_fused_dataset.py."
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

            # 4. Semantic Skeleton Channel Audit (Anti-Squashing & Anti-Leakage Check)
            # A valid skeleton mask is drawn on a black background and should be highly sparse.
            # A squashed tiled video (RGB + Skeleton) would have active RGB textures on the left, dropping sparsity.
            skel = pixels[:, :, 3].float()
            sparsity = (skel == 0.0).float().mean()
            if sparsity < 0.70:
                print(
                    f"❌ Semantic Violation in {fused_path.name}: Skeleton channel sparsity is too low ({sparsity.item():.2%}). "
                    "Expected at least 70.00% sparsity (black background). This indicates RGB leakage or squashed/unsplit tiled frames."
                )
                fused_failures += 1

            # Check that the skeleton is not completely dead (all zeros)
            if skel.max() < 100:
                print(
                    f"❌ Dead Channel Violation in {fused_path.name}: Skeleton channel max value is too low ({skel.max().item()}). "
                    "A valid skeleton mask must contain bright lines/joints."
                )
                fused_failures += 1

            # Check for RGB-to-Skeleton leakage using Pearson correlation
            r, g, b = (
                pixels[:, :, 0].float(),
                pixels[:, :, 1].float(),
                pixels[:, :, 2].float(),
            )
            rgb_gray = 0.299 * r + 0.587 * g + 0.114 * b
            mean_rgb = rgb_gray.mean()
            mean_skel = skel.mean()
            diff_rgb = rgb_gray - mean_rgb
            diff_skel = skel - mean_skel
            covariance = (diff_rgb * diff_skel).mean()
            std_rgb = torch.sqrt((diff_rgb**2).mean())
            std_skel = torch.sqrt((diff_skel**2).mean())
            if std_rgb > 0 and std_skel > 0:
                correlation = covariance / (std_rgb * std_skel)
                # If correlation is too high, it indicates leakage of RGB textures into the skeleton channel
                if correlation > 0.45:
                    print(
                        f"❌ Leakage Violation in {fused_path.name}: Skeleton channel is highly correlated with RGB channel "
                        f"({correlation.item():.2f}). Potential RGB-to-skeleton leakage or squashed frame resizing."
                    )
                    fused_failures += 1

            # 5. State, Action, and Waypoint Dynamics/Variance Checks
            # Check for dead/frozen states and actions (which indicate corrupted parquet/video logging)
            if state.std(dim=0).max() < 1e-4:
                print(
                    f"⚠️ Warning in {fused_path.name}: State exhibits near-zero temporal variance across all dimensions."
                )
            if action.std(dim=0).max() < 1e-4:
                print(
                    f"⚠️ Warning in {fused_path.name}: Action exhibits near-zero temporal variance across all dimensions."
                )
            if dino_waypoints.std() < 1e-4:
                print(
                    f"❌ Waypoint Violation in {fused_path.name}: Cached DINO waypoints exhibit zero variance."
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
