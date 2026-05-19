import os
import sys
import torch
import argparse
import numpy as np
from pathlib import Path
from datasets import load_dataset


def audit_dataset(repo_id, frames_dir):
    print(f"🔍 [AUDIT] Starting integrity check for {repo_id}...")

    # 1. Load Source Metadata
    ds = load_dataset(repo_id, split="train")
    num_frames = len(ds)
    print(f"📊 Source dataset has {num_frames} frames.")

    # 2. Check Directory and Metadata.pt
    out_dir = Path(frames_dir)
    if not out_dir.exists():
        print(f"❌ ERROR: Frames directory {out_dir} does not exist.")
        sys.exit(1)

    meta_path = out_dir / "metadata.pt"
    if not meta_path.exists():
        print(f"❌ ERROR: metadata.pt missing from {out_dir}.")
        sys.exit(1)

    meta = torch.load(meta_path, weights_only=False)

    # 3. Check Row Counts
    generated_files = list(out_dir.glob("frame_*.pt"))
    num_generated = len(generated_files)

    print(f"📁 Found {num_generated} generated frame files.")

    if num_generated != num_frames:
        print(
            f"❌ ERROR: Row count mismatch! Expected {num_frames}, but found {num_generated}."
        )
    else:
        print(f"✅ Row count matches perfectly.")

    # 4. Check Metadata Column Lengths
    for col, data in meta.items():
        if len(data) != num_frames:
            print(
                f"❌ ERROR: Metadata column '{col}' has invalid length {len(data)} (expected {num_frames})."
            )
        else:
            print(f"✅ Metadata column '{col}' is valid.")

    # 5. Spot Check Multiple Frames (Comprehensive Diagnostic)
    views = ["world_center", "world_left", "world_right", "world_top", "world_wrist"]
    check_indices = sorted(
        list(
            set(
                [0, num_frames - 1]
                + [int(i) for i in np.linspace(0, num_frames - 1, 20)]
            )
        )
    )

    print(
        f"\n🧐 Scanning {len(check_indices)} spot-check frames for rendering health..."
    )

    for check_idx in check_indices:
        frame_path = out_dir / f"frame_{check_idx:06d}.pt"
        if not frame_path.exists():
            print(f"❌ ERROR: Frame {check_idx} is missing.")
            continue

        try:
            frame = torch.load(frame_path, weights_only=False)
            if not all(v in frame for v in views):
                print(f"❌ ERROR: Frame {check_idx} is missing camera views.")
            else:
                print(f"✅ Frame {check_idx}:")
                for v in views:
                    tensor = frame[v]
                    f_shape = tensor.shape

                    # 1. Shape check
                    if len(f_shape) != 3 or f_shape[0] != 4:
                        print(
                            f"  ❌ SHAPE ERROR [{v}]: Expected (4, H, W), but got {tuple(f_shape)}"
                        )
                        continue

                    # 2. NaN/Inf check
                    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                        print(f"  ❌ VALUE ERROR [{v}]: Contains NaNs or Infs!")
                        continue

                    # 3. RGB activity check (channels 0-2)
                    rgb_channel = tensor[:3].float()
                    rgb_std = rgb_channel.std().item()
                    if rgb_std < 1.0:
                        print(
                            f"  🚨 CORRUPTION ALERT [{v}]: RGB image has static/blank content (std: {rgb_std:.4f})"
                        )

                    # 4. Skeleton rendering check (channel 3)
                    skel_channel = tensor[3].float()
                    non_zero_pixels = torch.sum(skel_channel > 0).item()
                    total_pixels = skel_channel.numel()
                    non_zero_pct = (non_zero_pixels / total_pixels) * 100.0

                    if non_zero_pixels == 0:
                        print(
                            f"  🚨 CORRUPTION ALERT [{v}]: Skeleton mask is completely empty (all 0s)!"
                        )
                    else:
                        print(
                            f"  ✅ View [{v}] - Shape: {tuple(f_shape)}, RGB Std: {rgb_std:.1f}, Skeleton: {non_zero_pixels}px ({non_zero_pct:.2f}%)"
                        )
        except Exception as e:
            print(f"❌ ERROR: Could not load frame {check_idx}: {e}")

    print("\n🏁 Audit complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--frames", type=str, default="dataset_skel_frames")
    args = parser.parse_args()

    audit_dataset(args.repo_id, args.frames)
