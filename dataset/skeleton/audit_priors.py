import os
import sys
import torch
import argparse
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

    # 5. Spot Check Random Frames
    views = ["world_center", "world_left", "world_right", "world_top", "world_wrist"]
    for check_idx in [0, num_frames // 2, num_frames - 1]:
        frame_path = out_dir / f"frame_{check_idx:06d}.pt"
        if not frame_path.exists():
            print(f"❌ ERROR: Frame {check_idx} is missing.")
            continue

        try:
            frame = torch.load(frame_path, weights_only=False)
            if not all(v in frame for v in views):
                print(f"❌ ERROR: Frame {check_idx} is missing camera views.")
            else:
                print(f"✅ Frame {check_idx} loaded successfully.")
                # --- NEW: Shape Validation ---
                for v in views:
                    f_shape = frame[v].shape
                    if len(f_shape) != 3 or f_shape[0] != 4:
                        print(
                            f"  ❌ SHAPE ERROR [{v}]: Expected (4, H, W), but got {tuple(f_shape)}"
                        )
                    elif f_shape[1] == 3 or f_shape[2] == 3:
                        print(
                            f"  🚨 CORRUPTION ALERT [{v}]: Found dimension '3' in H/W! Shape: {tuple(f_shape)}"
                        )
                    else:
                        print(f"  ✅ Shape Valid [{v}]: {tuple(f_shape)}")
        except Exception as e:
            print(f"❌ ERROR: Could not load frame {check_idx}: {e}")

    print("\n🏁 Audit complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--frames", type=str, default="dataset_skel_frames")
    args = parser.parse_args()

    audit_dataset(args.repo_id, args.frames)
