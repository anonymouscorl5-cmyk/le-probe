import os
import sys
import json
import cv2
import numpy as np
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

# --- Path Stabilization ---
CURRENT_FILE = Path(__file__).resolve()
ROOT_DIR = CURRENT_FILE.parents[2]  # le-probe/
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from lewm.lewm_data_plugin import LEWMDataPlugin


def visualize_audit(report_path, dataset_dir, output_dir="feature_gallery"):

    with open(report_path, "r") as f:
        report = json.load(f)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # 1. Initialize Plugin (to get frame_indices and boundary logic)
    # We use a placeholder repo_id since we'll override the local_dir if needed
    print(f"🔄 Initializing Data Plugin for index alignment...")
    plugin = LEWMDataPlugin(
        repo_id="vedpatwardhan/gr1_pickup_grasp", keys_to_load=["pixels"], num_steps=3
    )
    # Force the plugin to look at the local download directory
    plugin.root = Path(dataset_dir)

    # Model Spec
    tokens_per_moment = 771
    tokens_per_frame = 257
    grid_size = 16
    render_res = (480, 480)  # Native resolution for better quality
    camera_key = "observation.images.world_center"

    for fid, examples in report.items():
        print(f"🖼️ Generating gallery for Feature {fid}...")
        feat_dir = output_path / f"feature_{fid}"
        feat_dir.mkdir(exist_ok=True)

        for rank, (val, global_idx) in enumerate(examples):
            # 2. Map Activation Index to Plugin Index
            idx = global_idx // tokens_per_moment
            token_idx = global_idx % tokens_per_moment

            # 3. Apply the EXACT same boundary shift logic as the harvester
            buffer = 0  # No virtual actions in harvest config
            ep_start = plugin.episode_indices[idx]
            ep_end = plugin.episode_indices[idx + plugin.num_steps + buffer - 1]

            actual_idx = idx
            if ep_start != ep_end:
                actual_idx = idx - plugin.num_steps
                if actual_idx < 0:
                    actual_idx = 0

            episode_idx = int(plugin.episode_indices[actual_idx])
            base_frame_idx = int(plugin.frame_indices[actual_idx])

            # Find the specific frame in the T=3 window
            time_offset = token_idx // tokens_per_frame
            patch_idx = token_idx % tokens_per_frame
            actual_frame_idx = base_frame_idx + time_offset

            # 4. Extract Frame from Video
            video_path = (
                Path(dataset_dir)
                / "videos"
                / camera_key
                / f"chunk-000"
                / f"file-{episode_idx:03d}.mp4"
            )

            if video_path.exists():
                cap = cv2.VideoCapture(str(video_path))
                cap.set(cv2.CAP_PROP_POS_FRAMES, actual_frame_idx)
                ret, frame = cap.read()
                cap.release()

                if ret:
                    # Keep original resolution (480x480)
                    frame = cv2.resize(frame, render_res)
                    if patch_idx > 0:
                        p_idx = patch_idx - 1
                        row, col = p_idx // grid_size, p_idx % grid_size
                        # Scale grid to render_res
                        ph, pw = render_res[0] // grid_size, render_res[1] // grid_size
                        x, y = col * pw, row * ph
                        cv2.rectangle(frame, (x, y), (x + pw, y + ph), (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            f"Act: {val:.2f}",
                            (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (0, 255, 0),
                            1,
                        )

                    save_name = f"rank{rank}_ep{episode_idx}_f{actual_frame_idx}_t{time_offset}_{'patch' if patch_idx > 0 else 'CLS'}.jpg"
                    cv2.imwrite(str(feat_dir / save_name), frame)
                else:
                    print(
                        f"⚠️ Failed to read frame {actual_frame_idx} from {video_path}"
                    )
            else:
                print(f"⚠️ Video not found: {video_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=str, required=True, help="Path to audit JSON")
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset/gr1_pickup_grasp",
        help="Local path or Target path",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="vedpatwardhan/gr1_pickup_grasp",
        help="HF Repo ID for auto-download",
    )
    parser.add_argument(
        "--output", type=str, default="feature_gallery", help="Output folder"
    )
    args = parser.parse_args()

    # Auto-download if missing
    dataset_path = Path(args.dataset)
    if not (dataset_path / "videos").exists():
        print(f"📥 Dataset missing. Downloading videos from {args.repo_id}...")
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=str(dataset_path),
            allow_patterns=["videos/*"],
        )

    visualize_audit(args.report, str(dataset_path), args.output)
