import os
import sys
import numpy as np
import mujoco
import pandas as pd
import shutil
import torch
from PIL import Image, ImageDraw
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from huggingface_hub import snapshot_download
import argparse

# --- Path Stabilization ---
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from gr1_config import SCENE_PATH, COMPACT_WIRE_JOINTS
from gr1_protocol import StandardScaler


def process_frame(row, views, model, data, unscaler):
    """Processes a single frame and returns the 4-channel tensors"""
    qpos = row["observation.state"]
    qpos_raw = unscaler.unscale(np.array(qpos).reshape(1, -1), "observation.state")[0]

    data.qpos[: len(qpos_raw)] = qpos_raw
    mujoco.mj_forward(model, data)

    frame_data = {}
    for view_name in views:
        # 1. Get RGB
        raw_pixels = row[f"observation.images.{view_name}"]
        rgb = np.array(raw_pixels, dtype=np.uint8)
        H, W, _ = rgb.shape

        # 2. Render Skeleton
        mask = Image.new("L", (W, H), 0)
        draw = ImageDraw.Draw(mask)

        # Camera Projection
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, view_name)
        fovy = model.cam_fovy[cam_id]
        f = 0.5 * H / np.tan(fovy * np.pi / 360)
        cx, cy = W / 2, H / 2

        cam_pos = data.cam_xpos[cam_id]
        cam_rot = data.cam_xmat[cam_id].reshape(3, 3)

        def project(x):
            rel_pos = cam_rot.T @ (x - cam_pos)
            if rel_pos[2] <= 0:
                return None
            px = cx - f * rel_pos[0] / rel_pos[2]
            py = cy - f * rel_pos[1] / rel_pos[2]
            return (px, py)

        for joint_pair in COMPACT_WIRE_JOINTS:
            p1 = project(
                data.xpos[
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, joint_pair[0])
                ]
            )
            p2 = project(
                data.xpos[
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, joint_pair[1])
                ]
            )
            if p1 and p2:
                draw.line([p1, p2], fill=255, width=2)

        # 3. Stack into 4-channel tensor [C, H, W]
        skel = np.array(mask, dtype=np.uint8)
        combined = np.concatenate([rgb, skel[..., None]], axis=-1)
        # Convert to torch tensor immediately
        frame_data[view_name] = torch.from_numpy(combined).permute(2, 0, 1)

    return frame_data


def worker_task(chunk_data):
    """Worker process that handles a chunk of frames"""
    df_chunk, views, output_dir, start_idx = chunk_data
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)
    unscaler = StandardScaler()

    for i, (_, row) in enumerate(df_chunk.iterrows()):
        frame_tensors = process_frame(row, views, model, data, unscaler)
        frame_idx = start_idx + i
        torch.save(frame_tensors, output_dir / f"frame_{frame_idx:06d}.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo_id", type=str, required=True, help="HF Repo ID to download dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset_skel_frames",
        help="Dir to save .pt frames",
    )
    parser.add_argument("--cores", type=int, default=4)
    args = parser.parse_args()

    # 1. Download and Locate Parquet
    local_dir = Path(args.repo_id.split("/")[-1])
    print(f"📥 Syncing dataset from HF: {args.repo_id}...")
    snapshot_download(repo_id=args.repo_id, repo_type="dataset", local_dir=local_dir)

    parquet_matches = list(local_dir.rglob("dataset.parquet"))
    if not parquet_matches:
        raise FileNotFoundError(f"🚨 Could not find dataset.parquet inside {local_dir}")

    parquet_path = parquet_matches[0]
    df = pd.read_parquet(parquet_path)
    print(f"📊 Loaded {len(df)} frames from {parquet_path}")

    # 2. Setup Output Directories
    out_dir = Path(args.output_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # 3. Save Metadata
    meta_cols = ["progress", "episode_index", "frame_index"]
    metadata = {col: df[col].values.tolist() for col in meta_cols if col in df.columns}
    torch.save(metadata, out_dir / "metadata.pt")

    # 4. Multiprocess Frame Generation
    views = ["world_center", "world_left", "world_right", "world_top", "world_wrist"]
    chunks = np.array_split(df, args.cores)

    start_indices = [0]
    for c in chunks[:-1]:
        start_indices.append(start_indices[-1] + len(c))

    task_args = [
        (chunks[i], views, out_dir, start_indices[i]) for i in range(len(chunks))
    ]

    print(f"🚀 Generating {len(df)} skeletal frames across {args.cores} cores...")
    with Pool(args.cores) as p:
        list(
            tqdm(
                p.imap_unordered(worker_task, task_args),
                total=len(chunks),
                desc="Processing",
            )
        )

    print(f"✅ Success! All frames saved to {out_dir}")


if __name__ == "__main__":
    main()
