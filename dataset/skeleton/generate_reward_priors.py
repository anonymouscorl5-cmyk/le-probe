import os
import sys
import numpy as np
import mujoco
import torch
import shutil
from PIL import Image, ImageDraw
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
from datasets import load_dataset
import argparse

# --- Path Stabilization ---
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from gr1_config import SCENE_PATH, COMPACT_WIRE_JOINTS
from gr1_protocol import StandardScaler

# --- Global Worker Context ---
# This is populated once per worker process to avoid initialization overhead
_worker_context = {}


def init_worker(repo_id, views, output_dir):
    """Initializes the plugin and MuJoCo model once per process"""
    _worker_context["ds"] = load_dataset(repo_id, split="train")
    _worker_context["model"] = mujoco.MjModel.from_xml_path(SCENE_PATH)
    _worker_context["data"] = mujoco.MjData(_worker_context["model"])
    _worker_context["unscaler"] = StandardScaler()
    _worker_context["views"] = views
    _worker_context["out_dir"] = Path(output_dir)


def process_frame_task(idx):
    """Processes a single frame index using the worker's cached context"""
    ds = _worker_context["ds"]
    model = _worker_context["model"]
    data = _worker_context["data"]
    unscaler = _worker_context["unscaler"]
    views = _worker_context["views"]
    out_dir = _worker_context["out_dir"]

    row = ds[idx]

    # 1. Proprioception
    qpos = np.array(row["observation.state"])
    qpos_raw = unscaler.unscale_action(qpos)

    data.qpos[: len(qpos_raw)] = qpos_raw
    mujoco.mj_forward(model, data)

    frame_data = {}
    for view_name in views:
        # 2. Get RGB (Handle both HWC and CHW formats from datasets)
        rgb = np.array(row[f"observation.images.{view_name}"], dtype=np.uint8)
        if rgb.shape[0] == 3:
            rgb = rgb.transpose(1, 2, 0)
        H, W, _ = rgb.shape

        # 3. Render Skeleton
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

        # 4. Stack into 4-channel tensor [C, H, W]
        skel = np.array(mask, dtype=np.uint8)
        combined = np.concatenate([rgb, skel[..., None]], axis=-1)
        frame_data[view_name] = torch.from_numpy(combined).permute(2, 0, 1)

    torch.save(frame_data, out_dir / f"frame_{idx:06d}.pt")
    return idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True, help="HF Repo ID")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset_skel_frames",
        help="Dir to save .pt frames",
    )
    parser.add_argument("--cores", type=int, default=4)
    args = parser.parse_args()

    # 1. Initialize Dataset
    print(f"📥 Loading dataset via HuggingFace: {args.repo_id}...")
    ds = load_dataset(args.repo_id, split="train")
    num_frames = len(ds)
    print(f"📊 Dataset loaded: {num_frames} frames.")

    # 2. Setup Output Directories
    out_dir = Path(args.output_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # 3. Save Metadata
    print("💾 Saving metadata...")
    progress = ds["progress"] if "progress" in ds.column_names else ([0.0] * num_frames)
    ep_idx = (
        ds["episode_index"]
        if "episode_index" in ds.column_names
        else ([0] * num_frames)
    )

    if "frame_index" in ds.column_names:
        f_idx = ds["frame_index"]
    elif "step" in ds.column_names:
        f_idx = ds["step"]
    else:
        f_idx = list(range(num_frames))

    metadata = {
        "progress": progress,
        "episode_index": ep_idx,
        "frame_index": f_idx,
    }
    torch.save(metadata, out_dir / "metadata.pt")

    # 4. Multiprocess Frame Generation
    views = ["world_center", "world_left", "world_right", "world_top", "world_wrist"]

    print(f"🚀 Processing {num_frames} frames across {args.cores} cores...")

    # Using Pool with initializer ensures setup happens only once per process
    with Pool(
        processes=args.cores,
        initializer=init_worker,
        initargs=(args.repo_id, views, out_dir),
    ) as p:
        # chunksize=5 for a responsive progress bar with good efficiency
        results = p.imap_unordered(process_frame_task, range(num_frames), chunksize=5)

        for _ in tqdm(results, total=num_frames, desc="Rendering Skeletons"):
            pass

    print(f"✅ Success! All frames saved to {out_dir}")


if __name__ == "__main__":
    main()
