import os
import sys
import numpy as np
import mujoco
import torch
import shutil
import cv2
from PIL import Image, ImageDraw
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
from datasets import Dataset
from huggingface_hub import snapshot_download
import argparse

# --- Path Stabilization ---
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from gr1_config import SCENE_PATH, COMPACT_WIRE_JOINTS
from gr1_protocol import StandardScaler
from dataset.skeleton.projection_utils import (
    get_projection_matrix,
    project_point,
    is_allowed_action_chain,
)

# --- Global Worker Context ---
# This is populated once per worker process to avoid initialization overhead
_worker_context = {}


def check_cube_visibility(rgb_frame):
    hsv = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv, np.array([0, 100, 100]), np.array([10, 255, 255])
    ) + cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
    return np.sum(mask > 0) > 30


def find_initial_cube_pos_from_image(frame, K, R, t, table_z=0.82):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv, np.array([0, 100, 100]), np.array([10, 255, 255])
    ) + cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
    M = cv2.moments(mask)
    if M["m00"] == 0:
        return None
    u, v = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    p_cam = np.linalg.inv(K) @ np.array([u, v, 1])
    v_world = R @ p_cam
    return t + ((table_z - t[2]) / v_world[2]) * v_world


def draw_cube_wireframe(draw, cube_pos, K, R, t, color=255):
    size = 0.02
    corners = (
        np.array(
            [
                [-1, -1, -1],
                [1, -1, -1],
                [1, 1, -1],
                [-1, 1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [1, 1, 1],
                [-1, 1, 1],
            ]
        )
        * size
    ) + cube_pos
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    for s_idx, e_idx in edges:
        ps, _ = project_point(corners[s_idx], K, R, t)
        pe, _ = project_point(corners[e_idx], K, R, t)
        if ps is not None and pe is not None:
            draw.line([tuple(ps), tuple(pe)], fill=color, width=2)


def init_worker(parquet_path, views, output_dir):
    """Initializes the plugin and MuJoCo model once per process"""
    _worker_context["ds"] = Dataset.from_parquet(str(parquet_path))
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

    data.qpos[:] = model.qpos0
    data.qpos[
        model.jnt_qposadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        ] : model.jnt_qposadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        ]
        + 3
    ] = [0.0, 0.0, 0.95]
    for j, n in enumerate(COMPACT_WIRE_JOINTS):
        j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        if j_id != -1:
            data.qpos[model.jnt_qposadr[j_id]] = qpos_raw[j]
    mujoco.mj_forward(model, data)

    # 2. Detect cube position directly from observation.images.world_center for this snapshot
    center_rgb = np.array(row["observation.images.world_center"], dtype=np.uint8)
    if center_rgb.shape[0] == 3:
        center_rgb = center_rgb.transpose(1, 2, 0)
    H_c, W_c, _ = center_rgb.shape
    cam_id_center = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "world_center")
    K_center = get_projection_matrix(cam_id_center, model, W_c, H_c)
    t_center = data.cam_xpos[cam_id_center]
    R_center = data.cam_xmat[cam_id_center].reshape(3, 3) @ np.array(
        [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
    )
    bgr_center = cv2.cvtColor(center_rgb, cv2.COLOR_RGB2BGR)
    cube_pos = find_initial_cube_pos_from_image(
        bgr_center, K_center, R_center, t_center
    )

    frame_data = {}
    for view_name in views:
        # 3. Get RGB (Handle both HWC and CHW formats from datasets)
        rgb = np.array(row[f"observation.images.{view_name}"], dtype=np.uint8)
        if rgb.shape[0] == 3:
            rgb = rgb.transpose(1, 2, 0)
        H, W, _ = rgb.shape

        # 4. Render Skeleton
        mask = Image.new("L", (W, H), 0)
        draw = ImageDraw.Draw(mask)

        # Camera Projection
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, view_name)
        K = get_projection_matrix(cam_id, model, W, H)
        t_cam = data.cam_xpos[cam_id]
        R_cam = data.cam_xmat[cam_id].reshape(3, 3) @ np.array(
            [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
        )

        for b_id in range(1, model.nbody):
            p_id = model.body_parentid[b_id]
            if is_allowed_action_chain(b_id, model) and is_allowed_action_chain(
                p_id, model
            ):
                ps, _ = project_point(data.xpos[b_id], K, R_cam, t_cam)
                pp, _ = project_point(data.xpos[p_id], K, R_cam, t_cam)
                if ps is not None and pp is not None:
                    draw.line([tuple(ps), tuple(pp)], fill=255, width=2)

        # Draw cube wireframe
        if cube_pos is not None and check_cube_visibility(rgb):
            draw_cube_wireframe(draw, cube_pos, K, R_cam, t_cam)

        # 5. Stack into 4-channel tensor [C, H, W]
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

    # 1. Download and Locate Parquet File
    print(f"📥 Syncing dataset from HF: {args.repo_id}...")
    local_dir = Path(args.repo_id.split("/")[-1])
    snapshot_download(repo_id=args.repo_id, repo_type="dataset", local_dir=local_dir)

    parquet_matches = list(local_dir.rglob("*.parquet"))
    if not parquet_matches:
        raise FileNotFoundError(f"🚨 Could not find dataset.parquet inside {local_dir}")

    parquet_path = parquet_matches[0]
    print(f"📊 Loading dataset from parquet: {parquet_path}")
    ds = Dataset.from_parquet(str(parquet_path))
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
    ep_idx_list = (
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
        "episode_index": ep_idx_list,
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
        initargs=(parquet_path, views, out_dir),
    ) as p:
        # chunksize=5 for a responsive progress bar with good efficiency
        results = p.imap_unordered(process_frame_task, range(num_frames), chunksize=5)

        for _ in tqdm(results, total=num_frames, desc="Rendering Skeletons"):
            pass

    print(f"✅ Success! All frames saved to {out_dir}")


if __name__ == "__main__":
    main()
