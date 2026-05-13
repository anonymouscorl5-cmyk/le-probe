import os
import sys
import numpy as np
import mujoco
from PIL import Image, ImageDraw
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# --- Path Stabilization ---
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
# --------------------------

from gr1_config import SCENE_PATH, COMPACT_WIRE_JOINTS
from gr1_protocol import StandardScaler
from dataset.skeleton.projection_utils import (
    get_projection_matrix,
    project_point,
    is_allowed_action_chain,
)


def find_initial_cube_pos(video_path, K, R, t, table_z=0.82):
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
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


def check_cube_visibility(rgb_frame):
    hsv = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv, np.array([0, 100, 100]), np.array([10, 255, 255])
    ) + cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
    return np.sum(mask > 0) > 30


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


def generate_skeleton_video(
    ep_df, rgb_video_path, model, data, K, R, t, initial_cube_pos, output_path
):
    unscaler = StandardScaler()
    idx_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "R_index_tip_link")
    thm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "R_thumb_tip_link")
    cap = cv2.VideoCapture(str(rgb_video_path))
    frames = []

    for _, row in ep_df.iterrows():
        ret, rgb_frame = cap.read()
        if not ret:
            break
        unscaled = unscaler.unscale_action(row["observation.state"])
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
                data.qpos[model.jnt_qposadr[j_id]] = unscaled[j]
        mujoco.mj_forward(model, data)

        mask = Image.new("L", (480, 480), 0)
        draw = ImageDraw.Draw(mask)
        for b_id in range(1, model.nbody):
            p_id = model.body_parentid[b_id]
            if is_allowed_action_chain(b_id, model) and is_allowed_action_chain(
                p_id, model
            ):
                ps, _ = project_point(data.xpos[b_id], K, R, t)
                pp, _ = project_point(data.xpos[p_id], K, R, t)
                if ps is not None and pp is not None:
                    draw.line([tuple(ps), tuple(pp)], fill=255, width=2)

        if initial_cube_pos is not None and check_cube_visibility(rgb_frame):
            gripper_midpoint = (data.xpos[idx_id] + data.xpos[thm_id]) / 2.0
            cube_render_pos = (
                gripper_midpoint
                if np.linalg.norm(gripper_midpoint - initial_cube_pos) < 0.05
                else initial_cube_pos
            )
            draw_cube_wireframe(draw, cube_render_pos, K, R, t)
        frames.append(np.array(mask))
    cap.release()

    tmp_raw = output_path.with_suffix(".raw.mp4")
    video = cv2.VideoWriter(
        str(tmp_raw), cv2.VideoWriter_fourcc(*"mp4v"), 10, (480, 480), isColor=False
    )
    for f in frames:
        video.write(f)
    video.release()
    os.system(
        f"ffmpeg -y -i {tmp_raw} -vcodec libx264 -crf 30 -pix_fmt yuv420p {output_path} > /dev/null 2>&1"
    )
    if tmp_raw.exists():
        tmp_raw.unlink()


def main(repo_id="vedpatwardhan/gr1_pickup_grasp"):
    print(f"📦 [SKELETON GENERATOR] Initializing dataset from Hub: {repo_id}")
    dataset = LeRobotDataset(repo_id)
    dataset_path = Path(dataset.root)
    print(f"📁 Dataset Root: {dataset_path}")

    views = ["world_center", "world_left", "world_right", "world_top", "world_wrist"]
    for view in views:
        (dataset_path / f"videos/observation.images.{view}_skeleton/chunk-000").mkdir(
            parents=True, exist_ok=True
        )

    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)

    num_episodes = dataset.num_episodes
    print(
        f"📽️ Generating skeletal priors for {num_episodes} episodes across {len(views)} views..."
    )

    for i in tqdm(range(num_episodes), desc="Episodes"):
        parquet_file = dataset_path / f"data/chunk-000/file-{i:03d}.parquet"
        if not parquet_file.exists():
            continue

        center_rgb = (
            dataset_path
            / f"videos/observation.images.world_center/chunk-000/file-{i:03d}.mp4"
        )
        cam_id_center = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, "world_center"
        )
        mujoco.mj_forward(model, data)
        K_center = get_projection_matrix(cam_id_center, model, 480, 480)
        t_center, R_center = data.cam_xpos[cam_id_center], data.cam_xmat[
            cam_id_center
        ].reshape(3, 3) @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        initial_cube_pos = find_initial_cube_pos(
            center_rgb, K_center, R_center, t_center
        )

        df = pd.read_parquet(parquet_file)
        for view in views:
            cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, view)
            K = get_projection_matrix(cam_id, model, 480, 480)
            t_cam, R_cam = data.cam_xpos[cam_id], data.cam_xmat[cam_id].reshape(
                3, 3
            ) @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            rgb_video = (
                dataset_path
                / f"videos/observation.images.{view}/chunk-000/file-{i:03d}.mp4"
            )
            output_file = (
                dataset_path
                / f"videos/observation.images.{view}_skeleton/chunk-000/file-{i:03d}.mp4"
            )
            generate_skeleton_video(
                df,
                rgb_video,
                model,
                data,
                K,
                R_cam,
                t_cam,
                initial_cube_pos,
                output_file,
            )

    print(f"✅ Skeletal priors generation complete.")


if __name__ == "__main__":
    repo = sys.argv[1] if len(sys.argv) > 1 else "vedpatwardhan/gr1_pickup_grasp"
    main(repo)
