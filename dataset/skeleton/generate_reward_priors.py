import os
import sys
import numpy as np
import mujoco
import pandas as pd
import shutil
import pyarrow as pa
import pyarrow.parquet as pq
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
# --------------------------

from gr1_config import SCENE_PATH, COMPACT_WIRE_JOINTS
from gr1_protocol import StandardScaler
from dataset.skeleton.projection_utils import (
    get_projection_matrix,
    project_point,
    is_allowed_action_chain,
)


def process_chunk(df_chunk, views, img_size=480):
    """
    Processes a chunk of the Parquet dataframe to add skeletal priors.
    """
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)
    unscaler = StandardScaler()

    # Initialize Camera matrices
    cam_data = {}
    for view in views:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, view)
        K = get_projection_matrix(cam_id, model, img_size, img_size)
        # Note: We'll compute t/R inside the loop as they might depend on scene updates
        cam_data[view] = {"id": cam_id, "K": K}

    results = []
    for _, row in df_chunk.iterrows():
        # 1. Update MuJoCo state
        unscaled = unscaler.unscale_action(row["observation.state"])
        data.qpos[:] = model.qpos0

        # Set root height (standard for this protocol)
        root_jnt_adr = model.jnt_qposadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        ]
        data.qpos[root_jnt_adr : root_jnt_adr + 3] = [0.0, 0.0, 0.95]

        for j, n in enumerate(COMPACT_WIRE_JOINTS):
            j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
            if j_id != -1:
                data.qpos[model.jnt_qposadr[j_id]] = unscaled[j]

        mujoco.mj_forward(model, data)
        current_xpos = data.xpos.copy()

        # 2. Render skeletons for each view
        for view in views:
            cam_id = cam_data[view]["id"]
            K = cam_data[view]["K"]

            # Correct Coordinate System (MuJoCo -> OpenCV)
            t_cam = data.cam_xpos[cam_id]
            R_cam = data.cam_xmat[cam_id].reshape(3, 3) @ np.array(
                [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
            )

            mask = Image.new("L", (img_size, img_size), 0)
            draw = ImageDraw.Draw(mask)

            for b_id in range(1, model.nbody):
                p_id = model.body_parentid[b_id]
                if is_allowed_action_chain(b_id, model) and is_allowed_action_chain(
                    p_id, model
                ):
                    ps, _ = project_point(current_xpos[b_id], K, R_cam, t_cam)
                    pp, _ = project_point(current_xpos[p_id], K, R_cam, t_cam)
                    if ps is not None and pp is not None:
                        draw.line([tuple(ps), tuple(pp)], fill=255, width=2)

            # 3. Append to Parquet image list
            # The Parquet stores images as list of lists (usually 3 for RGB)
            # We append the skeleton as the 4th element.
            skel_array = np.array(mask, dtype=np.uint8)

            col_name = f"observation.images.{view}"
            if col_name in row:
                img_list = list(row[col_name])
                if len(img_list) == 3:
                    img_list.append(skel_array.tolist())
                    row[col_name] = img_list

        results.append(row)

    return pd.DataFrame(results)


def _process_and_save_chunk(chunk, views, output_path):
    """Worker function that processes a chunk and saves it immediately to disk"""
    processed_df = process_chunk(chunk, views)
    processed_df.to_parquet(output_path)
    # Return nothing to keep main process RAM empty
    return str(output_path)


def main(input_path, output_path, repo_id=None, num_cores=4):
    # 1. HF Snapshot Support
    parquet_file = Path(input_path)
    if not parquet_file.exists() and repo_id:
        print(f"📂 Dataset not found at {input_path}. Fetching from HF: {repo_id}...")
        local_dir = os.path.dirname(input_path) or "."
        snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=local_dir)
        # Re-verify after download
        if not parquet_file.exists():
            # Sometimes snapshots put things in subdirs, check recursively
            matches = list(Path(local_dir).rglob("dataset.parquet"))
            if matches:
                parquet_file = matches[0]
            else:
                raise FileNotFoundError(
                    f"❌ Failed to find dataset.parquet in {local_dir} after download."
                )

    print(f"📦 [REWARD PRIOR GENERATOR] Loading: {parquet_file}")
    df = pd.read_parquet(parquet_file)

    if "cube_qpos" in df.columns:
        df = df.drop(columns=["cube_qpos"])

    views = ["world_center", "world_left", "world_right", "world_top", "world_wrist"]

    # Split into more chunks for smoother progress tracking
    chunk_count = num_cores * 4
    chunks = np.array_split(df, chunk_count)

    # 3. Staging Pattern: Process and save individual chunks
    temp_dir = Path("temp_skel_chunks")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()

    print(f"🏗️ Staging {chunk_count} chunks to {temp_dir}...")

    # Update wrapper to include output path
    staged_tasks = []
    for i, chunk in enumerate(chunks):
        out_f = temp_dir / f"chunk_{i:04d}.parquet"
        staged_tasks.append((chunk, views, out_f))

    with Pool(num_cores) as p:
        # We use starmap here for simplicity since we're writing to disk anyway
        list(
            tqdm(
                p.starmap(_process_and_save_chunk, staged_tasks),
                total=len(chunks),
                desc="🎥 Processing Skeletons",
            )
        )

    # 4. Coalesce (Stream-to-Disk to save RAM)
    print("🖇️ Coalescing staged chunks (Memory Efficient)...")

    chunk_files = sorted(list(temp_dir.glob("*.parquet")))

    # --- Schema Hardening ---
    # Establish a master schema from the first chunk to ensure consistency
    sample_df = pd.read_parquet(chunk_files[0])
    master_schema = pa.Table.from_pandas(sample_df).schema
    writer = pq.ParquetWriter(output_path, master_schema)
    # ------------------------

    for f in tqdm(chunk_files, desc="Merging"):
        chunk_df = pd.read_parquet(f)

        # Force the chunk to match the master schema
        table = pa.Table.from_pandas(chunk_df, schema=master_schema)
        writer.write_table(table)

        # CLEAR RAM IMMEDIATELY
        del chunk_df
        del table

    if writer:
        writer.close()

    # Cleanup
    shutil.rmtree(temp_dir)
    print(f"✅ Done! Final dataset saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input dataset.parquet"
    )
    parser.add_argument("--output", type=str, help="Path to save upgraded parquet")
    parser.add_argument(
        "--repo_id", type=str, help="HF Repo ID to download if input missing"
    )
    parser.add_argument(
        "--cores", type=int, default=4, help="Number of CPU cores for processing"
    )
    args = parser.parse_args()

    out_path = (
        args.output if args.output else args.input.replace(".parquet", "_skel.parquet")
    )

    main(args.input, out_path, args.repo_id, args.cores)
