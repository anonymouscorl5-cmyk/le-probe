import json
import cv2
import argparse
from pathlib import Path
from fastapi import FastAPI, HTTPException, Response
import uvicorn

from lerobot.datasets.lerobot_dataset import LeRobotDataset

app = FastAPI()

# Global config to be set by CLI args
CONFIG = {"meta": None, "dataset": None}


@app.get("/api/robot-dataset/frames/{idx}.jpg")
async def get_frame(idx: int):
    """
    Maps a global token index to a dataset sample and extracts the corresponding frame.
    """
    meta = CONFIG["meta"]
    dataset = CONFIG["dataset"]

    if not dataset:
        raise HTTPException(status_code=500, detail="Dataset not initialized")

    # 1. Map global token index to sample index
    tokens_per_sample = meta.get("tokens_per_sample", 771)
    sample_idx = idx // tokens_per_sample

    if sample_idx >= len(dataset):
        raise HTTPException(status_code=404, detail="Sample index out of range")

    # 2. Extract Frame using LeRobot's internal mapping
    try:
        # We need to find which episode this sample belongs to
        # dataset.episode_data_index is a tensor/list of (start, end)
        ep_idx = -1
        for i, (start, end) in enumerate(
            zip(dataset.episode_data_index["from"], dataset.episode_data_index["to"])
        ):
            if start <= sample_idx < end:
                ep_idx = i
                local_frame_idx = sample_idx - start
                break

        if ep_idx == -1:
            raise HTTPException(status_code=404, detail="Episode not found for sample")

        ep_id = dataset.hf_dataset[ep_idx]["episode_id"]
        video_path = Path(dataset.root) / "videos" / f"{ep_id}.mp4"

        if not video_path.exists():
            # Try searching in subfolders if modality subfolders exist
            video_paths = list(Path(dataset.root).glob(f"**/videos/{ep_id}.mp4"))
            if video_paths:
                video_path = video_paths[0]
            else:
                raise HTTPException(
                    status_code=404, detail=f"Video not found for episode {ep_id}"
                )

        cap = cv2.VideoCapture(str(video_path))
        # local_frame_idx is 0-based in the episode, but videos might have offset
        # Usually for LeRobot, it's 1-to-1
        cap.set(cv2.CAP_PROP_POS_FRAMES, local_frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise HTTPException(
                status_code=500, detail="Failed to extract frame from video"
            )

        # 3. Resize for dashboard
        frame = cv2.resize(frame, (480, 480))

        # 4. Return as JPEG
        _, buffer = cv2.imencode(".jpg", frame)
        return Response(content=buffer.tobytes(), media_type="image/jpeg")

    except Exception as e:
        print(f"❌ Error extracting frame: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    parser = argparse.ArgumentParser(description="LeWM Colab Visual Bridge")
    parser.add_argument(
        "--meta", type=str, required=True, help="Path to encoder_L0.json metadata"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Local path to dataset directory (optional if --repo is used)",
    )
    parser.add_argument(
        "--repo", type=str, help="Hugging Face repo ID (e.g. lerobot/gr1_pickup_grasp)"
    )
    parser.add_argument("--port", type=int, default=8000, help="Local server port")
    args = parser.parse_args()

    if not args.dataset and not args.repo:
        parser.error("Either --dataset or --repo must be provided")

    # If dataset path not provided, infer from repo
    if not args.dataset:
        repo_name = args.repo.split("/")[-1]
        # Default to a 'dataset' folder in the current working directory
        dataset_path = Path("dataset") / repo_name
    else:
        dataset_path = Path(args.dataset)

    # Initialize Dataset (this will download if missing and repo is provided)
    print(f"📦 Initializing dataset from {dataset_path}...")
    dataset = LeRobotDataset(args.repo or str(dataset_path), root=dataset_path.parent)
    CONFIG["dataset"] = dataset

    # Load metadata
    with open(args.meta, "r") as f:
        CONFIG["meta"] = json.load(f)

    # Launch server
    print(f"📡 Image server starting on port {args.port}...")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
