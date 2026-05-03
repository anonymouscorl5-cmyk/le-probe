import json
import cv2
import argparse
from pathlib import Path
from fastapi import FastAPI, HTTPException, Response
import uvicorn

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

app = FastAPI()

# Global config to be set by CLI args
CONFIG = {"meta_path": None, "dataset_dir": None, "meta": None}


@app.get("/api/robot-dataset/frames/{idx}.jpg")
async def get_frame(idx: int):
    """
    Extracts a frame from the cloud-stored dataset (Google Drive) and serves it via HTTP.
    """
    meta = CONFIG["meta"]
    dataset_dir = CONFIG["dataset_dir"]

    # 1. Map index to episode and frame
    ep_idx = idx // (meta["frames_per_ep"] - 2)
    local_f = idx % (meta["frames_per_ep"] - 2)

    if ep_idx >= len(meta["episodes"]):
        raise HTTPException(status_code=404, detail="Index out of range")

    ep_id = meta["episodes"][ep_idx]
    video_path = dataset_dir / "videos" / f"{ep_id}.mp4"

    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {ep_id}")

    # 2. Extract Frame
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, local_f + 1)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise HTTPException(status_code=500, detail="Failed to extract frame")

    # 3. Resize for dashboard
    frame = cv2.resize(frame, (480, 480))

    # 4. Return as JPEG
    _, buffer = cv2.imencode(".jpg", frame)
    return Response(content=buffer.tobytes(), media_type="image/jpeg")


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

    CONFIG["dataset_dir"] = dataset_path

    # Download dataset if repo is provided and local path is missing
    if args.repo and not CONFIG["dataset_dir"].exists():
        print(f"📥 Dataset missing. Downloading {args.repo} from HF...")
        LeRobotDataset(args.repo, root=CONFIG["dataset_dir"].parent)

    # Load metadata
    with open(args.meta, "r") as f:
        CONFIG["meta"] = json.load(f)

    # Launch server
    print(f"📡 Image server starting on port {args.port}...")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
