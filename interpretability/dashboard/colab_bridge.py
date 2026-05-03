import json
import cv2
import argparse
from pathlib import Path
from fastapi import FastAPI, HTTPException, Response
import uvicorn

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
        "--dataset", type=str, required=True, help="Path to gr1_pickup_grasp directory"
    )
    parser.add_argument("--port", type=int, default=8000, help="Local server port")
    args = parser.parse_args()

    # Load metadata
    with open(args.meta, "r") as f:
        CONFIG["meta"] = json.load(f)
    CONFIG["dataset_dir"] = Path(args.dataset)

    # Launch server
    print(f"📡 Image server starting on port {args.port}...")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
