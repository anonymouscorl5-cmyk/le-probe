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

    # 2. Extract Frame using LeRobot's native indexing
    try:
        sample = dataset[sample_idx]

        # Find the image key (STRICT: world_center only)
        img_key = None
        for key in sample.keys():
            if "world_center" in key:
                img_key = key
                break

        if not img_key:
            raise HTTPException(status_code=404, detail="Strict modality 'world_center' not found in dataset")

        # LeRobot returns tensors (C, H, W)
        img_tensor = sample[img_key]

        # Convert to NumPy (H, W, C)
        # Handle both torch and numpy types
        if hasattr(img_tensor, "permute"):
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        else:
            img_np = img_tensor.transpose(1, 2, 0)

        # Ensure uint8 and BGR for OpenCV
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype("uint8")

        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # 3. Resize for dashboard
        img_bgr = cv2.resize(img_bgr, (480, 480))

        # 4. Return as JPEG
        _, buffer = cv2.imencode(".jpg", img_bgr)
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

    # Debug: show available keys
    if len(dataset) > 0:
        print(f"✅ Dataset loaded. Available keys: {list(dataset[0].keys())}")

    # Load metadata
    with open(args.meta, "r") as f:
        CONFIG["meta"] = json.load(f)

    # Launch server
    print(f"📡 Image server starting on port {args.port}...")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
