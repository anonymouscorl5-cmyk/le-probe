import json
import cv2
import argparse
from pathlib import Path
from fastapi import FastAPI, HTTPException, Response
import uvicorn
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

app = FastAPI()

# Global config to be set by CLI args
CONFIG = {"meta": None, "dataset": None}


@app.get("/api/robot-dataset/frames/{idx}.jpg")
async def get_frame(idx: int):
    """
    Maps a global token index to a dataset sample and extracts the corresponding frame.
    Also draws a red box if the token corresponds to a specific spatial patch.
    """
    meta = CONFIG["meta"]
    dataset = CONFIG["dataset"]

    if not dataset:
        raise HTTPException(status_code=500, detail="Dataset not initialized")

    # 1. Map global token index to sample index and patch
    tokens_per_sample = meta.get("tokens_per_sample", 771)
    sample_idx = idx // tokens_per_sample
    token_in_sample = idx % tokens_per_sample

    # 2. Determine frame offset and patch index
    # Sequence is [Frame0, Frame-1, Frame-2] (as per history_size=3)
    # Each frame has 257 tokens (1 CLS + 256 patches)
    frame_offset = token_in_sample // 257
    patch_token_idx = token_in_sample % 257  # 0 is CLS, 1-256 are patches

    target_sample_idx = max(0, sample_idx - frame_offset)

    if target_sample_idx >= len(dataset):
        raise HTTPException(status_code=404, detail="Sample index out of range")

    # 3. Extract Frame
    try:
        sample = dataset[target_sample_idx]

        # Find the image key (STRICT: world_center only)
        img_key = None
        for key in sample.keys():
            if "world_center" in key:
                img_key = key
                break

        if not img_key:
            raise HTTPException(
                status_code=404,
                detail="Strict modality 'world_center' not found in dataset",
            )

        # LeRobot returns tensors (C, H, W)
        img_tensor = sample[img_key]

        # Convert to NumPy (H, W, C)
        if hasattr(img_tensor, "permute"):
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        else:
            img_np = img_tensor.transpose(1, 2, 0)

        # Ensure uint8 and BGR for OpenCV
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype("uint8")

        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # 4. Resize for dashboard
        display_size = 480
        img_bgr = cv2.resize(img_bgr, (display_size, display_size))

        # 5. Draw Red Box if it's a spatial patch
        if patch_token_idx > 0:
            p = patch_token_idx - 1  # 0-255
            grid_size = 16  # 16x16 grid for 224x224 w/ patch_size 14
            patch_px = display_size // grid_size  # 480 / 16 = 30px

            row = p // grid_size
            col = p % grid_size

            x1, y1 = col * patch_px, row * patch_px
            x2, y2 = x1 + patch_px, y1 + patch_px

            # Draw red rectangle (BGR: 0, 0, 255)
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 3)
            # Add a subtle glow/label
            cv2.putText(
                img_bgr,
                f"P{p}",
                (x1 + 2, y1 + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1,
            )

        # 6. Return as JPEG
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

    if not args.dataset:
        repo_name = args.repo.split("/")[-1]
        dataset_path = Path("dataset") / repo_name
    else:
        dataset_path = Path(args.dataset)

    print(f"📦 Initializing dataset from {dataset_path}...")
    dataset = LeRobotDataset(args.repo or str(dataset_path), root=dataset_path.parent)
    CONFIG["dataset"] = dataset

    if len(dataset) > 0:
        print(f"✅ Dataset loaded. Available keys: {list(dataset[0].keys())}")

    with open(args.meta, "r") as f:
        CONFIG["meta"] = json.load(f)

    print(f"📡 Image server starting on port {args.port}...")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
