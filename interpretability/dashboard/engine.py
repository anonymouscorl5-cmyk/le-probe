import json
import cv2
import argparse
import torch
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException, Response
from typing import List, Dict, Any, Optional
import uvicorn

# LeWM / LeRobot Imports
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lewm.goal_mapper import GoalMapper
from interpretability.transcoders.universal_transcoder import Transcoder

app = FastAPI(title="LeWM Interpretability Engine")

# --- Global Engine State ---
STATE = {
    "model": None,
    "dataset": None,
    "transcoders": {},
    "meta": None,
}

# --- 1. VISUAL ENDPOINTS (Full Parity with colab_bridge.py) ---


@app.get("/api/robot-dataset/frames/{idx}.jpg")
async def get_frame(idx: int):
    """
    Maps a global token index to a dataset sample and extracts the corresponding frame.
    Ported logic handles temporal history (3 frames) and world_center modality.
    """
    meta = STATE["meta"]
    dataset = STATE["dataset"]
    if not dataset or not meta:
        raise HTTPException(status_code=500, detail="Engine resources not initialized")

    try:
        # 1. Map global token index to sample index and patch
        tokens_per_sample = meta.get("tokens_per_sample", 771)
        sample_idx = idx // tokens_per_sample
        token_in_sample = idx % tokens_per_sample

        # 2. Determine frame offset and patch index (History Size = 3)
        frame_offset = token_in_sample // 257
        patch_token_idx = token_in_sample % 257  # 0 is CLS, 1-256 are patches
        target_sample_idx = max(0, sample_idx - frame_offset)

        if target_sample_idx >= len(dataset):
            raise HTTPException(status_code=404, detail="Sample index out of range")

        sample = dataset[target_sample_idx]

        # 3. Extract Modality (STRICT: world_center)
        img_key = next((k for k in sample.keys() if "world_center" in k), None)
        if not img_key:
            raise HTTPException(
                status_code=404, detail="Modality 'world_center' not found"
            )

        img_tensor = sample[img_key]
        img_np = (
            img_tensor.permute(1, 2, 0).cpu().numpy()
            if hasattr(img_tensor, "permute")
            else img_tensor.transpose(1, 2, 0)
        )

        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype("uint8")
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        display_size = 480
        img_bgr = cv2.resize(img_bgr, (display_size, display_size))

        # 4. Draw Spatial Highlighting (Green)
        if patch_token_idx > 0:
            p = patch_token_idx - 1
            grid_size, patch_px = 16, display_size // 16
            row, col = p // grid_size, p % grid_size
            x1, y1, x2, y2 = (
                col * patch_px,
                row * patch_px,
                (col + 1) * patch_px,
                (row + 1) * patch_px,
            )

            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(
                img_bgr,
                f"P{p}",
                (x1 + 2, y1 + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
            )

        _, buffer = cv2.imencode(".jpg", img_bgr)
        return Response(content=buffer.tobytes(), media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- 2. ATTRIBUTION ENDPOINTS (New Phase 2 Features) ---


@app.post("/api/attribution/generate-graph")
async def generate_graph(request: Dict[str, Any]):
    """
    Placeholder for Phase 2: Hierarchical Circuit Tracing.
    Will utilize STATE['model'] and STATE['transcoders'] for IG calculation.
    """
    return {"status": "ready", "engine": "LeWM-Attributor-v1"}


# --- 3. MAIN BOOTSTRAP ---


def main():
    parser = argparse.ArgumentParser(description="LeWM Unified Interpretability Engine")
    parser.add_argument(
        "--meta", type=str, required=True, help="Path to layer metadata JSON"
    )
    parser.add_argument(
        "--repo", type=str, required=True, help="Hugging Face Dataset Repo"
    )
    parser.add_argument("--model", type=str, default="gr1_reward_tuned_v2.ckpt")
    parser.add_argument("--transcoders", type=str, default="transcoder_checkpoints")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    # Load Resources
    print(f"📦 Loading Dataset: {args.repo}")
    STATE["dataset"] = LeRobotDataset(args.repo)

    with open(args.meta, "r") as f:
        STATE["meta"] = json.load(f)

    print(f"📡 Engine starting on port {args.port}...")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
