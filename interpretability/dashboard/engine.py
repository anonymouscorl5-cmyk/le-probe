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


class LeWMAttributor:
    """
    Computes hierarchical attribution for the LeWM model.
    Traces influence from Action Logits -> Predictor Features -> Encoder Features -> Visual Tokens.
    """

    def __init__(self, model, transcoders, device="cuda"):
        self.model = model
        self.transcoders = transcoders
        self.device = device
        self.hooks = {}
        self.activations = {}
        self.gradients = {}

    def _register_hooks(self):
        """Registers forward and backward hooks to capture SAE latents and their gradients."""
        for layer_id, tc in self.transcoders.items():

            def forward_hook(module, input, output, lid=layer_id):
                # We need the underlying activation to feed into the transcoder
                val = output[0] if isinstance(output, tuple) else output
                # Transcode and capture latents
                with torch.no_grad():
                    self.activations[lid] = self.transcoders[lid].encode(val)

            def backward_hook(module, grad_input, grad_output, lid=layer_id):
                # Captured gradient of the layer activation
                g = grad_output[0] if isinstance(grad_output, tuple) else grad_output
                self.gradients[lid] = g

            # Find the actual module in LeWM (Encoder/Predictor)
            # This logic must match harvest_activations.py's discovery
            target_module = self._find_module(layer_id)
            if target_module:
                h_f = target_module.register_forward_hook(forward_hook)
                h_b = target_module.register_full_backward_hook(backward_hook)
                self.hooks[layer_id] = (h_f, h_b)

    def _find_module(self, layer_id):
        # Pattern: encoder_L3 or predictor_L1
        component, idx = layer_id.split("_L")
        idx = int(idx)
        if component == "encoder":
            return self.model.encoder.encoder.layer[idx]
        elif component == "predictor":
            return self.model.predictor.transformer.layers[idx]
        return None

    def attribute(self, sample, target_logit_idx, steps=20):
        """
        Runs Integrated Gradients to find feature importance.
        Returns a dictionary of nodes and edges for Neuronpedia.
        """
        self.activations.clear()
        self.gradients.clear()
        self._register_hooks()

        # 1. Setup Input (Standard LeRobot format)
        pixels = sample["pixels"].to(self.device).requires_grad_(True)
        actions = sample["action"].to(self.device)

        # 2. Forward & Backward
        # For simplicity, we'll use direct attribution (Act x Grad) for the MVP
        info = self.model.encode({"pixels": pixels, "action": actions})
        logits = self.model.predict(info["emb"], info["act_emb"])

        target = logits[0, -1, target_logit_idx]  # Final step, target joint/action
        target.backward()

        # 3. Build Graph
        nodes = []
        edges = []

        # Add Logit Node (Root)
        nodes.append(
            {
                "id": "logit_0",
                "type": "logit",
                "label": f"Action_{target_logit_idx}",
                "value": float(target),
            }
        )

        # Add Transcoder Features
        for lid, act in self.activations.items():
            grad = self.gradients.get(lid)
            if grad is None:
                continue

            # Decompose influence: Latent * (dLogit/dLatent)
            # We approximate dLogit/dLatent using the chain rule: dLogit/dAct * dAct/dLatent
            # For Linear Decoders: dAct/dLatent = Decoder.Weight
            # attr = act * (grad @ tc.decoder.weight.T)

            # Simple version: Top-K active features in this layer
            top_vals, top_idx = torch.topk(act.view(-1), k=20)
            for v, i in zip(top_vals, top_idx):
                nodes.append(
                    {
                        "id": f"feat_{lid}_{i}",
                        "type": "feature",
                        "layer": lid,
                        "index": int(i),
                        "value": float(v),
                    }
                )
                # Add edge to root (for now)
                edges.append(
                    {
                        "source": f"feat_{lid}_{i}",
                        "target": "logit_0",
                        "weight": float(v),
                    }
                )

        # Cleanup
        for h_f, h_b in self.hooks.values():
            h_f.remove()
            h_b.remove()
        self.hooks.clear()

        return {"nodes": nodes, "edges": edges}


# --- FastAPI Endpoints ---


@app.post("/api/attribution/generate-graph")
async def generate_graph(request: Dict[str, Any]):
    """
    Main entry point for Hierarchical Attribution Graphs.
    Traces causality from Action Head Logits -> Transcoder Features -> Visual Patches.
    """
    dataset = STATE["dataset"]
    model = STATE["model"]
    transcoders = STATE["transcoders"]

    if not all([dataset, model, transcoders]):
        raise HTTPException(
            status_code=500, detail="Engine resources (model/transcoders) not loaded"
        )

    sample_idx = request.get("sample_idx", 0)
    target_logit_idx = request.get("target_logit_idx", 0)

    try:
        attributor = LeWMAttributor(model, transcoders)
        sample = dataset[sample_idx]

        # Ensure batch dimension
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                sample[k] = v.unsqueeze(0)

        graph = attributor.attribute(sample, target_logit_idx)
        return graph

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Attribution failed: {str(e)}")


# --- 3. MAIN BOOTSTRAP ---


def load_engine_resources(model_path, dataset_repo, transcoder_dir, device="cuda"):
    print(f"🚀 Initializing Engine Resources | Device: {device}")

    # 1. Dataset
    print(f"📦 Loading Dataset: {dataset_repo}")
    STATE["dataset"] = LeRobotDataset(dataset_repo)

    # 2. Model
    print(f"🧠 Loading LeWM Model: {model_path}")
    mapper = GoalMapper(model_path=model_path, dataset_root=".")
    STATE["model"] = mapper.model.to(device).eval()

    # 3. Transcoders (Auto-Discovery)
    tc_path = Path(transcoder_dir)
    if tc_path.exists():
        print(f"🔍 Discovering Transcoders in {transcoder_dir}...")
        for path in tc_path.glob("*.pt"):
            # Expected format: encoder_L3_clt.pt or predictor_L1_clt.pt
            layer_id = path.stem.split("_clt")[0].split("_sae")[0]

            checkpoint = torch.load(path, map_location="cpu")
            config = checkpoint["config"]

            # Initialize Transcoder with correct dims
            tc = Transcoder(
                config["dict_size"], config["dict_size"]
            )  # Placeholder for real dims
            # Use state dict dims to be safe
            d_model = checkpoint["state_dict"]["encoder.weight"].shape[1]
            tc = Transcoder(d_model, config["dict_size"]).to(device)

            tc.load_state_dict(checkpoint["state_dict"])
            STATE["transcoders"][layer_id] = tc.eval()
            print(f"  ✅ Linked: {layer_id}")
    else:
        print(f"⚠️ Warning: Transcoder directory {transcoder_dir} not found.")


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
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    # 1. Load Metadata
    with open(args.meta, "r") as f:
        STATE["meta"] = json.load(f)

    # 2. Load Compute Resources
    load_engine_resources(args.model, args.repo, args.transcoders, device=args.device)

    # 3. Start Server
    print(f"📡 Engine starting on port {args.port}...")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
