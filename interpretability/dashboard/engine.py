import json
import cv2
import argparse
import torch
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
import uvicorn
import time
import torch.nn as nn

# LeWM / LeRobot Imports
from lewm.lewm_data_plugin import LEWMDataPlugin
from lewm.goal_mapper import GoalMapper
from interpretability.transcoders.universal_transcoder import Transcoder

app = FastAPI(title="LeWM Interpretability Engine")

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

    def __init__(self, model, transcoders, transform, device="cuda"):
        self.model = model
        self.transcoders = transcoders
        self.transform = transform
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
                # Use 'forward' as Transcoder doesn't have 'encode'
                res = self.transcoders[lid](val)
                self.activations[lid] = res["activations"].detach()

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
        Runs attribution to find feature and input importance.
        """
        self.activations.clear()
        self.gradients.clear()
        self._register_hooks()

        # 1. Setup Inputs with Gradient Tracking
        pixel_key = (
            "pixels" if "pixels" in sample else "observation.images.world_center"
        )
        raw_pixels = sample[pixel_key]  # [T, C, H, W]

        # Apply official preprocessor
        processed = self.transform({"pixels": raw_pixels})
        pixels = processed["pixels"].to(self.device).float().detach()

        # Ensure 5D: [B, T, C, H, W]
        if pixels.ndim == 3:  # [C, H, W]
            pixels = pixels.unsqueeze(0).unsqueeze(0)
        elif pixels.ndim == 4:  # [T, C, H, W]
            pixels = pixels.unsqueeze(0)
        pixels.requires_grad_(True)

        state_key = "action" if "action" in sample else "observation.state"
        state = sample[state_key].to(self.device).float().detach()
        # Ensure 3D: [B, T, D]
        if state.ndim == 1:
            state = state.unsqueeze(0).unsqueeze(0)
        elif state.ndim == 2:
            state = state.unsqueeze(0)
        state.requires_grad_(True)

        # 2. Forward Pass
        print(f"DEBUG: Input Pixels Shape: {pixels.shape}")
        print(f"DEBUG: Input State Shape: {state.shape}")

        info = self.model.encode({"pixels": pixels, "action": state})
        print(f"DEBUG: Encoded Emb Shape: {info['emb'].shape}")

        logits = self.model.predict(info["emb"], info["act_emb"])
        print(f"DEBUG: Logits Shape: {logits.shape}")

        # Target: Final action step, specific logit
        target = logits[0, -1, target_logit_idx]

        # 3. Backward Pass (Causal Trace)
        target.backward()

        # Capture Input Saliency
        pixel_grad = pixels.grad.detach().cpu()
        state_grad = state.grad.detach().cpu()

        # 4. Build Multi-Modal Graph
        nodes = []
        edges = []

        # A. Add Logit Node (Root)
        nodes.append(
            {
                "id": "logit_0",
                "type": "logit",
                "label": f"Action_{target_logit_idx}",
                "value": float(target),
            }
        )

        # B. Add Input Layer: Visual Patches (Top Saliency)
        patch_saliency = self._aggregate_spatial_grad(pixel_grad)
        top_patches = torch.topk(patch_saliency.view(-1), k=15)
        for v, idx in zip(top_patches.values, top_patches.indices):
            row, col = divmod(int(idx), 16)
            nodes.append(
                {
                    "id": f"patch_{idx}",
                    "type": "patch",
                    "label": f"Patch[{row},{col}]",
                    "value": float(v),
                    "metadata": {"row": row, "col": col},
                }
            )

        # C. Add Input Layer: Proprioception (State)
        state_saliency = state_grad.abs().view(-1)
        top_states = torch.topk(state_saliency, k=5)
        for v, idx in zip(top_states.values, top_states.indices):
            idx = int(idx)
            nodes.append(
                {
                    "id": f"state_{idx}",
                    "type": "state",
                    "label": self._get_state_label(idx),
                    "value": float(v),
                }
            )

        # D. Add Transcoder Features and link to Logit
        feature_nodes = []
        for lid, act in self.activations.items():
            grad = self.gradients.get(lid)
            # Project layer gradient to feature space using decoder weights
            # grad: [B, T, D_model], decoder.weight: [D_model, D_dict]
            # act: [B, T, D_dict]
            with torch.no_grad():
                W_dec = self.transcoders[lid].decoder.weight.data
                # Projects D_model grad to D_dict space
                feat_grad = torch.matmul(grad, W_dec)

            # Attribution = Activation * Feature Gradient
            influence = (act * feat_grad).view(-1)
            top_vals, top_idx = torch.topk(influence.abs(), k=15)

            for v, i in zip(top_vals, top_idx):
                i = int(i)
                node_id = f"feat_{lid}_{i}"
                nodes.append(
                    {
                        "id": node_id,
                        "type": "feature",
                        "layer": lid,
                        "index": i,
                        "value": float(v),
                    }
                )
                feature_nodes.append((node_id, lid, i, v))
                edges.append(
                    {"source": node_id, "target": "logit_0", "weight": float(v)}
                )

        # E. Link Inputs to Features (Hierarchical Circuit)
        for node_id, lid, feat_idx, feat_val in feature_nodes:
            tc = self.transcoders.get(lid)
            if tc is None:
                continue

            # Find which tokens contribute to this feature's activation
            weights = tc.encoder.weight[feat_idx].abs()
            top_weights = torch.topk(weights, k=3)

            for w_val, w_idx in zip(top_weights.values, top_weights.indices):
                w_idx = int(w_idx)

                if w_idx == 0:
                    # CLS Token link
                    source_id = "cls_token"
                    if not any(n["id"] == source_id for n in nodes):
                        nodes.append({"id": source_id, "type": "input", "label": "CLS"})
                elif 1 <= w_idx <= 256:
                    # Visual Patch link (Adjusted for CLS offset)
                    patch_idx = w_idx - 1
                    source_id = f"patch_{patch_idx}"
                    # Ensure patch node exists (might not be in top saliency but relevant to this feature)
                    if not any(n["id"] == source_id for n in nodes):
                        r, c = divmod(patch_idx, 16)
                        nodes.append(
                            {
                                "id": source_id,
                                "type": "patch",
                                "label": f"Patch[{r},{c}]",
                                "metadata": {"row": r, "col": c},
                            }
                        )
                else:
                    # Proprioception or other
                    source_id = f"state_{w_idx - 257}"

                edges.append(
                    {
                        "source": source_id,
                        "target": node_id,
                        "weight": float(w_val * feat_val),
                    }
                )

        # Cleanup hooks
        self._cleanup_hooks()
        return {"nodes": nodes, "edges": edges}

    def _aggregate_spatial_grad(self, pixel_grad):
        # pixel_grad: [1, 3, 224, 224]
        # Aggregate across color channels and take absolute value
        saliency = pixel_grad.abs().sum(dim=1)[0]
        # Downsample to 16x16 patches using area pooling
        saliency = saliency.unsqueeze(0).unsqueeze(0)
        patch_grad = torch.nn.functional.avg_pool2d(saliency, kernel_size=14, stride=14)
        return patch_grad.view(16, 16)

    def _get_state_label(self, idx):
        # Basic mapping for GR-1 / LeRobot standard
        labels = {
            0: "joint_0",
            1: "joint_1",
            2: "joint_2",
            3: "joint_3",
            4: "joint_4",
            5: "joint_5",
            6: "joint_6",
            7: "gripper_pos",
            8: "gripper_vel",
        }
        return labels.get(idx, f"dim_{idx}")

    def _cleanup_hooks(self):
        for h_f, h_b in self.hooks.values():
            h_f.remove()
            h_b.remove()
        self.hooks.clear()


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

    # Neuronpedia sends 'prompt'. We treat it as 'index' or 'index:joint'
    prompt = request.get("prompt", "0")
    try:
        clean_prompt = str(prompt).replace("<bos>", "").strip()
        if ":" in clean_prompt:
            parts = clean_prompt.split(":")
            sample_idx = int(parts[0])
            target_logit_idx = int(parts[1])
        else:
            sample_idx = int(clean_prompt)
            target_logit_idx = 7  # Default to Joint 7
    except Exception as e:
        print(f"Error parsing prompt '{prompt}': {e}")
        sample_idx = 0
        target_logit_idx = 7

    try:
        model = STATE["model"]
        transcoders = STATE["transcoders"]
        transform = STATE["transform"]
        attributor = LeWMAttributor(model, transcoders, transform)

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

    # 1. Dataset (High-Performance Direct Bypass)
    print(f"🚀 Initializing LEWM Data Plugin for {dataset_repo}")
    STATE["dataset"] = LEWMDataPlugin(
        repo_id=dataset_repo,
        keys_to_load=["pixels", "state", "action"],
        num_steps=1,
    )

    # 2. Model
    print(f"🧠 Loading LeWM Model: {model_path}")
    mapper = GoalMapper(model_path=model_path, dataset_root=".")
    STATE["model"] = mapper.model.to(device).eval()
    STATE["transform"] = mapper.transform

    # 3. Transcoders (Auto-Discovery)
    tc_path = Path(transcoder_dir)
    if tc_path.exists():
        print(f"🔍 Discovering Transcoders in {transcoder_dir}...")

        for path in tc_path.glob("*.pt"):
            layer_id = path.stem.split("_clt")[0].split("_sae")[0]

            t0 = time.time()
            # 1. Load checkpoint
            checkpoint = torch.load(path, map_location="cpu")
            state_dict = checkpoint["state_dict"]
            t_load = time.time() - t0

            # 2. Extract dimensions
            d_dict, d_model = state_dict["encoder.weight"].shape

            # 3. Initialize Transcoder
            t1 = time.time()
            tc = Transcoder(d_model, d_dict)
            t_init = time.time() - t1

            # 4. Move to Device
            t2 = time.time()
            tc = tc.to(device)
            t_dev = time.time() - t2

            # 5. Load weights
            t3 = time.time()
            tc.load_state_dict(state_dict)
            STATE["transcoders"][layer_id] = tc.eval()
            t_sd = time.time() - t3

            total = time.time() - t0
            print(
                f"  ✅ Linked: {layer_id:15} | Total: {total:.2f}s (Load: {t_load:.2f}s, Init: {t_init:.2f}s, Dev: {t_dev:.2f}s, SD: {t_sd:.2f}s)"
            )
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
