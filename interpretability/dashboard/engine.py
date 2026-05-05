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
        pixels_base = processed["pixels"].to(self.device).float().detach()

        # FORCE 4D: [B*T, C, H, W]
        # This is the key: we NEVER let the autograd graph see 5D
        if pixels_base.ndim == 5:
            pixels = pixels_base.view(-1, *pixels_base.shape[2:])
        elif pixels_base.ndim == 4:
            pixels = pixels_base
        else:  # 3D
            pixels = pixels_base.unsqueeze(0)

        pixels.requires_grad_(True)

        state_key = "action" if "action" in sample else "observation.state"
        state_base = sample[state_key].to(self.device).float().detach()
        # Ensure 2D: [B*T, D] for consistency
        if state_base.ndim == 3:
            state = state_base.view(-1, state_base.shape[-1])
        elif state_base.ndim == 2:
            state = state_base
        else:
            state = state_base.unsqueeze(0)
        state.requires_grad_(True)

        # 2. Forward Pass
        # Pixels are already 4D [B*T, C, H, W]
        # State is already 2D [B*T, D]
        print(f"DEBUG: Input Pixels Shape: {pixels.shape}")

        # 1. Vision Encoding (Directly on 4D)
        output = self.model.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]  # CLS token
        emb_flat = self.model.projector(pixels_emb)

        # 2. Unfold to [1, T, D] for the sequence-aware Predictor
        # We assume Batch=1 for attribution
        T_seq = pixels.shape[0]
        emb = emb_flat.view(1, T_seq, -1)

        # 3. Action Encoding
        # action_encoder expects [B, T, D]
        state_3d = state.view(1, T_seq, -1)
        act_emb = self.model.action_encoder(state_3d)

        print(f"DEBUG: Manually Encoded Emb Shape: {emb.shape}")

        # 4. Predict next state logits
        logits = self.model.predict(emb, act_emb)
        print(f"DEBUG: Logits Shape: {logits.shape}")

        # Target: Final action step, specific logit
        target = logits[0, -1, target_logit_idx]

        # 3. Backward Pass (Causal Trace)
        target.backward()

        # Capture Input Saliency
        pixel_grad = pixels.grad.detach().cpu()
        state_grad = state.grad.detach().cpu()

        # 4. Build compliant CLTGraph structure
        clt_nodes = []
        clt_links = []

        num_enc = 12
        num_pred = 6
        total_layers = num_enc + num_pred

        # Helper to track nodes for linking
        layer_to_nodes = {}  # layer_idx -> [node_data, ...]

        # A. Add Logit Node
        logit_id = "logit_0"
        logit_prob = float(torch.sigmoid(target))
        logit_node = {
            "node_id": logit_id,
            "feature": target_logit_idx,
            "layer": str(total_layers + 1),
            "ctx_idx": 0,
            "feature_type": "logit",
            "token_prob": logit_prob,
            "logitPct": logit_prob,
            "is_target_logit": True,
            "run_idx": 0,
            "reverse_ctx_idx": 0,
            "jsNodeId": logit_id,
            "streamIdx": total_layers + 1,
            "clerp": f"Action {target_logit_idx} (p={logit_prob:.3f})",
            "influence": float(target),
        }
        clt_nodes.append(logit_node)
        layer_to_nodes[total_layers + 1] = [logit_node]

        # B. Add Input Layer Nodes (Layer 0)
        layer_to_nodes[0] = []
        patch_saliency = self._aggregate_spatial_grad(pixel_grad)
        top_patches = torch.topk(patch_saliency.view(-1), k=15)
        for v, idx in zip(top_patches.values, top_patches.indices):
            idx = int(idx)
            row, col = divmod(idx, 16)
            node_id = f"patch_{idx}"
            node = {
                "node_id": node_id,
                "feature": idx,
                "layer": "E",
                "ctx_idx": 0,
                "feature_type": "patch",
                "token_prob": 1.0,
                "is_target_logit": False,
                "run_idx": 0,
                "reverse_ctx_idx": 0,
                "jsNodeId": node_id,
                "streamIdx": 0,
                "clerp": f"Patch[{row},{col}]",
                "influence": float(v),
            }
            clt_nodes.append(node)
            layer_to_nodes[0].append(node)

        state_saliency = state_grad.abs().view(-1)
        top_states = torch.topk(state_saliency, k=5)
        for v, idx in zip(top_states.values, top_states.indices):
            idx = int(idx)
            node_id = f"state_{idx}"
            node = {
                "node_id": node_id,
                "feature": idx,
                "layer": "E",
                "ctx_idx": 0,
                "feature_type": "state",
                "token_prob": 1.0,
                "is_target_logit": False,
                "run_idx": 0,
                "reverse_ctx_idx": 0,
                "jsNodeId": node_id,
                "streamIdx": 0,
                "clerp": self._get_state_label(idx),
                "influence": float(v),
            }
            clt_nodes.append(node)
            layer_to_nodes[0].append(node)

        # C. Process Transcoder Features
        # Order the layers properly
        layer_order = [f"encoder_L{i}" for i in range(num_enc)] + [
            f"predictor_L{i}" for i in range(num_pred)
        ]

        for lid in layer_order:
            act = self.activations.get(lid)
            grad = self.gradients.get(lid)
            if act is None or grad is None:
                continue

            try:
                comp, l_idx = lid.split("_L")
                l_idx = int(l_idx)
                if comp == "encoder":
                    stream_idx = l_idx + 1
                    layer_val = str(l_idx)
                else:  # predictor
                    stream_idx = l_idx + 1 + num_enc
                    layer_val = str(l_idx + num_enc)
            except:
                stream_idx = 1
                layer_val = "1"

            layer_to_nodes[stream_idx] = []

            with torch.no_grad():
                W_dec = self.transcoders[lid].decoder.weight.data  # [D_dict, D_model]
                feat_grad = torch.matmul(grad, W_dec)  # [T, D_dict]

            # act: [1, T, D_dict] where T is patches (encoder) or tokens (predictor)
            # Aggregate influence over the entire spatial/temporal dimension
            influence_per_feat = (act * feat_grad).sum(dim=1).view(-1)
            # Take max activation across tokens for the raw activation value
            max_act_per_feat = act.max(dim=1).values.view(-1)

            top_vals, top_idx = torch.topk(influence_per_feat.abs(), k=15)

            for v, feat_idx in zip(top_vals, top_idx):
                feat_idx = int(feat_idx)
                # Force ctx_idx to 0 to cluster all features vertically on the left
                token_idx = 0
                node_id = f"feat_{lid}_{feat_idx}"
                act_val = float(max_act_per_feat[feat_idx])

                node = {
                    "node_id": node_id,
                    "feature": feat_idx,
                    "layer": layer_val,
                    "ctx_idx": token_idx,
                    "feature_type": "feature",
                    "token_prob": act_val,
                    "is_target_logit": False,
                    "run_idx": 0,
                    "reverse_ctx_idx": 0,
                    "jsNodeId": node_id,
                    "streamIdx": stream_idx,
                    "clerp": f"F{feat_idx} ({lid})",
                    "influence": float(v),
                    "_raw_act": act_val,
                }
                clt_nodes.append(node)
                layer_to_nodes[stream_idx].append(node)

        # D. Build Causal Links (Path Tracing)
        for s_idx in range(total_layers + 1):
            curr_nodes = layer_to_nodes.get(s_idx + 1)
            prev_nodes = layer_to_nodes.get(s_idx)
            if not curr_nodes or not prev_nodes:
                continue

            for cn in curr_nodes:
                for pn in prev_nodes:
                    if cn["feature_type"] == "logit":
                        if cn["layer"] == str(total_layers + 1) and pn.get(
                            "layer"
                        ) == str(total_layers):
                            # Global influence for final action
                            clt_links.append(
                                {
                                    "source": pn["node_id"],
                                    "target": cn["node_id"],
                                    "weight": abs(pn["influence"]),
                                }
                            )
                    else:
                        try:
                            # 1. Handle Input -> Feature links
                            if pn["feature_type"] in ["patch", "state"]:
                                # Heuristic: combine raw influence
                                link_w = abs(pn["influence"] * cn["influence"])
                            else:
                                # 2. Handle Feature -> Feature links
                                lid_curr = (
                                    cn["node_id"].split("feat_")[1].rsplit("_", 1)[0]
                                )
                                lid_prev = (
                                    pn["node_id"].split("feat_")[1].rsplit("_", 1)[0]
                                )

                                f_idx_curr = int(cn["feature"])
                                f_idx_prev = int(pn["feature"])

                                W_dec_prev = self.transcoders[
                                    lid_prev
                                ].decoder.weight.data[:, f_idx_prev]
                                W_enc_curr = self.transcoders[
                                    lid_curr
                                ].encoder.weight.data[f_idx_curr]

                                # Connectivity * Source Activation * Target Influence (Grad)
                                cos_sim = torch.dot(W_dec_prev, W_enc_curr)
                                link_w = float(
                                    abs(cos_sim) * pn["_raw_act"] * abs(cn["influence"])
                                )

                            if link_w > 0.001:
                                clt_links.append(
                                    {
                                        "source": pn["node_id"],
                                        "target": cn["node_id"],
                                        "weight": link_w,
                                    }
                                )
                        except:
                            continue

        # Cleanup hooks
        self._cleanup_hooks()

        # E. Final Graph structure
        graph_data = {
            "metadata": {
                "slug": f"sample_{target_logit_idx}",
                "scan": "lewm-robot",
                "prompt_tokens": ["Robotic", "Frame"],
                "prompt": f"Sample {target_logit_idx}",
                "title_prefix": "Robotic Circuit",
                "schema_version": 0,
                "node_threshold": 99999,
                "neuronpedia_internal_model": {
                    "id": "lewm-robot",
                    "displayName": "LeWM Robotic Model",
                    "layers": total_layers,
                },
            },
            "qParams": {
                "linkType": "both",
                "pinnedIds": [],
                "clickedId": "",
                "supernodes": [],
                "sg_pos": "",
            },
            "nodes": clt_nodes,
            "links": clt_links,
        }
        return graph_data

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
            d_output = state_dict["decoder.weight"].shape[0]

            # 3. Initialize Transcoder
            t1 = time.time()
            tc = Transcoder(d_model, d_dict, d_output=d_output)
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

            if d_output > d_model:
                print(
                    f"  🔗 Info: {layer_id} is a Multi-Layer Crosscoder (d_output={d_output})"
                )

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
