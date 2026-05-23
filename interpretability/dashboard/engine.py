import json
import cv2
import io
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
import traceback

# LeWM / LeRobot Imports
from interpretability.lewm_experiment import (
    ExperimentConfig,
    VIEW_NAMES,
    add_experiment_args,
    build_data_plugin,
    build_goal_mapper,
    config_from_args,
    decode_token_index,
    prepare_pixels_6d,
    resolve_dataset_root,
    resolve_layer_module,
    sample_rgb_frame,
)
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
    "cfg": None,
    "min_k": 15,
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
    cfg: ExperimentConfig = STATE.get("cfg") or ExperimentConfig()
    if not dataset or not meta:
        raise HTTPException(status_code=500, detail="Engine resources not initialized")

    try:
        tokens_per_sample = meta.get(
            "tokens_per_sample", cfg.encoder_tokens_per_moment()
        )
        sample_idx = idx // tokens_per_sample
        token_in_sample = idx % tokens_per_sample

        frame_offset, view_idx, patch_token_idx = decode_token_index(
            token_in_sample, cfg
        )
        target_sample_idx = max(0, sample_idx - frame_offset)

        if target_sample_idx >= len(dataset):
            raise HTTPException(status_code=404, detail="Sample index out of range")

        sample = dataset[target_sample_idx]
        img_tensor = sample_rgb_frame(sample, cfg, view_idx=view_idx, time_idx=0)
        img_np = (
            img_tensor.permute(1, 2, 0).cpu().numpy()
            if hasattr(img_tensor, "permute")
            else img_tensor.transpose(1, 2, 0)
        )
        if img_np.dtype != np.uint8:
            img_np = np.clip(img_np, 0, 255).astype("uint8")
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        display_size = 480
        img_bgr = cv2.resize(img_bgr, (display_size, display_size))

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
            view_name = VIEW_NAMES[view_idx] if cfg.multi_view else "world_center"
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(
                img_bgr,
                f"{view_name} P{p}",
                (x1 + 2, y1 + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
            )

        _, buffer = cv2.imencode(".jpg", img_bgr)
        return Response(content=buffer.tobytes(), media_type="image/jpeg")

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# --- 2. ATTRIBUTION ENDPOINTS (New Phase 2 Features) ---


class LeWMAttributor:
    """
    Computes hierarchical attribution for the LeWM model.
    Traces influence from Action Logits -> Predictor Features -> Encoder Features -> Visual Tokens.
    """

    def __init__(
        self,
        model,
        transcoders,
        mapper,
        cfg: ExperimentConfig,
        device="cuda",
        min_k=15,
    ):
        self.model = model
        self.transcoders = transcoders
        self.mapper = mapper
        self.cfg = cfg
        self.device = device
        self.min_k = min_k
        self.hooks = {}
        self.activations = {}
        self.gradients = {}

    def _register_hooks(self):
        """Registers forward and backward hooks to capture SAE latents and their gradients."""
        for layer_id, tc_data in self.transcoders.items():
            tc = tc_data["model"]
            stats = tc_data["stats"]

            def forward_hook(
                module, input, output, lid=layer_id, model=tc, s_stats=stats
            ):
                # 1. Capture raw layer activation
                val = output[0] if isinstance(output, tuple) else output

                # 2. Apply Source Normalization (Matching train_transcoder.py)
                mean_s = s_stats.get(
                    "src_mean", s_stats.get("mean", torch.zeros(1))
                ).to(val.device)
                std_s = s_stats.get("src_std", s_stats.get("std", torch.ones(1))).to(
                    val.device
                )
                val_norm = (val - mean_s) / (std_s + 1e-6)

                # 3. Transcode and capture latents
                res = model(val_norm)
                self.activations[lid] = res["activations"].detach()

            def backward_hook(module, grad_input, grad_output, lid=layer_id):
                # Captured gradient of the layer activation
                g = grad_output[0] if isinstance(grad_output, tuple) else grad_output
                self.gradients[lid] = g

            # Find the actual module in LeWM (Encoder/Predictor)
            target_module = resolve_layer_module(self.model, layer_id)
            if target_module:
                h_f = target_module.register_forward_hook(forward_hook)
                h_b = target_module.register_full_backward_hook(backward_hook)
                self.hooks[layer_id] = (h_f, h_b)

    def _compute_target(self, pixels, actions, batch_tensors, attribution_target):
        info = self.model.encode({"pixels": pixels, "action": actions})
        logits = self.model.predict(info["emb"], info["act_emb"])

        if (
            attribution_target == "subgoal"
            and batch_tensors.get("dino_anchor") is not None
        ):
            phi = batch_tensors["dino_anchor"]
            phase = batch_tensors.get("phase_idx")
            if phase is None:
                phase = torch.zeros(
                    phi.shape[0], phi.shape[1], 1, device=phi.device, dtype=phi.dtype
                )
            B, T, _ = phi.shape
            phi_flat = phi.reshape(B * T, -1)
            z_target = self.model.project_dino(phi_flat).view(B, T, -1)
            z_pred = self.model.predict_subgoal(info["emb"], phase)
            return -(z_pred - z_target.detach()).pow(2).mean()

        reward_out = self.model.reward_head(logits[:, -1, :])
        return reward_out.squeeze()

    def attribute(self, sample, target_logit_idx, steps=20):
        """
        Runs attribution to find feature and input importance.
        """
        try:
            self.activations.clear()
            self.gradients.clear()
            self._register_hooks()

            batch_tensors = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in sample.items()
            }
            pixels, actions = prepare_pixels_6d(
                batch_tensors, self.mapper, self.device, self.cfg
            )
            pixels = pixels.float().detach().requires_grad_(True)
            actions = actions.float().detach().requires_grad_(True)

            batch_extra = {
                k: batch_tensors[k]
                for k in ("dino_anchor", "phase_idx", "is_checkpoint")
                if k in batch_tensors
            }

            print(
                f"📈 Computing Integrated Gradients ({steps} steps) "
                f"[target={self.cfg.attribution_target}]..."
            )

            total_pixel_grad = torch.zeros_like(pixels)
            total_action_grad = torch.zeros_like(actions)
            total_trans_grads = {lid: 0 for lid in self.transcoders.keys()}

            for step in range(steps):
                alpha = (step + 1) / steps
                curr_pixels = (pixels.detach() * alpha).requires_grad_(True)
                curr_actions = (actions.detach() * alpha).requires_grad_(True)

                self.model.zero_grad()
                target = self._compute_target(
                    curr_pixels,
                    curr_actions,
                    batch_extra,
                    self.cfg.attribution_target,
                )
                target.backward()

                if curr_pixels.grad is not None:
                    total_pixel_grad += curr_pixels.grad.detach()
                if curr_actions.grad is not None:
                    total_action_grad += curr_actions.grad.detach()
                for lid in self.transcoders.keys():
                    if lid in self.gradients:
                        total_trans_grads[lid] += self.gradients[lid].detach()

            pixel_grad = (total_pixel_grad / steps) * pixels
            state_grad = (total_action_grad / steps) * actions

            # Update self.gradients with the averaged path gradients for features
            for lid in self.transcoders.keys():
                self.gradients[lid] = total_trans_grads[lid] / steps

            # Capture Input Saliency
            pixel_grad = pixel_grad.detach().cpu()
            state_grad = state_grad.detach().cpu()

            # 4. Build compliant CLTGraph structure
            clt_nodes = []
            clt_links = []

            num_enc = 12
            num_pred = 6
            total_layers = num_enc + num_pred

            # --- COMPRESSED JEPA LAYOUT ---
            # 0: IMG, 1-12: Encoder, 12 (Stacked): JOINT, 13-18: Predictor, 19: SUCCESS
            layer_to_nodes = {}

            # A. Add Target Node (Grasp Success)
            if self.cfg.attribution_target == "subgoal":
                logit_id = "logit_subgoal"
                success_prob = float(torch.sigmoid(-target.detach()))
                target_label = f"DINO Subgoal (align={success_prob:.3f})"
            else:
                logit_id = "logit_success"
                success_prob = float(torch.sigmoid(target.detach()))
                target_label = f"Grasp Success (p={success_prob:.3f})"
            logit_node = {
                "node_id": logit_id,
                "feature": 0,
                "layer": "Lgt",
                "probe_location_idx": -1,
                "ctx_idx": 0,
                "feature_type": "logit",
                "token_prob": success_prob,
                "logitPct": success_prob,
                "is_target_logit": True,
                "run_idx": 0,
                "reverse_ctx_idx": 0,
                "jsNodeId": logit_id,
                "streamIdx": 19,
                "clerp": target_label,
                "ppClerp": target_label,
                "influence": float(target.detach()),
            }
            clt_nodes.append(logit_node)
            layer_to_nodes[19] = [logit_node]

            # B. Add Visual Input Layer (Layer 0)
            layer_to_nodes[0] = []
            patch_saliency = self._aggregate_spatial_grad(pixel_grad)
            top_patches = torch.topk(patch_saliency.view(-1), k=15)
            view_idx = 0
            for i, (v, idx) in enumerate(zip(top_patches.values, top_patches.indices)):
                idx = int(idx)
                row, col = divmod(idx, 16)
                node_id = f"patch_{idx}"
                view_name = (
                    VIEW_NAMES[view_idx] if self.cfg.multi_view else "world_center"
                )
                node = {
                    "node_id": node_id,
                    "feature": idx,
                    "layer": "IMG",
                    "ctx_idx": i,
                    "feature_type": "patch",
                    "token_prob": 1.0,
                    "is_target_logit": False,
                    "run_idx": 0,
                    "reverse_ctx_idx": 0,
                    "jsNodeId": node_id,
                    "streamIdx": 0,
                    "clerp": f"{view_name} Patch[{row},{col}]",
                    "influence": float(v),
                }
                clt_nodes.append(node)
                layer_to_nodes[0].append(node)

            # C. Add Action Input Layer (Stacked in Layer 12 - Encoder Boundary)
            if 12 not in layer_to_nodes:
                layer_to_nodes[12] = []
            state_saliency = state_grad.abs().view(-1)
            top_states = torch.topk(state_saliency, k=10)
            for i, (v, idx) in enumerate(zip(top_states.values, top_states.indices)):
                idx = int(idx)
                node_id = f"state_{idx}"
                node = {
                    "node_id": node_id,
                    "feature": idx,
                    "layer": "IMG",  # Labeled as IMG/INPUT
                    "ctx_idx": i + 15,  # Follows the 15 image patches (0-14)
                    "feature_type": "state",
                    "token_prob": 1.0,
                    "is_target_logit": False,
                    "run_idx": 0,
                    "reverse_ctx_idx": 0,
                    "jsNodeId": node_id,
                    "streamIdx": 0,  # Back to Input Row
                    "clerp": self._get_state_label(idx),
                    "ppClerp": self._get_state_label(idx),
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
                tc_data = self.transcoders.get(lid)
                if act is None or grad is None or tc_data is None:
                    continue

                tc = tc_data["model"]
                stats = tc_data["stats"]

                try:
                    comp, l_idx_str = lid.split("_L")
                    l_idx = int(l_idx_str.split("_")[0])
                    if comp == "encoder":
                        stream_idx = l_idx + 1
                        layer_val = str(l_idx)
                    else:  # predictor
                        # Predictor starts at 13 (since L11/Joints are at 12)
                        stream_idx = l_idx + 13
                        layer_val = str(l_idx + num_enc)
                except:
                    stream_idx = 1
                    layer_val = "1"

                layer_to_nodes[stream_idx] = []

                with torch.no_grad():
                    # 1. Multi-Layer Gradient Aggregation (Crosscoder Support)
                    num_target_layers = tc.d_output // tc.d_model
                    start_idx = -1
                    if num_target_layers > 1:
                        # Find window of layers predicted by this crosscoder
                        curr_idx = layer_order.index(lid)
                        # Heuristic for residual window: L-1, L, L+1
                        start_idx = max(0, curr_idx - 1)
                        if curr_idx == 0:
                            start_idx = 0

                        window_ids = layer_order[
                            start_idx : start_idx + num_target_layers
                        ]
                        grads_to_cat = []
                        for wid in window_ids:
                            g = self.gradients.get(wid)
                            if g is not None:
                                # Resolution Alignment (Funnel Fix)
                                # If current layer (grad) is spatial (257) but window layer (g) is global (1)
                                if g.shape[1] != grad.shape[1]:
                                    if g.shape[1] == 1 and grad.shape[1] == 257:
                                        # Broadcast global to spatial
                                        g = g.expand(-1, grad.shape[1], -1)
                                    elif g.shape[1] == 257 and grad.shape[1] == 1:
                                        # Pool spatial to global
                                        g = g.mean(dim=1, keepdim=True)
                                grads_to_cat.append(g)
                            else:
                                # Pad with zeros if gradient missing (e.g. at end of model)
                                grads_to_cat.append(torch.zeros_like(grad))

                        agg_grad = torch.cat(grads_to_cat, dim=-1)
                    else:
                        agg_grad = grad

                    # 2. Calculate feature influence: Grad * Act
                    W_dec = tc.decoder.weight.data  # [D_dict, D_output]
                    std_t = stats.get("tgt_std", stats.get("std", torch.ones(1))).to(
                        agg_grad.device
                    )

                    scaled_grad = agg_grad * std_t  # Chain rule for normalization
                    feat_grad = torch.matmul(scaled_grad, W_dec)  # [T, D_dict]

                    # 3. Use Transcoder Sparse Activations (already captured in forward_hook)
                    sparse_acts = act
                    if sparse_acts.ndim > 2:
                        sparse_acts = sparse_acts.squeeze(0)

                # Calculate total influence of each feature across all tokens
                # sparse_acts: [T, D], feat_grad: [1, T, D]
                influence_per_feat = (sparse_acts * feat_grad.squeeze(0)).sum(dim=0)
                # Max activation across tokens for visual scaling
                max_act_per_feat = sparse_acts.max(dim=0).values.view(-1)

                top_vals, top_idx = torch.topk(influence_per_feat.abs(), k=15)

                for i, (v, feat_idx) in enumerate(zip(top_vals, top_idx)):
                    feat_idx = int(feat_idx)
                    # Spread features horizontally across the X-axis
                    token_idx = i
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
                        "_start_idx": start_idx if num_target_layers > 1 else -1,
                    }
                    clt_nodes.append(node)
                    layer_to_nodes[stream_idx].append(node)

            # Causal Links (Scan across all layers: 0-19)
            print(f"🔗 Tracing Global Jump Connections with Min-{self.min_k} Filter...")
            all_potential_links = []

            for s_idx in range(20):
                curr_nodes = layer_to_nodes.get(s_idx)
                if not curr_nodes:
                    continue

                for next_idx in range(s_idx + 1, 20):
                    next_nodes = layer_to_nodes.get(next_idx)
                    if not next_nodes:
                        continue

                    for cn in next_nodes:
                        for pn in curr_nodes:
                            try:
                                link_w = 0.0
                                # 1. Input -> Feature links
                                if pn["feature_type"] in [
                                    "patch",
                                    "state",
                                    "embedding",
                                ]:
                                    # CAUSAL CONSTRAINT: Actions only feed into Predictor (L13+) or Success (L19)
                                    if pn["feature_type"] == "state" and next_idx < 13:
                                        continue

                                    link_w = abs(pn["influence"] * cn["influence"])
                                # 2. Feature -> Logit links
                                elif cn["feature_type"] == "logit":
                                    link_w = abs(pn["influence"])
                                # 3. Feature -> Feature links (THE JUMP MECHANISM)
                                else:
                                    lid_curr = (
                                        cn["node_id"]
                                        .split("feat_")[1]
                                        .rsplit("_", 1)[0]
                                    )
                                    lid_prev = (
                                        pn["node_id"]
                                        .split("feat_")[1]
                                        .rsplit("_", 1)[0]
                                    )
                                    f_idx_curr, f_idx_prev = int(cn["feature"]), int(
                                        pn["feature"]
                                    )

                                    W_dec_prev = self.transcoders[lid_prev][
                                        "model"
                                    ].decoder.weight.data[:, f_idx_prev]
                                    W_enc_curr = self.transcoders[lid_curr][
                                        "model"
                                    ].encoder.weight.data[f_idx_curr]

                                    if W_dec_prev.shape[0] > W_enc_curr.shape[0]:
                                        s_idx_prev = pn.get("_start_idx", -1)
                                        if s_idx_prev != -1:
                                            rel_idx = next_idx - (s_idx_prev + 1)
                                            d_model = W_enc_curr.shape[0]
                                            if (
                                                0
                                                <= rel_idx
                                                < (W_dec_prev.shape[0] // d_model)
                                            ):
                                                W_dec_prev = W_dec_prev[
                                                    rel_idx
                                                    * d_model : (rel_idx + 1)
                                                    * d_model
                                                ]
                                            else:
                                                W_dec_prev = W_dec_prev[:d_model]
                                        else:
                                            W_dec_prev = W_dec_prev[
                                                : W_enc_curr.shape[0]
                                            ]

                                    cos_sim = torch.dot(W_dec_prev, W_enc_curr)
                                    link_w = float(
                                        abs(cos_sim)
                                        * pn["_raw_act"]
                                        * abs(cn["influence"])
                                    )

                                if link_w > 1e-6:
                                    all_potential_links.append(
                                        {
                                            "source": pn["node_id"],
                                            "target": cn["node_id"],
                                            "weight": link_w,
                                        }
                                    )
                            except:
                                continue

            # Apply Top-K per node constraint (Separate Incoming/Outgoing)
            outgoing_map = {}  # node_id -> list of links where this is source
            incoming_map = {}  # node_id -> list of links where this is target

            for link in all_potential_links:
                s, t = link["source"], link["target"]
                if s not in outgoing_map:
                    outgoing_map[s] = []
                if t not in incoming_map:
                    incoming_map[t] = []
                outgoing_map[s].append(link)
                incoming_map[t].append(link)

            final_link_set = set()

            # Keep Top-K Outgoing
            for nid, links in outgoing_map.items():
                links.sort(key=lambda x: x["weight"], reverse=True)
                for l in links[: self.min_k]:
                    final_link_set.add((l["source"], l["target"], l["weight"]))

            # Keep Top-K Incoming
            for nid, links in incoming_map.items():
                links.sort(key=lambda x: x["weight"], reverse=True)
                for l in links[: self.min_k]:
                    final_link_set.add((l["source"], l["target"], l["weight"]))

            for s, t, w in final_link_set:
                clt_links.append({"source": s, "target": t, "weight": w})

            # Cleanup hooks
            self._cleanup_hooks()

            # E. Final Graph structure
            graph_data = {
                "metadata": {
                    "slug": f"robot_trace_{idx}",
                    "scan": "lewm-robot",
                    "trace_id": idx,
                    "prompt_tokens": ["Robotic", "Frame"],
                    "prompt": f"Robotic Trace {idx}",
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
                "_top_patch_indices": top_patches.indices.tolist(),
            }
        except Exception as e:
            print(f"❌ Error processing layer: {e}")
            traceback.print_exc()
            raise e
        return graph_data

    def _aggregate_spatial_grad(self, pixel_grad, view_idx=0):
        """Aggregate IG pixel gradients to a 16x16 patch grid (world_center by default)."""
        if pixel_grad.ndim == 6:
            g = pixel_grad[0, -1, view_idx, :3].abs().sum(dim=0)
        elif pixel_grad.ndim == 5:
            g = pixel_grad[0, -1, :3].abs().sum(dim=0)
        elif pixel_grad.ndim == 4:
            g = pixel_grad[0, :3].abs().sum(dim=0)
        else:
            g = pixel_grad.abs().sum(dim=-3)
        saliency = g.unsqueeze(0).unsqueeze(0)
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
        mapper = STATE["mapper"]
        cfg = STATE["cfg"]
        attributor = LeWMAttributor(
            model, transcoders, mapper, cfg, min_k=STATE["min_k"]
        )

        sample = dataset[sample_idx]

        # Ensure batch dimension
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                sample[k] = v.unsqueeze(0)

        graph = attributor.attribute(sample, target_logit_idx)
        # Add metadata for the robotic gallery without overwriting
        graph["metadata"]["trace_id"] = sample_idx
        graph["metadata"]["patch_indices"] = graph.get("_top_patch_indices", [])
        return graph

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Attribution failed: {str(e)}")


@app.get("/api/robot-dataset/gallery/{idx}.jpg")
async def get_gallery(idx: int, patches: Optional[str] = None):
    """
    Generates a 2x5 grid of the 10 frames in a sample, with specific patches highlighted.
    """
    dataset = STATE["dataset"]
    if not dataset:
        raise HTTPException(status_code=500, detail="Dataset not loaded")

    try:
        sample = dataset[idx]
        pixels = sample["pixels"]  # [T, C, H, W]
        if pixels.ndim == 5:
            pixels = pixels[0]

        T = pixels.shape[0]
        patch_list = []
        if patches:
            patch_list = [int(p) for p in patches.split(",")]

        frames = []
        for t in range(min(T, 10)):
            img_tensor = pixels[t]
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy().astype("uint8")
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            img_bgr = cv2.resize(img_bgr, (224, 224))

            # Draw patches
            for p in patch_list:
                grid_size, patch_px = 16, 224 // 16
                row, col = p // grid_size, p % grid_size
                x1, y1 = col * patch_px, row * patch_px
                cv2.rectangle(
                    img_bgr, (x1, y1), (x1 + patch_px, y1 + patch_px), (0, 255, 0), 1
                )

            frames.append(img_bgr)

        # Create 2x5 grid
        rows = []
        for i in range(0, len(frames), 5):
            rows.append(np.hstack(frames[i : i + 5]))

        composite = np.vstack(rows)
        _, buffer = cv2.imencode(".jpg", composite)
        return Response(content=buffer.tobytes(), media_type="image/jpeg")

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# --- 3. MAIN BOOTSTRAP ---


def load_engine_resources(
    model_path, dataset_repo, transcoder_dir, cfg: ExperimentConfig, device="cuda"
):
    print(f"🚀 Initializing Engine Resources | Device: {device}")
    print(
        f"🧪 Experiment: multi_view={cfg.multi_view}, skeleton={cfg.use_skeleton}, "
        f"dino={cfg.use_dino}, attribution={cfg.attribution_target}"
    )

    dataset_root = resolve_dataset_root(dataset_repo)
    print(f"🚀 Initializing data plugin for {dataset_repo}")
    STATE["dataset"] = build_data_plugin(dataset_repo, cfg, num_steps=cfg.history_size)
    STATE["cfg"] = cfg

    print(f"🧠 Loading LeWM Model: {model_path}")
    mapper = build_goal_mapper(model_path, dataset_root, cfg)
    STATE["model"] = mapper.model.to(device).eval()
    STATE["mapper"] = mapper

    # 3. Transcoders (Auto-Discovery)
    tc_path = Path(transcoder_dir)
    if tc_path.exists():
        print(f"🔍 Discovering Transcoders in {transcoder_dir}...")

        for path in tc_path.glob("*.pt"):
            layer_id = path.stem.split("_clt")[0].split("_sae")[0].split("_residual")[0]

            t0 = time.time()
            # 1. Load checkpoint (Include Norm Stats)
            checkpoint = torch.load(path, map_location="cpu")
            state_dict = checkpoint["state_dict"]
            norm_stats = checkpoint.get("norm_stats", {})
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

            # Store Model + Stats for Normalization-Aware Attribution
            STATE["transcoders"][layer_id] = {"model": tc.eval(), "stats": norm_stats}
            t_sd = time.time() - t3

            if d_output > d_model:
                print(
                    f"  🔗 Info: {layer_id} is a Multi-Layer Crosscoder (d_output={d_output})"
                )

            total = time.time() - t0
            print(
                f"  ✅ Linked: {layer_id:15} | Total: {total:.2f}s (Stats: {'Yes' if 'mean' in norm_stats else 'No'})"
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
    parser.add_argument(
        "--min-k", type=int, default=15, help="Min-K connections per node"
    )
    parser.add_argument(
        "--history_size",
        type=int,
        default=3,
        help="Temporal history for dataset samples",
    )
    add_experiment_args(parser, include_cls_only=False, include_attribution_target=True)
    args = parser.parse_args()

    STATE["min_k"] = args.min_k

    with open(args.meta, "r") as f:
        STATE["meta"] = json.load(f)

    cfg = config_from_args(args)
    cfg.history_size = args.history_size
    if "experiment" in STATE["meta"]:
        meta_cfg = ExperimentConfig.from_metadata(STATE["meta"])
        cfg.cls_only = meta_cfg.cls_only

    load_engine_resources(
        args.model, args.repo, args.transcoders, cfg, device=args.device
    )

    # 3. Start Server
    print(f"📡 Engine starting on port {args.port}...")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
