"""
ZERO-RAM FULL-STACK HARVESTER (Production Edition)
Role: Captures all 18 layers and streams directly to disk.
Output: .bin (raw data) and .json (metadata) for each layer.
"""

import sys
import torch
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader

# --- Path Stabilization (Robust Absolute Resolution) ---
# We use Path(__file__).resolve() to make the script immune to launch directory
CURRENT_FILE = Path(__file__).resolve()
ROOT_DIR = CURRENT_FILE.parents[2]  # le-probe/
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

LEWM_DIR = ROOT_DIR / "lewm"
if str(LEWM_DIR) not in sys.path:
    sys.path.append(str(LEWM_DIR))

LE_WM_DIR = LEWM_DIR / "le_wm"
if str(LE_WM_DIR) not in sys.path:
    sys.path.append(str(LE_WM_DIR))

from lewm.goal_mapper import GoalMapper
from lewm.lewm_data_plugin import LEWMDataPlugin


class TraceHook:
    """Native PyTorch hook for surgical activation capture."""

    def __init__(self):
        self.output = None

    def __call__(self, module, input, output):
        # Handle ViT output tuples (hidden_states, ...)
        val = output[0] if isinstance(output, tuple) else output
        self.output = val.detach().cpu().numpy()


def harvest_activations(
    model_path, dataset_repo, output_dir, num_episodes, shuffle=False, num_workers=2
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Initializing Production Harvest | Device: {device}")
    print(f"📁 Output Directory: {output_path}")

    # 1. Load Model
    mapper = GoalMapper(model_path=model_path, dataset_root=".")
    model = mapper.model.to(device).eval()

    # 2. Unified Discovery & Hooking (String-based resolution)
    hooks = {}
    handles = []

    print("🔍 Discovering model layers...")
    for name, module in model.named_modules():
        parts = name.split(".")
        # Pattern: encoder.encoder.layer.N or predictor.transformer.layers.N
        if (
            len(parts) > 1
            and (parts[-2] == "layer" or parts[-2] == "layers")
            and parts[-1].isdigit()
        ):
            layer_idx = parts[-1]
            component = "encoder" if "encoder" in name else "predictor"
            layer_id = f"{component}_L{layer_idx}"

            hook = TraceHook()
            handle = module.register_forward_hook(hook)
            hooks[layer_id] = hook
            handles.append(handle)
            print(f"  ⚓ Hooked: {layer_id} ({name})")

    if not hooks:
        raise RuntimeError(
            "🚨 Discovery Failure: No layers were identified for hooking!"
        )

    # 3. Initialize Data Plugin
    data_plugin = LEWMDataPlugin(
        repo_id=dataset_repo, keys_to_load=["pixels", "action"], num_steps=3
    )

    # 🧊 CACHE RESET: Prevent 'Invalid data' decoder contention across workers
    data_plugin.clear_cache()

    dataloader = DataLoader(
        data_plugin,
        batch_size=32,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    actual_total = num_episodes if num_episodes > 0 else len(dataloader)

    # Pre-flight Diagnostic: Full Causal Ghost Trace
    print("👻 Running System-Wide Ghost Trace (FP16 enabled)...")
    with torch.no_grad():
        with torch.amp.autocast("cuda"):  # Mixed Precision
            dummy_pixels = torch.randn(1, 3, 3, 224, 224).to(device)
            dummy_actions = torch.zeros(1, 3, 32).to(device)

            # Trigger Encoder hooks
            info = model.encode({"pixels": dummy_pixels, "action": dummy_actions})
            # Trigger Predictor hooks
            model.predict(info["emb"], info["act_emb"])

        for layer_id, hook in hooks.items():
            if hook.output is None:
                raise RuntimeError(
                    f"🚨 Pre-flight Failure: Layer {layer_id} is unresponsive!"
                )
    print(f"✅ System Green: All {len(hooks)} layers verified.")

    # 4. Open Streaming Files with large buffers
    file_handles = {
        layer_id: open(
            output_path / f"{layer_id}.bin", "wb", buffering=1024 * 1024
        )  # 1MB buffer
        for layer_id in hooks.keys()
    }

    total_samples = {layer_id: 0 for layer_id in hooks.keys()}
    tokens_per_sample = {}
    last_shape = {}

    # 5. Harvesting Loop
    print(f"📊 Streaming vertical slices from {actual_total} batches...")

    try:
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Harvesting", total=actual_total)
            for i, batch in enumerate(pbar):
                if num_episodes > 0 and i >= num_episodes:
                    break

                raw_pixels = batch["pixels"].to(device)
                actions = batch["action"].to(device)

                # --- 🎯 OFFICIAL TRANSFORM (ImageNet + Resizing) ---
                # This ensures the model sees EXACTLY what it saw during v8 training.
                B, T, C, H, W = raw_pixels.shape
                # Flatten B and T for the preprocessor
                raw_pixels_flat = raw_pixels.view(B * T, C, H, W)
                processed_pixels = mapper.transform({"pixels": raw_pixels_flat})[
                    "pixels"
                ]
                # Unflatten back to (B, T, C, 224, 224)
                pixels = processed_pixels.view(B, T, C, 224, 224)
                # ---------------------------------------------------

                if torch.isnan(actions).any():
                    actions = torch.nan_to_num(actions, 0.0)

                # TRIGGER THE CAUSAL CHAIN in Mixed Precision
                with torch.amp.autocast("cuda"):
                    # 1. Perception (Encoder)
                    info = model.encode({"pixels": pixels, "action": actions})
                    # 2. Intention (Predictor)
                    model.predict(info["emb"], info["act_emb"])

                # Stream results to disk in FP16
                for layer_id, hook in hooks.items():
                    acts = hook.output  # (B, T, D)
                    if acts is None:
                        raise RuntimeError(
                            f"🚨 Trace Failure: Layer {layer_id} did not report activations!"
                        )

                    # Explicit cast to float16 to save 50% disk IO/Space
                    acts_flat = acts.reshape(-1, acts.shape[-1]).astype(np.float16)
                    file_handles[layer_id].write(acts_flat.tobytes())
                    total_samples[layer_id] += acts_flat.shape[0]
                    last_shape[layer_id] = acts_flat.shape[1]
                    tokens_per_sample[layer_id] = acts_flat.shape[0] // pixels.shape[0]

    finally:
        # 6. Cleanup hooks and Close files
        for h in handles:
            h.remove()
        for f in file_handles.values():
            f.close()

    # 7. Save Metadata Sidecars
    print("💾 Finalizing metadata headers...")
    for layer_id in hooks.keys():
        # Calculate tokens per sample (Spatial * Temporal)
        # We use the last known shape but divided by the logic
        metadata = {
            "shape": [total_samples[layer_id], last_shape[layer_id]],
            "tokens_per_sample": tokens_per_sample[layer_id],
            "dtype": "float16",
            "layer_id": layer_id,
        }
        with open(output_path / f"{layer_id}.json", "w") as f:
            json.dump(metadata, f)

    print(f"✨ Harvest Complete. Streaming data stored in {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gr1_reward_tuned_v2.ckpt")
    parser.add_argument("--dataset", type=str, default="vedpatwardhan/gr1_pickup_grasp")
    parser.add_argument("--output_dir", type=str, default="harvested_activations")
    parser.add_argument("--batches", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    harvest_activations(
        model_path=args.model,
        dataset_repo=args.dataset,
        output_dir=args.output_dir,
        num_episodes=args.batches,
        shuffle=args.shuffle,
        num_workers=args.workers,
    )
