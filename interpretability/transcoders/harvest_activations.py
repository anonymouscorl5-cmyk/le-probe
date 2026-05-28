"""
ZERO-RAM FULL-STACK HARVESTER (Production Edition)
Role: Captures all 18 layers and streams directly to disk.
Output: .bin (raw data) and .json (metadata) for each layer.

Experiment flags (parity with lewm_server.py / harvest_manifold.py):
  --multi_view --use_skeleton --use_dino
"""

import sys
import torch
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader

CURRENT_FILE = Path(__file__).resolve()
ROOT_DIR = CURRENT_FILE.parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from interpretability.lewm_experiment import (
    ExperimentConfig,
    TraceHook,
    add_experiment_args,
    build_data_plugin,
    build_goal_mapper,
    config_from_args,
    discover_layer_ids,
    flatten_activation,
    forward_harvest,
    ghost_trace_batch,
    prepare_pixels_6d,
    resolve_dataset_root,
)
from interpretability.transcoders.profile_harvest import profile_harvest


def harvest_activations(
    model_path,
    dataset_repo,
    output_dir,
    num_episodes,
    cfg: ExperimentConfig,
    shuffle=False,
    num_workers=2,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Initializing Production Harvest | Device: {device}")
    print(f"📁 Output Directory: {output_path}")
    print(
        f"🧪 Experiment: multi_view={cfg.multi_view}, skeleton={cfg.use_skeleton}, "
        f"dino={cfg.use_dino}, views={cfg.num_views}, cls_only={cfg.cls_only}"
    )

    dataset_root = resolve_dataset_root(dataset_repo)
    mapper = build_goal_mapper(model_path, dataset_root, cfg)
    model = mapper.model.to(device).eval()

    layer_paths = discover_layer_ids(model)
    hooks = {}
    handles = []

    print("🔍 Discovering model layers...")
    for layer_id, module_path in sorted(layer_paths.items()):
        module = dict(model.named_modules())[module_path]
        hook = TraceHook()
        handle = module.register_forward_hook(hook)
        hooks[layer_id] = hook
        handles.append(handle)
        print(f"  ⚓ Hooked: {layer_id} ({module_path})")

    if not hooks:
        raise RuntimeError(
            "🚨 Discovery Failure: No layers were identified for hooking!"
        )

    data_plugin = build_data_plugin(dataset_repo, cfg, num_steps=cfg.history_size)

    dataloader = DataLoader(
        data_plugin,
        batch_size=32,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    actual_total = num_episodes if num_episodes > 0 else len(dataloader)

    print("👻 Running System-Wide Ghost Trace (FP16 enabled)...")
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            g_pixels, g_actions, g_extra = ghost_trace_batch(device, cfg)
            g_batch = g_extra or {}
            forward_harvest(model, g_pixels, g_actions, cfg, g_batch)

        for layer_id, hook in hooks.items():
            if hook.output is None:
                raise RuntimeError(
                    f"🚨 Pre-flight Failure: Layer {layer_id} is unresponsive!"
                )
    print(f"✅ System Green: All {len(hooks)} layers verified.")

    file_handles = {
        layer_id: open(output_path / f"{layer_id}.bin", "wb", buffering=1024 * 1024)
        for layer_id in hooks.keys()
    }

    total_samples = {layer_id: 0 for layer_id in hooks.keys()}
    tokens_per_sample = {}
    last_shape = {}

    print(f"📊 Streaming vertical slices from {actual_total} batches...")

    try:
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Harvesting", total=actual_total)
            for i, batch in enumerate(pbar):
                if num_episodes > 0 and i >= num_episodes:
                    break

                pixels, actions = prepare_pixels_6d(batch, mapper, device, cfg)

                if i == 0:
                    print(
                        f"\n🔍 [BATCH 0] pixels={tuple(pixels.shape)} "
                        f"actions={tuple(actions.shape)}\n"
                    )

                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    forward_harvest(model, pixels, actions, cfg, batch)

                B = pixels.shape[0]
                for layer_id, hook in hooks.items():
                    acts = hook.output
                    if acts is None:
                        raise RuntimeError(
                            f"🚨 Trace Failure: Layer {layer_id} did not report activations!"
                        )

                    acts_flat = flatten_activation(acts, layer_id, cfg)
                    file_handles[layer_id].write(acts_flat.tobytes())
                    total_samples[layer_id] += acts_flat.shape[0]
                    last_shape[layer_id] = acts_flat.shape[1]
                    tokens_per_sample[layer_id] = max(1, acts_flat.shape[0] // B)

    finally:
        for h in handles:
            h.remove()
        for f in file_handles.values():
            f.close()
        data_plugin.clear_cache()

    print("💾 Finalizing metadata headers...")
    exp_meta = cfg.to_metadata()
    for layer_id in hooks.keys():
        metadata = {
            "shape": [total_samples[layer_id], last_shape[layer_id]],
            "tokens_per_sample": tokens_per_sample[layer_id],
            "dtype": "float16",
            "layer_id": layer_id,
            "experiment": exp_meta,
        }
        with open(output_path / f"{layer_id}.json", "w") as f:
            json.dump(metadata, f)

    print(f"✨ Harvest Complete. Streaming data stored in {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Harvest LeWM activations for transcoder training"
    )
    parser.add_argument("--model", type=str, default="gr1_reward_tuned_v2.ckpt")
    parser.add_argument("--dataset", type=str, default="gr1_pickup_grasp")
    parser.add_argument("--output_dir", type=str, default="activations_granular")
    parser.add_argument("--batches", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--history_size",
        type=int,
        default=3,
        help="Temporal history length (matches training)",
    )
    parser.add_argument(
        "--profile_batches",
        type=int,
        default=0,
        help="If >0, run profile_harvest for N batches and exit (no disk harvest)",
    )
    add_experiment_args(parser, include_cls_only=True)
    args = parser.parse_args()

    cfg = config_from_args(args)
    cfg.history_size = args.history_size

    if args.profile_batches > 0:
        profile_harvest(
            args.model,
            args.dataset,
            cfg,
            num_batches=args.profile_batches,
            num_workers=args.workers,
        )
        raise SystemExit(0)

    harvest_activations(
        model_path=args.model,
        dataset_repo=args.dataset,
        output_dir=args.output_dir,
        num_episodes=args.batches,
        cfg=cfg,
        shuffle=args.shuffle,
        num_workers=args.workers,
    )
