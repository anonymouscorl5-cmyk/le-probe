"""
Profile harvest_activations bottlenecks on a few batches.

Usage (Colab, from interpretability/transcoders):
  python profile_harvest.py --model gr1_reward_tuned_v2.ckpt --multi_view --batches 5
  python profile_harvest.py --model gr1_reward_tuned_v2.ckpt --multi_view --cls_only --batches 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
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
    prepare_pixels_6d,
    resolve_dataset_root,
)

PATCHES_PER_FRAME = 257


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _estimate_encoder_rows(cfg: ExperimentConfig, batch_size: int = 32) -> int:
    per_sample = cfg.history_size * cfg.num_views
    if not cfg.cls_only:
        per_sample *= PATCHES_PER_FRAME
    return batch_size * per_sample


def _estimate_disk_mb_per_batch(
    cfg: ExperimentConfig, batch_size: int = 32, dim: int = 192
) -> float:
    """Rough FP16 bytes written per batch (all 18 layers)."""
    enc_rows = _estimate_encoder_rows(cfg, batch_size)
    pred_rows = batch_size * cfg.history_size
    enc_bytes = 12 * enc_rows * dim * 2
    pred_bytes = 6 * pred_rows * dim * 2
    return (enc_bytes + pred_bytes) / (1024 * 1024)


def profile_harvest(
    model_path: str,
    dataset_repo: str,
    cfg: ExperimentConfig,
    num_batches: int = 5,
    num_workers: int = 2,
    batch_size: int = 32,
    warmup_batches: int = 1,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📊 Harvest profile | device={device} | batches={num_batches}")
    print(
        f"   multi_view={cfg.multi_view} views={cfg.num_views} "
        f"cls_only={cfg.cls_only} skeleton={cfg.use_skeleton} dino={cfg.use_dino}"
    )
    print(
        f"   ~{_estimate_disk_mb_per_batch(cfg, batch_size):.0f} MB written/batch (FP16, 18 layers)"
    )

    dataset_root = resolve_dataset_root(dataset_repo)
    mapper = build_goal_mapper(model_path, dataset_root, cfg)
    model = mapper.model.to(device).eval()

    layer_paths = discover_layer_ids(model)
    hooks = {lid: TraceHook() for lid in layer_paths}
    handles = []
    for layer_id, module_path in sorted(layer_paths.items()):
        module = dict(model.named_modules())[module_path]
        handles.append(module.register_forward_hook(hooks[layer_id]))

    data_plugin = build_data_plugin(dataset_repo, cfg, num_steps=cfg.history_size)
    loader = DataLoader(
        data_plugin,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    totals = {
        "data": 0.0,
        "prepare": 0.0,
        "forward": 0.0,
        "export": 0.0,
    }
    n_timed = 0

    def run_batch(batch):
        nonlocal n_timed
        t0 = time.perf_counter()
        pixels, actions = prepare_pixels_6d(batch, mapper, device, cfg)
        t1 = time.perf_counter()

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                forward_harvest(model, pixels, actions, cfg, batch)
        _sync(device)
        t2 = time.perf_counter()

        nbytes = 0
        b = pixels.shape[0]
        for layer_id, hook in hooks.items():
            acts = hook.output
            flat = flatten_activation(acts, layer_id, cfg)
            nbytes += flat.nbytes
        _sync(device)
        t3 = time.perf_counter()

        totals["prepare"] += t1 - t0
        totals["forward"] += t2 - t1
        totals["export"] += t3 - t2
        n_timed += 1
        return nbytes, b

    it = iter(loader)
    for _ in range(warmup_batches):
        try:
            run_batch(next(it))
        except StopIteration:
            break

    for i in range(num_batches):
        t_data = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            print(f"⚠️ Dataset exhausted after {i} batches")
            break
        totals["data"] += time.perf_counter() - t_data
        nbytes, b = run_batch(batch)
        if i == 0:
            print(f"   batch pixels → export: {nbytes / 1e6:.1f} MB activations")

    for h in handles:
        h.remove()
    data_plugin.clear_cache()

    if n_timed == 0:
        print("No batches timed.")
        return

    per_batch = sum(totals.values()) / n_timed
    print("\n--- Per-batch timing (seconds, mean after warmup) ---")
    for key in ("data", "prepare", "forward", "export"):
        sec = totals[key] / n_timed
        pct = 100.0 * sec / per_batch if per_batch else 0
        print(f"  {key:8s}: {sec:6.2f}s  ({pct:5.1f}%)")
    print(f"  {'TOTAL':8s}: {per_batch:6.2f}s")

    full_batches = len(loader)
    est_200 = per_batch * 200 / 60
    est_full = per_batch * full_batches / 60
    print(f"\n--- Extrapolation ---")
    print(f"  200 batches: ~{est_200:.0f} min")
    print(f"  full loader ({full_batches} batches): ~{est_full:.0f} min")

    if not cfg.cls_only and cfg.multi_view:
        cls_cfg = ExperimentConfig(
            multi_view=cfg.multi_view,
            use_skeleton=cfg.use_skeleton,
            use_dino=cfg.use_dino,
            num_views=cfg.num_views,
            history_size=cfg.history_size,
            cls_only=True,
        )
        ratio = _estimate_disk_mb_per_batch(cfg, batch_size) / max(
            _estimate_disk_mb_per_batch(cls_cfg, batch_size), 1e-6
        )
        print(
            f"\n💡 --cls_only would cut encoder disk ~{ratio:.0f}x "
            f"(~{_estimate_disk_mb_per_batch(cls_cfg, batch_size):.0f} MB/batch vs "
            f"{_estimate_disk_mb_per_batch(cfg, batch_size):.0f} MB/batch)"
        )
    if num_workers > 2:
        print("💡 Colab often warns workers>2; try --workers 2")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile activation harvest")
    parser.add_argument("--model", type=str, default="gr1_reward_tuned_v2.ckpt")
    parser.add_argument("--dataset", type=str, default="gr1_pickup_grasp")
    parser.add_argument("--batches", type=int, default=5, help="Timed batches")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    add_experiment_args(parser, include_cls_only=True)
    args = parser.parse_args()
    cfg = config_from_args(args)
    profile_harvest(
        args.model,
        args.dataset,
        cfg,
        num_batches=args.batches,
        num_workers=args.workers,
        batch_size=args.batch_size,
        warmup_batches=args.warmup,
    )
