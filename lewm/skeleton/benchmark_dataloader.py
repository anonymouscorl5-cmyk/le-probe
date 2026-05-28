import os
import sys
import time
import torch
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf

# --- Path Stabilization ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
LEWM_DIR = os.path.join(REPO_ROOT, "lewm")
if LEWM_DIR not in sys.path:
    sys.path.append(LEWM_DIR)
# --------------------------

from lewm.skeleton.data import SkeletonDataPlugin


def main():
    print("🏎️  Dataloader Bottleneck Benchmark")
    print("====================================")

    # 1. Config Simulation (Mirroring trainer.py)
    num_workers = 6
    batch_size = 32
    num_batches = 20

    cfg = OmegaConf.create(
        {
            "img_size": 224,
            "wm": {"history_size": 1, "num_preds": 4},
            "data": {"dataset": {"repo_id": "gr1_pickup_grasp"}},
        }
    )

    keys_to_load = [
        "observation.state",
        "action",
        "world_center",
        "world_left",
        "world_right",
        "world_top",
        "world_wrist",
    ]

    # 2. Dataset Initialization
    print(f"📦 Initializing dataset: {cfg.data.dataset.repo_id}")
    dataset = SkeletonDataPlugin(
        repo_id=cfg.data.dataset.repo_id,
        keys_to_load=keys_to_load,
        num_steps=cfg.wm.history_size + cfg.wm.num_preds,
        use_multi_view=True,
        img_size=cfg.img_size,
    )

    # 3. Dataloader Initialization
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    # 4. Benchmarking Loop
    print(f"🚀 Benchmarking {num_batches} batches with {num_workers} workers...")

    # Clear profile CSV if exists
    if os.path.exists("dataloader_profile.csv"):
        os.remove("dataloader_profile.csv")

    start_time = time.perf_counter()
    batch_times = []

    # Warmup
    print("🔥 Warming up...")
    it = iter(loader)
    next(it)

    for i in range(num_batches):
        b_start = time.perf_counter()
        batch = next(it)
        b_end = time.perf_counter()
        batch_times.append(b_end - b_start)
        print(f"  Batch {i+1}/{num_batches}: {batch_times[-1]:.3f}s")

    total_time = time.perf_counter() - start_time
    avg_batch_time = np.mean(batch_times)
    throughput = 1.0 / avg_batch_time

    print("\n📊 Results Summary")
    print("------------------")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Avg Batch Time: {avg_batch_time:.3f}s")
    print(f"Throughput: {throughput:.2f} batches/s (it/s)")

    # 5. Bottleneck Analysis from CSV
    if os.path.exists("dataloader_profile.csv"):
        import pandas as pd

        df = pd.read_csv("dataloader_profile.csv")
        print("\n🔍 Granular Timing Breakdown (Averages in ms)")
        # Calculate means for all columns ending in _ms
        ms_cols = [c for c in df.columns if "_ms" in c]
        if ms_cols:
            means = df[ms_cols].mean().sort_values(ascending=False)
            print(means)

        # Calculate per-view loading if present
        load_cols = [c for c in df.columns if "load_" in c]
        if load_cols:
            print("\n📸 Per-View Loading Breakdown (ms):")
            print(df[load_cols].mean())


if __name__ == "__main__":
    main()
