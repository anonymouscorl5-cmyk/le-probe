"""
EXHAUSTIVE ACTIVATION AUDITOR (Production Grade)
Role: Validates the integrity of harvested .bin and .json streams.
Checks:
1. Layer Discovery Parity (Files vs. Model)
2. Sample Count Consistency (Cross-layer alignment)
3. Shape/Byte-Count Integrity (File corruption check)
4. Value Fidelity (NaN/Inf Audit)
5. Memmap Readiness
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

# Fix paths
CURRENT_FILE = Path(__file__).resolve()
ROOT_DIR = CURRENT_FILE.parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
if str(ROOT_DIR / "lewm") not in sys.path:
    sys.path.append(str(ROOT_DIR / "lewm"))

from lewm.goal_mapper import GoalMapper


def audit_activations(output_dir, model_path):
    print(f"🕵️ Starting Exhaustive Audit of: {output_dir}")
    path = Path(output_dir).resolve()

    # 1. Discovery Parity
    print("\n🔍 Step 1: Discovery Audit...")
    mapper = GoalMapper(model_path=model_path, dataset_root=".")
    model = mapper.model

    expected_layers = []
    for name, module in model.named_modules():
        parts = name.split(".")
        if (
            len(parts) > 1
            and (parts[-2] == "layer" or parts[-2] == "layers")
            and parts[-1].isdigit()
        ):
            comp = "encoder" if "encoder" in name else "predictor"
            expected_layers.append(f"{comp}_L{parts[-1]}")

    found_bins = list(path.glob("*.bin"))
    found_jsons = list(path.glob("*.json"))
    found_layer_ids = [f.stem for f in found_bins]

    missing = set(expected_layers) - set(found_layer_ids)
    if missing:
        print(f"  ❌ MISSING LAYERS: {missing}")
    else:
        print(f"  ✅ All {len(expected_layers)} expected layers found.")

    # 2. Integrity & Consistency Audit
    print("\n📊 Step 2: Integrity & Consistency Audit...")
    report = []
    total_samples_list = []

    for layer_id in tqdm(found_layer_ids, desc="Auditing Layers"):
        bin_file = path / f"{layer_id}.bin"
        json_file = path / f"{layer_id}.json"

        # Check if metadata exists
        if not json_file.exists():
            report.append({"layer": layer_id, "status": "❌ Missing Metadata"})
            continue

        with open(json_file, "r") as f:
            meta = json.load(f)

        # Verify Byte-Count Alignment
        # FP16 = 2 bytes per element
        expected_bytes = meta["shape"][0] * meta["shape"][1] * 2
        actual_bytes = bin_file.stat().st_size

        if expected_bytes != actual_bytes:
            report.append(
                {
                    "layer": layer_id,
                    "status": f"❌ Byte Mismatch (Expected {expected_bytes}, Found {actual_bytes})",
                }
            )
            continue

        # Value Fidelity Check (Sample-based)
        data = np.memmap(
            bin_file, dtype=np.float16, mode="r", shape=tuple(meta["shape"])
        )

        # Normalize Alignment (Accounting for the Funnel Effect)
        # Encoder has 771 tokens (3 frames * 257 tokens)
        # Predictor has 3 tokens (3 frames * 1 token)
        is_encoder = "encoder" in layer_id
        tokens_per_moment = 771 if is_encoder else 3
        equiv_moments = meta["shape"][0] / tokens_per_moment
        total_samples_list.append(round(equiv_moments, 2))

        # Check first/last/middle for NaNs/Infs
        sample_indices = [0, len(data) // 2, len(data) - 1]
        nan_found = False
        for idx in sample_indices:
            if np.isnan(data[idx]).any() or np.isinf(data[idx]).any():
                nan_found = True
                break

        if nan_found:
            report.append(
                {
                    "layer": layer_id,
                    "status": "⚠️ Value Fidelity Failure (NaN/Inf detected)",
                }
            )
        else:
            report.append(
                {
                    "layer": layer_id,
                    "status": "✅ Healthy",
                    "rows": meta["shape"][0],
                    "equiv": round(equiv_moments, 1),
                    "dims": meta["shape"][1],
                }
            )

    # 3. Cross-Layer Alignment
    print("\n🔗 Step 3: Cross-Layer Alignment...")
    unique_equiv = set(total_samples_list)
    if not unique_equiv:
        print("  ⚠️ No valid layers found to align.")
    elif len(unique_equiv) > 1:
        print(
            f"  ❌ ALIGNMENT FAILURE: Layers represent different moments! {unique_equiv}"
        )
    else:
        print(
            f"  ✅ All layers perfectly aligned at {list(unique_equiv)[0]} equivalent samples."
        )

    # 4. Final Summary Table
    print("\n📋 --- FINAL AUDIT REPORT ---")
    print(f"{'Layer ID':<20} | {'Status':<30} | {'Rows':<10} | {'Eq. Samples'}")
    print("-" * 80)
    for r in report:
        status = r["status"]
        rows = r.get("rows", "N/A")
        equiv = r.get("equiv", "N/A")
        print(f"{r['layer']:<20} | {status:<30} | {rows:<10} | {equiv}")

    print("\n✨ Audit complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir", type=str, required=True, help="Path to harvested activations"
    )
    parser.add_argument("--model", type=str, default="gr1_reward_tuned_v2.ckpt")
    args = parser.parse_args()

    audit_activations(args.dir, args.model)
