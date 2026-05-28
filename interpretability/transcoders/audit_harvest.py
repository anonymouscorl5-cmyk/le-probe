"""
EXHAUSTIVE ACTIVATION AUDITOR (Production Grade)
Validates harvested .bin / .json streams for a given experiment configuration.
"""

import sys
import json
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

CURRENT_FILE = Path(__file__).resolve()
ROOT_DIR = CURRENT_FILE.parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
if str(ROOT_DIR / "lewm") not in sys.path:
    sys.path.append(str(ROOT_DIR / "lewm"))

from interpretability.lewm_experiment import (
    ExperimentConfig,
    add_experiment_args,
    build_goal_mapper,
    config_from_args,
    discover_layer_ids,
    resolve_dataset_root,
)


def audit_activations(
    output_dir,
    model_path,
    cfg: ExperimentConfig,
    dataset_root: str = ".",
):
    print(f"🕵️ Starting Exhaustive Audit of: {output_dir}")
    path = Path(output_dir).resolve()

    print("\n🔍 Step 1: Discovery Audit...")
    mapper = build_goal_mapper(model_path, dataset_root, cfg)
    expected_layers = sorted(discover_layer_ids(mapper.model).keys())

    found_bins = list(path.glob("*.bin"))
    found_layer_ids = sorted({f.stem for f in found_bins})

    missing = set(expected_layers) - set(found_layer_ids)
    if missing:
        print(f"  ❌ MISSING LAYERS: {missing}")
    else:
        print(f"  ✅ All {len(expected_layers)} expected layers found.")

    enc_tpm = cfg.encoder_tokens_per_moment()
    pred_tpm = cfg.predictor_tokens_per_moment()

    print("\n📊 Step 2: Integrity & Consistency Audit...")
    report = []
    total_samples_list = []

    for layer_id in tqdm(found_layer_ids, desc="Auditing Layers"):
        bin_file = path / f"{layer_id}.bin"
        json_file = path / f"{layer_id}.json"

        if not json_file.exists():
            report.append({"layer": layer_id, "status": "❌ Missing Metadata"})
            continue

        with open(json_file, "r") as f:
            meta = json.load(f)

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

        data = np.memmap(
            bin_file, dtype=np.float16, mode="r", shape=tuple(meta["shape"])
        )

        is_encoder = layer_id.startswith("encoder")
        tokens_per_moment = enc_tpm if is_encoder else pred_tpm
        recorded_tps = meta.get("tokens_per_sample", tokens_per_moment)
        equiv_moments = meta["shape"][0] / max(recorded_tps, 1)
        total_samples_list.append(round(equiv_moments, 2))

        sample_indices = [0, len(data) // 2, len(data) - 1]
        nan_found = any(
            np.isnan(data[idx]).any() or np.isinf(data[idx]).any()
            for idx in sample_indices
        )

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
                    "tps": recorded_tps,
                }
            )

    print("\n🔗 Step 3: Cross-Layer Alignment...")
    unique_equiv = set(total_samples_list)
    if not unique_equiv:
        print("  ⚠️ No valid layers found to align.")
    elif len(unique_equiv) > 1:
        print(
            f"  ❌ ALIGNMENT FAILURE: Layers represent different moments! {unique_equiv}"
        )
    else:
        print(f"  ✅ All layers aligned at {list(unique_equiv)[0]} equivalent moments.")

    print("\n📋 --- FINAL AUDIT REPORT ---")
    print(f"{'Layer ID':<20} | {'Status':<30} | {'Rows':<10} | {'Eq. Samples'}")
    print("-" * 80)
    for r in report:
        print(
            f"{r['layer']:<20} | {r['status']:<30} | "
            f"{r.get('rows', 'N/A'):<10} | {r.get('equiv', 'N/A')}"
        )

    print("\n✨ Audit complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit harvested activation streams")
    parser.add_argument(
        "--dir", type=str, required=True, help="Path to harvested activations"
    )
    parser.add_argument("--model", type=str, default="gr1_reward_tuned_v2.ckpt")
    parser.add_argument("--dataset", type=str, default="gr1_pickup_grasp")
    add_experiment_args(parser, include_cls_only=True)
    args = parser.parse_args()

    cfg = config_from_args(args)
    root = resolve_dataset_root(args.dataset)
    audit_activations(args.dir, args.model, cfg, dataset_root=root)
