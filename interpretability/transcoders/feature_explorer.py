import os
import torch
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm


def explore_features(
    weight_path, activation_dir, feature_ids=None, top_k=10, auto_top=None
):
    """
    Finds the top-activating examples for specific features.
    If auto_top is provided, it first finds the N features with the highest max activations.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")

    # 1. Load Model & Stats
    data = torch.load(weight_path, map_location="cpu")
    sd = data["state_dict"]
    norm_stats = {k: v.to(device) for k, v in data["norm_stats"].items()}
    config = data["config"]
    layer_id = config["source_layer"]

    # Extract Encoder weights: (d_model, d_sae)
    w_enc = sd["encoder.weight"].T.to(device)
    b_enc = sd["encoder.bias"].to(device)

    # 2. Load Activation Metadata
    # Standard flat structure from harvest_activations.py
    json_path = os.path.join(activation_dir, f"{layer_id}.json")
    bin_path = os.path.join(activation_dir, f"{layer_id}.bin")

    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"❌ Could not find activation files for {layer_id} in {activation_dir}"
        )

    with open(json_path, "r") as f:
        meta = json.load(f)

    # Memory map the activations
    acts = np.memmap(bin_path, dtype=np.float16, mode="r", shape=tuple(meta["shape"]))

    # 3. Discovery Mode: Find Top-N features if requested
    if auto_top:
        print(f"🔭 Discovering Top {auto_top} features with highest peak activation...")
        max_acts = torch.zeros(w_enc.shape[1], device=device)
        batch_size = 20000
        for i in tqdm(range(0, acts.shape[0], batch_size)):
            batch = torch.from_numpy(acts[i : i + batch_size].copy()).to(device).float()
            batch = (batch - norm_stats["mean"]) / norm_stats["std"]
            feat_acts = torch.relu(batch @ w_enc + b_enc)
            batch_max, _ = torch.max(feat_acts, dim=0)
            max_acts = torch.max(max_acts, batch_max)

        top_val, top_feat_ids = torch.topk(max_acts, auto_top)
        feature_ids = top_feat_ids.tolist()
        print(
            f"✅ Found top features: {feature_ids[:10]}... (Total {len(feature_ids)})"
        )

    results = {fid: [] for fid in feature_ids}
    print(f"🔍 Scanning activations for top examples of {len(feature_ids)} features...")

    # Batch process for speed
    batch_size = 10000
    for i in tqdm(range(0, acts.shape[0], batch_size)):
        batch = torch.from_numpy(acts[i : i + batch_size].copy()).to(device).float()
        batch = (batch - norm_stats["mean"]) / norm_stats["std"]

        # Compute activations only for relevant features
        feat_acts = torch.relu(batch @ w_enc[:, feature_ids] + b_enc[feature_ids])

        for idx, fid in enumerate(feature_ids):
            vals = feat_acts[:, idx]
            # Optimization: only process if there's a potential top_k entry
            if torch.max(vals) > 0:
                top_vals, top_idx = torch.topk(vals, min(top_k, vals.shape[0]))
                for val, local_idx in zip(top_vals, top_idx):
                    if val > 0:
                        results[fid].append((val.item(), i + local_idx.item()))

                # Prune to top_k
                results[fid] = sorted(results[fid], key=lambda x: x[0], reverse=True)[
                    :top_k
                ]

    # 4. Output Findings
    output_report = Path("feature_audit_report.json")
    with open(output_report, "w") as f:
        json.dump(results, f, indent=4)

    print(f"✨ Audit complete. Results saved to {output_report}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", type=str, required=True, help="Path to .pt weight file"
    )
    parser.add_argument(
        "--acts", type=str, required=True, help="Path to activations directory"
    )
    parser.add_argument(
        "--features", type=int, nargs="+", help="Specific feature IDs to audit"
    )
    parser.add_argument(
        "--auto_top", type=int, default=None, help="Automatically find top N features"
    )
    args = parser.parse_args()

    f_ids = args.features if args.features else []
    explore_features(args.weights, args.acts, feature_ids=f_ids, auto_top=args.auto_top)
