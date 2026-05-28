import argparse
import torch
import os
import sys
import numpy as np

# --- Path Stabilization ---
# Project Root is the base for data and weights
ROOT_DIR = "/workspace/cortex-os"
LE_PROBE_DIR = os.path.join(ROOT_DIR, "le-probe")
if LE_PROBE_DIR not in sys.path:
    sys.path.insert(0, LE_PROBE_DIR)
# --------------------------

from interpretability.clt.clt_model import CrossLayerTranscoder


def find_triggers(clt_path, activations_path, feature_id=90):
    device = torch.device("cpu")

    # 1. Load CLT and Data
    checkpoint = torch.load(clt_path, map_location=device)
    config = checkpoint["config"]
    norm = checkpoint["norm_stats"]

    clt = CrossLayerTranscoder(d_model=config["d_model"], d_sae=config["d_sae"]).to(
        device
    )
    clt.load_state_dict(checkpoint["state_dict"])
    clt.eval()

    data = torch.load(activations_path, map_location=device)
    x_L = data["enc"].float().to(device)

    # 2. Get CLT Activations
    x_L_norm = (x_L - norm["mean_L"].to(device)) / norm["std_L"].to(device)
    with torch.no_grad():
        out = clt(x_L_norm)
        clt_acts = out["activations"]  # (N, 1024)

    feat_values = clt_acts[:, feature_id]

    # 3. Find Top Triggering Frames
    top_vals, top_indices = torch.topk(feat_values, k=10)

    print(f"\n🧠 Top Triggers for Feature {feature_id} (Reach Intent):")
    print("-" * 80)
    print(f"{'Rank':<5} | {'Frame':<8} | {'Activation':<12} | {'Source Context'}")
    print("-" * 80)
    for i in range(10):
        idx = top_indices[i].item()
        val = top_vals[i].item()

        # Mapping ranges from the harvest summary
        if idx < 2002:
            source = "Snapshot Audit (Success Case)"
        elif 2002 <= idx < 8414:
            source = f"Pickup Cup | Episode {(idx-2002)//32} | Frame {(idx-2002)%32}"
        else:
            source = f"Success Grasp | Episode {(idx-8414)//32} | Frame {(idx-8414)%32}"

        print(f"{i+1:<5} | {idx:<8} | {val:<12.4f} | {source}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", type=int, default=90, help="Feature ID to audit")
    args = parser.parse_args()

    clt_path = os.path.join(ROOT_DIR, "clt_weights.pt")
    data_path = os.path.join(ROOT_DIR, "activations_dual_14k.pt")

    if os.path.exists(clt_path) and os.path.exists(data_path):
        find_triggers(clt_path, data_path, feature_id=args.feature)
    else:
        print(f"❌ Data or weights missing at: {clt_path} or {data_path}")
