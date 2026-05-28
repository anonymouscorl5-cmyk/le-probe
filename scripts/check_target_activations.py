import torch
import os
import sys

# Path Setup
ROOT_DIR = "/workspace/cortex-os"
LE_PROBE_DIR = os.path.join(ROOT_DIR, "le-probe")
if LE_PROBE_DIR not in sys.path:
    sys.path.insert(0, LE_PROBE_DIR)

from interpretability.clt.clt_model import CrossLayerTranscoder


def check():
    device = torch.device("cpu")
    clt_path = os.path.join(ROOT_DIR, "clt_weights.pt")
    data_path = os.path.join(ROOT_DIR, "activations_dual_14k.pt")

    checkpoint = torch.load(clt_path, map_location=device)
    norm = checkpoint["norm_stats"]
    clt = CrossLayerTranscoder(
        d_model=checkpoint["config"]["d_model"], d_sae=checkpoint["config"]["d_sae"]
    )
    clt.load_state_dict(checkpoint["state_dict"])
    clt.eval()

    data = torch.load(data_path, map_location=device)
    x_L = data["enc"].float()
    x_L_norm = (x_L - norm["mean_L"]) / norm["std_L"]

    with torch.no_grad():
        out = clt(x_L_norm)
        acts = out["activations"]

    targets = {"358": 11993, "90": 12117, "743": 9047}

    print("\n🔍 VERIFYING HARVESTED ACTIVATIONS:")
    for fid, idx in targets.items():
        val = acts[idx, int(fid)].item()
        print(f"Feature {fid} | Global Index {idx} | Harvested Value: {val:.4f}")

        # Also check other audited features for this frame
        all_fid = [90, 358, 743]
        print(f"  All: { {f: round(float(acts[idx, f]), 4) for f in all_fid} }")


if __name__ == "__main__":
    check()
