import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# --- Path Stabilization ---
ROOT_DIR = "/workspace/cortex-os"
LE_PROBE_DIR = os.path.join(ROOT_DIR, "le-probe")
LEWM_DIR = os.path.join(LE_PROBE_DIR, "lewm")
LE_WM_DIR = os.path.join(LEWM_DIR, "le_wm")

for p in [ROOT_DIR, LE_PROBE_DIR, LEWM_DIR, LE_WM_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from lewm.goal_mapper import GoalMapper
from interpretability.clt.clt_model import CrossLayerTranscoder
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Paths
MODEL_PATH = os.path.join(ROOT_DIR, "gr1_reward_tuned_v2.ckpt")
CLT_PATH = os.path.join(ROOT_DIR, "clt_weights.pt")
DATASET_ROOT = os.path.join(
    LE_PROBE_DIR, "datasets", "anonymouscorl5-cmyk", "gr1_pickup_grasp"
)
ASSETS_DIR = os.path.join(LE_PROBE_DIR, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

# Correct Mappings based on Audit
# Global Index -> Dataset Local Index
# 358: 11993 -> 3579
# 90: 12117 -> 3703
# 743: 9047 -> 633
TARGETS = {
    "tactile_engagement": {"local_idx": 3703, "label": "Tactile Engagement"},
    "spatial_lockdown": {"local_idx": 3579, "label": "Spatial Lockdown"},
    "alignment_precision": {"local_idx": 633, "label": "Alignment Precision"},
}

AUDIT_FEATURES = {
    90: "Tactile Engagement",
    358: "Spatial Lockdown",
    743: "Alignment Precision",
}


def generate():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"🧠 Loading interpretability stack on {device}...")

    # 1. Load Models
    agent = GoalMapper(MODEL_PATH, dataset_root=ROOT_DIR)
    nn_model = agent.model.to(device).eval()

    checkpoint = torch.load(CLT_PATH, map_location=device)
    norm = checkpoint["norm_stats"]
    clt = (
        CrossLayerTranscoder(
            d_model=checkpoint["config"]["d_model"], d_sae=checkpoint["config"]["d_sae"]
        )
        .to(device)
        .eval()
    )
    clt.load_state_dict(checkpoint["state_dict"])

    # 2. Load Dataset
    print(f"📂 Loading LeRobot Dataset for Parity...")
    ds = LeRobotDataset("gr1_pickup_grasp", root=DATASET_ROOT)

    print("🚀 Generating Canonical Triptychs (Bit-Perfect)...")

    for key, info in TARGETS.items():
        local_idx = info["local_idx"]

        # 3. Extract Frames directly from LeRobot (Ensures same decoder/preprocessing)
        row_after = ds[local_idx]
        img_after = row_after["observation.images.world_center"]  # Tensor [3, 480, 480]

        row_before = ds[local_idx - 8]
        img_before = row_before["observation.images.world_center"]

        # 4. Compute Activations
        # Transform exactly like harvesting
        batch = agent.transform({"pixels": img_after})
        x = batch["pixels"].unsqueeze(0).to(device)

        with torch.no_grad():
            enc_out = nn_model.encoder(x, interpolate_pos_encoding=True)
            pixels_emb = enc_out.last_hidden_state[:, 0]
            enc_latent = nn_model.projector(pixels_emb)

            x_norm = (enc_latent - norm["mean_L"].to(device)) / norm["std_L"].to(device)
            x_centered = x_norm - clt.b_dec
            acts = torch.nn.functional.relu(
                clt.encoder(x_centered) + clt.b_enc
            ).squeeze()

        # 5. Render Triptych
        plt.style.use("dark_background")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor="#111111")

        # Convert tensors to displayable numpy
        after_disp = img_after.permute(1, 2, 0).cpu().numpy()
        before_disp = img_before.permute(1, 2, 0).cpu().numpy()

        # Panel 1: Before
        axes[0].imshow(before_disp)
        axes[0].set_title(
            "MIND'S EYE (8 FRAMES BEFORE)",
            fontsize=10,
            fontweight="bold",
            color="#AAAAAA",
        )
        axes[0].axis("off")

        # Panel 2: After
        axes[1].imshow(after_disp)
        axes[1].set_title(
            f"MIND'S EYE (PEAK STATE - EP {row_after['episode_index']})",
            fontsize=10,
            fontweight="bold",
            color="#FFFFFF",
        )
        axes[1].axis("off")

        # Panel 3: Activations
        ids = sorted(AUDIT_FEATURES.keys())
        names = [AUDIT_FEATURES[fid] for fid in ids]
        vals = [float(acts[fid]) for fid in ids]
        colors = ["#FF4B4B", "#4BFF4B", "#4B4BFF"]
        bars = axes[2].barh(
            names, vals, color=colors, alpha=0.8, edgecolor="white", linewidth=0.5
        )

        axes[2].set_xlim(0, 5.0)
        axes[2].set_title(
            "MECHANISTIC ACTIVATIONS", fontsize=10, fontweight="bold", color="#AAAAAA"
        )
        axes[2].grid(axis="x", linestyle="--", alpha=0.3)

        for bar in bars:
            width = bar.get_width()
            axes[2].text(
                width + 0.1,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.2f}",
                va="center",
                fontsize=9,
                color="white",
                fontweight="bold",
            )

        # Save to assets
        output_path = os.path.join(ASSETS_DIR, f"canonical_{key}.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=120, bbox_inches="tight", facecolor="#111111")
        plt.close()
        print(f"✅ Canonical triptych (Parity) saved: {output_path}")


if __name__ == "__main__":
    generate()
