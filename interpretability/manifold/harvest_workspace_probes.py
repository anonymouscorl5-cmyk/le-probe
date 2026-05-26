#!/usr/bin/env python3
"""B5: Encode workspace probe bundle with a reward-tuned LeWM checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

CURRENT_FILE = Path(__file__).resolve()
ROOT_DIR = CURRENT_FILE.parents[2]
LE_PROBE_ROOT = CURRENT_FILE.parents[2]
if str(LE_PROBE_ROOT) not in sys.path:
    sys.path.insert(0, str(LE_PROBE_ROOT))
LEWM_DIR = LE_PROBE_ROOT / "lewm"
if str(LEWM_DIR) not in sys.path:
    sys.path.append(str(LEWM_DIR))

from lewm.goal_mapper import GoalMapper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle",
        type=str,
        default=str(
            LE_PROBE_ROOT / "datasets/workspace_probe_grasp/workspace_probe_bundle.pt"
        ),
    )
    parser.add_argument("--model", type=str, default="gr1_reward_tuned_v2.ckpt")
    parser.add_argument("--dataset_root", type=str, default=".")
    parser.add_argument("--multi_view", action="store_true")
    parser.add_argument("--use_skeleton", action="store_true")
    parser.add_argument("--use_dino", action="store_true")
    parser.add_argument("--tag", type=str, default="mv")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Default: workspace_probe_latents_{tag}.pt next to bundle",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle_path = Path(args.bundle)
    data = torch.load(bundle_path, map_location="cpu", weights_only=False)

    cam_names = list(data["cam_names"])
    rgb = data["rgb"]  # N,V,H,W,3 uint8
    states = data["state_norm"].float()
    n = rgb.shape[0]

    mapper = GoalMapper(
        model_path=args.model,
        dataset_root=args.dataset_root,
        use_multi_view=args.multi_view,
        num_views=len(cam_names) if args.multi_view else 1,
        use_skeleton=args.use_skeleton,
        use_dino=args.use_dino,
    )
    model = mapper.model.to(device).eval()

    use_skel = args.use_skeleton and "skeleton" in data
    skel = data.get("skeleton") if use_skel else None

    latents = []
    batch_size = 32

    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size), desc="Encode probes"):
            end = min(start + batch_size, n)
            B = end - start

            # (B, V, H, W, 3) -> (B, 1, V, C, H, W)
            px = rgb[start:end].permute(0, 1, 4, 2, 3).float() / 255.0
            if not args.multi_view:
                px = px[:, :1]
            pixels_6d = px.unsqueeze(1)

            if use_skel and skel is not None:
                sk = skel[start:end].permute(0, 1, 4, 2, 3).float() / 255.0
                if sk.shape[-3] != 1:
                    sk = sk.mean(dim=-3, keepdim=True)
                skel_6d = sk.unsqueeze(1)
                V, H, W = pixels_6d.shape[2], pixels_6d.shape[4], pixels_6d.shape[5]
                sk_flat = skel_6d.reshape(B * V, 1, H, W)
                if sk_flat.shape[-2:] != (224, 224):
                    sk_flat = torch.nn.functional.interpolate(
                        sk_flat, size=(224, 224), mode="nearest"
                    )
                sk_final = sk_flat.view(B, 1, V, 1, 224, 224)
                flat_rgb = pixels_6d.reshape(B * V, 3, H, W)
                proc = mapper.transform({"pixels": flat_rgb})["pixels"]
                pixels = proc.view(B, 1, V, 3, 224, 224)
                pixels = torch.cat([pixels, sk_final.to(pixels.dtype)], dim=-3)
            else:
                Bt, Tt, Vt, C, H, W = pixels_6d.shape
                flat = pixels_6d.reshape(Bt * Tt * Vt, C, H, W)
                proc = mapper.transform({"pixels": flat})["pixels"]
                pixels = proc.view(B, 1, Vt, C, 224, 224)

            actions = states[start:end].unsqueeze(1).to(device)
            pixels = pixels.to(device)

            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                info = model.encode({"pixels": pixels, "action": actions})
                emb = info["emb"].squeeze(1).cpu().numpy().astype(np.float32)
            latents.append(emb)

    latents_np = np.concatenate(latents, axis=0)
    out_path = (
        Path(args.out)
        if args.out
        else bundle_path.parent / f"workspace_probe_latents_{args.tag}.pt"
    )
    torch.save(
        {
            "latents": latents_np,
            "probe_ids": data["probe_ids"].numpy(),
            "segment_hint": data.get("segment_hint"),
            "ee_achieved_xyz": data["ee_achieved_xyz"].numpy(),
            "tag": args.tag,
            "model": args.model,
            "multi_view": args.multi_view,
            "use_skeleton": args.use_skeleton,
            "use_dino": args.use_dino,
        },
        out_path,
    )
    print(f"✅ Probe latents ({latents_np.shape}) → {out_path}")


if __name__ == "__main__":
    main()
