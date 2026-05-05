# --- Path Stabilization ---
# Ensures that 'le-probe' is in the python path for absolute imports
import os
import sys
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
ROOT_DIR = CURRENT_FILE.parents[2]  # To le-probe/
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
# --------------------------

import json
import torch
import torch.optim as optim
import argparse
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from interpretability.transcoders.universal_transcoder import Transcoder


class StreamingActivationsDataset(Dataset):
    """
    High-performance streaming dataset for .bin activation files.
    Supports memory mapping for zero-RAM overhead.
    """

    def __init__(self, bin_path, json_path):
        with open(json_path, "r") as f:
            self.meta = json.load(f)

        self.shape = tuple(self.meta["shape"])
        self.tokens_per_sample = self.meta.get("tokens_per_sample", 1)
        # Load as float16 to save RAM and speed up transfer
        self.data = np.memmap(bin_path, dtype=np.float16, mode="r", shape=self.shape)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        # We don't use this in the hot loop (we use get_batch_raw)
        return torch.from_numpy(self.data[idx])


class MultiLayerStreamingDataset(Dataset):
    """
    Synchronized dataset for multiple .bin activation files.

    Optimized for Crosscoder mode: Handles multiple file pointers and de-duplicates
    disk reads for overlapping source/target layers.
    """

    def __init__(self, source_dir, layers):
        self.datasets = []
        self.layer_to_idx = {}
        print("🔍 MultiLayer Startup Audit:")
        for i, layer in enumerate(layers):
            bin_path = os.path.join(source_dir, f"{layer}.bin")
            json_path = os.path.join(source_dir, f"{layer}.json")
            ds = StreamingActivationsDataset(bin_path, json_path)
            self.datasets.append(ds)
            self.layer_to_idx[layer] = i
            print(f"  - [{layer}] Linked to: {bin_path} | Shape: {ds.shape}")

        self.total_samples = len(self.datasets[0])
        self.d_model_per_layer = self.datasets[0].shape[1]
        self.tokens_per_sample = self.datasets[0].tokens_per_sample

    def __len__(self):
        return self.total_samples

    def get_batch_raw(self, indices):
        """
        Pull raw float16 tensors. ZERO COPY on CPU.
        """
        return [torch.from_numpy(ds.data[indices]) for ds in self.datasets]


def train_transcoder(
    source_dir,
    source_layers_str,
    target_layers_str,
    output_path,
    dict_size=12288,
    l1_coeff=1e-3,
    epochs=5,
    batch_size=4096,
    lr=1e-4,
):
    """
    Core training loop for SAEs and Crosscoders.
    Optimized for maximum GPU utilization and minimal CPU-to-GPU latency.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Bare-Metal Optimized Training | Device: {device}")

    # Parse layer lists for multi-layer support
    src_list = [s.strip() for s in source_layers_str.split(",")]
    tgt_list = [t.strip() for t in target_layers_str.split(",")]

    # Identify unique layers to avoid redundant disk reads
    unique_layers = sorted(list(set(src_list + tgt_list)))
    layer_to_pos = {l: i for i, l in enumerate(unique_layers)}

    print(f"📦 Unique Layers in play: {unique_layers}")
    print(f"🔗 {source_layers_str} ⮕ {target_layers_str}")

    # 1. Initialize Master Dataset
    master_ds = MultiLayerStreamingDataset(source_dir, unique_layers)

    # 2. Normalization Pass (Vectorized)
    # Calculate running stats for BOTH source and target to ensure balanced gradients
    print("📈 Calculating Normalization Stats (Source & Target)...")
    sample_size = min(len(master_ds), 500_000)
    indices = np.random.choice(len(master_ds), sample_size, replace=False)

    # Source Stats
    src_data = []
    for l in src_list:
        raw = torch.from_numpy(
            master_ds.datasets[master_ds.layer_to_idx[l]].data[indices]
        )
        src_data.append(raw.to(device).float())
    src_subset = torch.cat(src_data, dim=-1)
    mean_s, std_s = src_subset.mean(dim=0), src_subset.std(dim=0) + 1e-6

    # Target Stats
    tgt_data = []
    for l in tgt_list:
        raw = torch.from_numpy(
            master_ds.datasets[master_ds.layer_to_idx[l]].data[indices]
        )
        tgt_data.append(raw.to(device).float())
    tgt_subset = torch.cat(tgt_data, dim=-1)
    mean_t, std_t = tgt_subset.mean(dim=0), tgt_subset.std(dim=0) + 1e-6

    # 3. Model Setup
    d_in = len(src_list) * master_ds.d_model_per_layer
    d_out = len(tgt_list) * master_ds.d_model_per_layer

    model = Transcoder(
        d_model=d_in, d_dict=dict_size, d_output=d_out, l1_coeff=l1_coeff
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Diagnostics tracking: tracks which features have fired at least once
    has_activated = torch.zeros(dict_size, dtype=torch.bool, device=device)

    # 4. Training Loop
    for epoch in range(epochs):
        model.train()
        num_tokens = len(master_ds)
        indices = np.arange(num_tokens)
        np.random.shuffle(indices)

        # Funnel check for Encoder/Predictor alignment
        src_tokens = master_ds.datasets[
            master_ds.layer_to_idx[src_list[0]]
        ].tokens_per_sample
        tgt_tokens = master_ds.datasets[
            master_ds.layer_to_idx[tgt_list[0]]
        ].tokens_per_sample
        is_funnel = src_tokens != tgt_tokens

        pbar = tqdm(range(0, num_tokens, batch_size), desc=f"Epoch {epoch+1}/{epochs}")
        for i in pbar:
            batch_idx = indices[i : i + batch_size]

            # STEP 1: Pull float16 raw tensors from disk
            raw_tensors = master_ds.get_batch_raw(batch_idx)

            # STEP 2 & 3: Move to GPU and cast
            gpu_tensors = [t.to(device, non_blocking=True).float() for t in raw_tensors]

            # STEP 4: Construct batches
            s_batch = torch.cat(
                [gpu_tensors[layer_to_pos[l]] for l in src_list], dim=-1
            )
            # Normalize Source
            s_batch_norm = (s_batch - mean_s) / std_s

            if source_layers_str == target_layers_str:
                t_batch_norm = s_batch_norm
            else:
                if not is_funnel:
                    t_batch = torch.cat(
                        [gpu_tensors[layer_to_pos[l]] for l in tgt_list], dim=-1
                    )
                else:
                    # Funnel Logic (Summary token mapping)
                    src_idx_start = batch_idx // src_tokens
                    token_offset = batch_idx % src_tokens
                    tgt_batch_idx = src_idx_start * tgt_tokens + (token_offset // 257)

                    t_raw = [
                        torch.from_numpy(
                            master_ds.datasets[master_ds.layer_to_idx[l]].data[
                                tgt_batch_idx
                            ]
                        )
                        for l in tgt_list
                    ]
                    t_batch = torch.cat(
                        [t.to(device, non_blocking=True).float() for t in t_raw], dim=-1
                    )

                # Normalize Target (Ensures L1 penalty isn't drowned out by raw MSE scale)
                t_batch_norm = (t_batch - mean_t) / std_t

            optimizer.zero_grad()
            res = model(s_batch_norm, t_batch_norm)
            loss = res["loss"]
            loss.backward()

            # Normalize decoder weights
            model.normalize_decoder()
            optimizer.step()

            # --- Diagnostics and Residual Audit ---
            if i % 100 == 0:
                with torch.no_grad():
                    if "activations" in res:
                        has_activated |= res["activations"].sum(dim=0) > 0

                # TOTAL EV (Now based on normalized variance, which is 1.0)
                mse_total = res["l2_loss"].item()
                ev_total = 1 - (mse_total / 1.0)  # Var is 1.0 after normalization

                # PER-LAYER AUDIT: Shows normalized MSE per layer (Target < 0.1)
                layer_mses = []
                for j, layer_name in enumerate(tgt_list):
                    start = j * master_ds.d_model_per_layer
                    end = (j + 1) * master_ds.d_model_per_layer
                    l_hat = res["output"][:, start:end]
                    l_target = t_batch_norm[:, start:end]
                    l_mse = torch.nn.functional.mse_loss(l_hat, l_target).item()
                    layer_mses.append(f"{layer_name}:{l_mse:.3f}")

                audit_str = " ".join(layer_mses)
                l0 = (
                    (res["activations"] > 0).float().sum(dim=1).mean().item()
                    if "activations" in res
                    else 0
                )
                dead_cnt = (has_activated == 0).sum().item()

                pbar.set_postfix(
                    {
                        "ev": f"{ev_total:.1%}",
                        "l0": f"{l0:.1f}",
                        "dead": dead_cnt,
                        "audit": f"[{audit_str}]",
                    }
                )

    # 5. Save Model
    print(f"💾 Saving to {output_path}")
    save_dict = {
        "state_dict": model.state_dict(),
        "norm_stats": {
            "mean": mean_s.cpu(),
            "std": std_s.cpu(),
            "tgt_mean": mean_t.cpu(),
            "tgt_std": std_t.cpu(),
        },
        "meta": {
            "d_model": d_in,
            "d_output": d_out,
            "d_dict": dict_size,
            "source_layers": src_list,
            "target_layers": tgt_list,
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(save_dict, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir", type=str, required=True, help="Directory with .bin/.json files"
    )
    parser.add_argument(
        "--source_layer",
        type=str,
        required=True,
        help="Layer(s) to read from (comma-sep)",
    )
    parser.add_argument(
        "--target_layer",
        type=str,
        required=True,
        help="Layer(s) to reconstruct (comma-sep)",
    )
    parser.add_argument("--output", type=str, required=True, help="Output .pt path")
    parser.add_argument("--dict_size", type=int, default=12288)
    parser.add_argument("--l1", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()

    train_transcoder(
        source_dir=args.dir,
        source_layers_str=args.source_layer,
        target_layers_str=args.target_layer,
        output_path=args.output,
        dict_size=args.dict_size,
        l1_coeff=args.l1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
