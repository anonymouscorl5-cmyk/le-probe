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
        self.data = np.memmap(bin_path, dtype=np.float16, mode="r", shape=self.shape)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        # Return as float32 for training to avoid precision issues
        return torch.from_numpy(self.data[idx].copy()).float()


class MultiLayerStreamingDataset(Dataset):
    """
    Synchronized dataset for multiple .bin activation files.
    Concatenates layers along the feature dimension using vectorized indexing.

    Optimized for Crosscoder mode: Handles multiple file pointers and de-duplicates
    disk reads for overlapping source/target layers.
    """

    def __init__(self, source_dir, layers):
        self.datasets = []
        self.layer_to_idx = {}
        for i, layer in enumerate(layers):
            bin_path = os.path.join(source_dir, f"{layer}.bin")
            json_path = os.path.join(source_dir, f"{layer}.json")
            self.datasets.append(StreamingActivationsDataset(bin_path, json_path))
            self.layer_to_idx[layer] = i

        self.total_samples = len(self.datasets[0])
        self.d_model_per_layer = self.datasets[0].shape[1]
        self.tokens_per_sample = self.datasets[0].tokens_per_sample

    def __len__(self):
        return self.total_samples

    def get_batch(self, indices, layers_to_extract=None):
        """
        Vectorized batch loading. If layers_to_extract is provided,
        it only pulls and concatenates those specific layers.

        Note: NumPy advanced indexing is significantly faster than Python loops.
        """
        acts = []
        target_datasets = self.datasets
        if layers_to_extract:
            target_datasets = [
                self.datasets[self.layer_to_idx[l]] for l in layers_to_extract
            ]

        for ds in target_datasets:
            batch = torch.from_numpy(ds.data[indices].copy()).float()
            acts.append(batch)
        return torch.cat(acts, dim=-1)


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

    Arguments:
        source_layers_str: Comma-separated list of layers for the input (e.g. 'L1')
        target_layers_str: Comma-separated list of layers for reconstruction (e.g. 'L0,L1,L2')
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Master-Dataset Optimized Training | Device: {device}")

    # Parse layer lists for multi-layer support
    src_list = [s.strip() for s in source_layers_str.split(",")]
    tgt_list = [t.strip() for t in target_layers_str.split(",")]

    # IDENTIFY ALL UNIQUE LAYERS TO AVOID REDUNDANT READS
    unique_layers = []
    for l in src_list + tgt_list:
        if l not in unique_layers:
            unique_layers.append(l)

    print(f"📦 Unique Layers in play: {unique_layers}")
    print(f"🔗 Source: {src_list} ⮕ Target: {tgt_list}")

    # 1. Initialize Master Dataset
    master_ds = MultiLayerStreamingDataset(source_dir, unique_layers)

    # 2. Normalization Pass (Vectorized)
    # Calculate running stats on a large subset to ensure stable training
    print("📈 Calculating Normalization Stats...")
    sample_size = min(len(master_ds), 500_000)
    indices = np.random.choice(len(master_ds), sample_size, replace=False)
    src_subset = master_ds.get_batch(indices, layers_to_extract=src_list)
    mean_s, std_s = src_subset.mean(dim=0), src_subset.std(dim=0) + 1e-6

    # 3. Model Setup
    d_in = len(src_list) * master_ds.d_model_per_layer
    d_out = len(tgt_list) * master_ds.d_model_per_layer

    model = Transcoder(
        d_model=d_in, d_dict=dict_size, d_output=d_out, l1_coeff=l1_coeff
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 4. Training Loop
    for epoch in range(epochs):
        model.train()
        num_tokens = len(master_ds)
        indices = np.arange(num_tokens)
        np.random.shuffle(indices)

        # Funnel check: Ensures we map Encoder Summary tokens to Predictor tokens correctly
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

            # Load only the unique layers once (The core I/O optimization)
            full_batch = master_ds.get_batch(batch_idx).to(device)

            # Slicing from the Master Batch instead of re-reading from disk
            src_indices = []
            for l in src_list:
                idx = unique_layers.index(l)
                start = idx * master_ds.d_model_per_layer
                src_indices.append(
                    torch.arange(start, start + master_ds.d_model_per_layer)
                )
            src_batch = full_batch[:, torch.cat(src_indices)]

            # Center and scale to unit variance for stable training
            s_batch_norm = (src_batch - mean_s.to(device)) / std_s.to(device)

            if source_layers_str == target_layers_str:
                t_batch_norm = s_batch_norm
            else:
                if not is_funnel:
                    # Optimized slice for target layers
                    tgt_indices = []
                    for l in tgt_list:
                        idx = unique_layers.index(l)
                        start = idx * master_ds.d_model_per_layer
                        tgt_indices.append(
                            torch.arange(start, start + master_ds.d_model_per_layer)
                        )
                    t_batch = full_batch[:, torch.cat(tgt_indices)]
                else:
                    # Funnel Logic (Requires a separate read for target due to index mapping)
                    # We assume summary tokens (CLS) are at index 0 of every 257-token block
                    src_idx_start = batch_idx // src_tokens
                    token_offset = batch_idx % src_tokens
                    tgt_batch_idx = src_idx_start * tgt_tokens + (token_offset // 257)
                    t_batch = master_ds.get_batch(
                        tgt_batch_idx, layers_to_extract=tgt_list
                    ).to(device)

                t_batch_norm = t_batch

            optimizer.zero_grad()
            res = model(s_batch_norm, t_batch_norm)
            loss = res["loss"]
            loss.backward()

            # Normalize decoder weights to unit norm (Critical for dictionary stability)
            model.normalize_decoder()
            optimizer.step()

            if i % 100 == 0:
                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "l2": f"{res['l2_loss'].item():.4f}"}
                )

    # 5. Save Model
    print(f"💾 Saving to {output_path}")
    save_dict = {
        "state_dict": model.state_dict(),
        "norm_stats": {"mean": mean_s, "std": std_s},
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
