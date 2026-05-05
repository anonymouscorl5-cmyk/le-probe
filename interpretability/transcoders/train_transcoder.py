# --- Path Stabilization ---
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
        # Return as float32 for training
        return torch.from_numpy(self.data[idx].copy()).float()


class MultiLayerStreamingDataset(Dataset):
    """
    Synchronized dataset for multiple .bin activation files.
    Concatenates layers along the feature dimension.
    """

    def __init__(self, source_dir, layers):
        self.datasets = []
        for layer in layers:
            bin_path = os.path.join(source_dir, f"{layer}.bin")
            json_path = os.path.join(source_dir, f"{layer}.json")
            self.datasets.append(StreamingActivationsDataset(bin_path, json_path))

        # We assume all layers have the same number of samples (aligned by harvest_activations)
        self.total_samples = len(self.datasets[0])
        self.d_model_per_layer = self.datasets[0].shape[1]
        self.tokens_per_sample = self.datasets[0].tokens_per_sample
        self.total_d_model = self.d_model_per_layer * len(layers)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Concatenate all layers for this index
        acts = [ds[idx] for ds in self.datasets]
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Multi-Layer Streaming Training | Device: {device}")

    # Parse layer lists
    src_list = [s.strip() for s in source_layers_str.split(",")]
    tgt_list = [t.strip() for t in target_layers_str.split(",")]

    print(f"📦 Source Layers: {src_list}")
    print(f"🎯 Target Layers: {tgt_list}")

    # 1. Initialize Datasets
    src_ds = MultiLayerStreamingDataset(source_dir, src_list)
    if source_layers_str == target_layers_str:
        print(f"🔄 SAE Mode: Identity reconstruction")
        tgt_ds = src_ds
    else:
        print(f"🔀 Crosscoder Mode")
        tgt_ds = MultiLayerStreamingDataset(source_dir, tgt_list)

    # 2. Normalization Pass (Sample-based)
    print("📈 Calculating Normalization Stats...")
    sample_size = min(len(src_ds), 500_000)
    indices = np.random.choice(len(src_ds), sample_size, replace=False)

    src_samples = []
    for idx in tqdm(indices, desc="Sampling"):
        src_samples.append(src_ds[idx].unsqueeze(0))
    src_subset = torch.cat(src_samples, dim=0)

    mean_s, std_s = src_subset.mean(dim=0), src_subset.std(dim=0) + 1e-6

    # 3. Model Setup
    model = Transcoder(
        d_model=src_ds.total_d_model,
        d_dict=dict_size,
        d_output=tgt_ds.total_d_model,
        l1_coeff=l1_coeff,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 4. Training Loop
    print(f"🏋️ Training: {source_layers_str} ⮕ {target_layers_str}")

    for epoch in range(epochs):
        model.train()
        num_tokens = len(src_ds)
        indices = np.arange(num_tokens)
        np.random.shuffle(indices)

        is_funnel = src_ds.tokens_per_sample != tgt_ds.tokens_per_sample

        pbar = tqdm(range(0, num_tokens, batch_size), desc=f"Epoch {epoch+1}/{epochs}")
        for i in pbar:
            batch_idx = indices[i : i + batch_size]

            # Load batches (Concatenated)
            s_batch = torch.stack([src_ds[idx] for idx in batch_idx]).to(device)
            s_batch_norm = (s_batch - mean_s.to(device)) / std_s.to(device)

            if source_layers_str == target_layers_str:
                t_batch_norm = s_batch_norm
            else:
                # Load Target Batch
                if not is_funnel:
                    t_batch = torch.stack([tgt_ds[idx] for idx in batch_idx]).to(device)
                else:
                    # Funnel Logic: Map large token count to small token count
                    # We assume summary tokens (e.g. CLS) are at index 0 of each patch-grid
                    # In LeWM Encoder (771 tokens), CLS is every 257 tokens.
                    # In Predictor (3 tokens), every token is a summary token.
                    src_idx_start = batch_idx // src_ds.tokens_per_sample
                    token_offset = batch_idx % src_ds.tokens_per_sample

                    # Target index mapping
                    tgt_batch_idx = src_idx_start * tgt_ds.tokens_per_sample + (
                        token_offset // 257
                    )
                    t_batch = torch.stack([tgt_ds[idx] for idx in tgt_batch_idx]).to(
                        device
                    )

                t_batch_norm = t_batch

            optimizer.zero_grad()
            res = model(s_batch_norm, t_batch_norm)
            loss = res["loss"]
            loss.backward()

            # Normalize decoder weights to unit norm
            model.normalize_decoder()
            optimizer.step()

            if i % 100 == 0:
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "l2": f"{res['l2_loss'].item():.4f}",
                        "l1": f"{res['l1_loss'].item():.4f}",
                    }
                )

    # 5. Save Model
    print(f"💾 Saving to {output_path}")
    save_dict = {
        "state_dict": model.state_dict(),
        "norm_stats": {"mean": mean_s, "std": std_s},
        "meta": {
            "d_model": src_ds.total_d_model,
            "d_output": tgt_ds.total_d_model,
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
