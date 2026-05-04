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


def train_transcoder(
    source_dir,
    source_layer,
    target_layer,
    output_path,
    dict_size=12288,
    l1_coeff=1e-3,
    epochs=5,
    batch_size=4096,
    lr=1e-4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Streaming Training | Device: {device}")

    src_bin = os.path.join(source_dir, f"{source_layer}.bin")
    src_json = os.path.join(source_dir, f"{source_layer}.json")
    tgt_bin = os.path.join(source_dir, f"{target_layer}.bin")
    tgt_json = os.path.join(source_dir, f"{target_layer}.json")

    # 1. Initialize Datasets
    src_ds = StreamingActivationsDataset(src_bin, src_json)
    if source_layer == target_layer:
        print(f"🔄 SAE Mode: Training identity reconstruction for {source_layer}")
        tgt_ds = src_ds
    else:
        print(f"🔀 Transcoder Mode: {source_layer} ⮕ {target_layer}")
        tgt_ds = StreamingActivationsDataset(tgt_bin, tgt_json)

    # 2. Normalization Pass (Streaming)
    print("📈 Calculating Normalization Stats...")
    sample_size = min(len(src_ds), 1_000_000)
    indices = np.random.choice(len(src_ds), sample_size, replace=False)

    src_subset = torch.from_numpy(src_ds.data[indices].copy()).float()
    mean_s, std_s = src_subset.mean(dim=0), src_subset.std(dim=0) + 1e-6

    # 3. Model Setup
    d_model = src_ds.shape[1]
    model = Transcoder(d_model, dict_size, l1_coeff).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 4. Training Loop
    print(f"🏋️ Training: {source_layer} ⮕ {target_layer}")

    for epoch in range(epochs):
        model.train()

        num_tokens = len(src_ds)
        indices = np.arange(num_tokens)
        np.random.shuffle(indices)

        is_funnel = src_ds.tokens_per_sample != tgt_ds.tokens_per_sample

        pbar = tqdm(range(0, num_tokens, batch_size), desc=f"Epoch {epoch+1}/{epochs}")
        for i in pbar:
            batch_idx = indices[i : i + batch_size]

            s_batch = torch.from_numpy(src_ds.data[batch_idx].copy()).float().to(device)
            # Center and scale to unit variance for stable SAE training
            s_batch_norm = (s_batch - mean_s.to(device)) / std_s.to(device)

            if source_layer == target_layer:
                t_batch_norm = s_batch_norm
            else:
                # Load target activations
                if not is_funnel:
                    t_batch = (
                        torch.from_numpy(tgt_ds.data[batch_idx].copy())
                        .float()
                        .to(device)
                    )
                else:
                    # Funnel Logic: Map large token count to small token count
                    # We assume the user wants to map Encoder Summary tokens to Predictor tokens
                    # Find which 'moment' each token belongs to
                    moments = batch_idx // src_ds.tokens_per_sample
                    target_idx = moments * tgt_ds.tokens_per_sample + (
                        batch_idx % tgt_ds.tokens_per_sample
                    )

                    # Ensure we don't go out of bounds if encoder has more tokens than predictor can map
                    valid_mask = (
                        batch_idx % src_ds.tokens_per_sample
                    ) < tgt_ds.tokens_per_sample
                    if not valid_mask.any():
                        continue

                    s_batch_norm = s_batch_norm[valid_mask]
                    t_batch = (
                        torch.from_numpy(tgt_ds.data[target_idx[valid_mask]].copy())
                        .float()
                        .to(device)
                    )

                # Transcoder target is normalized the same way as source if layers differ but represent same manifold
                # but for simplicity we'll use raw for now unless source==target
                t_batch_norm = t_batch

            optimizer.zero_grad()
            out = model(s_batch_norm, target=t_batch_norm)
            loss = out["loss"]
            loss.backward()
            optimizer.step()
            model.normalize_decoder()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    # 5. Save
    save_dict = {
        "state_dict": model.state_dict(),
        "norm_stats": {"mean": mean_s, "std": std_s},
        "config": {
            "dict_size": dict_size,
            "l1": l1_coeff,
            "source_layer": source_layer,
            "target_layer": target_layer,
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(save_dict, output_path)
    print(f"✨ Model saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--source_layer", type=str, required=True)
    parser.add_argument("--target_layer", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--dict_size", type=int, default=12288)
    parser.add_argument("--l1", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    train_transcoder(
        args.dir,
        args.source_layer,
        args.target_layer,
        args.output,
        args.dict_size,
        l1_coeff=args.l1,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )
