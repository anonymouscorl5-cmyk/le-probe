# --- Path Stabilization ---
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# --------------------------

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path


def test_dataloader_repro():
    parquet_file = Path(
        "le-probe/datasets/anonymouscorl5-cmyk/gr1_reward_pred/dataset.parquet"
    )
    if not parquet_file.exists():
        print(f"❌ File not found: {parquet_file}")
        return

    print(f"📊 Loading Parquet Dataset: {parquet_file}...")
    df = pd.read_parquet(parquet_file)
    print(f"📦 Dataset Loaded: {len(df)} samples.")

    class ParquetDataset(Dataset):
        def __init__(self, dataframe):
            self.df = dataframe

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            raw_img = row["observation.images.world_center"]

            # THE LOGIC FROM TUNE_REWARD_HEAD.PY (v9)
            try:
                img_np = np.stack(
                    [
                        np.array(raw_img[0].tolist(), dtype=np.uint8),
                        np.array(raw_img[1].tolist(), dtype=np.uint8),
                        np.array(raw_img[2].tolist(), dtype=np.uint8),
                    ],
                    axis=-1,
                )
                print(img_np.shape, img_np.dtype)

                return torch.from_numpy(img_np), torch.tensor([row["progress"]])
            except Exception as e:
                print(f"\n❌ Error at Index {idx}: {e}")
                raise e

    dataset = ParquetDataset(df)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    print("🚀 Iterating through first 5 batches...")
    try:
        for i, (imgs, rewards) in enumerate(loader):
            print(f"  Batch {i+1} loaded. Shape: {imgs.shape}")
            # if i >= 4:
            #     break
        print("\n✅ Local Data Loading Test PASSED. No errors reproduced.")
    except Exception as e:
        print(f"\n🔥 Local Data Loading Test FAILED.")


if __name__ == "__main__":
    test_dataloader_repro()
