# --- Path Stabilization ---
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# --------------------------

import hashlib
import os
from huggingface_hub import hf_hub_download
from pathlib import Path


def get_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def compare_files():
    local_path = Path(
        "le-probe/datasets/anonymouscorl5-cmyk/gr1_reward_pred/dataset.parquet"
    )
    print(f"📥 Downloading from HF...")
    hf_path = hf_hub_download(
        repo_id="gr1_reward_pred",
        filename="dataset.parquet",
        repo_type="dataset",
    )

    local_hash = get_hash(local_path)
    hf_hash = get_hash(hf_path)

    print(f"Local Path: {local_path}")
    print(f"HF Path:    {hf_path}")
    print(f"Local Hash: {local_hash}")
    print(f"HF Hash:    {hf_hash}")
    print(f"Match:      {local_hash == hf_hash}")


if __name__ == "__main__":
    compare_files()
