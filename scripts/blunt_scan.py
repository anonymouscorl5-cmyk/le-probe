# --- Path Stabilization ---
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# --------------------------

import pandas as pd
import numpy as np


def blunt_scan():
    df = pd.read_parquet(
        "le-probe/datasets/anonymouscorl5-cmyk/gr1_reward_pred/dataset.parquet"
    )
    item = df.iloc[0]["observation.images.world_center"]

    # Check Index 0
    print(f"Index 0 Audit:")
    for c_idx in range(len(item)):
        c = item[c_idx]
        print(f"  Channel {c_idx} (len={len(c)}):")
        for h_idx in range(len(c)):
            h = c[h_idx]
            if not hasattr(h, "__len__"):
                print(
                    f"    ERROR: C{c_idx} H{h_idx} is NOT a sequence (type={type(h)})"
                )
            elif len(h) != 224:
                print(f"    ERROR: C{c_idx} H{h_idx} length is {len(h)} (Expected 224)")

    print("\nScan Complete.")


if __name__ == "__main__":
    blunt_scan()
