# --- Path Stabilization ---
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# --------------------------

import pandas as pd
import numpy as np
from collections import Counter


def find_ragged():
    df = pd.read_parquet(
        "le-probe/datasets/anonymouscorl5-cmyk/gr1_reward_pred/dataset.parquet"
    )
    item = df.iloc[0]["observation.images.world_center"]

    print(f"Index 0 Top Len: {len(item)}")
    c_lengths = [len(c) for c in item]
    if len(set(c_lengths)) > 1:
        print(f"  ❌ VARIANCE IN CHANNEL LENGTHS: {c_lengths}")

    for c_idx, c in enumerate(item):
        h_lengths = [len(h) for h in c]
        if len(set(h_lengths)) > 1:
            print(f"  ❌ VARIANCE IN ROW LENGTHS (C{c_idx}): {set(h_lengths)}")
            print(f"     Counts: {Counter(h_lengths)}")

        for h_idx, h in enumerate(c):
            if hasattr(h, "__len__"):
                w_lengths = [1 if not hasattr(w, "__len__") else len(w) for w in h]
                if len(set(w_lengths)) > 1:
                    print(
                        f"  ❌ VARIANCE IN PIXEL LENGTHS (C{c_idx} H{h_idx}): {set(w_lengths)}"
                    )


if __name__ == "__main__":
    find_ragged()
