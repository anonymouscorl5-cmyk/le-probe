# --- Path Stabilization ---
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# --------------------------

from tqdm import tqdm

from inference_http import InferenceHTTPClient, TELEOP_PATH, TELEOP_TIMEOUT_S


def automate_wild_harvest(num_samples=1000, base_url="http://127.0.0.1:5556"):
    client = InferenceHTTPClient(
        base_url,
        timeout_s=TELEOP_TIMEOUT_S,
        endpoint=TELEOP_PATH,
    )

    print(f"🌀 Starting Automated Wild Harvest ({num_samples} samples) → {base_url}")

    for i in tqdm(range(num_samples)):
        resp = client.command({"command": "wild_randomize"})
        if resp.get("status") != "wild_randomize_ok":
            print(f"❌ Error: Wild randomize failed at iteration {i}")
            break

        resp = client.command({"command": "store_snapshot"})
        if resp.get("status") != "snapshot_ok":
            print(f"❌ Error: Snapshot failed at iteration {i}")
            break

    print(f"✅ Automated Wild Harvest complete! {num_samples} failure modes captured.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://127.0.0.1:5556",
        help="Teleop HTTP server (POST /teleop)",
    )
    parser.add_argument("-n", type=int, default=1000)
    args = parser.parse_args()
    automate_wild_harvest(args.n, base_url=args.base_url)
