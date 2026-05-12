# --- Path Stabilization ---
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# --------------------------

import zmq
import msgpack
import time
from tqdm import tqdm


def automate_wild_harvest(num_samples=1000, port=5556):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://127.0.0.1:{port}")

    print(f"🌀 Starting Automated Wild Harvest ({num_samples} samples)...")

    for i in tqdm(range(num_samples)):
        # 1. Wild Randomize
        socket.send(msgpack.packb({"command": "wild_randomize"}))
        resp = msgpack.unpackb(socket.recv(), raw=False)
        if resp.get("status") != "wild_randomize_ok":
            print(f"❌ Error: Wild randomize failed at iteration {i}")
            break

        # 2. Store Snapshot
        socket.send(msgpack.packb({"command": "store_snapshot"}))
        resp = msgpack.unpackb(socket.recv(), raw=False)
        if resp.get("status") != "snapshot_ok":
            print(f"❌ Error: Snapshot failed at iteration {i}")
            break

        # Optional: slight delay to allow renderer to breathe
        # time.sleep(0.01)

    print(f"✅ Automated Wild Harvest complete! 1,000 failure modes captured.")


if __name__ == "__main__":
    automate_wild_harvest(1000)
