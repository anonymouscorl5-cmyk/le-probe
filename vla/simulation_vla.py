# --- Path Stabilization ---
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# --------------------------

import os
import datetime
import numpy as np
import time
import argparse
import rerun as rr
import json
import traceback
from PIL import Image
import mujoco
from simulation_base import GR1MuJoCoBase
from gr1_protocol import StandardScaler
from inference_http import InferenceHTTPClient, pack_np


class GR1VLAClient(GR1MuJoCoBase):
    """
    Proactive Autonomous Driver (REQ Client).
    Matches the Genesis simulation_gr1.py workflow.
    """

    def __init__(
        self,
        scene_path=None,
        base_url="http://127.0.0.1:5555",
    ):
        super().__init__(scene_path) if scene_path else super().__init__()

        # ✅ Canonical Normalization (Min-Max [-1, 1])
        self.unscaler = StandardScaler()

        # Enable VLA-specific diagnostic logging
        self.debug_log_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "vla_debug.log"
        )
        with open(self.debug_log_path, "w") as f:
            f.write(f"--- VLA DEBUG LOG INITIALIZED: {datetime.datetime.now()} ---\n")

        self.vla_client = InferenceHTTPClient(base_url)
        if not self.vla_client.health():
            print(
                f"⚠️ VLA server health check failed at {base_url} (will retry on first plan)"
            )
        print(f"🔗 VLA HTTP client → {base_url}")

    def capture_vla_observation(self, instruction):
        """Captures 5-cam view (resized to 224x224) and raw state."""
        mujoco.mj_forward(self.model, self.data)

        # ✅ UNIVERSAL HANDSHAKE: Scale state based on mode
        raw_state = self.get_state_32()
        norm_state = self.unscaler.scale_state(raw_state)
        norm_state = np.clip(norm_state, -1.0, 1.0)

        obs = {"instruction": instruction, "state": pack_np(norm_state)}
        for cam in self.cam_names:
            self.renderer.update_scene(self.data, camera=cam)
            img_raw = self.renderer.render()

            # ✅ RESIZE TO 224x224 (Model Native Resolution)
            # Using Image.Resampling.LANCZOS to match LeRobot data augmentation pipelines
            img_resized = np.array(
                Image.fromarray(img_raw).resize((224, 224), Image.Resampling.LANCZOS)
            )
            # ✅ PROTOCOL ALIGNMENT: LeRobot expects (C, H, W) PyTorch-style tensors
            img_chw = np.transpose(img_resized, (2, 0, 1))

            # ✅ PROTOCOL ALIGNMENT: LeRobot expects keys prefixed with 'observation.images.'
            obs[f"observation.images.{cam}"] = pack_np(img_chw)
        return obs

    def run(self, instruction="Pick up the red cube", max_chunks=10):
        print(f"🚀 Starting Autonomous Mission (Canonical Protocol): '{instruction}'")
        self.reset_env()

        # Audit History for Joint-Level verification
        audit_history = []

        for chunk_idx in range(max_chunks):
            # 1. Perception
            obs_payload = self.capture_vla_observation(instruction)

            # 2. Planning
            print(
                f"[{time.strftime('%H:%M:%S')}] 🧠 Requesting Chunk {chunk_idx+1}/{max_chunks}..."
            )
            try:
                resp = self.vla_client.plan(obs_payload)

                if "action" in resp:
                    actions = resp["action"]

                    # ✅ ACTION SAFETY: Clip to [-1, 1] to match simulation_replay.py
                    actions_np = np.clip(actions, -1.0, 1.0)
                    print(
                        f"   🚀 Executing Chunk ({chunk_idx+1}/{max_chunks}): {actions_np.shape[0]} temporal steps..."
                    )

                    # Experiment: Only take the first action
                    first_action = actions_np[0]
                    print(
                        f"   🚀 Executing ONLY first action of chunk for full duration..."
                    )

                    # Protocol Alignment: Process only this one target
                    self.process_target_32(first_action)

                    # Record for audit (single entry for the chunk)
                    audit_history.append(
                        {
                            "chunk": chunk_idx,
                            "frame": 0,
                            "brain_slots": first_action.tolist(),
                            "sim_joints": self.get_state_32().tolist(),
                        }
                    )

                    # Execute for 1600 steps (16 frames * 100 steps/frame)
                    # This holds the model's primary intent for the full 1.6s
                    self.dispatch_action(
                        first_action,
                        self.last_target_q,
                        n_steps=240,
                        render_freq=30,
                        reset_start=True,
                    )
                else:
                    print(f"❌ VLA Server Error: {resp.get('error', 'Unknown error')}")
                    break
            except Exception as e:
                print(f"❌ Mission Error: {e}")
                traceback.print_exc()
                break

        # Save Detailed Audit
        os.makedirs("inference_history_new", exist_ok=True)
        audit_path = "inference_history_new/joint_level_audit.json"
        with open(audit_path, "w") as f:
            json.dump(audit_history, f)
        print(f"💾 Full joint-level audit saved to: {audit_path}")
        print("🏁 Mission Complete. Exit.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GR-1 Autonomous Mission Driver")
    parser.add_argument("--instruction", type=str, default="Pick up the red cube")
    parser.add_argument("--chunks", type=int, default=10)
    parser.add_argument(
        "--base_url",
        type=str,
        default=None,
        help="VLA server base URL (e.g. https://xxxx.ngrok-free.app)",
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Legacy: builds base URL"
    )
    parser.add_argument(
        "--port", "-p", type=int, default=5555, help="Legacy: builds base URL"
    )
    args = parser.parse_args()

    # Re-init Rerun for standalone local run
    rr.init("gr1_vla", spawn=False)
    rr.connect_grpc("rerun+http://127.0.0.1:9876/proxy")

    base_url = args.base_url or f"http://{args.host}:{args.port}"
    sim = GR1VLAClient(base_url=base_url)
    sim.run(instruction=args.instruction, max_chunks=args.chunks)
