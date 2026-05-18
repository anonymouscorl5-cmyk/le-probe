"""
ORACLE MPC SIMULATION DRIVER
Role: Client for the LEWM MPC Server. Drives the MuJoCo robot in closed-loop.
"""

# --- Path Stabilization ---
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# --------------------------


import os
import datetime
import json
import numpy as np
import zmq
import msgpack
import time
import argparse
import rerun as rr
import traceback
import mujoco
from PIL import Image
from simulation_base import GR1MuJoCoBase
from gr1_protocol import StandardScaler


class GR1LEWMClient(GR1MuJoCoBase):
    def __init__(
        self,
        server_host="localhost",
        server_port=5555,
        use_multi_view=False,
        use_skeleton=False,
        use_dino=False,
    ):
        super().__init__()
        self.scaler = StandardScaler()
        self.use_multi_view = use_multi_view
        self.use_skeleton = use_skeleton
        self.use_dino = use_dino

        # ZMQ Context
        self.context = zmq.Context()
        self.client = self.context.socket(zmq.REQ)
        self.client.setsockopt(zmq.RCVTIMEO, 120000)
        self.client.connect(f"tcp://{server_host}:{server_port}")

        print(
            f"🔗 Connected to MPC Server at {server_host}:{server_port} (Multi-View: {use_multi_view}, Skeleton: {use_skeleton}, DINO: {use_dino})"
        )

    def capture_observation(self, instruction):
        """Captures required camera views and state."""

        def pack_np(arr):
            return {
                "data": arr.tobytes(),
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
            }

        state = self.get_state_32()
        payload = {
            "instruction": instruction,
            "state": pack_np(state),
        }

        # Extract ground-truth cube position for server-side skeletal prior rendering
        try:
            cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
            if cube_id != -1:
                cube_pos = self.data.xpos[cube_id].copy()
                payload["cube_pos"] = pack_np(cube_pos)
        except Exception as e:
            print(f"⚠️ Could not extract cube position: {e}")

        if self.use_multi_view:
            cam_names = [
                "world_center",
                "world_left",
                "world_right",
                "world_top",
                "world_wrist",
            ]
            for cam in cam_names:
                self.renderer.update_scene(self.data, camera=cam)
                img = self.renderer.render()
                img_resized = np.array(
                    Image.fromarray(img).resize((224, 224), Image.Resampling.LANCZOS)
                )
                payload[f"observation.images.{cam}"] = pack_np(img_resized)
        else:
            # Legacy Single View
            self.renderer.update_scene(self.data, camera="world_center")
            img = self.renderer.render()
            img_resized = np.array(
                Image.fromarray(img).resize((224, 224), Image.Resampling.LANCZOS)
            )
            payload["observation.images.world_center"] = pack_np(img_resized)

        return payload


def run_mission(
    server_host,
    server_port,
    use_multi_view,
    use_skeleton=False,
    use_dino=False,
    instruction="Pick up the red cube",
    max_steps=100,
):
    sim = GR1LEWMClient(
        server_host=server_host,
        server_port=server_port,
        use_multi_view=use_multi_view,
        use_skeleton=use_skeleton,
        use_dino=use_dino,
    )
    print(f"🚀 Starting Omni-MPC Autonomous Mission: '{instruction}'")
    sim.reset_env(randomize_cube=False)

    # Audit History for Parity verification
    audit_history = []

    step_idx = 0
    try:
        while step_idx < max_steps:
            # 1. Perception
            obs_payload = sim.capture_observation(instruction)

            # 2. Planning (Requesting the next optimized chunk)
            print(
                f"[{time.strftime('%H:%M:%S')}] 🧠 Requesting MPC Plan (Universal Gallery)..."
            )
            sim.client.send(msgpack.packb(obs_payload, use_bin_type=True))
            resp = msgpack.unpackb(sim.client.recv(), raw=False)

            if "action" in resp:
                # Received normalized plan (Horizon, 32) in [-1, 1]
                plan_norm = np.array(resp["action"], dtype=np.float32)
                diag = resp.get("diagnostics", {})

                print(
                    "   🚀 Executing first action from plan (Solve Time: "
                    f"{diag.get('plan_time_ms')}ms, Horizon: {plan_norm.shape[0]})"
                )

                # MPC Chunking: Execute 5 steps before re-planning
                chunk_size = min(5, len(plan_norm))
                for i in range(chunk_size):
                    curr_action_norm = plan_norm[i]

                    # --- 🔌 PROTOCOL HANDSHAKE: Unscale to Radians ---
                    curr_action_raw = sim.scaler.unscale_action(curr_action_norm)

                    # Record for audit (All numbers go to JSON, not Rerun)
                    audit_history.append(
                        {
                            "step": step_idx,
                            "action_norm": curr_action_norm.tolist(),
                            "action_raw": curr_action_raw.tolist(),
                            "sim_state": sim.get_state_32().tolist(),
                        }
                    )

                    sim.process_target_32(curr_action_norm)
                    sim.dispatch_action(
                        curr_action_norm,
                        sim.last_target_q,
                        n_steps=50,
                        render_freq=10,
                    )
                    step_idx += 1
            else:
                print(f"❌ Server Error: {resp.get('error')}")
                break
    except KeyboardInterrupt:
        print("\n🛑 Mission interrupted by user.")
    except Exception as e:
        print(f"❌ Mission Error: {e}")
        traceback.print_exc()
    finally:
        # Save Detailed Audit
        os.makedirs("inference_history_lewm", exist_ok=True)
        audit_path = "inference_history_lewm/joint_level_audit.json"

        with open(audit_path, "w") as f:
            json.dump(audit_history, f)
        print(f"💾 Full joint-level audit saved to: {audit_path}")
        print("🏁 Mission Complete. Exit.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--multi_view", action="store_true", default=False)
    parser.add_argument("--use_skeleton", action="store_true", default=False)
    parser.add_argument("--use_dino", action="store_true", default=False)
    args = parser.parse_args()

    # Re-init Rerun for standalone local run
    rr.init("gr1_lewm", spawn=False)
    rr.connect_grpc("rerun+http://127.0.0.1:9876/proxy")

    run_mission(
        args.host,
        args.port,
        args.multi_view,
        use_skeleton=args.use_skeleton,
        use_dino=args.use_dino,
    )
