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
import threading
import rerun as rr
import traceback
import mujoco
from PIL import Image
from simulation_base import GR1MuJoCoBase
from gr1_protocol import StandardScaler
from gr1_config import SCENE_PATH

try:
    from dataset.gr1_reachability import (
        GR1ReachabilityEngine,
        TELEOP_ACTIVE_WIRE_IDX,
        draw_polytope_on_rgb,
        teleop_reachability_config,
    )

    REACHABILITY_AVAILABLE = True
except ImportError as e:
    REACHABILITY_AVAILABLE = False
    _REACHABILITY_IMPORT_ERROR = e


class GR1LEWMClient(GR1MuJoCoBase):
    def __init__(
        self,
        server_host="localhost",
        server_port=5555,
        use_multi_view=False,
        use_skeleton=False,
        use_dino=False,
        show_reachability=True,
        reachability_horizon=0.25,
        reachability_refresh_every=5,
        reachability_include_hand=False,
        reachability_limit_mode="hybrid",
        reachability_dq_max=1.5,
        reachability_quality="fast",
        reachability_n_samples=None,
        reachability_facet_dim=None,
        reachability_fill_alpha=0.15,
    ):
        super().__init__()
        self.scaler = StandardScaler()
        self.use_multi_view = use_multi_view
        self.use_skeleton = use_skeleton
        self.use_dino = use_dino

        self.show_reachability = show_reachability and REACHABILITY_AVAILABLE
        self.reachability_refresh_every = max(1, reachability_refresh_every)
        self._reach_step_counter = 0
        self._reach_engine = None
        self._reach_cfg = None
        self._reach_lock = threading.Lock()
        self._reach_busy = False
        self._last_reach_poly = None
        self.reachability_fill_alpha = reachability_fill_alpha

        if show_reachability and not REACHABILITY_AVAILABLE:
            print(f"⚠️ Reachability overlay disabled: {_REACHABILITY_IMPORT_ERROR}")
        elif self.show_reachability:
            active_idx = (
                list(range(16, 26))
                if reachability_include_hand
                else TELEOP_ACTIVE_WIRE_IDX
            )
            self._reach_engine = GR1ReachabilityEngine(
                SCENE_PATH,
                active_wire_idx=active_idx,
            )
            self._reach_cfg = teleop_reachability_config(
                horizon=reachability_horizon,
                limit_mode=reachability_limit_mode,
                dq_max_rad_s=reachability_dq_max,
                quality=reachability_quality,
                n_samples=reachability_n_samples,
                facet_dim=reachability_facet_dim,
            )
            c = self._reach_cfg
            print(
                f"🌐 LeWM sim: reachable polytope on all cameras (same as teleop; "
                f"horizon={c.time_horizon}s, limits={c.limit_mode}, "
                f"refresh every {self.reachability_refresh_every} plan cycles)"
            )

        # ZMQ Context
        self.context = zmq.Context()
        self.client = self.context.socket(zmq.REQ)
        self.client.setsockopt(zmq.RCVTIMEO, 120000)
        self.client.connect(f"tcp://{server_host}:{server_port}")

        print(
            f"🔗 Connected to MPC Server at {server_host}:{server_port} (Multi-View: {use_multi_view}, Skeleton: {use_skeleton}, DINO: {use_dino})"
        )

    def _post_render_hook(self, name, rgb, depth=None):
        poly = None
        if self.show_reachability:
            with self._reach_lock:
                poly = self._last_reach_poly
        if poly is not None:
            drawn = draw_polytope_on_rgb(
                rgb,
                poly,
                name,
                self.model,
                self.data,
                depth_buffer=depth,
                fill_alpha=self.reachability_fill_alpha,
            )
            if drawn is not rgb:
                rgb[:] = drawn
        super()._post_render_hook(name, rgb, depth=depth)

    def _update_reachability_overlay(self, force=False):
        if not self._reach_engine:
            return
        self._reach_step_counter += 1
        if (
            not force
            and self._reach_step_counter % self.reachability_refresh_every != 0
        ):
            return
        if self._reach_busy and not force:
            return

        wire32 = self.get_state_32().copy()
        engine = self._reach_engine
        cfg = self._reach_cfg

        def _compute():
            with self._reach_lock:
                self._reach_busy = True
            try:
                engine.set_baseline_from_wire32(wire32)
                poly = engine.compute(cfg=cfg)
                with self._reach_lock:
                    self._last_reach_poly = poly
            except Exception as e:
                print(f"⚠️ Reachability compute failed: {e}")
            finally:
                with self._reach_lock:
                    self._reach_busy = False

        threading.Thread(target=_compute, daemon=True).start()

    def capture_observation(self, instruction):
        """Captures required camera views and state."""

        def pack_np(arr):
            return {
                "data": arr.tobytes(),
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
            }

        state = self.get_state_32()

        # # Calculate dynamic phase_idx based on hand-to-cube distance
        payload = {
            "instruction": instruction,
            "state": pack_np(state),
        }

        if self.use_dino:
            physics = self.get_physics_state()
            dist = physics["target_dist"]
            if dist > 0.2:
                phase_idx = 0
            elif dist > 0.1:
                phase_idx = 1
            else:
                phase_idx = 2
            payload["phase_idx"] = phase_idx

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
    show_reachability=True,
    reachability_horizon=0.25,
    reachability_refresh_every=5,
    reachability_include_hand=False,
    reachability_limit_mode="hybrid",
    reachability_dq_max=1.5,
    reachability_quality="fast",
    reachability_n_samples=None,
    reachability_facet_dim=None,
    reachability_fill_alpha=0.15,
):
    sim = GR1LEWMClient(
        server_host=server_host,
        server_port=server_port,
        use_multi_view=use_multi_view,
        use_skeleton=use_skeleton,
        use_dino=use_dino,
        show_reachability=show_reachability,
        reachability_horizon=reachability_horizon,
        reachability_refresh_every=reachability_refresh_every,
        reachability_include_hand=reachability_include_hand,
        reachability_limit_mode=reachability_limit_mode,
        reachability_dq_max=reachability_dq_max,
        reachability_quality=reachability_quality,
        reachability_n_samples=reachability_n_samples,
        reachability_facet_dim=reachability_facet_dim,
        reachability_fill_alpha=reachability_fill_alpha,
    )
    print(f"🚀 Starting Omni-MPC Autonomous Mission: '{instruction}'")
    sim.reset_env(randomize_cube=False)

    if sim.show_reachability:
        sim._update_reachability_overlay(force=True)

    # Audit History for Parity verification
    audit_history = []

    step_idx = 0
    try:
        while step_idx < max_steps:
            if sim.show_reachability:
                sim._update_reachability_overlay()

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

                reach_viol = diag.get("reach_violation_final")
                reach_msg = (
                    f", reach_viol={reach_viol:.4f}" if reach_viol is not None else ""
                )
                print(
                    "   🚀 Executing first action from plan (Solve Time: "
                    f"{diag.get('plan_time_ms')}ms, Horizon: {plan_norm.shape[0]}{reach_msg})"
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
    parser.add_argument(
        "--no-reachability",
        action="store_true",
        help="Disable reachable-workspace overlay on camera views",
    )
    parser.add_argument(
        "--reachability-horizon",
        type=float,
        default=0.25,
        help="Time horizon (seconds) for local reachable set",
    )
    parser.add_argument(
        "--reachability-refresh-every",
        type=int,
        default=5,
        help="Recompute reachability every N MPC plan requests",
    )
    parser.add_argument("--reachability-include-hand", action="store_true")
    parser.add_argument(
        "--reachability-limit-mode",
        choices=["hybrid", "velocity", "position"],
        default="hybrid",
    )
    parser.add_argument("--reachability-dq-max", type=float, default=1.5)
    parser.add_argument(
        "--reachability-quality",
        choices=["fast", "balanced", "high"],
        default="fast",
    )
    parser.add_argument("--reachability-n-samples", type=int, default=None)
    parser.add_argument("--reachability-facet-dim", type=int, default=None)
    parser.add_argument("--reachability-fill-alpha", type=float, default=0.15)
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
        show_reachability=not args.no_reachability,
        reachability_horizon=args.reachability_horizon,
        reachability_refresh_every=args.reachability_refresh_every,
        reachability_include_hand=args.reachability_include_hand,
        reachability_limit_mode=args.reachability_limit_mode,
        reachability_dq_max=args.reachability_dq_max,
        reachability_quality=args.reachability_quality,
        reachability_n_samples=args.reachability_n_samples,
        reachability_facet_dim=args.reachability_facet_dim,
        reachability_fill_alpha=args.reachability_fill_alpha,
    )
