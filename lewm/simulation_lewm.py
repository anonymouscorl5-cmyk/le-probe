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
import time
import argparse
import rerun as rr
import traceback
import mujoco
from PIL import Image
from simulation_base import GR1MuJoCoBase
from gr1_protocol import StandardScaler
from gr1_config import SCENE_PATH
from inference_http import InferenceHTTPClient, pack_np

try:
    from dataset.polytope_utils import (
        draw_polytope_on_rgb,
        draw_world_points_on_rgb,
        log_polytope_rerun,
    )
    from lewm.task_workspace import TaskWorkspaceMPCConstraint

    # BGR: blue dot = server FK of final plan step (CEM gate check); green = live EE in draw_polytope
    PLAN_FINAL_EE_BGR = (0, 0, 255)

    TASK_WORKSPACE_AVAILABLE = True
except ImportError as e:
    TASK_WORKSPACE_AVAILABLE = False
    _TASK_WORKSPACE_IMPORT_ERROR = e


class GR1LEWMClient(GR1MuJoCoBase):
    def __init__(
        self,
        base_url="http://127.0.0.1:5555",
        use_multi_view=False,
        use_skeleton=False,
        use_dino=False,
        show_task_workspace=False,
        task_workspace_fill_alpha=0.15,
    ):
        super().__init__()
        self.scaler = StandardScaler()
        self.use_multi_view = use_multi_view
        self.use_skeleton = use_skeleton
        self.use_dino = use_dino

        self.show_task_workspace = show_task_workspace and TASK_WORKSPACE_AVAILABLE
        self.task_workspace_fill_alpha = task_workspace_fill_alpha
        self._task_ws = None
        self._plan_final_ee_xyz: np.ndarray | None = None

        if show_task_workspace and not TASK_WORKSPACE_AVAILABLE:
            print(f"⚠️ Task workspace overlay disabled: {_TASK_WORKSPACE_IMPORT_ERROR}")
        elif self.show_task_workspace:
            self._task_ws = TaskWorkspaceMPCConstraint()
            p = self._task_ws.poly
            print(
                f"🌐 LeWM sim: fixed task polytope viz (local copy, {len(p.corner_points)} corners, "
                f"{p.face_indices.shape[0]} faces) — not sent to server"
            )

        self.client = InferenceHTTPClient(base_url)
        if not self.client.health():
            print(
                f"⚠️ MPC server health check failed at {base_url} (will retry on first plan)"
            )

        print(
            f"🔗 MPC HTTP client → {base_url} (Multi-View: {use_multi_view}, "
            f"Skeleton: {use_skeleton}, DINO: {use_dino})"
        )

    def _log_task_workspace_rerun(self):
        if not self._task_ws:
            return
        log_polytope_rerun(
            self._task_ws.get_draw_polytope(),
            entity_path="world/task_workspace",
            wireframe_path="world/task_workspace_wireframe",
        )

    def _post_render_hook(self, name, rgb, depth=None):
        if self._task_ws is not None:
            drawn = draw_polytope_on_rgb(
                rgb,
                self._task_ws.get_draw_polytope(),
                name,
                self.model,
                self.data,
                depth_buffer=depth,
                fill_alpha=self.task_workspace_fill_alpha,
            )
            if drawn is not rgb:
                rgb[:] = drawn
        if self._plan_final_ee_xyz is not None:
            drawn = draw_world_points_on_rgb(
                rgb,
                self._plan_final_ee_xyz.reshape(1, 3),
                name,
                self.model,
                self.data,
                depth_buffer=depth,
                color=PLAN_FINAL_EE_BGR,
                radius=8,
                label_points=False,
            )
            if drawn is not rgb:
                rgb[:] = drawn
        super()._post_render_hook(name, rgb, depth=depth)

    def capture_observation(self, instruction):
        """Captures required camera views and state."""
        state = self.get_state_32()

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
            self.renderer.update_scene(self.data, camera="world_center")
            img = self.renderer.render()
            img_resized = np.array(
                Image.fromarray(img).resize((224, 224), Image.Resampling.LANCZOS)
            )
            payload["observation.images.world_center"] = pack_np(img_resized)

        return payload


def run_mission(
    base_url,
    use_multi_view,
    use_skeleton=False,
    use_dino=False,
    instruction="Pick up the red cube",
    max_steps=100,
    show_task_workspace=False,
    task_workspace_fill_alpha=0.15,
):
    sim = GR1LEWMClient(
        base_url=base_url,
        use_multi_view=use_multi_view,
        use_skeleton=use_skeleton,
        use_dino=use_dino,
        show_task_workspace=show_task_workspace,
        task_workspace_fill_alpha=task_workspace_fill_alpha,
    )
    print(f"🚀 Starting Omni-MPC Autonomous Mission: '{instruction}'")
    sim.reset_env(randomize_cube=False)

    if sim.show_task_workspace:
        sim._log_task_workspace_rerun()

    audit_history = []

    step_idx = 0
    try:
        while step_idx < max_steps:
            obs_payload = sim.capture_observation(instruction)

            print(
                f"[{time.strftime('%H:%M:%S')}] 🧠 Requesting MPC Plan (Universal Gallery)..."
            )
            resp = sim.client.plan(obs_payload)

            if "action" in resp:
                plan_norm = np.array(resp["action"], dtype=np.float32)
                diag = resp.get("diagnostics", {})

                tw_viol = diag.get("task_workspace_violation")
                tw_feas = diag.get("task_workspace_feasible")
                ee_xyz = diag.get("plan_final_ee_xyz")
                if ee_xyz is not None and len(ee_xyz) == 3:
                    sim._plan_final_ee_xyz = np.asarray(ee_xyz, dtype=np.float64)
                    rr.log(
                        "world/plan_final_ee",
                        rr.Points3D(
                            sim._plan_final_ee_xyz.reshape(1, 3),
                            radii=0.02,
                            colors=[0, 80, 255],
                        ),
                    )
                tw_msg = ""
                if tw_viol is not None:
                    tw_msg = f", task_viol(final)={tw_viol:.4f}"
                if tw_feas is not None:
                    tw_msg += f", feasible={tw_feas}"
                if ee_xyz is not None:
                    tw_msg += f", plan_final_ee=({ee_xyz[0]:.3f}, {ee_xyz[1]:.3f}, {ee_xyz[2]:.3f})"
                print(
                    "   🚀 Executing first action from plan (Solve Time: "
                    f"{diag.get('plan_time_ms')}ms, Horizon: {plan_norm.shape[0]}{tw_msg})"
                )

                chunk_size = min(5, len(plan_norm))
                for i in range(chunk_size):
                    curr_action_norm = plan_norm[i]
                    curr_action_raw = sim.scaler.unscale_action(curr_action_norm)

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
        os.makedirs("inference_history_lewm", exist_ok=True)
        audit_path = "inference_history_lewm/joint_level_audit.json"

        with open(audit_path, "w") as f:
            json.dump(audit_history, f)
        print(f"💾 Full joint-level audit saved to: {audit_path}")
        print("🏁 Mission Complete. Exit.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="MPC server base URL (e.g. https://xxxx.ngrok-free.app or http://127.0.0.1:5555)",
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Legacy: builds base URL"
    )
    parser.add_argument(
        "--port", type=int, default=5555, help="Legacy: builds base URL"
    )
    parser.add_argument("--multi_view", action="store_true", default=False)
    parser.add_argument("--use_skeleton", action="store_true", default=False)
    parser.add_argument("--use_dino", action="store_true", default=False)
    parser.add_argument(
        "--task_workspace",
        action="store_true",
        help="Show fixed task workspace polytope on cameras + Rerun (viz only)",
    )
    parser.add_argument(
        "--task_workspace_fill_alpha",
        type=float,
        default=0.15,
        help="Semi-transparent fill on camera overlay",
    )
    args = parser.parse_args()

    rr.init("gr1_lewm", spawn=False)
    rr.connect_grpc("rerun+http://127.0.0.1:9876/proxy")

    base_url = args.base_url or f"http://{args.host}:{args.port}"
    run_mission(
        base_url,
        args.multi_view,
        use_skeleton=args.use_skeleton,
        use_dino=args.use_dino,
        show_task_workspace=args.task_workspace,
        task_workspace_fill_alpha=args.task_workspace_fill_alpha,
    )
