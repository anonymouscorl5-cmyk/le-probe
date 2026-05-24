# --- Path Stabilization ---
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# --------------------------

import numpy as np
import rerun as rr
import mujoco
import os
import json
import argparse
from PIL import Image
from simulation_base import GR1MuJoCoBase
from gr1_config import SCENE_PATH
from gr1_protocol import StandardScaler
from gr1_config import COMPACT_WIRE_JOINTS
from inference_http import InferenceHTTPClient, pack_np, serve_http, TELEOP_PATH
from dataset.polytope_utils import draw_polytope_on_rgb, log_polytope_rerun
from lewm.task_workspace import get_task_workspace_draw_polytope


class GR1TeleopServer(GR1MuJoCoBase):
    """
    Reactive Teleoperation Server (HTTP + msgpack).
    Dedicated to the Streamlit Dashboard and IK Calibration.
    """

    def __init__(
        self,
        scene_path=None,
        port=5556,
        lock_posture=False,
        show_task_workspace=False,
        task_workspace_fill_alpha=0.15,
        query_lewm_reward=False,
        lewm_base_url="http://127.0.0.1:5555",
        lewm_multi_view=False,
    ):
        super().__init__(scene_path or SCENE_PATH, restrict_ik=True)
        self.port = port
        self.lock_posture = lock_posture
        self.is_running = True
        self.show_task_workspace = show_task_workspace
        self.task_workspace_fill_alpha = task_workspace_fill_alpha
        self.query_lewm_reward = query_lewm_reward
        self.lewm_multi_view = lewm_multi_view
        self._lewm_client = None
        if self.query_lewm_reward:
            self._lewm_client = InferenceHTTPClient(lewm_base_url)
            print(
                f"🎯 LeWM reward probe ON → {lewm_base_url} "
                f"(multi_view={lewm_multi_view}; match lewm_server flags)"
            )
        self._tw_poly = None

        if self.show_task_workspace:
            self._tw_poly = get_task_workspace_draw_polytope()
            print(
                f"🌐 Task workspace overlay ON (fixed hull, {len(self._tw_poly.corner_points)} corners, "
                f"{self._tw_poly.face_indices.shape[0]} faces)"
            )

    def _render_needs_depth(self) -> bool:
        return self.show_task_workspace

    def _log_task_workspace_rerun(self):
        if self._tw_poly is not None:
            log_polytope_rerun(
                self._tw_poly,
                entity_path="world/task_workspace",
                wireframe_path="world/task_workspace_wireframe",
            )

    def _post_render_hook(self, name, rgb, depth=None):
        if self._tw_poly is not None:
            drawn = draw_polytope_on_rgb(
                rgb,
                self._tw_poly,
                name,
                self.model,
                self.data,
                depth_buffer=depth,
                fill_alpha=self.task_workspace_fill_alpha,
            )
            if drawn is not rgb:
                rgb[:] = drawn
        super()._post_render_hook(name, rgb, depth=depth)

    def _build_lewm_reward_payload(self) -> dict:
        """Observation packet for ``POST /reward`` (same layout as simulation_lewm)."""
        state = self.get_state_32()
        payload = {"state": pack_np(state)}
        try:
            cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
            if cube_id != -1:
                payload["cube_pos"] = pack_np(self.data.xpos[cube_id].copy())
        except Exception:
            pass

        cam_names = (
            ["world_center", "world_left", "world_right", "world_top", "world_wrist"]
            if self.lewm_multi_view
            else ["world_center"]
        )
        for cam in cam_names:
            self.renderer.update_scene(self.data, camera=cam)
            img = self.renderer.render()
            img_resized = np.array(
                Image.fromarray(img).resize((224, 224), Image.Resampling.LANCZOS)
            )
            payload[f"observation.images.{cam}"] = pack_np(img_resized)
        return payload

    def _maybe_attach_lewm_reward(self, payload: dict) -> dict:
        if not self.query_lewm_reward or self._lewm_client is None:
            return payload
        try:
            reward_resp = self._lewm_client.reward(self._build_lewm_reward_payload())
            if "error" in reward_resp:
                payload["lewm_reward_error"] = reward_resp["error"]
            else:
                physics = payload.get("physics") or self.get_physics_state()
                teleop_progress = (1.0 - physics["target_dist"]) * 10.0
                payload["lewm_reward"] = {
                    **reward_resp,
                    "teleop_progress_proxy": float(teleop_progress),
                }
        except Exception as e:
            payload["lewm_reward_error"] = str(e)
        return payload

    def _enrich_response(self, payload: dict) -> dict:
        payload.update(
            {
                "upload_queue": self.recorder.pending_uploads,
                "total_episodes": self.recorder.total_episodes,
                "batch_status": self.recorder.episodes_since_sync,
                "physics": self.get_physics_state(),
            }
        )
        return self._maybe_attach_lewm_reward(payload)

    def process_request(self, data: dict) -> dict:
        cmd = data.get("command")

        if cmd == "reset":
            self.reset_env(lock_posture=self.lock_posture)
            norm_state = StandardScaler().scale_state(self.get_state_32())
            return self._enrich_response(
                {"status": "reset_ok", "joints": norm_state.tolist()}
            )

        if cmd == "wild_randomize":
            self.wild_reset()
            norm_state = StandardScaler().scale_state(self.get_state_32())
            return self._enrich_response(
                {"status": "wild_randomize_ok", "joints": norm_state.tolist()}
            )

        if cmd == "sync":
            self.recorder.force_sync()
            return self._enrich_response({"status": "sync_started"})

        if cmd == "start_recording":
            self.recorder.start_episode(data.get("task", "Pick up red cube"))
            self.is_recording = True
            return self._enrich_response({"status": "recording_started"})

        if cmd == "stop_recording":
            self.recorder.stop_episode()
            self.is_recording = False
            return self._enrich_response({"status": "recording_stopped"})

        if cmd == "discard_recording":
            self.recorder.discard_episode()
            self.is_recording = False
            return self._enrich_response({"status": "recording_discarded"})

        if cmd == "poll_status":
            return self._enrich_response({"status": "status_ok"})

        if cmd == "ik_pickup":
            phase = data.get("phase", 0)
            offset_cm = data.get("offset_cm", 5)
            self._handle_ik_pickup_logic(phase=phase, offset_cm=offset_cm)
            norm_state = StandardScaler().scale_state(self.get_state_32())
            return self._enrich_response(
                {"status": "ik_pickup_ok", "joints": norm_state.tolist()}
            )

        if cmd == "set_cube_pose":
            pose = np.array(data["pose"], dtype=np.float32)
            cube_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint"
            )
            if cube_id != -1:
                q_idx = self.model.jnt_qposadr[cube_id]
                self.data.qpos[q_idx : q_idx + 7] = pose
                mujoco.mj_forward(self.model, self.data)
            return self._enrich_response({"status": "cube_pose_ok"})

        if "target" in data:
            action_32 = np.array(data["target"], dtype=np.float32)
            self.process_target_32(action_32)
            self.dispatch_action(action_32, self.last_target_q)
            return self._enrich_response({"status": "step_ok"})

        if cmd == "store_snapshot":
            raw_state = self.get_state_32()
            norm_state = StandardScaler().scale_state(raw_state)
            physics = self.get_physics_state()

            cube_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint"
            )
            cube_qpos = []
            if cube_id != -1:
                q_idx = self.model.jnt_qposadr[cube_id]
                cube_qpos = self.data.qpos[q_idx : q_idx + 7].tolist()

            snapshot = {
                "observation.state": norm_state.tolist(),
                "action": norm_state.tolist(),
                "progress": (1.0 - physics["target_dist"]) * 10.0,
                "cube_qpos": cube_qpos,
            }

            cam_mapping = {
                "observation.images.world_center": "world_center",
                "observation.images.world_left": "world_left",
                "observation.images.world_right": "world_right",
                "observation.images.world_top": "world_top",
                "observation.images.world_wrist": "world_wrist",
            }

            for key, cam_name in cam_mapping.items():
                self.renderer.update_scene(self.data, camera=cam_name)
                rgb = self.renderer.render()
                img = Image.fromarray(rgb).resize((224, 224))
                snapshot[key] = np.array(img).transpose(2, 0, 1).tolist()

            snap_dir = os.path.join(
                ROOT_DIR,
                "datasets",
                "vedpatwardhan",
                "gr1_reward_pred_v2",
            )
            os.makedirs(snap_dir, exist_ok=True)

            existing_wild = [f for f in os.listdir(snap_dir) if f.startswith("wild_")]
            next_idx = len(existing_wild)
            snap_path = os.path.join(snap_dir, f"wild_{next_idx:04d}.json")

            with open(snap_path, "w") as f:
                json.dump(snapshot, f)

            print(
                f"📸 Snapshot {next_idx:04d} stored at {snap_path} "
                f"(Reward: {snapshot['progress']:.4f})"
            )
            return self._enrich_response({"status": "snapshot_ok", "index": next_idx})

        return self._enrich_response({"status": "unknown"})

    def run(self, host: str = "0.0.0.0"):
        rr.init("gr1_teleop", spawn=False)
        rr.connect_grpc("rerun+http://127.0.0.1:9876/proxy")
        if self.show_task_workspace:
            self._log_task_workspace_rerun()
        serve_http(
            self.process_request,
            host=host,
            port=self.port,
            rpc_path=TELEOP_PATH,
            title="GR-1 Teleop Server",
        )

    def _handle_ik_pickup_logic(self, phase=0, offset_cm=5):
        """Hardened multi-phase IK solver for red cube (Extreme Constraint Edition)."""
        self.current_phase = phase + 1
        print(
            f"🎯 Executing IK Pickup Phase {phase} (Global ID: {self.current_phase})..."
        )

        cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        cube_pos = self.data.qpos[
            self.model.jnt_qposadr[cube_id] : self.model.jnt_qposadr[cube_id] + 3
        ].copy()
        quat_down = [0, 1, 0, 0]

        if phase == 0:
            # Phase 1: Lift (Approach)
            pos_i_h, pos_t_h, pos_w_h = (
                cube_pos + [0.02, 0.02, 0.02 + offset_cm / 100.0],
                cube_pos + [-0.02, 0, 0.02 + offset_cm / 100.0],
                cube_pos + [0, 0, 0.08 + offset_cm / 100.0],
            )
            q_reach_h = self.solve_ik(
                pos_w_h, quat_down, pos_i_h, pos_t_h, posture_cost=1e-6
            )
            self.dispatch_action(
                self.qpos_to_action_32(q_reach_h),
                q_reach_h,
                n_steps=240,
                render_freq=30,
            )

        elif phase == 1:
            # Phase 2: Descent
            pos_i_l, pos_t_l, pos_w_l = (
                cube_pos + [-0.02, 0.02, 0],
                cube_pos + [-0.06, 0, 0],
                cube_pos + [0, 0, 0.06],
            )
            q_reach_l = self.solve_ik(
                pos_w_l, quat_down, pos_i_l, pos_t_l, posture_cost=1e-6
            )
            # ✅ WIDE OPEN HAND: Force fingers to 0.0 (Open)
            for f_idx in [50, 51, 52, 53, 54, 55, 56]:
                if f_idx < len(q_reach_l):
                    q_reach_l[f_idx] = 0.0

            self.dispatch_action(
                self.qpos_to_action_32(q_reach_l),
                q_reach_l,
                n_steps=240,
                render_freq=30,
            )

        elif phase == 2:
            # Phase 3: Grasp
            pos_i_l, pos_t_l, pos_w_l = (
                cube_pos + [0, 0.02, 0],
                cube_pos + [0, 0, 0],
                cube_pos + [0, 0, 0],
            )
            q_reach_l = self.solve_ik(
                pos_w_l, quat_down, pos_i_l, pos_t_l, posture_cost=1e-6
            )
            q_grasp = q_reach_l.copy()
            q_grasp[48] = 1.1
            for g_id in [50, 52, 54, 56]:
                q_grasp[g_id] = -1.1
            self.dispatch_action(
                self.qpos_to_action_32(q_grasp), q_grasp, n_steps=240, render_freq=30
            )

        elif phase == 3:
            # Phase 4: Lift (Retract)
            pos_i_up, pos_t_up, pos_w_up = (
                cube_pos + [0, 0.02, 0.25],
                cube_pos + [0, 0, 0.25],
                cube_pos + [0, 0, 0.25],
            )
            q_lift = self.solve_ik(
                pos_w_up, quat_down, pos_i_up, pos_t_up, posture_cost=1e-6
            )
            q_lift[48] = 1.1
            for g_id in [50, 52, 54, 56]:
                q_lift[g_id] = -1.1
            self.dispatch_action(
                self.qpos_to_action_32(q_lift), q_lift, n_steps=240, render_freq=30
            )

        self._log_phase(phase + 1)

    def _log_phase(self, phase_num):
        """Snapshots unnormalized, normalized, and scene states for the current phase."""
        # 1. Capture states
        raw_state = self.get_state_32()
        norm_state = StandardScaler().scale_state(raw_state)

        # Capture Cube State
        cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        cube_qpos = []
        if cube_id != -1:
            q_idx = self.model.jnt_qposadr[cube_id]
            cube_qpos = self.data.qpos[q_idx : q_idx + 7].tolist()

        # 2. Map to Names
        unnorm_dict = {
            name: float(val) for name, val in zip(COMPACT_WIRE_JOINTS, raw_state)
        }
        norm_dict = {
            name: float(val) for name, val in zip(COMPACT_WIRE_JOINTS, norm_state)
        }

        # 3. Update internal registry
        if not hasattr(self, "phase_lifecycle"):
            self.phase_lifecycle = {}

        self.phase_lifecycle[f"phase_{phase_num}"] = {
            "unnormalized": unnorm_dict,
            "normalized": norm_dict,
            "cube_qpos": cube_qpos,
        }

        # 4. Save to target file
        log_path = os.path.join(ROOT_DIR, "phase_lifecycle.json")
        with open(log_path, "w") as f:
            json.dump(self.phase_lifecycle, f, indent=4)
        print(f"📝 Phase {phase_num} lifecycle saved to phase_lifecycle.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GR-1 Teleop Server")
    parser.add_argument("--port", type=int, default=5556, help="HTTP listen port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="HTTP bind address")
    parser.add_argument(
        "--lock-posture",
        action="store_true",
        default=True,
        help="Lock IK joints to specific targets",
    )
    parser.add_argument(
        "--task-workspace",
        action="store_true",
        help="Show fixed task workspace polytope on all cameras + Rerun",
    )
    parser.add_argument(
        "--task-workspace-fill-alpha",
        type=float,
        default=0.15,
        help="Semi-transparent fill on camera overlay (0 = wireframe only)",
    )
    parser.add_argument(
        "--query-lewm-reward",
        action="store_true",
        help="On poll_status, query LeWM POST /reward (reward head only, no MPC)",
    )
    parser.add_argument(
        "--lewm-base-url",
        type=str,
        default="http://127.0.0.1:5555",
        help="LeWM server base URL when --query-lewm-reward is set",
    )
    parser.add_argument(
        "--lewm-multi-view",
        action="store_true",
        help="Send 5 camera views to LeWM (must match lewm_server --multi_view)",
    )
    args = parser.parse_args()

    GR1TeleopServer(
        port=args.port,
        lock_posture=args.lock_posture,
        show_task_workspace=args.task_workspace,
        task_workspace_fill_alpha=args.task_workspace_fill_alpha,
        query_lewm_reward=args.query_lewm_reward,
        lewm_base_url=args.lewm_base_url,
        lewm_multi_view=args.lewm_multi_view,
    ).run(host=args.host)
