# --- Path Stabilization ---
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# --------------------------

import numpy as np
import zmq
import msgpack
import rerun as rr
import mujoco
import os
import json
import argparse
import threading
from PIL import Image
from simulation_base import GR1MuJoCoBase
from gr1_config import SCENE_PATH
from gr1_protocol import StandardScaler
from gr1_config import COMPACT_WIRE_JOINTS

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


class GR1TeleopServer(GR1MuJoCoBase):
    """
    Reactive Teleoperation Server (REP Socket).
    Dedicated to the Streamlit Dashboard and IK Calibration.
    """

    def __init__(
        self,
        scene_path=None,
        port=5556,
        lock_posture=False,
        show_reachability=True,
        reachability_horizon=0.25,
        reachability_refresh_every=5,
        reachability_include_hand=False,
    ):
        super().__init__(scene_path or SCENE_PATH, restrict_ik=True)
        self.port = port
        self.lock_posture = lock_posture
        self.is_running = True
        self.show_reachability = show_reachability and REACHABILITY_AVAILABLE
        self.reachability_refresh_every = max(1, reachability_refresh_every)
        self._reach_step_counter = 0
        self._reach_engine = None
        self._reach_cfg = None
        self._reach_lock = threading.Lock()
        self._reach_busy = False
        self._last_reach_poly = None

        if show_reachability and not REACHABILITY_AVAILABLE:
            print(f"⚠️ Reachability overlay disabled: {_REACHABILITY_IMPORT_ERROR}")
        elif self.show_reachability:
            active_idx = (
                list(range(16, 26))
                if reachability_include_hand
                else TELEOP_ACTIVE_WIRE_IDX
            )
            self._reach_engine = GR1ReachabilityEngine(
                scene_path or SCENE_PATH,
                active_wire_idx=active_idx,
            )
            self._reach_cfg = teleop_reachability_config(horizon=reachability_horizon)
            dof_label = "arm+hand 10-DoF" if reachability_include_hand else "arm 7-DoF"
            print(
                f"🌐 Reachability 2D overlay on all cameras (depth-occluded) "
                f"({dof_label}, horizon={reachability_horizon}s, "
                f"refresh every {self.reachability_refresh_every} steps)"
            )

    def _post_render_hook(self, name, rgb, depth=None):
        """Draw latest reachable polytope wireframe on each camera frame before Rerun log."""
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
                fill_alpha=0.15,
            )
            if drawn is not rgb:
                rgb[:] = drawn
        super()._post_render_hook(name, rgb, depth=depth)

    def _update_reachability_overlay(self, force=False):
        """Recompute reachable workspace in a background thread; cache for 2D draw."""
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

        qpos_snapshot = self.data.qpos.copy()
        engine = self._reach_engine
        cfg = self._reach_cfg

        def _compute_and_log():
            with self._reach_lock:
                self._reach_busy = True
            try:
                engine.set_baseline_from_qpos(qpos_snapshot)
                poly = engine.compute(cfg=cfg)
                with self._reach_lock:
                    self._last_reach_poly = poly
            except Exception as e:
                print(f"⚠️ Reachability compute failed: {e}")
            finally:
                with self._reach_lock:
                    self._reach_busy = False

        threading.Thread(target=_compute_and_log, daemon=True).start()

    def run(self):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://*:{self.port}")

        rr.init("gr1_teleop", spawn=False)
        rr.connect_grpc("rerun+http://127.0.0.1:9876/proxy")
        print(
            f"🚀 Teleop Server Running on port {self.port} (Lock Posture: {self.lock_posture})"
        )
        if self.show_reachability:
            self._update_reachability_overlay(force=True)

        while self.is_running:
            msg = socket.recv()
            data = msgpack.unpackb(msg, raw=False)
            cmd = data.get("command")

            def send_resp(payload):
                payload.update(
                    {
                        "upload_queue": self.recorder.pending_uploads,
                        "total_episodes": self.recorder.total_episodes,
                        "batch_status": self.recorder.episodes_since_sync,
                        "physics": self.get_physics_state(),
                    }
                )
                socket.send(msgpack.packb(payload))

            if cmd == "reset":
                self.reset_env(lock_posture=self.lock_posture)
                self._update_reachability_overlay(force=True)
                # Server is the Source of Normalized Truth
                norm_state = StandardScaler().scale_state(self.get_state_32())
                send_resp({"status": "reset_ok", "joints": norm_state.tolist()})

            elif cmd == "wild_randomize":
                self.wild_reset()
                self._update_reachability_overlay(force=True)
                norm_state = StandardScaler().scale_state(self.get_state_32())
                send_resp(
                    {"status": "wild_randomize_ok", "joints": norm_state.tolist()}
                )

            elif cmd == "sync":
                self.recorder.force_sync()
                send_resp({"status": "sync_started"})

            elif cmd == "start_recording":
                self.recorder.start_episode(data.get("task", "Pick up red cube"))
                self.is_recording = True
                send_resp({"status": "recording_started"})

            elif cmd == "stop_recording":
                self.recorder.stop_episode()
                self.is_recording = False
                send_resp({"status": "recording_stopped"})

            elif cmd == "discard_recording":
                self.recorder.discard_episode()
                self.is_recording = False
                send_resp({"status": "recording_discarded"})

            elif cmd == "poll_status":
                send_resp({"status": "status_ok"})

            elif cmd == "ik_pickup":
                phase = data.get("phase", 0)
                offset_cm = data.get("offset_cm", 5)
                self._handle_ik_pickup_logic(phase=phase, offset_cm=offset_cm)
                self._update_reachability_overlay(force=True)

                # Server is the Source of Normalized Truth
                norm_state = StandardScaler().scale_state(self.get_state_32())
                send_resp({"status": "ik_pickup_ok", "joints": norm_state.tolist()})

            elif cmd == "set_cube_pose":
                pose = np.array(data["pose"], dtype=np.float32)
                cube_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint"
                )
                if cube_id != -1:
                    q_idx = self.model.jnt_qposadr[cube_id]
                    self.data.qpos[q_idx : q_idx + 7] = pose
                    mujoco.mj_forward(self.model, self.data)
                self._update_reachability_overlay(force=True)
                send_resp({"status": "cube_pose_ok"})

            elif "target" in data:
                action_32 = np.array(data["target"], dtype=np.float32)
                self.process_target_32(action_32)
                self.dispatch_action(action_32, self.last_target_q)
                self._update_reachability_overlay()
                send_resp({"status": "step_ok"})

            elif cmd == "store_snapshot":
                # 1. Capture All Data
                raw_state = self.get_state_32()
                norm_state = StandardScaler().scale_state(raw_state)
                physics = self.get_physics_state()

                # Capture Cube Pose (Full 7-DoF for future-proof restoration)
                cube_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint"
                )
                cube_qpos = []
                if cube_id != -1:
                    q_idx = self.model.jnt_qposadr[cube_id]
                    cube_qpos = self.data.qpos[q_idx : q_idx + 7].tolist()

                # 2. Build Payload with all 5 Camera Views
                snapshot = {
                    "observation.state": norm_state.tolist(),
                    "action": norm_state.tolist(),  # Dummy action (current state)
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
                    # Store as (C, H, W) list
                    snapshot[key] = np.array(img).transpose(2, 0, 1).tolist()

                # 3. Save to Next Available Index in v2 Dataset
                snap_dir = os.path.join(
                    ROOT_DIR,
                    "datasets",
                    "vedpatwardhan",
                    "gr1_reward_pred_v2",
                )
                os.makedirs(snap_dir, exist_ok=True)

                # Check for "wild_" prefix to distinguish from harvested spectrum
                existing_wild = [
                    f for f in os.listdir(snap_dir) if f.startswith("wild_")
                ]
                next_idx = len(existing_wild)
                snap_path = os.path.join(snap_dir, f"wild_{next_idx:04d}.json")

                with open(snap_path, "w") as f:
                    json.dump(snapshot, f)

                print(
                    f"📸 Snapshot {next_idx:04d} stored at {snap_path} (Reward: {snapshot['progress']:.4f})"
                )
                send_resp({"status": "snapshot_ok", "index": next_idx})

            else:
                send_resp({"status": "unknown"})

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
    parser.add_argument("--port", type=int, default=5556, help="ZMQ Port")
    parser.add_argument(
        "--lock-posture",
        action="store_true",
        default=True,
        help="Lock IK joints to specific targets",
    )
    parser.add_argument(
        "--show-reachability",
        action="store_true",
        default=True,
        help="Overlay reachable workspace wireframe on all 5 camera views",
    )
    parser.add_argument(
        "--no-reachability",
        action="store_true",
        help="Disable reachability overlay",
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
        help="Recompute reachability every N teleop steps",
    )
    parser.add_argument(
        "--reachability-include-hand",
        action="store_true",
        help="Include thumb+index joints (slower, ~8s per update)",
    )
    args = parser.parse_args()

    refresh_every = args.reachability_refresh_every
    if args.reachability_include_hand and refresh_every == 5:
        refresh_every = 20

    GR1TeleopServer(
        port=args.port,
        lock_posture=args.lock_posture,
        show_reachability=args.show_reachability and not args.no_reachability,
        reachability_horizon=args.reachability_horizon,
        reachability_refresh_every=refresh_every,
        reachability_include_hand=args.reachability_include_hand,
    ).run()
