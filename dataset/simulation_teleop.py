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
from PIL import Image
from simulation_base import GR1MuJoCoBase
from gr1_config import SCENE_PATH
from gr1_protocol import StandardScaler
from gr1_config import COMPACT_WIRE_JOINTS


class GR1TeleopServer(GR1MuJoCoBase):
    """
    Reactive Teleoperation Server (REP Socket).
    Dedicated to the Streamlit Dashboard and IK Calibration.
    """

    def __init__(self, scene_path=None, port=5556, lock_posture=False):
        super().__init__(scene_path or SCENE_PATH, restrict_ik=True)
        self.port = port
        self.lock_posture = lock_posture
        self.is_running = True

    def run(self):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://*:{self.port}")

        rr.init("gr1_teleop", spawn=False)
        rr.connect_grpc("rerun+http://127.0.0.1:9876/proxy")
        print(
            f"🚀 Teleop Server Running on port {self.port} (Lock Posture: {self.lock_posture})"
        )

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
                # Server is the Source of Normalized Truth
                norm_state = StandardScaler().scale_state(self.get_state_32())
                send_resp({"status": "reset_ok", "joints": norm_state.tolist()})

            elif cmd == "wild_randomize":
                self.wild_reset()
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
                send_resp({"status": "cube_pose_ok"})

            elif "target" in data:
                action_32 = np.array(data["target"], dtype=np.float32)
                self.process_target_32(action_32)
                self.dispatch_action(action_32, self.last_target_q)
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
    args = parser.parse_args()

    # Pass the toggle to the server
    GR1TeleopServer(port=args.port, lock_posture=args.lock_posture).run()
