import numpy as np
import mujoco
import rerun as rr
import os
import datetime
import warnings
from pathlib import Path
from PIL import Image
import mink

from dataset.lerobot_manager import LeRobotManager
from gr1_config import (
    COMPACT_WIRE_JOINTS,
    JOINT_LIMITS_MIN,
    JOINT_LIMITS_MAX,
    SCENE_PATH,
    FROZEN_JOINTS,
    IK_POSTURE_LOCKS,
)
from gr1_protocol import StandardScaler

# Suppress performance warnings from qpsolvers
warnings.filterwarnings("ignore", category=UserWarning, module="qpsolvers")


class GR1MuJoCoBase:
    """
    Shared Physical Foundation for GR-1 MuJoCo Simulations.
    Handles XML loading, IK solving, State extraction, and Perception.
    """

    def __init__(self, scene_path=SCENE_PATH, restrict_ik=True):
        print(f"--- GR-1 MODULAR BASE (MuJoCo) ---")
        self.restrict_ik = restrict_ik
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)

        # Perception & Cameras
        self.cam_names = [
            "world_top",
            "world_left",
            "world_right",
            "world_center",
            "world_wrist",
        ]
        self.frame_indices = {cam: 0 for cam in self.cam_names}

        # Renderer
        self.res = (480, 480)
        self.renderer = mujoco.Renderer(
            self.model, height=self.res[1], width=self.res[0]
        )

        # Mapping Protocol Names -> Joint IDs
        self.wire_min = np.array(JOINT_LIMITS_MIN)
        self.wire_max = np.array(JOINT_LIMITS_MAX)
        self._init_joint_mappings()
        self._init_finger_coupling()

        # Diagnostic Logging
        self.debug_log_path = None

        # LeRobot Manager
        self.recorder = LeRobotManager(
            repo_id="gr1_pickup_grasp", fps=10, upload_interval=20
        )

        # Canonical Scaling Logic
        self.unscaler = StandardScaler()

        # IK Setup (Mink 1.1.0)
        self._init_ik_solver()

        # Internal state
        root_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        self.root_q_idx = self.model.jnt_qposadr[root_id]
        self.data.qpos[self.root_q_idx : self.root_q_idx + 3] = [0.0, 0.0, 0.95]

        self.last_target_q = self.data.qpos.copy()
        self.active_joints_this_command = set()
        self.is_recording = False
        self.current_phase = 0  # 0: Neutral, 1: Approach, 2: Descent, 3: Grasp, 4: Lift
        self.rerun_count = 0
        self.render_step_idx = 0
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_log_dir = (
            Path(os.path.dirname(os.path.abspath(__file__)))
            / "temp_images"
            / self.session_id
        )
        self._init_joint_mappings()
        self._init_finger_coupling()

    def _init_joint_mappings(self):
        self.ik_names = set()
        base_path = os.path.dirname(os.path.abspath(__file__))

        # Load IK whitelist
        i_path = os.path.join(base_path, "ik_joints.txt")
        with open(i_path, "r") as f:
            self.ik_names.update(
                [l.strip().split("#")[0].strip() for l in f if l.strip()]
            )

        print(f"✅ Loaded {len(self.ik_names)} IK joint names.")

        self.protocol_joint_ids = []
        self.v_allowed_mask = np.zeros(32)
        for i, name in enumerate(COMPACT_WIRE_JOINTS):
            try:
                j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                self.protocol_joint_ids.append(j_id)
                # v_allowed_mask is ALWAYS 1.0 for valid joints to allow full protocol freedom
                # (randomization, VLA actions, etc.), even if IK is restricted.
                if j_id != -1:
                    self.v_allowed_mask[i] = 1.0
            except:
                self.protocol_joint_ids.append(-1)

    def _init_finger_coupling(self):
        self.coupling_map = {}
        for i, name in enumerate(COMPACT_WIRE_JOINTS):
            if "proximal" in name.lower():
                base_prefix = name.split("_proximal")[0]
                for j in range(self.model.njnt):
                    j_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
                    if j_name and base_prefix in j_name and j_name != name:
                        if "thumb" in name.lower() and (
                            ("yaw" in name.lower() and "pitch" in j_name.lower())
                            or ("pitch" in name.lower() and "yaw" in j_name.lower())
                        ):
                            continue
                        if i not in self.coupling_map:
                            self.coupling_map[i] = []
                        # Store qpos index instead of joint id!
                        q_idx = self.model.jnt_qposadr[j]
                        self.coupling_map[i].append(q_idx)

    def _init_ik_solver(self):
        self.ee_index_link = "R_index_tip_link"
        self.ee_thumb_link = "R_thumb_tip_link"
        self.ee_wrist_link = "right_hand_pitch_link"
        self.configuration = mink.Configuration(self.model)

        # Determine authorized velocity indices (DOFs) for the IK Solver
        self.auth_dofs = set()
        whitelist = self.ik_names if self.restrict_ik else COMPACT_WIRE_JOINTS
        for name in whitelist:
            j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if j_id != -1:
                v_idx = self.model.jnt_dofadr[j_id]
                # Map joint type (free=0, ball=1, slide=2, hinge=3) to DOF count
                dof_counts = [6, 3, 1, 1]
                j_type = self.model.jnt_type[j_id]
                dnum = dof_counts[j_type]
                for d in range(dnum):
                    self.auth_dofs.add(int(v_idx + d))
        print(f"IK auth_dofs (restricted={self.restrict_ik}):", self.auth_dofs)

        # Always allow root for relative calculations if needed,
        # but here we strictly follow the whitelist for robot bones.

        all_dofs = set(range(self.model.nv))
        frozen_dofs = list(all_dofs - self.auth_dofs)

        self.tasks = [
            mink.FrameTask(
                frame_name=self.ee_index_link,
                frame_type="body",
                position_cost=5.0,
                orientation_cost=0.0,
                lm_damping=0.001,
            ),
            mink.FrameTask(
                frame_name=self.ee_thumb_link,
                frame_type="body",
                position_cost=5.0,
                orientation_cost=0.0,
                lm_damping=0.001,
            ),
            mink.FrameTask(
                frame_name=self.ee_wrist_link,
                frame_type="body",
                position_cost=2.0,
                orientation_cost=0.0,
                lm_damping=0.001,
            ),
            mink.PostureTask(model=self.model, cost=1e-6),
            mink.DofFreezingTask(model=self.model, dof_indices=frozen_dofs),
        ]
        self.tasks[3].set_target(self.model.qpos0)

        # Build physical constraints (Hard Limits 🔗)
        # This ensures the solver itself is aware of the XML boundaries
        self.limits = [
            mink.ConfigurationLimit(model=self.model, min_distance_from_limits=0.01),
            mink.VelocityLimit(model=self.model),
        ]

    def _debug_log(self, msg):
        """Helper for timestamped diagnostic logging. Only writes if debug_log_path is set."""
        if self.debug_log_path is None:
            return
        t_str = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        with open(self.debug_log_path, "a") as f:
            f.write(f"[{t_str}] {msg}\n")

    def _render_needs_depth(self) -> bool:
        """Subclasses override when overlays need a depth pass (task workspace, plan EE dot)."""
        return False

    def _post_render_hook(self, name, rgb, depth=None):
        """Saves camera views to the filesystem for diagnostic-verification (Mirrors original teleop logic)."""
        # Save in a subdirectory for this specific camera inside the session folder
        cam_dir = self.base_log_dir / name
        cam_dir.mkdir(parents=True, exist_ok=True)

        img_path = cam_dir / f"{self.frame_indices[name]:04d}.png"
        Image.fromarray(rgb).save(img_path)

    def get_state_32(self):
        return self.qpos_to_action_32(self.data.qpos)

    def qpos_to_action_32(self, qpos):
        """Converts a 76-dim qpos vector to a raw 32-dim protocol action/state (Radians)."""
        state = np.zeros(32, dtype=np.float32)
        for i, j_id in enumerate(self.protocol_joint_ids):
            if j_id != -1:
                q_idx = self.model.jnt_qposadr[j_id]
                state[i] = qpos[q_idx]
        return state

    def solve_ik(
        self,
        pos_wrist,
        quat,
        pos_index=None,
        pos_thumb=None,
        posture_target=None,
        posture_cost=None,
    ):
        """Hardened IK solver for a 3-tip end effector (Index, Thumb, Wrist)."""
        # 1. Convert input rotation (wxyz) to a NumPy array for mathematical operations
        quat = np.array(quat)
        # Validate that the quaternion has the correct length (MuJoCo/Mink expects 4: wxyz)
        if quat.shape[0] != 4:
            raise ValueError(f"solve_ik: quat must be length 4 (wxyz), got {len(quat)}")

        # 2. Convert target positions to NumPy arrays
        pos_wrist = np.array(pos_wrist)
        # If fingertip targets aren't provided, default them to a small offset above the wrist
        if pos_index is None:
            pos_index = pos_wrist + np.array([0, 0, 0.05])
        if pos_thumb is None:
            pos_thumb = pos_wrist + np.array([0, 0, 0.05])

        pos_index = np.array(pos_index)
        pos_thumb = np.array(pos_thumb)

        # 3. IK POSTURE ENFORCEMENT: Force restricted joints (like shoulder yaw) to specific waypoints
        # This keeps the arm 'stiff' and predictable during demonstrations
        q_start = self.data.qpos.copy()
        for j_name, target_val in IK_POSTURE_LOCKS.items():
            # Look up the MuJoCo joint ID by its XML name
            j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
            if j_id != -1:
                # Update the starting configuration with the locked value
                q_start[self.model.jnt_qposadr[j_id]] = target_val

        # Update the solver's internal configuration with the snapped starting pose
        self.configuration.update(q_start)

        # Convert the target quaternion into a Mink rotation object
        rotation = mink.SO3(quat)

        # 4. DYNAMIC POSTURE BIASING: Influence the solver toward a 'natural' or neutral pose
        if posture_target is not None:
            # If a specific target is provided, use it to guide the whole body
            self.tasks[3].set_target(posture_target)
        else:
            # Otherwise, use the model's default neutral pose (qpos0) as a reference
            self.tasks[3].set_target(self.model.qpos0)

        # Adjust the 'strength' of the postural bias (how hard the solver tries to stay neutral)
        if posture_cost is not None:
            self.tasks[3].cost = np.array([posture_cost])
        else:
            self.tasks[3].cost = np.array([1e-6])  # Use a very low cost by default

        # 5. CONSTRUCT SPATIAL TASKS: Map the XYZ + Rotation targets to the MuJoCo sites
        # These tell the solver WHERE and HOW oriented each tip should be
        self.tasks[0].set_target(
            mink.SE3.from_rotation_and_translation(rotation, pos_index)
        )
        self.tasks[1].set_target(
            mink.SE3.from_rotation_and_translation(rotation, pos_thumb)
        )
        self.tasks[2].set_target(
            mink.SE3.from_rotation_and_translation(rotation, pos_wrist)
        )

        # 6. ITERATIVE SOLVER LOOP: Refine the joint angles to minimize distance to targets
        for i in range(1500):
            # Calculate the velocity vector needed to move toward targets
            solver = mink.solve_ik(
                self.configuration,
                self.tasks,
                dt=0.15,  # Time step size for the solver
                solver="osqp",  # Quadratic programming solver backend
                limits=self.limits,  # Respect physical joint limits defined in XML
            )

            # Extract the current joint state to calculate the next step
            q_ref = self.configuration.q.copy()
            full_vel = solver.copy()

            # 7. VELOCITY FILTERING: Prevent movement in 'Unauthorized' or locked joints
            # This is the physical enforcement of your stiff-arm demonstrations
            for d in range(len(full_vel)):
                if d not in self.auth_dofs:
                    full_vel[d] = (
                        0.0  # Zero out the velocity for unauthorized/frozen degrees-of-freedom
                    )

            # Integrate the velocity over a small 'tick' to find new joint positions
            mujoco.mj_integratePos(self.model, q_ref, full_vel, 0.05)

            # Update the configuration state with the new calculated positions
            self.configuration.update(q_ref)

            # 8. CONVERGENCE CHECK: Stop early if the index finger is within 1cm of the target
            err = self.tasks[0].compute_error(self.configuration)
            if np.linalg.norm(err) < 0.01:
                break

        # Return the final calculated joint sequence (qpos)
        return self.configuration.q.copy()

    def sync_ctrl_to_qpos(self, q):
        for a_id in range(self.model.nu):
            j_id = self.model.actuator_trnid[a_id, 0]
            q_idx = self.model.jnt_qposadr[j_id]
            self.data.ctrl[a_id] = q[q_idx]

    def get_physics_state(self):
        """Calculates current ground-truth metrics for RA-BC conditioning and UI monitoring."""
        # 1. Extract Cube Position
        cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        cube_pos = np.zeros(3)
        if cube_id != -1:
            cube_pos = self.data.qpos[
                self.model.jnt_qposadr[cube_id] : self.model.jnt_qposadr[cube_id] + 3
            ].copy()

        # 2. Extract Hand Center (Midpoint of index and thumb tips)
        index_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "R_index_tip_link"
        )
        thumb_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "R_thumb_tip_link"
        )

        hand_pos = np.zeros(3)
        if index_id != -1 and thumb_id != -1:
            p_index = self.data.xpos[index_id]
            p_thumb = self.data.xpos[thumb_id]
            hand_pos = (p_index + p_thumb) / 2.0

        # 3. Calculate L2 Distance
        target_dist = float(np.linalg.norm(hand_pos - cube_pos))

        # 4. Phase-based Grasp Detection
        is_grasping = self.current_phase >= 3

        return {
            "cube_z": float(cube_pos[2]),
            "is_grasping": is_grasping,
            "target_dist": target_dist,
        }

    def render_and_record(self, action_32):
        """Captures camera views, logs to Rerun, and records to dataset if active."""
        views = {}
        rr.set_time("sim_step", sequence=self.render_step_idx)

        need_depth = self._render_needs_depth()
        for name in self.cam_names:
            self.renderer.update_scene(self.data, camera=name)
            rgb = self.renderer.render()
            depth = None
            if need_depth:
                self.renderer.enable_depth_rendering()
                depth = self.renderer.render().copy()
                self.renderer.disable_depth_rendering()
            self._post_render_hook(name, rgb, depth=depth)
            views[name] = rgb
            rr.log(name, rr.Image(rgb))
            self.frame_indices[name] += 1

        self.render_step_idx += 1
        self.rerun_count += 1
        if self.is_recording:
            # Extract ground-truth physics for RA-BC Progress Weighting
            physics = self.get_physics_state()
            self.recorder.add_frame(views, self.get_state_32(), action_32, physics)

    def dispatch_action(
        self, action_32_norm, target_q, n_steps=None, render_freq=None, reset_start=True
    ):
        # Backward Compatible Defaults
        total_steps = n_steps if n_steps is not None else 200
        rf = render_freq if render_freq is not None else 16

        # ✅ VLA FIX: Trajectory Threading
        # If reset_start is True (default), we interpolate from current actual physics pose.
        # If False, we preserve the last target to ensure smooth trajectory flow across chunks.
        if reset_start or not hasattr(self, "_last_interp_q"):
            self._last_interp_q = self.data.qpos.copy()

        start_q = self._last_interp_q

        root_target = target_q[self.root_q_idx : self.root_q_idx + 7]
        for step in range(total_steps):
            alpha = (step + 1) / float(total_steps)
            current_target_q = start_q + alpha * (target_q - start_q)
            self.sync_ctrl_to_qpos(current_target_q)
            self.data.qpos[self.root_q_idx : self.root_q_idx + 7] = root_target
            self.data.qvel[:6] = 0.0
            mujoco.mj_step(self.model, self.data)

            # [AUDIT:STAGE2] Physical Reality for R-Shoulder Roll
            if step == total_steps - 1:
                q_val = self.data.qpos[
                    self.model.jnt_qposadr[
                        mujoco.mj_name2id(
                            self.model,
                            mujoco.mjtObj.mjOBJ_JOINT,
                            "right_shoulder_roll_joint",
                        )
                    ]
                ]

            # Legacy Periodic Rendering (e.g., for VLA/Teleop)
            if rf > 0 and step % rf == 0:
                self.render_and_record(action_32_norm)

        # Ensure at least one render at the end if rf was too large
        if rf >= total_steps or rf == 0:
            self.render_and_record(action_32_norm)

        # Save target for next chunk (VLA Trajectory Threading)
        self._last_interp_q = target_q.copy()

    def reset_env(self, lock_posture=False, randomize_cube=True):
        """Resets the simulation with optional postural locking."""
        print(f"🎲 Resetting environment (Lock Posture: {lock_posture})...")
        self.current_phase = 0
        # Constrain cube randomization to table bounds (X: 0.25-0.65, Y: ±0.25)
        # Using [0.27, 0.63] and ±0.23 for a small safety margin from the edges
        if randomize_cube:
            rx, ry = np.random.uniform(0.27, 0.63), np.random.uniform(-0.23, 0.23)
        else:
            rx, ry = 0.5, 0
        print(f"rx: {rx}, ry: {ry}")
        cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        cube_q_idx = self.model.jnt_qposadr[cube_id]
        self.data.qpos[cube_q_idx : cube_q_idx + 3] = [rx, ry, 0.82]
        self.data.qpos[cube_q_idx + 3 : cube_q_idx + 7] = [1, 0, 0, 0]
        home_q = self.model.qpos0.copy()
        home_q[cube_q_idx : cube_q_idx + 7] = self.data.qpos[
            cube_q_idx : cube_q_idx + 7
        ].copy()
        for i, j_id in enumerate(self.protocol_joint_ids):
            if j_id != -1 and self.v_allowed_mask[i] > 0.5:
                # Protocols: Exempt frozen joints from randomization
                name = COMPACT_WIRE_JOINTS[i]
                if name in FROZEN_JOINTS:
                    home_q[self.model.jnt_qposadr[j_id]] = FROZEN_JOINTS[name]
                elif name in IK_POSTURE_LOCKS:
                    target = IK_POSTURE_LOCKS[name]
                    # Snap if locked, otherwise add small jitter for data diversity
                    home_q[self.model.jnt_qposadr[j_id]] = (
                        target
                        if lock_posture
                        else target + np.random.uniform(-0.1, 0.1)
                    )
                else:
                    # Always randomize other active joints (waist, etc.)
                    center = (self.wire_max[i] + self.wire_min[i]) / 2.0
                    home_q[self.model.jnt_qposadr[j_id]] = center + np.random.uniform(
                        -0.2, 0.2
                    )
        home_q[self.root_q_idx : self.root_q_idx + 3] = [0.0, 0.0, 0.95]

        # --- INSTANT HARD RESET ---
        # Teleport instantly instead of interpolating to avoid 'batting' the cube
        self.data.qpos[:] = home_q.copy()
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # Sync the interpolation state so the NEXT movement starts from here
        self._last_interp_q = self.data.qpos.copy()
        self.last_target_q = home_q.copy()

        # Render one frame to update the UI immediately
        self.render_and_record(None)

    def wild_reset(self):
        """High-variance joint randomization while PRESERVING current cube position."""
        print(f"🌀 WILD RANDOMIZING robot pose...")
        self.current_phase = 0

        # 1. Capture Current Cube Pose (Do not randomize)
        cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        cube_q_idx = self.model.jnt_qposadr[cube_id]
        current_cube_qpos = self.data.qpos[cube_q_idx : cube_q_idx + 7].copy()

        # 2. Full-Range Joint Randomization
        home_q = self.model.qpos0.copy()
        home_q[cube_q_idx : cube_q_idx + 7] = current_cube_qpos

        for i, j_id in enumerate(self.protocol_joint_ids):
            if j_id != -1:
                q_idx = self.model.jnt_qposadr[j_id]
                # SYNC with lewm_server.py: Freeze 0-15 (Left side + Head)
                if i < 16:
                    # PRESERVE PREVIOUS STATE for frozen joints
                    home_q[q_idx] = self.data.qpos[q_idx]
                else:
                    # EXPLORE FULL RANGE for indices 16-31 (Right side + Waist)
                    home_q[q_idx] = np.random.uniform(
                        self.wire_min[i], self.wire_max[i]
                    )

        # Keep base stable
        home_q[self.root_q_idx : self.root_q_idx + 3] = [0.0, 0.0, 0.95]

        # Instant Teleport
        self.data.qpos[:] = home_q.copy()
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._last_interp_q = self.data.qpos.copy()
        self.last_target_q = home_q.copy()

        self.render_and_record(None)

    def process_target_32(self, action_32_norm):
        """Receives the normalized action and updates the target qpos."""
        self.active_joints_this_command.clear()

        # Canonical Handshake: unscale incoming [-1, 1] to raw Radians
        action_32_rad = self.unscaler.unscale_action(action_32_norm)

        for i, val_norm in enumerate(action_32_norm):
            if (
                not np.isnan(val_norm)
                and self.v_allowed_mask[i] > 0
                and self.protocol_joint_ids[i] != -1
            ):
                self.active_joints_this_command.add(i)
                q_idx = self.model.jnt_qposadr[self.protocol_joint_ids[i]]

                # Update target in raw radians
                rad = float(action_32_rad[i])
                self.last_target_q[q_idx] = rad

                if i in self.coupling_map:
                    for distal_idx in self.coupling_map[i]:
                        self.last_target_q[distal_idx] = rad
