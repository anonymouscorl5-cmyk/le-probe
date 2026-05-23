"""
ORACLE MPC INFERENCE SERVER (Gallery Edition)
Role: Standalone HTTP server hosting the JEPA world model and CEM solver.
Mandatory: Requires goal_gallery.pth
"""

# --- Path Stabilization ---
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# --------------------------


import sys
import os
from pathlib import Path


import torch
import numpy as np
import time
import argparse
import traceback
import json
from gymnasium.spaces import Box

# Local imports
from lewm.goal_mapper import GoalMapper
from stable_worldmodel.solver.cem import CEMSolver
from gr1_protocol import StandardScaler
from lewm.skeleton.skeletal_utils import reconstruct_4ch_frame

# Skeletal prior imports
import mujoco
from PIL import Image, ImageDraw
from gr1_config import SCENE_PATH, COMPACT_WIRE_JOINTS
from dataset.skeleton.projection_utils import (
    get_projection_matrix,
    project_point,
    is_allowed_action_chain,
)
from lewm.task_workspace import TaskWorkspaceMPCConstraint
from lewm.planning_constraints import (
    CEM_NUM_SAMPLES_DEFAULT,
    CEM_NUM_SAMPLES_HARD_ARM_GATE,
)
from inference_http import serve_http, unpack_np, _to_msgpack_safe

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(DEVICE)
PORT = 5555


class MockConfig:
    def __init__(self, horizon, init_var=1.0):
        self.horizon = horizon
        self.init_var = init_var
        self.action_block = 1


class MockSpace:
    def __init__(self, shape):
        self.shape = shape
        self.low = -1.0
        self.high = 1.0


class LEWMInferenceServer:
    def __init__(
        self,
        model_path,
        gallery_path="goal_gallery.pth",
        use_multi_view=False,
        use_skeleton=False,
        use_dino=False,
        use_task_workspace=False,
    ):
        print(
            f"--- Initializing Oracle MPC Server (Gallery Only, Multi-View: {use_multi_view}, "
            f"Skeleton: {use_skeleton}, DINO: {use_dino}, TaskWorkspace: {use_task_workspace}) ---"
        )
        self.use_task_workspace = use_task_workspace
        self.scaler = StandardScaler()
        self.initial_pose = None
        self.use_multi_view = use_multi_view
        self.use_skeleton = use_skeleton
        self.use_dino = use_dino

        gallery_file = Path(gallery_path)
        if not gallery_file.exists():
            print(f"❌ Error: Gallery not found at {gallery_file}")
            print("💡 Run 'python research/harvest_goals.py' first.")
            exit(1)

        # 1. Load the Universal Gallery
        print(f"💎 Loading Gallery: {gallery_file}")
        self.gallery = torch.load(gallery_file, map_location=DEVICE)
        print(f"✅ Success: {len(self.gallery['goals'])} goal latents ready.")

        # 2. Initialize Brain (Gallery doesn't need data root)
        cem_samples = (
            CEM_NUM_SAMPLES_DEFAULT
            if use_task_workspace
            else CEM_NUM_SAMPLES_HARD_ARM_GATE
        )
        self.agent = GoalMapper(
            model_path,
            dataset_root=".",
            use_multi_view=use_multi_view,
            num_views=5 if use_multi_view else 1,
            use_skeleton=use_skeleton,
            use_dino=use_dino,
            use_task_workspace=use_task_workspace,
        )

        # Initialize MuJoCo for server-side skeletal prior rendering
        if self.use_skeleton:
            print(
                f"🦴 [Skeletal Fusion] Initializing server-side MuJoCo scene: {SCENE_PATH}"
            )
            self.mj_model = mujoco.MjModel.from_xml_path(SCENE_PATH)
            self.mj_data = mujoco.MjData(self.mj_model)
            self.compact_wire_joints = COMPACT_WIRE_JOINTS
            self.idx_id = mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "R_index_tip_link"
            )
            self.thm_id = mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "R_thumb_tip_link"
            )

        # 3. Load Entire Gallery into Brain (Omni-Goal mode)
        goal_list = [
            self.gallery["goals"][eid] for eid in sorted(self.gallery["goals"].keys())
        ]
        self.agent.goal_latent = torch.stack(goal_list).to(DEVICE)
        print(f"🚀 Brain Prime: Loaded all {len(goal_list)} goals for Omni-MPC.")

        # 4. CEM Solver Hyperparameters (Graceful Multi-View)
        self.solver = CEMSolver(
            model=self.agent,
            num_samples=cem_samples,
            var_scale=0.3,
            n_steps=5,
            topk=100,
            device=DEVICE,
        )
        self.solver.configure(
            action_space=Box(low=-1.0, high=1.0, shape=(4, 32)),
            n_envs=1,
            config=MockConfig(horizon=4, init_var=0.1),
        )

        self.task_workspace = None
        if self.use_task_workspace:
            self.task_workspace = TaskWorkspaceMPCConstraint()
            p = self.task_workspace.poly
            print(
                "🌐 Task workspace gate ON (fixed hull, final plan step only) "
                f"— {len(p.corner_points)} corners, {p.H.shape[0]} facets, "
                f"ε={self.task_workspace.feasibility_eps:g}, "
                "FK=sim-sync (root z=0.95 + client cube_pos)"
            )
        else:
            print(
                f"🌐 Task workspace OFF — right-arm wire gate before LeWM "
                f"({cem_samples} CEM samples). Pass --task-workspace for EE hull."
            )

        # 5. State Buffering
        self.history = {"pixels": [], "actions": []}

        # 6. Input Audit Configuration
        self.audit_dir = os.path.join(ROOT_DIR, "temp_images", "inputs_audit")
        os.makedirs(self.audit_dir, exist_ok=True)
        self.step_counter = 0

    @staticmethod
    def _cube_xyz_from_request(req: dict) -> np.ndarray | None:
        """World-frame cube position from client (simulation_lewm sends cube_pos)."""
        raw = req.get("cube_pos")
        if raw is None:
            return None
        pos = unpack_np(raw)
        if pos is None or len(pos) < 3:
            return None
        return np.asarray(pos[:3], dtype=np.float64)

    def process_request(self, req: dict) -> dict:
        """Handle one planning request; returns action + diagnostics or error."""
        try:
            self.step_counter += 1

            # 0. Get state of robot + scene (cube for FK / gate alignment with sim)
            raw_sim_state = unpack_np(req.get("state"))
            cube_xyz = self._cube_xyz_from_request(req)

            # 1. Perception Unpacking
            if self.use_multi_view:
                cam_keys = [
                    "observation.images.world_center",
                    "observation.images.world_left",
                    "observation.images.world_right",
                    "observation.images.world_top",
                    "observation.images.world_wrist",
                ]
                views = []
                for k in cam_keys:
                    raw_img = unpack_np(req.get(k))
                    # Transform to (C, H, W)
                    transformed = self.agent.transform({"pixels": raw_img})
                    if self.use_skeleton:
                        # Get state of cube
                        raw_cube_pos = None
                        if "cube_pos" in req:
                            raw_cube_pos = unpack_np(req.get("cube_pos"))

                        # Render skeleton for this view on-the-fly!
                        skel_mask = self.render_skeleton_mask(
                            view_name=k.split(".")[-1],
                            raw_sim_state=raw_sim_state,
                            raw_cube_pos=raw_cube_pos,
                        )

                        # Concatenate normalized RGB with float skeleton mask
                        transformed_rgb = transformed["pixels"].to(DEVICE)
                        skel_tensor = (
                            torch.from_numpy(skel_mask).float().unsqueeze(0).to(DEVICE)
                            / 255.0
                        )
                        transformed["pixels"] = torch.cat(
                            [transformed_rgb, skel_tensor], dim=0
                        )

                        # Inputs Visual Audit Saving Hook
                        view_name = k.split(".")[-1]
                        rgb_vis = raw_img
                        skel_vis = skel_mask
                        skel_3ch = np.stack([skel_vis] * 3, axis=-1)
                        side_by_side = np.hstack([rgb_vis, skel_3ch])
                        audit_path = os.path.join(
                            self.audit_dir,
                            f"step_{self.step_counter:03d}_{view_name}.png",
                        )
                        Image.fromarray(side_by_side).save(audit_path)
                    views.append(transformed["pixels"])

                # Current frame is (V, C, H, W)
                current_pixels = torch.stack(views, dim=0).to(DEVICE)
            else:
                raw_image = unpack_np(req.get("observation.images.world_center"))
                transformed = self.agent.transform({"pixels": raw_image})
                # Current frame is (1, C, H, W) for single-view consistency
                current_pixels = transformed["pixels"].unsqueeze(0).to(DEVICE)

            print(
                f"📷 Received {5 if self.use_multi_view else 1}-View Frame. "
                f"current_pixels: {current_pixels.shape} ({current_pixels.dtype}) |"
                f" raw_sim_state: {raw_sim_state.shape} ({raw_sim_state.dtype})"
            )

            # Grounding: Normalize the current state for history alignment
            norm_state = self.scaler.scale_state(raw_sim_state)

            # 🧊 DYNAMIC FREEZE ANCHOR: Capture initial pose on first step
            if self.initial_pose is None:
                self.initial_pose = norm_state.copy()
                self.agent.frozen_pose = torch.tensor(
                    self.initial_pose, device=DEVICE
                ).float()
                print("🧊 Initial Pose Anchored.")

            # Pixels History (V, C, H, W)
            try:
                self.history["pixels"].pop(0)
            except IndexError:
                pass
            while len(self.history["pixels"]) < 3:
                self.history["pixels"].append(current_pixels.clone())

            # Action History: Pad to size 3 on step 0, then slide naturally at the end of the loop
            if not self.history["actions"]:
                while len(self.history["actions"]) < 3:
                    self.history["actions"].append(norm_state.copy())

            # pixels_stacked: (B=1, T_history=3, V, C, H, W)
            pixels_stacked = torch.stack(self.history["pixels"]).unsqueeze(0).to(DEVICE)
            # (B=1, S=1, T_history=3, V, C, H, W)
            pixels_stacked = pixels_stacked.unsqueeze(1)

            actions_stacked = (
                torch.tensor(np.stack(self.history["actions"]), dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(DEVICE)
            )

            print(
                f"📊 [Telemetry] Queue Context: Pixels History={len(self.history['pixels'])}, Actions History={len(self.history['actions'])} | "
                f"Pixels Stacked: {pixels_stacked.shape} ({pixels_stacked.dtype}) | "
                f"Actions Stacked: {actions_stacked.shape} ({actions_stacked.dtype})"
            )
            print(
                f"🧠 Step: Planning ({self.solver.num_samples} parallel samples, "
                f"Shape: {pixels_stacked.shape})..."
            )
            start_time = time.time()
            with torch.inference_mode():
                # 🚀 WARM-START CEM: Pass previous action as initial guess
                last_executed_action = actions_stacked[:, :, -1:, :]  # (1, 1, 1, 32)

                # (1, 1, 1, 32) -> (1, 32) -> (4, 32) -> (1, 4, 32)
                # Use .repeat() instead of .expand() to avoid memory aliasing errors in CEM
                init_guess = (
                    last_executed_action.squeeze(0)
                    .squeeze(0)
                    .repeat(4, 1)  # Updated to match horizon 4
                    .to(DEVICE)
                    .float()
                )

                # 🧊 FREEZE SAMPLING SPACE (0-15) 🧊
                # We ensure the solver starts with the joints frozen to their initial pose
                init_guess[..., 0:16] = self.agent.frozen_pose[0:16]
                init_guess = init_guess.unsqueeze(0)

                obs_dict = {
                    "pixels": pixels_stacked,
                    "action": actions_stacked,
                }
                if self.use_task_workspace:
                    self.task_workspace.set_baseline_from_wire32(
                        raw_sim_state, cube_xyz=cube_xyz
                    )
                    tw_H, tw_d = self.task_workspace.get_halfspaces()
                    obs_dict["task_workspace_H"] = tw_H.astype(np.float32)[
                        np.newaxis, ...
                    ]
                    obs_dict["task_workspace_d"] = tw_d.astype(np.float32)[
                        np.newaxis, ...
                    ]
                    obs_dict["task_workspace_wire32"] = raw_sim_state.astype(
                        np.float32
                    )[np.newaxis, ...]
                    obs_dict["task_workspace_check_final_only"] = np.array(
                        [[True]], dtype=np.bool_
                    )
                    if cube_xyz is not None:
                        obs_dict["task_workspace_cube_xyz"] = cube_xyz.astype(
                            np.float32
                        )[np.newaxis, np.newaxis, :]
                        print(
                            f"🌐 FK scene cube_xyz=({cube_xyz[0]:.3f}, {cube_xyz[1]:.3f}, "
                            f"{cube_xyz[2]:.3f})"
                        )
                if "phase_idx" in req:
                    p_idx = int(req.get("phase_idx"))
                    obs_dict["phase_idx"] = torch.tensor(
                        [[p_idx]], dtype=torch.long, device=DEVICE
                    )

                outputs = self.solver.solve(
                    obs_dict,
                    init_action=init_guess,
                )

            best_plan = outputs["actions"].cpu().numpy()
            if best_plan.ndim == 4:
                best_plan = best_plan[0, 0]  # (B, S, T, D) -> (T, D)
            elif best_plan.ndim == 3:
                best_plan = best_plan[0]  # (S, T, D) -> (T, D)

            # Freeze left + head; keep protocol [-1, 1] on active joints
            best_plan[:, 0:16] = self.initial_pose[0:16]
            best_plan[:, 16:] = np.clip(best_plan[:, 16:], -1.0, 1.0)
            diagnostics = {"plan_time_ms": 0}
            if self.use_task_workspace:
                fk_report = self.task_workspace.fk_debug_report(
                    raw_sim_state,
                    best_plan,
                    check_final_only=True,
                    cube_xyz=cube_xyz,
                )
                plan_final_ee = np.asarray(
                    fk_report["ee_full_chain_final_xyz"], dtype=np.float64
                )
                tw_viol = float(fk_report["violation_full_chain"])
                tw_feasible = bool(fk_report["feasible_full_chain"])
                print(
                    f"   🌐 Task workspace violation (final step): {tw_viol:.4f}, "
                    f"feasible={tw_feasible}, "
                    f"plan_final_ee=({plan_final_ee[0]:.3f}, {plan_final_ee[1]:.3f}, "
                    f"{plan_final_ee[2]:.3f})"
                )
                diagnostics["task_workspace_violation"] = tw_viol
                diagnostics["task_workspace_feasible"] = tw_feasible
                diagnostics["plan_final_ee_xyz"] = [
                    float(plan_final_ee[0]),
                    float(plan_final_ee[1]),
                    float(plan_final_ee[2]),
                ]
                diagnostics["fk_debug"] = fk_report
                diagnostics["request_wire32_rad"] = raw_sim_state.astype(float).tolist()
                diagnostics["sim_scene_sync"] = True
                if cube_xyz is not None:
                    diagnostics["scene_cube_xyz"] = cube_xyz.astype(float).tolist()

            plan_time = time.time() - start_time
            diagnostics["plan_time_ms"] = int(plan_time * 1000)

            # Diagnostic Logging: Action Stats
            print(
                f"🧠 Planning Stats -> Solve Time: {plan_time:.2f}s, "
                f"Max Action: {np.abs(best_plan).max():.4f}, "
                f"Mean Action: {np.abs(best_plan).mean():.4f}"
            )

            # --- 📡 Rerun Telemetry & Audit ---
            # For now, log the center view for diagnostics
            self.log_diagnostics(
                raw_image=(
                    pixels_stacked[0, 0, -1, 0, :3].cpu().numpy().transpose(1, 2, 0)
                ),
                best_plan=best_plan,
                plan_time=plan_time,
                instruction=req.get("instruction", "Unknown"),
                diagnostics=diagnostics,
            )

            # Plan/run 4-4: commit each step of the executed horizon (3-frame action queue).
            for t in range(best_plan.shape[0]):
                self.history["actions"].append(best_plan[t])
                if len(self.history["actions"]) > 3:
                    self.history["actions"].pop(0)

            return {"action": best_plan.tolist(), "diagnostics": diagnostics}

        except Exception as e:
            print(f"❌ Server Error: {e}")
            traceback.print_exc()
            return {"error": str(e)}

    def run(self, host="0.0.0.0", port: int | None = None):
        serve_http(self.process_request, host=host, port=port or PORT)

    def log_diagnostics(
        self, raw_image, best_plan, plan_time, instruction, diagnostics
    ):
        """Pure JSONL logging for 'Wild Movement' debugging."""
        try:
            # Lifecycle Audit (JSONL)
            log_entry = {
                "timestamp": time.time(),
                "instruction": instruction,
                "solve_time": plan_time,
                "action_max": float(np.abs(best_plan).max()),
                "action_mean": float(np.abs(best_plan).mean()),
                "best_plan_norm": best_plan.tolist(),
                "best_plan_raw": [
                    self.scaler.unscale_action(plan).tolist() for plan in best_plan
                ],
                **diagnostics,
            }
            log_file = os.path.join(ROOT_DIR, "lewm_lifecycle_audit.json")
            with open(log_file, "a") as f:
                f.write(json.dumps(_to_msgpack_safe(log_entry)) + "\n")

        except Exception as e:
            print(f"⚠️ Diagnostic logging failed: {e}")

    def render_skeleton_mask(self, view_name, raw_sim_state, raw_cube_pos=None):
        """Generates dynamic 1-channel skeletal prior on the server."""
        # 1. Update the MuJoCo model to the current proprioceptive state
        self.mj_data.qpos[:] = self.mj_model.qpos0
        root_adr = self.mj_model.jnt_qposadr[
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        ]
        self.mj_data.qpos[root_adr : root_adr + 3] = [0.0, 0.0, 0.95]

        for j, n in enumerate(self.compact_wire_joints):
            j_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, n)
            if j_id != -1:
                self.mj_data.qpos[self.mj_model.jnt_qposadr[j_id]] = raw_sim_state[j]

        mujoco.mj_forward(self.mj_model, self.mj_data)

        # 2. Get camera matrices for projection (resized to 224x224)
        cam_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, view_name)
        K = get_projection_matrix(cam_id, self.mj_model, 224, 224)
        t_cam = self.mj_data.cam_xpos[cam_id]
        R_cam = self.mj_data.cam_xmat[cam_id].reshape(3, 3) @ np.array(
            [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
        )

        # 3. Draw robot joints onto mask
        mask = Image.new("L", (224, 224), 0)
        draw = ImageDraw.Draw(mask)

        for b_id in range(1, self.mj_model.nbody):
            p_id = self.mj_model.body_parentid[b_id]
            if is_allowed_action_chain(b_id, self.mj_model) and is_allowed_action_chain(
                p_id, self.mj_model
            ):
                ps, _ = project_point(self.mj_data.xpos[b_id], K, R_cam, t_cam)
                pp, _ = project_point(self.mj_data.xpos[p_id], K, R_cam, t_cam)
                if ps is not None and pp is not None:
                    draw.line([tuple(ps), tuple(pp)], fill=255, width=1)

        # 4. Draw cube wireframe if cube_pos is provided (snapped to hand if within grasping range)
        if raw_cube_pos is not None:
            # Grasp Snapping Hysteresis:
            gripper_mid = (
                self.mj_data.xpos[self.idx_id] + self.mj_data.xpos[self.thm_id]
            ) / 2.0
            if np.linalg.norm(gripper_mid - raw_cube_pos) < 0.05:
                cube_pos = gripper_mid
            else:
                cube_pos = raw_cube_pos

            size = 0.02
            corners = (
                np.array(
                    [
                        [-1, -1, -1],
                        [1, -1, -1],
                        [1, 1, -1],
                        [-1, 1, -1],
                        [-1, -1, 1],
                        [1, -1, 1],
                        [1, 1, 1],
                        [-1, 1, 1],
                    ]
                )
                * size
            ) + cube_pos

            edges = [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),  # Bottom
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 4),  # Top
                (0, 4),
                (1, 5),
                (2, 6),
                (3, 7),  # Verticals
            ]
            for s_idx, e_idx in edges:
                ps, _ = project_point(corners[s_idx], K, R_cam, t_cam)
                pe, _ = project_point(corners[e_idx], K, R_cam, t_cam)
                if ps is not None and pe is not None:
                    draw.line([tuple(ps), tuple(pe)], fill=255, width=1)

        return np.array(mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", "-p", type=int, default=PORT, help="HTTP listen port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="HTTP bind address")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--gallery", type=str, default="goal_gallery.pth")
    parser.add_argument("--multi_view", action="store_true", default=False)
    parser.add_argument("--use_skeleton", action="store_true", default=False)
    parser.add_argument("--use_dino", action="store_true", default=False)
    parser.add_argument(
        "--task_workspace",
        action="store_true",
        help="Enable fixed task workspace gate on CEM final-step EE",
    )
    args = parser.parse_args()
    server = LEWMInferenceServer(
        args.model,
        args.gallery,
        use_multi_view=args.multi_view,
        use_skeleton=args.use_skeleton,
        use_dino=args.use_dino,
        use_task_workspace=args.task_workspace,
    )
    server.run(host=args.host, port=args.port)
