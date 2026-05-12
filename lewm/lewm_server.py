"""
ORACLE MPC INFERENCE SERVER (Gallery Edition)
Role: Standalone ZMQ server hosting the JEPA world model and CEM solver.
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


import zmq
import msgpack
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
        self, model_path, gallery_path="goal_gallery.pth", use_multi_view=False
    ):
        print(
            f"--- Initializing Oracle MPC Server (Gallery Only, Multi-View: {use_multi_view}) ---"
        )
        self.scaler = StandardScaler()
        self.initial_pose = None
        self.use_multi_view = use_multi_view

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
        self.agent = GoalMapper(
            model_path,
            dataset_root=".",
            use_multi_view=use_multi_view,
            num_views=5 if use_multi_view else 1,
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
            num_samples=800,
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

        # 5. State Buffering
        self.history = {"pixels": [], "actions": []}

    def run(self, host="0.0.0.0"):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://{host}:{PORT}")
        print(f"🚀 LEWM Server LISTENING on port {PORT}...")

        while True:
            try:
                message = socket.recv()
                req = msgpack.unpackb(message, raw=False)

                def unpack_np(d):
                    return np.frombuffer(d["data"], dtype=d["dtype"]).reshape(
                        d["shape"]
                    )

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
                        views.append(transformed["pixels"])

                    # Current frame is (V, C, H, W)
                    current_pixels = torch.stack(views, dim=0).to(DEVICE)
                else:
                    raw_image = unpack_np(req.get("observation.images.world_center"))
                    transformed = self.agent.transform({"pixels": raw_image})
                    # Current frame is (1, C, H, W) for single-view consistency
                    current_pixels = transformed["pixels"].unsqueeze(0).to(DEVICE)

                raw_sim_state = unpack_np(req.get("state"))

                # Grounding: Normalize the current state for history alignment
                norm_state = self.scaler.scale_state(raw_sim_state)

                # 🧊 DYNAMIC FREEZE ANCHOR: Capture initial pose on first step
                if self.initial_pose is None:
                    self.initial_pose = norm_state.copy()
                    self.agent.frozen_pose = torch.tensor(
                        self.initial_pose, device=DEVICE
                    ).float()
                    print(
                        f"🧊 Initial Pose Anchored: Left Shoulder Pitch = {self.initial_pose[0]:.4f}"
                    )

                self.history["pixels"].append(current_pixels)
                if len(self.history["pixels"]) > 3:
                    self.history["pixels"].pop(0)

                # Action History: Ground the model in the current normalized pose
                while len(self.history["actions"]) < 3:
                    self.history["actions"].append(norm_state)

                # pixels_stacked: (B=1, T=3, V, C, H, W)
                pixels_stacked = (
                    torch.stack(self.history["pixels"]).unsqueeze(0).to(DEVICE)
                )

                actions_stacked = (
                    torch.tensor(np.stack(self.history["actions"]), dtype=torch.float32)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(DEVICE)
                )

                print(
                    f"🧠 Step: Planning (8,000 parallel samples, Shape: {pixels_stacked.shape})..."
                )
                start_time = time.time()
                with torch.inference_mode():
                    # 🚀 WARM-START CEM: Pass previous action as initial guess
                    last_executed_action = actions_stacked[
                        :, :, -1:, :
                    ]  # (1, 1, 1, 32)

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
                    init_guess[:, 0:16] = self.agent.frozen_pose[0:16]
                    init_guess = init_guess.unsqueeze(0)

                    outputs = self.solver.solve(
                        {"pixels": pixels_stacked, "action": actions_stacked},
                        init_action=init_guess,
                    )

                best_plan = outputs["actions"].cpu().numpy()
                if best_plan.ndim == 4:
                    best_plan = best_plan[0, 0]  # (B, S, T, D) -> (T, D)
                elif best_plan.ndim == 3:
                    best_plan = best_plan[0]  # (S, T, D) -> (T, D)

                # 🧊 MANIFOLD ENFORCEMENT (0-15) 🧊
                # Explicitly zero out any leakage for the frozen joints in the full plan
                best_plan[:, 0:16] = self.initial_pose[0:16]

                target_action = best_plan[0]  # (32,)

                # 🛡️ THE GOVERNOR: Delta Capping & Smoothing 🛡️
                if len(self.history["actions"]) > 0:
                    prev_action = self.history["actions"][-1]

                    # 1. Delta Capping (Max 0.05 normalized units per step)
                    # We ONLY apply delta capping to the active joints (16-31)
                    max_delta = 0.05
                    delta = target_action[16:] - prev_action[16:]
                    clipped_delta = np.clip(delta, -max_delta, max_delta)
                    target_action[16:] = prev_action[16:] + clipped_delta

                    # 2. 🛡️ PRECISION MAPPING & PROTOCOL ENFORCEMENT 🛡️
                    arm_min = np.array([-0.312, -0.098, -0.156, -0.098])
                    arm_max = np.array([1.172, 0.098, 0.236, 0.098])

                    target_action[17:21] = (
                        arm_min
                        + (target_action[17:21] + 1.0) * (arm_max - arm_min) / 2.0
                    )

                    # Ensure indices 0-15 remain exactly at their initial pose
                    target_action[0:16] = self.initial_pose[0:16]

                    # Final safety clip for active joints
                    target_action[16:] = np.clip(target_action[16:], -1.0, 1.0)

                best_plan[0] = target_action

                plan_time = time.time() - start_time

                # Diagnostic Logging: Action Stats
                print(
                    f"🧠 Planning Stats -> Solve Time: {plan_time:.2f}s, "
                    f"Max Action: {np.abs(best_plan).max():.4f}, "
                    f"Mean Action: {np.abs(best_plan).mean():.4f}"
                )

                # --- 📡 Rerun Telemetry & Audit ---
                # For now, log the center view for diagnostics
                self.log_diagnostics(
                    raw_image=pixels_stacked[0, -1, 0].cpu().numpy().transpose(1, 2, 0),
                    best_plan=best_plan,
                    plan_time=plan_time,
                    instruction=req.get("instruction", "Unknown"),
                )

                # Update history with the first action of the plan (which is normalized)
                self.history["actions"].append(best_plan[0])
                if len(self.history["actions"]) > 3:
                    self.history["actions"].pop(0)

                socket.send(
                    msgpack.packb(
                        {
                            "action": best_plan.tolist(),
                            "diagnostics": {"plan_time_ms": int(plan_time * 1000)},
                        },
                        use_bin_type=True,
                    )
                )

            except Exception as e:
                print(f"❌ Server Error: {e}")
                traceback.print_exc()
                socket.send(msgpack.packb({"error": str(e)}, use_bin_type=True))

    def log_diagnostics(self, raw_image, best_plan, plan_time, instruction):
        """Pure JSONL logging for 'Wild Movement' debugging."""
        try:
            # Lifecycle Audit (JSONL)
            log_entry = {
                "timestamp": time.time(),
                "instruction": instruction,
                "solve_time": plan_time,
                "action_max": float(np.abs(best_plan).max()),
                "action_mean": float(np.abs(best_plan).mean()),
                "first_action_norm": best_plan[0].tolist(),
                "first_action_raw": self.scaler.unscale_action(best_plan[0]).tolist(),
            }
            log_file = os.path.join(ROOT_DIR, "lewm_lifecycle_audit.json")
            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        except Exception as e:
            print(f"⚠️ Diagnostic logging failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--gallery", type=str, default="goal_gallery.pth")
    parser.add_argument("--multi_view", action="store_true", default=False)
    args = parser.parse_args()
    server = LEWMInferenceServer(
        args.model, args.gallery, use_multi_view=args.multi_view
    )
    server.run()
