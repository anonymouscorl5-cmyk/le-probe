"""
UNIFIED PLANNING AGENT (The "Brain")
Role: Wrapper for the Oracle World Model and Planning Cost Logic.

This class serves as the primary interface for the CEM Solver. It:
1. Loads the v17 Oracle weights and maintains the JEPA model instance.
2. Manages the "Goal Memory" (encoding success frames once).
3. Implements high-performance windowed rollouts with VRAM de-duplication.
"""

# --- Path Stabilization ---
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# --------------------------


import torch
from pathlib import Path

# Project-specific imports
import stable_pretraining as spt
from jepa import JEPA
from module import ARPredictor
from gr1_modules import GR1Embedder, GR1MLP, MultiViewJEPA
from lewm.le_wm.utils import get_img_preprocessor
from lewm.goal_utils import get_goal_pixels, get_episode_video_path
from lewm.train_lewm import RewardPredictor
from lewm.multi_view_encoder import LateFusionEncoder


class GoalMapper:
    """
    Utility to map task success (last frame of dataset) to World Model latent embeddings.
    Used for Zero-Shot Goal-Conditioned MPC.
    """

    def __init__(
        self,
        model_path,
        dataset_root,
        img_size=224,
        use_multi_view=True,
        num_views=5,
        fusion_type="linear",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset_root = Path(dataset_root)
        self.img_size = img_size
        self.use_multi_view = use_multi_view
        self.num_views = num_views

        # 🎯 OFFICIAL TRAINING TRANSFORMS (Strict Parity)
        self.transform = get_img_preprocessor(
            source="pixels", target="pixels", img_size=img_size
        )

        # Initialize Model Architecture (Always Wrapped for Parity)
        backbone = spt.backbone.utils.vit_hf(
            "tiny", patch_size=14, image_size=224, pretrained=False
        )
        self.encoder = LateFusionEncoder(
            backbone, embed_dim=192, fusion=fusion_type, num_views=num_views
        )
        self.predictor = ARPredictor(
            num_frames=3,
            input_dim=192,
            hidden_dim=192,
            output_dim=192,
            depth=6,
            heads=16,
            mlp_dim=2048,
        )
        self.action_encoder = GR1Embedder(input_dim=32, emb_dim=192)
        self.projector = GR1MLP(input_dim=192, output_dim=192, hidden_dim=2048)
        self.projector_proj = GR1MLP(input_dim=192, output_dim=192, hidden_dim=2048)

        self.model = (
            MultiViewJEPA(
                encoder=self.encoder,
                predictor=self.predictor,
                action_encoder=self.action_encoder,
                projector=self.projector,
                pred_proj=self.projector_proj,
            )
            .to(self.device)
            .eval()
        )

        self.model.reward_head = (
            RewardPredictor(input_dim=192, hidden_dim=512).to(self.device).eval()
        )

        # 🧊 Dynamic Freeze Anchor (Initial Pose)
        self.frozen_pose = None

        # Load Weights
        print(f"🧠 Loading Oracle Weights: {Path(model_path).name}")
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        new_sd = {k.replace("model.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(new_sd, strict=False)

    @torch.no_grad()
    def set_goal(self, episode_idx=0):
        """
        Fetches the success state (last frame) from the dataset.
        In Multi-View mode, it loads all 5 camera views and stacks them.
        """
        if self.use_multi_view:
            cam_keys = [
                "observation.images.world_center",
                "observation.images.world_left",
                "observation.images.world_right",
                "observation.images.world_top",
                "observation.images.world_wrist",
            ]
            views = []
            for cam in cam_keys:
                video_path = get_episode_video_path(
                    self.dataset_root, episode_idx, camera_key=cam
                )
                pixels = get_goal_pixels(video_path)
                if pixels is None:
                    print(f"⚠️ Warning: Missing view {cam} for episode {episode_idx}")
                    return False

                # Transform each view individually
                # Transform expects a dict with 'pixels' as (C, H, W) or (H, W, C)
                # Goal Utils returns (C, H, W)
                transformed = self.transform({"pixels": pixels})["pixels"]
                views.append(transformed)

            # Stack to (V, C, H, W)
            processed_pixels = torch.stack(views, dim=0).to(self.device)
            # Add T=1 and B=1: (1, 1, V, C, H, W)
            processed_pixels = processed_pixels.unsqueeze(0).unsqueeze(0)
        else:
            video_path = get_episode_video_path(self.dataset_root, episode_idx)
            pixels = get_goal_pixels(video_path)

            if pixels is None:
                return False

            # Transform and Batch (Force 6D: B, T, V, C, H, W)
            batch = self.transform({"pixels": pixels})
            processed_pixels = batch["pixels"].to(self.device)
            # (C, H, W) -> (1, 1, 1, C, H, W)
            processed_pixels = processed_pixels.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # 2. Encode to Latent once
        info = self.model.encode({"pixels": processed_pixels})
        self.goal_latent = info["emb"].detach()  # (1, 1, D)

        print(
            f"✅ Goal Latent Cached: {self.goal_latent.shape} (Multi-View: {self.use_multi_view})"
        )
        return True

    def predict(self, *args, **kwargs):
        """Proxy to the internal World Model's prediction logic."""
        return self.model.predict(*args, **kwargs)

    def manifold_squash(self, actions):
        """
        PRECISION MANIFOLD MAPPING:
        1. Restricts sampling to Right Arm (16-22), Right Hand (23-28), and Waist (29-31).
        2. Freezes Left Arm/Hand (0-12) and Head (13-15) to -1.0.
        3. Applies buffered remapping to Right Arm joints (17-20).
        """
        # 1. Freeze Left Side and Head (Indices 0-15)
        # We set these to the frozen_pose (Initial Pose).
        # Error if not set, as we must never default to -1.0.
        if self.frozen_pose is None:
            raise ValueError(
                "❌ GoalMapper Error: frozen_pose not set before planning!"
            )

        actions[..., 0:16] = self.frozen_pose[..., 0:16]

        # 2. Global Safety Clamp for Active Joints
        actions = torch.clamp(actions, -1.0, 1.0)

        # 3. Right Arm Precision Mapping (Indices 17, 18, 19, 20)
        # Buffered Ranges (Orig Min/Max + 20% Range Buffer):
        arm_min = torch.tensor([-0.312, -0.098, -0.156, -0.098], device=actions.device)
        arm_max = torch.tensor([1.172, 0.098, 0.236, 0.098], device=actions.device)

        # Select the joints for all steps in the plan
        arm_joints = actions[..., 17:21]

        # Apply the linear transformation
        remapped_arm = arm_min + (arm_joints + 1.0) * (arm_max - arm_min) / 2.0

        # Re-insert into the action vector
        actions[..., 17:21] = remapped_arm

        return actions

    @torch.no_grad()
    def get_cost(self, obs_dict, actions):
        """
        FAST-PATH MPC COST (Ironclad 5D Protocol)
        Guarantees that the World Model only ever sees 5D (B, T, C, H, W).
        """
        # 1. Extract and Force 5D Observation
        raw_pixels = obs_dict["pixels"]
        B, S = actions.size(0), actions.size(1)

        # Force 6D Protocol: (Batch, T, V, C, H, W)
        pixels_input = raw_pixels
        target_ndim = 6

        while pixels_input.ndim > target_ndim:
            pixels_input = pixels_input[:, 0]  # Squeeze the Sample/S dimensions

        # 2. Optimized Encoding
        # All samples S share the same history. We encode the batch once.
        info = self.model.encode({"pixels": pixels_input})
        init_emb = info["emb"]  # (B, T, D)

        # Expand latents for the solver: (B, T, D) -> (B*S, T, D)
        curr_emb = init_emb.repeat_interleave(S, dim=0)

        # 3. History Actions Normalization
        raw_hist_actions = obs_dict.get("action", None)
        if raw_hist_actions is None:
            raw_hist_actions = torch.zeros(B, init_emb.size(1), actions.size(-1)).to(
                self.device
            )

        hist_actions_5d = raw_hist_actions
        while hist_actions_5d.ndim > 3:  # (B, S, T, D) -> (B, T, D)
            hist_actions_5d = hist_actions_5d[:, 0]

        # Flatten to BS space
        flat_hist_actions = (
            hist_actions_5d.repeat_interleave(S, dim=0).to(self.device).float()
        )

        # 4. Prepare Plan Actions (BS, T, D) with MANIFOLD SQUASHING
        raw_actions = actions.view(B * S, -1, actions.size(-1)).to(self.device).float()
        flat_plan_actions = self.manifold_squash(raw_actions)
        all_actions = torch.cat([flat_hist_actions, flat_plan_actions], dim=1)

        # 5. Sliding Window Rollout (Flattened BS space)
        history_size = init_emb.size(1)
        T_horizon = actions.size(2)

        # Track all predicted latents for dense costing (BS, T_horizon, D)
        pred_latents = []

        for t in range(T_horizon):
            emb_window = curr_emb[:, -history_size:, :]
            act_window = all_actions[:, t : t + history_size, :]

            act_emb = self.model.action_encoder(act_window)
            pred_emb = self.model.predict(emb_window, act_emb)

            last_pred = pred_emb[:, -1:, :]
            curr_emb = torch.cat([curr_emb, last_pred], dim=1)
            pred_latents.append(last_pred)

        # 6. Optimized Planning Cost Logic
        # Combine all predictions: (BS, T_horizon, D)
        all_preds = torch.cat(pred_latents, dim=1)

        # CHOICE: Use the Reward Head for task-specific optimization
        with torch.no_grad():
            # (BS, T_horizon, D) -> (BS, T_horizon, 1) -> (BS, T_horizon)
            # -----------------------------------------------------------------
            # 1. PRIMARY REWARD: Predicted Proximity to Cube (0.0 to 10.0)
            # -----------------------------------------------------------------
            # Scaled by 10.0 so the ~7.0 delta of an episode translates to
            # a massive cost reduction, effectively drowning out the smoothness penalty.
            reward_pred = self.model.reward_head(all_preds).squeeze(-1)
            reward_weight = 50.0
            dist = (10.0 - reward_pred) * reward_weight

            # -----------------------------------------------------------------
            # 2. GLOBAL COMPASS: Euclidean distance to Latent Goal Success
            # -----------------------------------------------------------------
            # Provides a steady gradient from any distance, matching the
            # 'Smart Reward' logic used during training.
            if self.goal_latent is not None:
                # all_preds: (BS, T, D), goal_latent: (N_goals, 1, D)
                # Compare every step in horizon to the final successful goal state
                # goal_target shape: (N_goals, D) -> (1, N_goals, D)
                goal_target = (
                    self.goal_latent.squeeze(1).unsqueeze(0).to(all_preds.dtype)
                )

                # all_preds shape: (BS, T, D)
                # We want dist from every step in T to every goal in N_goals
                # To do this efficiently, we flatten BS * T
                flat_preds = all_preds.reshape(-1, all_preds.size(-1))  # (BS*T, D)

                # dists shape: (BS*T, N_goals)
                dists_to_latents = torch.cdist(flat_preds, goal_target.squeeze(0))

                # Reshape back to (BS, T, N_goals), find min goal dist per step
                # Shape: (BS, T)
                min_latent_dist_per_step = (
                    dists_to_latents.view(all_preds.size(0), all_preds.size(1), -1)
                    .min(dim=-1)
                    .values
                )

                # Global Compass Weight: Reduce to 0.5 to let the Reward Head lead.
                dist = dist + min_latent_dist_per_step * 0.5

            # Reduce dist to (BS,) by averaging over horizon
            dist = dist.mean(dim=-1)

            # -----------------------------------------------------------------
            # 3. PHYSICAL GRACE: Smoothness Penalty (STABILIZED)
            # -----------------------------------------------------------------
            # a. Jitter from Last Real Pose (Start of plan)
            # flat_hist_actions: (BS, T_hist, D)
            last_real_action = flat_hist_actions[:, -1, :]
            jump_start = torch.mean(
                (flat_plan_actions[:, 0, :] - last_real_action) ** 2, dim=-1
            )

            # b. Delta within the Plan Horizon (BS, T-1, D)
            if T_horizon > 1:
                jitters = torch.mean(
                    (flat_plan_actions[:, 1:, :] - flat_plan_actions[:, :-1, :]) ** 2,
                    dim=-1,
                )
                jump_internal = torch.mean(jitters, dim=1)
            else:
                jump_internal = 0.0

            # Smoothness Weight (10.0): Reduced from 300.0 to allow the solver to move the arm.
            smoothness_weight = 10.0
            dist = dist + (jump_start + jump_internal) * smoothness_weight

        # 7. Unflatten back to (B, S) for the Solver
        return dist.view(B, S)
