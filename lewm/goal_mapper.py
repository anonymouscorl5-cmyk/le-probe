# --- Path Stabilization ---
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# --------------------------

import torch
import torch.nn.functional as F
from stable_pretraining import data as dt
from pathlib import Path

try:
    import torchcodec

    HAS_TORCHCODEC = True
except ImportError:
    HAS_TORCHCODEC = False

# Local imports
from lewm.le_wm.jepa import JEPA
from lewm.le_wm.module import ARPredictor
from lewm.skeleton.skeletal_utils import reconstruct_4ch_frame
from lewm.gr1_modules import MultiViewJEPA, GR1Embedder, GR1MLP
from lewm.train_lewm import RewardPredictor
from lewm.multi_view_encoder import get_multi_view_encoder
from lewm.skeleton.encoder import patch_vit_for_skeleton
from lewm.task_workspace import TaskWorkspaceMPCConstraint, INFEASIBLE_COST
from omegaconf import OmegaConf
import numpy as np


class GoalMapper:
    """
    GoalMapper: The Brain Wrapper for LeWM.
    Handles goal encoding, state prediction, and cost calculation for CEM planning.
    """

    def __init__(
        self,
        model_path,
        dataset_root=".",
        use_multi_view=False,
        num_views=1,
        use_skeleton=False,
        use_dino=False,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset_root = Path(dataset_root) if dataset_root is not None else None
        self.use_multi_view = use_multi_view
        self.num_views = num_views
        self.use_skeleton = use_skeleton
        self.use_dino = use_dino

        # 1. Initialize the Model
        if str(model_path).endswith(".pt") or str(model_path).endswith(".ckpt"):
            print(f"🧠 Loading Model from Checkpoint: {model_path}")
            raw_data = torch.load(model_path, map_location=self.device)

            # If it's a dict, we need to instantiate the architecture
            if isinstance(raw_data, dict):
                print(
                    "🧬 Checkpoint detected as Dict. Instantiating MultiViewJEPA backbone..."
                )
                cfg = OmegaConf.create(
                    {
                        "backbone": "vit_tiny_patch14_224",
                        "use_multi_view": use_multi_view,
                        "num_views": num_views,
                        "img_size": 224,
                        "fusion_type": "linear",
                        "encoder_scale": "tiny",
                        "patch_size": 14,
                        "wm": {"history_size": 3, "action_dim": 32},
                        "predictor": {
                            "depth": 6,
                            "heads": 16,
                            "mlp_dim": 2048,
                            "dim_head": 64,
                        },
                    }
                )
                encoder = get_multi_view_encoder(cfg)
                if use_skeleton:
                    patch_vit_for_skeleton(encoder.backbone)

                hidden_dim = encoder.config.hidden_size
                embed_dim = 192  # Standard LeWM embedding dim

                self.model = MultiViewJEPA(
                    encoder=encoder,
                    predictor=ARPredictor(
                        num_frames=cfg.wm.history_size,
                        input_dim=embed_dim,
                        hidden_dim=hidden_dim,
                        output_dim=hidden_dim,
                        **cfg.predictor,
                    ),
                    action_encoder=GR1Embedder(input_dim=32, emb_dim=embed_dim),
                    projector=GR1MLP(
                        input_dim=hidden_dim, output_dim=embed_dim, hidden_dim=2048
                    ),
                    pred_proj=GR1MLP(
                        input_dim=hidden_dim, output_dim=embed_dim, hidden_dim=2048
                    ),
                )
                self.model.reward_head = RewardPredictor(
                    input_dim=embed_dim, hidden_dim=512
                )

                # Load weights (handles model. prefix)
                sd = raw_data.get("state_dict", raw_data)
                # Strip model. prefix and filter out training-only sigreg keys
                clean_sd = {}
                for k, v in sd.items():
                    clean_key = k.replace("model.", "") if k.startswith("model.") else k
                    if not clean_key.startswith("sigreg."):
                        clean_sd[clean_key] = v
                self.model.load_state_dict(clean_sd, strict=True)
            else:
                self.model = raw_data
        else:
            raise ValueError(f"Unsupported model path: {model_path}")

        self.model.to(self.device).eval()
        self.goal_latent = None
        self.frozen_pose = None

        self._task_ws = TaskWorkspaceMPCConstraint()

        # 2. Image Transform (Standard LeWM 224x224)
        imagenet_stats = dt.dataset_stats.ImageNet
        self.transform = dt.transforms.Compose(
            dt.transforms.ToImage(**imagenet_stats, source="pixels", target="pixels"),
            dt.transforms.Resize(224, source="pixels", target="pixels"),
        )

        # print(
        #     "✅ GoalMapper initialized "
        #     f"(Skeleton: {use_skeleton}, DINO: {self.use_dino})"
        # )
        print(f"✅ GoalMapper initialized (Skeleton: {use_skeleton})")

    def set_goal(self, episode_idx, frame_idx):
        """Encodes a specific frame from the dataset as the target goal."""
        # 1. Identify Goal Source
        # We prefer tiled videos if use_skeleton is True
        image_key = "world_center"
        if self.use_skeleton:
            image_key = "world_center_tiled"

        video_path = (
            self.dataset_root
            / "videos"
            / image_key
            / "chunk-000"
            / f"file-{episode_idx:03d}.mp4"
        )

        if not video_path.exists():
            # Fallback to non-tiled if tiled is missing
            video_path = (
                self.dataset_root
                / "videos"
                / "world_center"
                / "chunk-000"
                / f"file-{episode_idx:03d}.mp4"
            )

        if not video_path.exists():
            raise FileNotFoundError(f"🚨 Goal video not found: {video_path}")

        # 2. Decode Goal Frame
        if not HAS_TORCHCODEC:
            raise RuntimeError("torchcodec is required for GoalMapper goal loading.")

        decoder = torchcodec.decoders.VideoDecoder(str(video_path))
        frames = decoder.get_frames_at(indices=[frame_idx])
        raw_frame = frames.data[0]  # (C, H, W)

        # 3. Reconstruct & Transform
        if self.use_skeleton and "_tiled" in video_path.name:
            full_frame = reconstruct_4ch_frame(raw_frame, transform_fn=self.transform)
        else:
            # Fallback to RGB-only or already 4ch (unlikely from raw video)
            full_frame = self.transform({"pixels": raw_frame})["pixels"]
            if self.use_skeleton and full_frame.shape[0] == 3:
                # Add empty skeleton if model expects 4 channels
                skel = torch.zeros((1, 224, 224), device=full_frame.device)
                full_frame = torch.cat([full_frame, skel], dim=0)

        # 4. Encode Goal Latent
        # Input to model.encode expects (B, T, V, C, H, W)
        # We simulate a single frame: (1, 1, 1, C, H, W)
        with torch.no_grad():
            pixels_batch = full_frame.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            if self.use_multi_view:
                # Repeat for all views if the model expects multi-view
                pixels_batch = pixels_batch.repeat(1, 1, self.num_views, 1, 1, 1)

            output = self.model.encode({"pixels": pixels_batch.to(self.device)})
            self.goal_latent = output["emb"].detach()

        print(
            f"💎 Goal Set: Episode {episode_idx}, Frame {frame_idx} (Latent Shape: {self.goal_latent.shape})"
        )

    def encode_goal_from_pixels(self, pixels, skeleton=None):
        """
        Directly encodes a goal from a pre-processed pixel tensor.
        Input 'pixels' should be (C, H, W) or (V, C, H, W).
        If 'skeleton' is provided, it is fused into the 4th channel.
        """
        with torch.no_grad():
            # 1. Handle Skeletal Fusion
            if self.use_skeleton and skeleton is not None:
                # pixels: (3, H, W), skeleton: (1, H, W) -> (4, H, W)
                if pixels.ndim == 3:
                    pixels = torch.cat([pixels, skeleton], dim=0)
                elif pixels.ndim == 4:
                    # (V, 3, H, W) + (V, 1, H, W) -> (V, 4, H, W)
                    pixels = torch.cat([pixels, skeleton], dim=1)

            # 2. Batching
            if pixels.ndim == 3:
                # (C, H, W) -> (1, 1, 1, C, H, W)
                pixels_batch = pixels.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                if self.use_multi_view:
                    pixels_batch = pixels_batch.repeat(1, 1, self.num_views, 1, 1, 1)
            elif pixels.ndim == 4:
                # (V, C, H, W) -> (1, 1, V, C, H, W)
                pixels_batch = pixels.unsqueeze(0).unsqueeze(0)
            else:
                raise ValueError(f"Invalid pixels shape: {pixels.shape}")

            # 3. Encode
            output = self.model.encode({"pixels": pixels_batch.to(self.device)})
            self.goal_latent = output["emb"].detach()
            return self.goal_latent

    @torch.no_grad()
    def get_cost(self, obs_dict, actions):
        """
        EVIDENCE-BASED COST CALCULATION (Transparent Adapter)
        Flattens (B, S) into a single dimension to satisfy the 6D Library requirement.
        """
        # 1. Extract and Force 7D Observation (B, S, T_history, V, C, H, W)
        B, S = actions.size(0), actions.size(1)
        pixels_input = obs_dict["pixels"].to(self.device)
        target_ndim = 7

        if pixels_input.ndim > target_ndim:
            # (B, S, 1, T_history, V, C, H, W) -> (B, S, T_history, V, C, H, W)
            pixels_input = pixels_input.squeeze(2)
        if actions.ndim > 4:
            # (1, B, S, T_history, 32) -> (B, S, T_history, 32)
            actions = actions.squeeze(0)

        # 2. Optimized Encoding
        # All samples S share the same history. We encode the batch once.
        info = self.model.encode({"pixels": pixels_input[:, 0]})
        init_emb = info["emb"]  # (B, T_history, 192)
        curr_emb = init_emb.repeat_interleave(S, dim=0)  # (B * S, T_history, 192)

        # 3. History Actions Normalization
        hist_actions = obs_dict.get("action", None)  # (B, S, T_history, 32)

        if hist_actions is None:
            hist_actions = torch.zeros(B, S, init_emb.size(1), actions.size(-1)).to(
                self.device
            )
        if hist_actions.ndim > 4:
            # (B, S, 1, T_history, 32) -> (B, S, T_history, 32)
            hist_actions = hist_actions.squeeze(2)

        # (B, S, T_history, 32) -> (B * S, T_history, 32)
        flat_hist_actions = hist_actions[:, 0].repeat_interleave(S, dim=0)

        # 4. Prepare Plan Actions (B * S, T, D) with MANIFOLD SQUASHING
        # flat_hist_actions: (B * S, T_history, 32), flat_plan_actions: (B * S, T_horizon, 32)
        # all_actions: (B * S, T, 32)
        flat_plan_actions = self.manifold_squash(
            actions.view(B * S, -1, actions.size(-1))
        )
        all_actions = torch.cat([flat_hist_actions, flat_plan_actions], dim=1)

        # 5. Sliding Window Rollout (Flattened BS space)
        pred_latents = []
        T_history = flat_hist_actions.size(1)
        T_horizon = flat_plan_actions.size(1)
        for T in range(T_horizon):
            emb_window = curr_emb[:, -T_history:, :]  # (B, T_history, 192)
            act_window = all_actions[:, T : T + T_history, :]  # (B * S, T_history, 32)
            act_emb = self.model.action_encoder(act_window)  # (B * S, T_history, 192)
            pred_emb = self.model.predict(
                emb_window, act_emb
            )  # (B * S, T_history, 192)
            last_pred = pred_emb[:, -1:, :]  # (B * S, 1, 192)
            curr_emb = torch.cat(
                [curr_emb, last_pred], dim=1
            )  # (B * S, curr_emb.size(1) + 1, 192)
            pred_latents.append(last_pred)

        # 6. Optimized Planning Cost Logic
        all_preds = torch.cat(pred_latents, dim=1)  # (B * S, T_horizon, 192)

        # 7. Smart Cost Calculation
        # (B * S, T_horizon)
        reward_pred = self.model.reward_head(all_preds).squeeze(-1)
        reward_weight = 50.0
        dist = (10.0 - reward_pred) * reward_weight  # (B * S, T_horizon)

        if self.use_dino:
            # Hierarchical Macro Subgoal Planning Cost
            # A. Extract current phase index
            phase_idx = obs_dict.get("phase_idx", None)
            if phase_idx is None:
                # Default to Phase 0
                phase_idx = torch.zeros((B, 1), device=self.device)
            else:
                if not isinstance(phase_idx, torch.Tensor):
                    phase_idx = torch.tensor(phase_idx, device=self.device)

                # Flatten first
                phase_idx = phase_idx.flatten()

                # Bulletproof slice/repeat to exactly match B
                if phase_idx.numel() == B * S:
                    # CEM solver repeats observations using repeat_interleave(S, dim=0)
                    phase_idx = phase_idx[::S].view(B, 1).to(self.device)
                elif phase_idx.numel() >= B:
                    phase_idx = phase_idx[:B].view(B, 1).to(self.device)
                else:
                    phase_idx = phase_idx[0].repeat(B).view(B, 1).to(self.device)

            # B. Query High-Level Predictor for Macro Subgoal Coordinate
            curr_state = init_emb[:, -1, :]  # (B, 192)
            subgoal = self.model.predict_subgoal(curr_state, phase_idx)  # (B, 192)
            subgoal_target = subgoal.repeat_interleave(S, dim=0)  # (B * S, 192)

            # C. Euclidean distance between rollout states and the subgoal
            # Final-step distance (Targeting the checkpoint bottleneck at step H)
            final_dist = torch.norm(
                all_preds[:, -1, :] - subgoal_target, p=2, dim=-1
            )  # (B * S,)

            # Step-by-step distance (Ensuring smooth semantic progress)
            step_dists = torch.norm(
                all_preds - subgoal_target.unsqueeze(1), p=2, dim=-1
            )  # (B * S, T_horizon)

            # Combine costs (B * S, T_horizon)
            # We map final_dist across the sequence dim to match the base dist structure
            subgoal_cost = final_dist.unsqueeze(-1) + 0.1 * step_dists
            dist = dist + subgoal_cost * 1.0  # Weight the visual subgoal guidance
            dist = dist.mean(dim=-1)  # (B * S,)
        else:
            # Standard Flat Goal Planning Cost
            # repeat (current B * S) // (number of unique goals) times.
            num_goals = self.goal_latent.view(-1, self.goal_latent.size(-1)).size(0)
            goal_target = self.goal_latent.view(num_goals, 1, -1).to(all_preds.dtype)

            expansion_factor = (B * S) // num_goals
            if expansion_factor > 1:
                goal_target = goal_target.repeat_interleave(expansion_factor, dim=0)
            # (B * S, 1, 192) or (N_goals, 1, 192) if no expansion needed

            # (B * S, T_horizon)
            dists_to_latents = torch.cdist(all_preds, goal_target).squeeze(-1)
            # (B * S, 1)
            min_latent_dist_per_step = dists_to_latents.min(dim=-1).values.unsqueeze(-1)

            # Global Compass Weight: Increase to 0.5 to pull the robot out of pose-saturation.
            dist = dist + min_latent_dist_per_step * 0.5  # (B * S, T_horizon)
            dist = dist.mean(dim=-1)  # (B * S,)

        # 8. Smoothness Penalty
        last_real_action = flat_hist_actions[:, -1, :]  # (B * S, 32)
        jump_start = torch.mean(
            (flat_plan_actions[:, 0, :] - last_real_action) ** 2, dim=-1
        )  # (B * S,)

        # Delta within the Plan Horizon (B * S, T_horizon - 1, D)
        if T_horizon > 1:
            jitters = torch.mean(
                (flat_plan_actions[:, 1:, :] - flat_plan_actions[:, :-1, :]) ** 2,
                dim=-1,
            )  # (B * S, T_horizon - 1)
            jump_internal = torch.mean(jitters, dim=1)
        else:
            jump_internal = 0.0

        smoothness_weight = 100.0
        dist = dist + (jump_start + jump_internal) * smoothness_weight  # (B,)

        dist = self._apply_task_workspace_gate(obs_dict, flat_plan_actions, dist)

        return dist.view(B, S)

    def predict(self, *args, **kwargs):
        """Proxy to the internal World Model's prediction logic."""
        return self.model.predict(*args, **kwargs)

    def _apply_task_workspace_gate(
        self, obs_dict, flat_plan_actions, dist: torch.Tensor
    ):
        """Only samples with final-step EE inside fixed task hull compete on reward."""
        tw_H = obs_dict.get("task_workspace_H")
        tw_d = obs_dict.get("task_workspace_d")
        tw_wire32 = obs_dict.get("task_workspace_wire32")
        tw_cube = obs_dict.get("task_workspace_cube_xyz")
        if tw_H is None or tw_d is None or tw_wire32 is None:
            return dist
        cube_xyz = None
        if tw_cube is not None:
            if isinstance(tw_cube, torch.Tensor):
                cube_xyz = tw_cube[0, 0].detach().cpu().numpy().reshape(3)
            else:
                cube_xyz = np.asarray(tw_cube[0, 0], dtype=np.float64).reshape(3)

        if isinstance(tw_H, torch.Tensor):
            H = tw_H[0, 0].detach().cpu().numpy()
            d = tw_d[0, 0].detach().cpu().numpy().reshape(-1)
            wire32 = tw_wire32[0, 0].detach().cpu().numpy()
        else:
            H = np.asarray(tw_H[0, 0], dtype=np.float64)
            d = np.asarray(tw_d[0, 0], dtype=np.float64).reshape(-1)
            wire32 = np.asarray(tw_wire32[0, 0], dtype=np.float64)

        plan_np = flat_plan_actions.detach().cpu().numpy()
        eps = self._task_ws.feasibility_eps
        final_only = obs_dict.get("task_workspace_check_final_only", True)
        if isinstance(final_only, torch.Tensor):
            final_only = bool(final_only.reshape(-1)[0].item())
        elif isinstance(final_only, np.ndarray):
            final_only = bool(np.asarray(final_only).reshape(-1)[0])
        else:
            final_only = bool(final_only)

        n_total = int(plan_np.shape[0])
        print(
            f"[FK_DEBUG/gate] obs shapes H={np.asarray(tw_H).shape} "
            f"d={np.asarray(tw_d).shape} wire32={np.asarray(tw_wire32).shape} "
            f"plans={plan_np.shape} final_only={final_only} eps={eps:g}"
        )

        feasible, violations = self._task_ws.feasible_mask_batch(
            wire32,
            plan_np,
            check_all_steps=not final_only,
            cube_xyz=cube_xyz,
        )
        n_feas = int(feasible.sum())
        viol = violations.astype(np.float64)
        print(
            f"[FK_DEBUG/gate] feasible={n_feas}/{n_total} "
            f"viol min={viol.min():.6f} med={np.median(viol):.6f} max={viol.max():.6f}"
        )

        # CEM lane 0 (includes mean candidate): FK audit for winner semantics
        if n_total > 0:
            lane0 = self._task_ws.fk_debug_report(
                wire32,
                plan_np[0],
                check_final_only=final_only,
                cube_xyz=cube_xyz,
            )
            self._task_ws.log_fk_debug_report(lane0, prefix="[FK_DEBUG/gate/lane0]")

        # Lowest-violation sample (closest to hull boundary from inside)
        best_vi_idx = int(np.argmin(viol))
        if best_vi_idx != 0 and n_total > 1:
            lane_best_v = self._task_ws.fk_debug_report(
                wire32,
                plan_np[best_vi_idx],
                check_final_only=final_only,
                cube_xyz=cube_xyz,
            )
            self._task_ws.log_fk_debug_report(
                lane_best_v, prefix=f"[FK_DEBUG/gate/lane_min_viol={best_vi_idx}]"
            )

        if n_feas == 0:
            relaxed = violations <= eps * 100.0
            if int(relaxed.sum()) > 0:
                print(
                    f"⚠️ Task workspace gate: 0/{n_total} feasible at ε={eps:g}; "
                    f"using relaxed ε={eps * 100:g} ({int(relaxed.sum())} samples)"
                )
                feasible = relaxed
            else:
                print(
                    f"⚠️ Task workspace gate: 0/{n_total} feasible even at relaxed ε; "
                    "skipping gate for this cost eval"
                )
                return dist

        # Among reward-feasible elites, log FK for lowest-cost feasible sample
        if n_feas > 0:
            feas_idx = np.where(feasible)[0]
            dist_np = dist.detach().cpu().numpy().reshape(-1)
            best_cost_idx = int(feas_idx[np.argmin(dist_np[feas_idx])])
            if best_cost_idx not in (0, best_vi_idx):
                lane_best_c = self._task_ws.fk_debug_report(
                    wire32,
                    plan_np[best_cost_idx],
                    check_final_only=final_only,
                    cube_xyz=cube_xyz,
                )
                self._task_ws.log_fk_debug_report(
                    lane_best_c,
                    prefix=f"[FK_DEBUG/gate/lane_min_cost_feasible={best_cost_idx}]",
                )

        feasible_t = torch.from_numpy(feasible).to(device=dist.device, dtype=torch.bool)
        infeasible_cost = torch.tensor(
            INFEASIBLE_COST, device=dist.device, dtype=dist.dtype
        )
        return torch.where(feasible_t, dist, infeasible_cost)

    def manifold_squash(self, actions):
        """
        Freeze left side + head; clamp active joints to [-1, 1].
        Task workspace is enforced via final-step EE gate in get_cost (no joint remap).
        """
        actions = actions.clone()

        if self.frozen_pose is None:
            raise ValueError(
                "❌ GoalMapper Error: frozen_pose not set before planning!"
            )

        actions[..., 0:16] = self.frozen_pose[..., 0:16]
        actions = torch.clamp(actions, -1.0, 1.0)
        return actions

    def encode(self, pixels):
        """
        Standardized encoder entrypoint for world model interaction.
        Expects pixels: (B, T, V, C, H, W)
        """
        return self.model.encode(pixels)

    def project(self, x):
        """Standardized projection entrypoint."""
        return self.model.projector(x)
