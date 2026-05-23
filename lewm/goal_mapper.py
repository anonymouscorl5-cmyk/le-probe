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
from lewm.planning_constraints import (
    freeze_and_clamp_actions,
    right_arm_norm_feasible_mask,
    task_workspace_feasible_mask,
    scatter_infeasible_costs,
)
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
        use_task_workspace=False,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset_root = Path(dataset_root) if dataset_root is not None else None
        self.use_multi_view = use_multi_view
        self.num_views = num_views
        self.use_skeleton = use_skeleton
        self.use_dino = use_dino
        self.use_task_workspace = use_task_workspace

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
                    use_dino=use_dino,
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
                    if (
                        not clean_key.startswith("sigreg.")
                        and clean_key != "encoder.backbone.embeddings.mask_token"
                    ):
                        clean_sd[clean_key] = v
                self.model.load_state_dict(clean_sd, strict=True)
            else:
                self.model = raw_data
        else:
            raise ValueError(f"Unsupported model path: {model_path}")

        self.model.to(self.device).eval()
        self.goal_latent = None
        self.frozen_pose = None

        self._task_ws = TaskWorkspaceMPCConstraint() if use_task_workspace else None

        # 2. Image Transform (Standard LeWM 224x224)
        imagenet_stats = dt.dataset_stats.ImageNet
        self.transform = dt.transforms.Compose(
            dt.transforms.ToImage(**imagenet_stats, source="pixels", target="pixels"),
            dt.transforms.Resize(224, source="pixels", target="pixels"),
        )

        gate = "task_workspace" if use_task_workspace else "right_arm_norm"
        print(
            f"✅ GoalMapper initialized "
            f"(Skeleton: {use_skeleton}, DINO: {use_dino}, MPC gate: {gate})"
        )

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
            # (1, B, S, T_horizon, 32) -> (B, S, T_horizon, 32)
            actions = actions.squeeze(0)

        # 3. History actions (CEM expands sample dim S; history is shared per batch row)
        hist_actions = obs_dict.get("action", None)  # (B, S, T_history, 32)
        t_hist = 3
        if hist_actions is not None and hist_actions.ndim >= 3:
            t_hist = int(hist_actions.shape[-2])
        if hist_actions is None:
            hist_actions = torch.zeros(B, S, t_hist, actions.size(-1)).to(self.device)
        if hist_actions.ndim > 4:
            # (B, S, 1, T_history, 32) -> (B, S, T_history, 32)
            hist_actions = hist_actions.squeeze(2)

        # (B, S, T_history, 32) -> (B * S, T_history, 32)
        flat_hist_actions = hist_actions[:, 0].repeat_interleave(S, dim=0)

        # 4. Plan actions: freeze 0–15, clamp [-1, 1] — no joint remap
        # actions: (B, S, T_horizon, 32) -> flat_plan_actions: (B * S, T_horizon, 32)
        flat_plan_actions = freeze_and_clamp_actions(
            actions.view(B * S, -1, actions.size(-1)), self.frozen_pose
        )

        # 5. Feasibility gate (right-arm norm envelope or task workspace) — before LeWM
        feasible_np = self._precheck_plan_feasibility(
            obs_dict, flat_plan_actions
        )  # (B * S,)
        if not feasible_np.any():
            return torch.full(
                (B, S), INFEASIBLE_COST, device=self.device, dtype=torch.float32
            )

        f_idx = torch.from_numpy(np.nonzero(feasible_np)[0]).to(
            device=self.device, dtype=torch.long
        )  # (K,) indices into flattened B * S
        dist_feas = self._rollout_planning_cost(
            obs_dict,
            pixels_input,
            flat_hist_actions[f_idx],
            flat_plan_actions[f_idx],
            B,
            S,
            f_idx,
        )
        if feasible_np.all():
            return dist_feas.view(B, S)  # (K,) == (B * S,) when all feasible

        # Scatter feasible costs back; infeasible slots stay INFEASIBLE_COST
        dist = scatter_infeasible_costs(
            B * S,
            feasible_np,
            dist_feas,
            device=self.device,
            dtype=dist_feas.dtype,
        )  # (B * S,)
        return dist.view(B, S)

    def _precheck_plan_feasibility(
        self, obs_dict, flat_plan_actions: torch.Tensor
    ) -> np.ndarray:
        """
        Reject infeasible CEM samples before any LeWM forward pass.

        flat_plan_actions: (B * S, T_horizon, 32) -> returns (B * S,) bool mask.
        """
        plan_np = flat_plan_actions.detach().cpu().numpy()  # (B * S, T_horizon, 32)
        print(
            f"plan_np: {plan_np.min()} {plan_np.max()} {plan_np.mean()} {plan_np.std()}"
        )

        if self.use_task_workspace:
            tw_wire32 = obs_dict.get("task_workspace_wire32")
            tw_H = obs_dict.get("task_workspace_H")
            if tw_wire32 is None or tw_H is None or self._task_ws is None:
                return np.ones(plan_np.shape[0], dtype=bool)

            wire32 = np.asarray(tw_wire32[0, 0], dtype=np.float64).reshape(-1)
            final_only = obs_dict.get("task_workspace_check_final_only", True)
            if isinstance(final_only, torch.Tensor):
                final_only = bool(final_only.reshape(-1)[0].item())
            else:
                final_only = bool(np.asarray(final_only).reshape(-1)[0])

            cube_xyz = None
            tw_cube = obs_dict.get("task_workspace_cube_xyz")
            if tw_cube is not None:
                cube_xyz = np.asarray(tw_cube[0, 0], dtype=np.float64).reshape(3)

            self._task_ws.set_baseline_from_wire32(wire32, cube_xyz=cube_xyz)
            return task_workspace_feasible_mask(
                self._task_ws,
                wire32,
                plan_np,
                check_final_only=final_only,
                cube_xyz=cube_xyz,
            )

        return right_arm_norm_feasible_mask(plan_np)

    def _rollout_planning_cost(
        self,
        obs_dict,
        pixels_input: torch.Tensor,
        flat_hist_actions: torch.Tensor,
        flat_plan_actions: torch.Tensor,
        B: int,
        S: int,
        flat_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        LeWM rollout + reward cost for feasible CEM samples only.

        When K < B * S, flat_* tensors are indexed by f_idx; encode uses unique batch rows.
        """
        K = flat_plan_actions.shape[0]
        batch_ids = flat_indices // S  # (K,) env row per feasible sample

        # Encode once per unique batch row (history shared across S for that row)
        info = self.model.encode({"pixels": pixels_input[batch_ids, 0]})
        init_emb = info["emb"]  # (K, T_history, 192)
        curr_emb = init_emb  # grows along time as rollout proceeds

        # flat_hist: (K, T_history, 32), flat_plan: (K, T_horizon, 32)
        # all_actions: (K, T_history + T_horizon, 32)
        all_actions = torch.cat([flat_hist_actions, flat_plan_actions], dim=1)

        pred_latents = []
        T_history = flat_hist_actions.size(1)
        T_horizon = flat_plan_actions.size(1)
        for _T in range(T_horizon):
            emb_window = curr_emb[:, -T_history:, :]  # (K, T_history, 192)
            act_window = all_actions[:, _T : _T + T_history, :]  # (K, T_history, 32)
            act_emb = self.model.action_encoder(act_window)  # (K, T_history, 192)
            pred_emb = self.model.predict(emb_window, act_emb)  # (K, T_history, 192)
            last_pred = pred_emb[:, -1:, :]  # (K, 1, 192)
            curr_emb = torch.cat(
                [curr_emb, last_pred], dim=1
            )  # (K, T_history + step, 192)
            pred_latents.append(last_pred)

        all_preds = torch.cat(pred_latents, dim=1)  # (K, T_horizon, 192)

        # Reward head cost — (K, T_horizon) then reduced
        reward_pred = self.model.reward_head(all_preds).squeeze(-1)
        reward_weight = 50.0
        dist = (10.0 - reward_pred) * reward_weight  # (K, T_horizon)

        if self.use_dino:
            phase_idx = obs_dict.get("phase_idx")
            if phase_idx is None:
                phase_idx = torch.zeros((B, 1), device=self.device)
            else:
                if not isinstance(phase_idx, torch.Tensor):
                    phase_idx = torch.tensor(phase_idx, device=self.device)
                phase_idx = phase_idx.flatten()
                if phase_idx.numel() == B * S:
                    phase_idx = phase_idx[::S].view(B, 1).to(self.device)
                elif phase_idx.numel() >= B:
                    phase_idx = phase_idx[:B].view(B, 1).to(self.device)
                else:
                    phase_idx = phase_idx[0].repeat(B).view(B, 1).to(self.device)
            phase_idx = phase_idx[batch_ids].view(K, 1)  # (K, 1)

            curr_state = init_emb[:, -1, :]  # (K, 192)
            subgoal = self.model.predict_subgoal(curr_state, phase_idx)  # (K, 192)
            final_dist = torch.norm(all_preds[:, -1, :] - subgoal, p=2, dim=-1)  # (K,)
            step_dists = torch.norm(
                all_preds - subgoal.unsqueeze(1), p=2, dim=-1
            )  # (K, T_horizon)
            subgoal_cost = final_dist.unsqueeze(-1) + 0.1 * step_dists
            dist = dist + subgoal_cost
            dist = dist.mean(dim=-1)  # (K,)
        else:
            num_goals = self.goal_latent.view(-1, self.goal_latent.size(-1)).size(0)
            expansion_factor = max(1, (B * S) // num_goals)
            flat_idx_np = flat_indices.detach().cpu().numpy()
            goal_ids = np.minimum(flat_idx_np // expansion_factor, num_goals - 1)
            goal_target = (
                self.goal_latent.view(num_goals, -1)[goal_ids]
                .unsqueeze(1)
                .to(all_preds.dtype)
            )  # (K, 1, 192)
            dists_to_latents = torch.cdist(all_preds, goal_target).squeeze(
                -1
            )  # (K, T_horizon)
            min_latent_dist = dists_to_latents.min(dim=-1).values.unsqueeze(
                -1
            )  # (K, 1)
            dist = dist + min_latent_dist * 0.5
            dist = dist.mean(dim=-1)  # (K,)

        # Smoothness penalty on the actual (un-remapped) plan actions
        last_real = flat_hist_actions[:, -1, :]  # (K, 32)
        jump_start = torch.mean(
            (flat_plan_actions[:, 0, :] - last_real) ** 2, dim=-1
        )  # (K,)
        if T_horizon > 1:
            jitters = torch.mean(
                (flat_plan_actions[:, 1:, :] - flat_plan_actions[:, :-1, :]) ** 2,
                dim=-1,
            )  # (K, T_horizon - 1)
            jump_internal = torch.mean(jitters, dim=1)  # (K,)
        else:
            jump_internal = 0.0
        dist = dist + (jump_start + jump_internal) * 100.0  # (K,)
        return dist

    def predict(self, *args, **kwargs):
        """Proxy to the internal World Model's prediction logic."""
        return self.model.predict(*args, **kwargs)

    def manifold_squash(self, actions):
        """Alias for freeze_and_clamp_actions (no right-arm remap)."""
        if self.frozen_pose is None:
            raise ValueError(
                "❌ GoalMapper Error: frozen_pose not set before planning!"
            )
        return freeze_and_clamp_actions(actions, self.frozen_pose)

    def encode(self, pixels):
        """
        Standardized encoder entrypoint for world model interaction.
        Expects pixels: (B, T, V, C, H, W)
        """
        return self.model.encode(pixels)

    def project(self, x):
        """Standardized projection entrypoint."""
        return self.model.projector(x)
