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
from omegaconf import OmegaConf


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
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset_root = Path(dataset_root) if dataset_root is not None else None
        self.use_multi_view = use_multi_view
        self.num_views = num_views
        self.use_skeleton = use_skeleton

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

        # 2. Image Transform (Standard LeWM 224x224)
        imagenet_stats = dt.dataset_stats.ImageNet
        self.transform = dt.transforms.Compose(
            dt.transforms.ToImage(**imagenet_stats, source="pixels", target="pixels"),
            dt.transforms.Resize(224, source="pixels", target="pixels"),
        )

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
        print(f"[GOAL_MAPPER] pixels_input shape before: {pixels_input.shape}")
        print(f"[GOAL_MAPPER] actions shape before: {actions.shape}")
        if pixels_input.ndim > target_ndim:
            # (B, S, 1, T_history, V, C, H, W) -> (B, S, T_history, V, C, H, W)
            pixels_input = pixels_input.squeeze(2)
        if actions.ndim > 4:
            # (1, B, S, T_history, 32) -> (B, S, T_history, 32)
            actions = actions.squeeze(0)
        print(f"[GOAL_MAPPER] pixels_input shape after: {pixels_input.shape}")
        print(f"[GOAL_MAPPER] actions shape after: {actions.shape}")

        # 2. Optimized Encoding
        # All samples S share the same history. We encode the batch once.
        info = self.model.encode({"pixels": pixels_input[:, 0]})
        init_emb = info["emb"]  # (B, T_history, 192)
        print(f"[GOAL_MAPPER] init_emb shape: {init_emb.shape}")
        curr_emb = init_emb.repeat_interleave(S, dim=0)  # (B * S, T_history, 192)
        print(f"[GOAL_MAPPER] curr_emb shape: {curr_emb.shape}")

        # 3. History Actions Normalization
        hist_actions = obs_dict.get("action", None)  # (B, S, T_history, 32)
        print(f"[GOAL_MAPPER] hist_actions shape before: {hist_actions.shape}")
        if hist_actions is None:
            hist_actions = torch.zeros(B, S, actions.size(-1)).to(self.device)
        if hist_actions.ndim > 4:
            # (B, S, 1, T_history, 32) -> (B, S, T_history, 32)
            hist_actions = hist_actions.squeeze(2)
        print(f"[GOAL_MAPPER] hist_actions shape after: {hist_actions.shape}")

        # (B, S, T_history, 32) -> (B * S, T_history, 32)
        flat_hist_actions = hist_actions[:, 0].repeat_interleave(S, dim=0)
        print(f"[GOAL_MAPPER] flat_hist_actions shape: {flat_hist_actions.shape}")

        # 4. Prepare Plan Actions (B * S, T, D) with MANIFOLD SQUASHING
        # flat_hist_actions: (B * S, T_history, 32), flat_plan_actions: (B * S, T_horizon, 32)
        # all_actions: (B * S, T, 32)
        flat_plan_actions = self.manifold_squash(
            actions.view(B * S, -1, actions.size(-1))
        )
        all_actions = torch.cat([flat_hist_actions, flat_plan_actions], dim=1)
        print(f"[GOAL_MAPPER] all_actions shape: {all_actions.shape}")

        # 5. Sliding Window Rollout (Flattened BS space)
        pred_latents = []
        T_history = flat_hist_actions.size(1)
        T_horizon = flat_plan_actions.size(1)
        for T in range(T_horizon):
            print(f"============ {T} ============")
            emb_window = curr_emb[:, -T_history:, :]  # (B, T_history, 192)
            print(f"[GOAL_MAPPER] emb_window shape: {emb_window.shape}")
            act_window = all_actions[:, T : T + T_history, :]  # (B * S, T_history, 32)
            print(f"[GOAL_MAPPER] act_window shape: {act_window.shape}")
            act_emb = self.model.action_encoder(act_window)  # (B * S, T_history, 192)
            print(f"[GOAL_MAPPER] act_emb shape: {act_emb.shape}")
            pred_emb = self.model.predict(
                emb_window, act_emb
            )  # (B * S, T_history, 192)
            print(f"[GOAL_MAPPER] pred_emb shape: {pred_emb.shape}")
            last_pred = pred_emb[:, -1:, :]  # (B * S, 1, 192)
            print(f"[GOAL_MAPPER] last_pred shape: {last_pred.shape}")
            curr_emb = torch.cat(
                [curr_emb, last_pred], dim=1
            )  # (B * S, curr_emb.size(1) + 1, 192)
            print(f"[GOAL_MAPPER] curr_emb shape: {curr_emb.shape}")
            pred_latents.append(last_pred)

        # 6. Optimized Planning Cost Logic
        print(f"============   ============")
        all_preds = torch.cat(pred_latents, dim=1)  # (B * S, T_horizon, 192)
        print(f"[GOAL_MAPPER] all_preds shape: {all_preds.shape}")

        # 7. Smart Cost Calculation
        # (B * S, T_horizon)
        reward_pred = self.model.reward_head(all_preds).squeeze(-1)
        reward_weight = 50.0
        dist = (10.0 - reward_pred) * reward_weight  # (B * S, T_horizon)
        print(f"[GOAL_MAPPER] dist shape: {dist.shape}")

        # Latent distance cost
        # repeat (current B * S) // (original B * S) times.
        goal_target = self.goal_latent.to(all_preds.dtype).repeat_interleave(
            B * S // (self.goal_latent.size(0) * self.goal_latent.size(1)), dim=0
        )  # (B * S, 1, 192)
        print(f"[GOAL_MAPPER] goal_target shape: {goal_target.shape}")

        # (B * S, T_horizon)
        dists_to_latents = torch.cdist(all_preds, goal_target).squeeze(-1)
        print(f"[GOAL_MAPPER] dists_to_latents shape: {dists_to_latents.shape}")

        # (B * S, 1)
        min_latent_dist_per_step = dists_to_latents.min(dim=-1).values.unsqueeze(-1)
        print(
            "[GOAL_MAPPER] min_latent_dist_per_step shape: "
            f"{min_latent_dist_per_step.shape}"
        )

        # Global Compass Weight: Reduce to 0.5 to let the Reward Head lead.
        dist = dist + min_latent_dist_per_step * 0.5  # (B * S, T_horizon)
        print(f"[GOAL_MAPPER] dist shape before: {dist.shape}")
        dist = dist.mean(dim=-1)  # (B * S,)
        print(f"[GOAL_MAPPER] dist shape after: {dist.shape}")

        # 8. Smoothness Penalty
        last_real_action = flat_hist_actions[:, -1, :]  # (B * S, 32)
        print(f"[GOAL_MAPPER] last_real_action shape: {last_real_action.shape}")
        jump_start = torch.mean(
            (flat_plan_actions[:, 0, :] - last_real_action) ** 2, dim=-1
        )  # (B * S,)
        print(f"[GOAL_MAPPER] jump_start shape: {jump_start.shape}")

        # Delta within the Plan Horizon (B * S, T_horizon - 1, D)
        if T_horizon > 1:
            jitters = torch.mean(
                (flat_plan_actions[:, 1:, :] - flat_plan_actions[:, :-1, :]) ** 2,
                dim=-1,
            )  # (B * S, T_horizon - 1)
            jump_internal = torch.mean(jitters, dim=1)
        else:
            jump_internal = 0.0
        print(f"[GOAL_MAPPER] jump_internal shape: {jump_internal.shape}")

        smoothness_weight = 100.0
        dist = dist + (jump_start + jump_internal) * smoothness_weight  # (B,)
        print(f"[GOAL_MAPPER] dist shape 2: {dist.shape}")

        return dist.view(B, S)

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

    def encode(self, pixels):
        """
        Standardized encoder entrypoint for world model interaction.
        Expects pixels: (B, T, V, C, H, W)
        """
        return self.model.encode(pixels)

    def project(self, x):
        """Standardized projection entrypoint."""
        return self.model.projector(x)
