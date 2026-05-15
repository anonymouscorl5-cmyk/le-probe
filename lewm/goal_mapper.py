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
    @torch.no_grad()
    def get_cost(self, obs_dict, action_candidates):
        """
        PRECISION PLANNING COST (Manifold-Aligned)
        Handles 7D MPC inputs by flattening Batch/Samples for library compatibility.
        """
        if self.goal_latent is None:
            raise ValueError("❌ GoalMapper Error: Goal not set before planning!")

        # 1. Flatten Batch and Samples to satisfy the 6D Library requirement
        # pixels: (B, S, T, V, C, H, W) -> (B*S, T, V, C, H, W)
        pixels = obs_dict["pixels"].to(self.device)
        B, S, T_obs, V, C, H, W = pixels.shape
        flat_pixels = pixels.reshape(B * S, T_obs, V, C, H, W)

        # Actions: (B, S, T_plan, D) -> (B*S, T_plan, D)
        actions = action_candidates.to(self.device)
        T_plan = actions.size(2)
        flat_actions = actions.reshape(B * S, T_plan, -1)

        # 2. Execute Prediction Rollout (Using 6D Flattened Tensors)
        info_dict = {
            "pixels": flat_pixels,
            "action": obs_dict.get("action", torch.zeros_like(actions[:, :, :1]))
            .reshape(B * S, -1, actions.size(-1))
            .to(self.device),
            "goal_emb": self.goal_latent,  # (B, 1, D) - will be handled by criterion or manual logic
        }

        info_dict = self.model.rollout(info_dict, flat_actions)

        # 3. Unfold Results back to (B, S)
        # predicted_emb: (B*S, T_plan, D) -> (B, S, T_plan, D)
        pred_latents = info_dict["predicted_emb"].reshape(B, S, T_plan, -1)
        _, _, _, D_feat = pred_latents.shape

        # 4. Smart Cost Calculation
        # a. PRIMARY REWARD: Tuned Reward Head (Task Progress)
        flat_preds = pred_latents.reshape(-1, D_feat)
        reward_pred = self.model.reward_head(flat_preds).reshape(B, S, T_plan)
        reward_cost = (10.0 - reward_pred).mean(dim=-1) * 50.0

        # b. GLOBAL COMPASS: Latent Euclidean Distance
        # Expand goal_latent (B, 1, D) to (B, S, T_plan, D)
        goal_expanded = (
            self.goal_latent.unsqueeze(1).unsqueeze(1).expand(B, S, T_plan, D_feat)
        )
        latent_cost = F.mse_loss(pred_latents, goal_expanded, reduction="none").mean(
            dim=(2, 3)
        )

        # c. PHYSICAL GRACE: Smoothness Penalty
        jitters = (actions[:, :, 1:] - actions[:, :, :-1]).pow(2).mean(dim=(2, 3))
        smoothness_weight = 100.0

        total_cost = reward_cost + (latent_cost * 0.5) + (jitters * smoothness_weight)

        return total_cost

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
