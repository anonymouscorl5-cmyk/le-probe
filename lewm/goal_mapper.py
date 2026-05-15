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
from lewm.skeleton.skeletal_utils import load_skeletal_state_dict, reconstruct_4ch_frame


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
        skel_frames_dir=None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset_root = Path(dataset_root)
        self.use_multi_view = use_multi_view
        self.num_views = num_views
        self.use_skeleton = use_skeleton
        self.skel_frames_dir = Path(skel_frames_dir) if skel_frames_dir else None

        # 1. Initialize the Model
        # We assume the model is a JEPA object or a checkpoint containing it
        if str(model_path).endswith(".pt") or str(model_path).endswith(".ckpt"):
            print(f"🧠 Loading Model from Checkpoint: {model_path}")
            if use_skeleton:
                self.model = load_skeletal_state_dict(model_path)
            else:
                self.model = torch.load(model_path, map_location=self.device)
        else:
            # Assume it's a directory or a generic path (not supported here)
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

    def encode_goal_from_pixels(self, pixels):
        """
        Directly encodes a goal from a pre-processed pixel tensor.
        Input 'pixels' should be (C, H, W) or (V, C, H, W).
        """
        with torch.no_grad():
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

            output = self.model.encode({"pixels": pixels_batch.to(self.device)})
            self.goal_latent = output["emb"].detach()
            return self.goal_latent

    def get_cost(self, info_dict, action_candidates):
        """
        CEM PLANNING ENTRYPOINT:
        Calculates the distance between predicted future states and the goal.
        """
        if self.goal_latent is None:
            raise ValueError("❌ GoalMapper Error: Goal not set before planning!")

        # 1. Ensure goal_latent is repeated for the sampling dimension (S)
        # JEPA.rollout expects info_dict to have pixels for initial state encoding
        # and info_dict['goal_emb'] for cost calculation.

        # goal_latent is (1, 1, D) or (B, 1, D)
        # We need to expand it for S samples in the criterion
        info_dict["goal_emb"] = self.goal_latent

        # 2. Rollout the future state predictions
        info_dict = self.model.rollout(info_dict, action_candidates)

        # 3. Calculate cost (MSE between predicted and goal latent)
        return self.model.criterion(info_dict)

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
