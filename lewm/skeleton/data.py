import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from lewm.lewm_data_plugin import LEWMDataPlugin


class SkeletonDataPlugin(LEWMDataPlugin):
    """
    High-Efficiency Tiled Skeleton Data Plugin with PT Tensor Caching.
    Supports high-speed direct disk caching to bypass H.264 video decoding,
    while retaining standard video decoding/splitting fallback.
    """

    def __init__(self, *args, **kwargs):
        # 1. Identify base views
        keys = kwargs.get("keys_to_load", ["world_center"])
        self.base_views = [
            k
            for k in keys
            if k
            in ["world_center", "world_left", "world_right", "world_top", "world_wrist"]
        ]

        # Initialize base class
        super().__init__(*args, **kwargs)

        # 2. Setup key_map to point these views to the tiled video files
        for view in self.base_views:
            self.key_map[view] = f"observation.images.{view}_tiled"

        # 3. Wrap transform to handle splitting before standard transforms run
        self.orig_transform = self.transform
        self.transform = self.tiled_transform_wrapper

        # 4. Check for High-Speed direct PT cache directory
        self.cache_dir = self.root / "cache"
        self.use_tensor_cache = self.cache_dir.exists()

        # Worker-local single-episode RAM cache to avoid redundant disk reads
        self._last_loaded_ep = -1
        self._last_loaded_data = None

        if self.use_tensor_cache:
            print(f"⚡ High-Speed Direct Disk Cache DETECTED at: {self.cache_dir}")
            print(
                "🚀 Bypassing on-the-fly video decoding. Training speed will be maximized!"
            )
        else:
            print(
                f"🚀 Tiled SkeletonDataPlugin initialized (Video Fallback) for views: {self.base_views}"
            )

    def _run_filtered_transforms(self, tf, batch):
        if tf is None:
            return batch

        transforms_list = []
        if hasattr(tf, "transforms"):
            transforms_list = tf.transforms
        elif isinstance(tf, (list, tuple)):
            transforms_list = tf
        else:
            transforms_list = [tf]

        for t in transforms_list:
            if hasattr(t, "transforms"):
                batch = self._run_filtered_transforms(t, batch)
                continue

            source = getattr(t, "source", None)
            if source is not None:
                if any(k in source for k in ["pixels", "images", "world_"]):
                    continue
            try:
                batch = t(batch)
            except KeyError:
                pass
        return batch

    def tiled_transform_wrapper(self, nested_batch):
        """
        Intercepts the transform call to split tiled videos and handle 4-channel fusion.
        """
        # If cache is used, tensors are already resized, normalized, and fused
        if self.use_tensor_cache:
            if "pixels" in nested_batch:
                px = nested_batch["pixels"]
                if px.dtype == torch.uint8:
                    rgb = px[:, :, :3].float() / 255.0
                    skel = px[:, :, 3:].float() / 255.0

                    # ImageNet stats normalization: mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
                    mean = torch.tensor(
                        [0.485, 0.456, 0.406], dtype=rgb.dtype, device=rgb.device
                    ).view(1, 1, 3, 1, 1)
                    std = torch.tensor(
                        [0.229, 0.224, 0.225], dtype=rgb.dtype, device=rgb.device
                    ).view(1, 1, 3, 1, 1)
                    rgb = (rgb - mean) / std

                    nested_batch["pixels"] = torch.cat([rgb, skel], dim=2)

            if self.orig_transform:
                nested_batch = self._run_filtered_transforms(
                    self.orig_transform, nested_batch
                )
            return nested_batch

        skeletons = {}

        # Split Tiled Videos into RGB + Skeleton
        for view in self.base_views:
            path = None
            tensor = None
            if view in nested_batch:
                tensor = nested_batch[view]
                path = [view]
            elif (
                "observation" in nested_batch
                and "images" in nested_batch["observation"]
                and view in nested_batch["observation"]["images"]
            ):
                tensor = nested_batch["observation"]["images"][view]
                path = ["observation", "images", view]

            if tensor is not None and tensor.shape[-1] > tensor.shape[-2]:
                mid = tensor.shape[-1] // 2
                rgb_raw = tensor[..., :mid]  # [T, 3, H, W]
                skel_raw = tensor[..., mid:]

                # Resize RGB immediately on Worker (CPU) to 224x224
                if rgb_raw.shape[-2:] != (self.img_size, self.img_size):
                    rgb = torch.nn.functional.interpolate(
                        rgb_raw.float(),
                        size=(self.img_size, self.img_size),
                        mode="bilinear",
                        align_corners=False,
                    ).byte()
                else:
                    rgb = rgb_raw

                # Store RGB back
                d = nested_batch
                for p in path[:-1]:
                    d = d[p]
                d[path[-1]] = rgb

                # Downsample Skeleton to 224x224
                if skel_raw.shape[-2:] != (self.img_size, self.img_size):
                    skeletons[view] = torch.nn.functional.interpolate(
                        skel_raw.float(),
                        size=(self.img_size, self.img_size),
                        mode="nearest",
                    ).byte()
                else:
                    skeletons[view] = skel_raw

        # Run Original Transforms
        if self.orig_transform:
            nested_batch = self.orig_transform(nested_batch)

        # Attach raw skeletons for GPU fusion
        for view, skel in skeletons.items():
            nested_batch[f"{view}_skel_raw"] = skel

        return nested_batch

    def __getitem__(self, idx):
        # Determine temporal step sequence boundaries for phase/checkpoint mapping
        frame_idx = int(self.frame_indices[idx])

        # Calculate Phase Index and Checkpoint Frame Index for each time step
        # Phase boundaries are at static 8-frame intervals: [0..7] -> P0, [8..15] -> P1, etc.
        seq_steps = torch.arange(frame_idx, frame_idx + self.num_steps)
        phase_idx = seq_steps // 8
        checkpoint_frame_idx = (phase_idx + 1) * 8 - 1

        # --- PATH A: High-Speed Direct PT Cache ---
        if self.use_tensor_cache:
            episode_idx = int(self.episode_indices[idx])

            # Single-episode worker-local caching to avoid loading the same file 32 times
            if self._last_loaded_ep != episode_idx:
                cache_path = self.cache_dir / f"episode_{episode_idx:03d}_fused.pt"
                # If cached episode file is missing, fallback to raw loading
                if cache_path.exists():
                    self._last_loaded_data = torch.load(cache_path, map_location="cpu")
                    self._last_loaded_ep = episode_idx
                else:
                    self._last_loaded_data = None
                    self._last_loaded_ep = -1

            if self._last_loaded_data is not None:
                # Clamp sequence steps to stay within valid cached episode bounds (usually 32 frames)
                max_frame_idx = self._last_loaded_data["pixels"].shape[0] - 1
                clamped_steps = torch.clamp(seq_steps, 0, max_frame_idx)

                phase_idx = clamped_steps // 8
                checkpoint_frame_idx = (phase_idx + 1) * 8 - 1

                # Fetch pre-computed DINO anchors
                dino_waypoints = self._last_loaded_data.get(
                    "dino_waypoints", torch.zeros((4, 384))
                )
                phase_anchors = dino_waypoints[phase_idx]

                batch = {
                    "observation.state": self._last_loaded_data["state"][clamped_steps],
                    "action": self._last_loaded_data["action"][clamped_steps],
                    "pixels": self._last_loaded_data["pixels"][
                        clamped_steps
                    ],  # Shape [T, V, 4, 224, 224]
                    "phase_idx": phase_idx.unsqueeze(-1),  # Shape [T, 1]
                    "checkpoint_frame_idx": checkpoint_frame_idx.unsqueeze(
                        -1
                    ),  # Shape [T, 1]
                    "dino_anchor": phase_anchors,  # Shape [T, 384]
                }

                if self.has_progress:
                    batch["progress"] = self.cached_progress[idx : idx + self.num_steps]

                # Apply transforms (normalized)
                nested_batch = self.nest_dict(batch)
                if self.transform:
                    nested_batch = self.transform(nested_batch)
                return self.flatten_dict(nested_batch)

        # --- PATH B: Video Decoding Fallback ---
        batch = super().__getitem__(idx)
        episode_idx = int(self.episode_indices[idx])

        # Multi-view Stacking
        if self.use_multi_view:
            views = self.base_views
            pixels = []
            skels = []

            for vn in views:
                # RGB
                pix = batch.get(f"observation.images.{vn}") or batch.get(vn)
                if pix is not None:
                    pixels.append(pix)

                # Skeletons
                sk = batch.get(f"{vn}_skel_raw")
                if sk is not None:
                    skels.append(sk)

            if pixels:
                batch["pixels"] = torch.stack(pixels, dim=1)
            if skels:
                batch["skeletons_raw"] = torch.stack(skels, dim=1)

        # Fetch pre-computed DINO anchors from raw file fallback
        dino_pt_path = (
            self.root / f"cache_dino/chunk-000/file-{episode_idx:03d}_dino.pt"
        )
        if dino_pt_path.exists():
            dino_waypoints = torch.load(dino_pt_path, map_location="cpu")
        else:
            dino_waypoints = torch.zeros((4, 384))
        phase_anchors = dino_waypoints[phase_idx]

        # Attach phase and anchor tracking arrays
        batch["phase_idx"] = phase_idx.unsqueeze(-1)
        batch["checkpoint_frame_idx"] = checkpoint_frame_idx.unsqueeze(-1)
        batch["dino_anchor"] = phase_anchors

        return batch
