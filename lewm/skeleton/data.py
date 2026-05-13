import torch
import numpy as np
from lewm.lewm_data_plugin import LEWMDataPlugin
import torchvision.transforms.functional as TF


class SkeletonDataPlugin(LEWMDataPlugin):
    """
    Subclass of LEWMDataPlugin that loads both RGB and Skeletal Mask streams.
    Stacks them into a 4-channel [T, V, 4, H, W] tensor for training.
    """

    def __init__(self, *args, **kwargs):
        # 1. Capture and remove transform from kwargs to handle it manually
        self.original_transform = kwargs.get("transform", None)
        kwargs["transform"] = None

        # 2. Extract base keys and skeleton keys
        self.base_keys = kwargs.get("keys_to_load", [])
        self.view_names = [k for k in self.base_keys if "world_" in k]
        self.skel_keys = [f"{k}_skeleton" for k in self.view_names]

        # We DO NOT add skel_keys to kwargs["keys_to_load"] because the base
        # LEWMDataPlugin doesn't know how to decode them as videos.

        super().__init__(*args, **kwargs)

        # 3. Register skeleton mappings for our manual loader
        for view in self.view_names:
            skel_key = f"observation.images.{view}_skeleton"
            self.key_map[f"{view}_skeleton"] = skel_key

    def __getitem__(self, idx):
        # A. Call base loader (Decodes RGB videos, loads states/actions)
        # Transform is None, so no KeyError yet.
        batch = super().__getitem__(idx)

        # B. Manually decode Skeleton videos
        episode_idx = int(self.episode_indices[idx])
        frame_idx = int(self.frame_indices[idx])

        for vn in self.view_names:
            skel_target = f"{vn}_skeleton"
            skel_source = self.key_map[skel_target]

            try:
                video_path = self._get_video_path(episode_idx, skel_source)
                if video_path.exists():
                    decoder = self._get_decoder(video_path)
                    seq_indices = list(range(frame_idx, frame_idx + self.num_steps))
                    frames = decoder.get_frames_at(indices=seq_indices)

                    # Resize and add to batch
                    batch[skel_target] = TF.resize(
                        frames.data, [self.img_size, self.img_size], antialias=True
                    ).byte()
            except Exception as e:
                # Fallback: If skeleton is missing, we'll handle it during fusion
                pass

        # C. Run Nesting and Manual Transform
        # This is where 'world_center_skeleton' is now safely found
        nested_batch = self.nest_dict(batch)
        if self.original_transform:
            nested_batch = self.original_transform(nested_batch)

        # Flatten back for the rest of the pipeline
        batch = self.flatten_dict(nested_batch)

        # D. View Fusion (Multi-View stacking into 4-channel pixels)
        if self.use_multi_view:
            fused_views = []
            for vn in self.view_names:
                rgb_key = f"observation.images.{vn}"
                skel_key = f"observation.images.{vn}_skeleton"

                if rgb_key in batch:
                    rgb = batch[rgb_key]
                    if rgb.dtype == torch.uint8:
                        rgb = rgb.float() / 255.0

                    if skel_key in batch:
                        skel = batch[skel_key]
                        if skel.dtype == torch.uint8:
                            skel = skel.float() / 255.0
                    else:
                        # Fallback: Zero-mask if skeleton is missing
                        skel = torch.zeros(
                            (rgb.shape[0], 1, rgb.shape[2], rgb.shape[3]),
                            device=rgb.device,
                        )

                    # Ensure shapes match before concatenation
                    if skel.shape[2:] != rgb.shape[2:]:
                        skel = TF.resize(skel, rgb.shape[2:])

                    fused = torch.cat([rgb, skel], dim=1)  # [T, 4, H, W]
                    fused_views.append(fused)

            if fused_views:
                batch["pixels"] = torch.stack(fused_views, dim=1)

        return batch
