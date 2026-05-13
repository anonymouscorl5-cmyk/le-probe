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
        # 1. Expand keys_to_load to include skeletal masks
        # This tells the base class to decode them as video streams
        if "keys_to_load" in kwargs:
            base_keys = kwargs["keys_to_load"]
            self.view_names = [k for k in base_keys if "world_" in k]
            skel_keys = [f"{k}_skeleton" for k in self.view_names]
            kwargs["keys_to_load"] = base_keys + skel_keys

        super().__init__(*args, **kwargs)

        # 2. Register skeleton mappings so base class knows where to find .mp4s
        for view in self.view_names:
            skel_key = f"observation.images.{view}_skeleton"
            self.key_map[f"{view}_skeleton"] = skel_key

    def __getitem__(self, idx):
        # A. Call base loader (Now decodes BOTH RGB and Skeleton videos natively)
        batch = super().__getitem__(idx)

        # B. View Fusion (Multi-View stacking into 4-channel pixels)
        if self.use_multi_view:
            fused_views = []
            for vn in self.view_names:
                # The base class has already decoded and resized these
                rgb_key = vn
                skel_key = f"{vn}_skeleton"

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
                            dtype=rgb.dtype,
                        )

                    # Ensure shapes match (base class handles this, but for safety)
                    if skel.shape[1] != 1:
                        # Skeleton is likely RGB (3-channel), take mean
                        skel = skel.mean(dim=1, keepdim=True)

                    fused = torch.cat([rgb, skel], dim=1)  # [T, 4, H, W]
                    fused_views.append(fused)

            if fused_views:
                # [T, V, 4, H, W]
                batch["pixels"] = torch.stack(fused_views, dim=1)

        return batch
