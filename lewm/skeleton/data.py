import torch
import os
from lewm.lewm_data_plugin import LEWMDataPlugin


class SkeletonDataPlugin(LEWMDataPlugin):
    """
    High-Efficiency Tiled Skeleton Data Plugin.
    Decodes a single [RGB | Skeleton] video per view, halving I/O and decoder overhead.
    """

    def __init__(self, *args, **kwargs):
        # Intercept keys_to_load to use tiled videos
        if "keys_to_load" in kwargs:
            keys = kwargs["keys_to_load"]
            self.base_views = [
                k
                for k in keys
                if k
                in [
                    "world_center",
                    "world_left",
                    "world_right",
                    "world_top",
                    "world_wrist",
                ]
            ]
            # Replace original views with _tiled versions
            new_keys = [k for k in keys if k not in self.base_views]
            for view in self.base_views:
                new_keys.append(f"{view}_tiled")
            kwargs["keys_to_load"] = new_keys
        else:
            self.base_views = ["world_center"]

        super().__init__(*args, **kwargs)

        # Cache tiled mapping
        self.tiled_keys = {vn: f"{vn}_tiled" for vn in self.base_views}
        print(f"🚀 Tiled SkeletonDataPlugin initialized for views: {self.base_views}")

    def __getitem__(self, idx):
        # A. Call base loader (Decodes the tiled videos natively)
        batch = super().__getitem__(idx)

        # B. View Splitting and Fusion
        if self.use_multi_view:
            fused_views = []
            for vn in self.base_views:
                tiled_key = self.tiled_keys[vn]
                tiled_tensor = batch.get(tiled_key)

                if tiled_tensor is None:
                    continue

                # Standardize to float32 [0,1]
                if tiled_tensor.dtype == torch.uint8:
                    tiled_tensor = tiled_tensor.float() / 255.0

                # Horizontal Split: [T, C, H, 2*W] -> [RGB(480) | Skel(480)]
                # The image width is 960 (480*2)
                mid = tiled_tensor.shape[-1] // 2
                rgb = tiled_tensor[..., :mid]
                skel_3ch = tiled_tensor[..., mid:]

                # Convert 3-channel skeleton (saved as BGR/RGB) to 1-channel mask
                skel = skel_3ch.mean(dim=1, keepdim=True)

                # Concatenate along channel dimension: [T, 4, H, W]
                fused = torch.cat([rgb, skel], dim=1)
                fused_views.append(fused)

            if fused_views:
                # Update pixels with fused 4-channel multi-view tensor [T, V, 4, H, W]
                batch["pixels"] = torch.stack(fused_views, dim=1)

        return batch
