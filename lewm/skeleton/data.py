import torch
from lewm.lewm_data_plugin import LEWMDataPlugin


class SkeletonDataPlugin(LEWMDataPlugin):
    """
    Subclass of LEWMDataPlugin that loads both RGB and Skeletal Mask streams.
    Stacks them into a 4-channel [T, V, 4, H, W] tensor for training.
    """

    def __init__(self, *args, **kwargs):
        # Expand keys_to_load to include skeletal masks
        if "keys_to_load" in kwargs:
            base_keys = kwargs["keys_to_load"]
            skel_keys = [f"{k}_skeleton" for k in base_keys if "world_" in k]
            kwargs["keys_to_load"] = base_keys + skel_keys

        super().__init__(*args, **kwargs)

        # Update key_map for skeleton streams
        for view in [
            "world_center",
            "world_left",
            "world_right",
            "world_top",
            "world_wrist",
        ]:
            skel_key = f"observation.images.{view}_skeleton"
            self.key_map[f"{view}_skeleton"] = skel_key

    def __getitem__(self, idx):
        # Call base loader to get all raw view frames
        batch = super().__getitem__(idx)

        if not self.use_multi_view:
            return batch

        view_names = [
            "world_center",
            "world_left",
            "world_right",
            "world_top",
            "world_wrist",
        ]
        fused_views = []

        for vn in view_names:
            rgb_key = f"observation.images.{vn}"
            skel_key = f"observation.images.{vn}_skeleton"

            if rgb_key in batch:
                rgb = batch[rgb_key]
                if rgb.dtype == torch.uint8:
                    rgb = rgb.float() / 255.0

                # Check for skeleton mask
                if skel_key in batch:
                    skel = batch[skel_key]
                    if skel.dtype == torch.uint8:
                        skel = skel.float() / 255.0
                else:
                    # Fallback: Zero-mask if skeleton is missing
                    skel = torch.zeros(
                        (rgb.shape[0], 1, rgb.shape[2], rgb.shape[3]), device=rgb.device
                    )

                fused = torch.cat([rgb, skel], dim=1)  # [T, 4, H, W]
                fused_views.append(fused)

        if fused_views:
            batch["pixels"] = torch.stack(fused_views, dim=1)

        return batch
