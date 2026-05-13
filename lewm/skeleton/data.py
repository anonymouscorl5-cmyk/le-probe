import torch
import torchvision.transforms.functional as TF
from lewm.lewm_data_plugin import LEWMDataPlugin


class SkeletonDataPlugin(LEWMDataPlugin):
    """
    High-Efficiency Tiled Skeleton Data Plugin.
    Decodes a single [RGB | Skeleton] video per view, halving I/O and decoder overhead.
    Wraps transforms to handle splitting and prevents squashing.
    """

    def __init__(self, *args, **kwargs):
        # 1. Identify base views (e.g. world_center)
        keys = kwargs.get("keys_to_load", ["world_center"])
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

        # Initialize base class
        super().__init__(*args, **kwargs)

        # 2. Setup key_map to point these views to the tiled video files
        # We modify self.key_map AFTER super().__init__ because the base class
        # hardcodes its own key_map and doesn't take it in kwargs.
        for view in self.base_views:
            # We map 'world_center' to 'observation.images.world_center_tiled'
            # The base class will find the video for the tiled folder
            # but store it under the key 'world_center' in the flat batch.
            self.key_map[view] = f"observation.images.{view}_tiled"

        # 3. Wrap transform to handle splitting before standard transforms run
        self.orig_transform = self.transform
        self.transform = self.tiled_transform_wrapper
        print(
            f"🚀 Tiled SkeletonDataPlugin initialized (Wrapped Transform) for views: {self.base_views}"
        )

    def tiled_transform_wrapper(self, nested_batch):
        """
        Intercepts the transform call to split tiled videos and handle 4-channel fusion.
        """
        skeletons = {}

        # A. Pre-Transform: Split Tiled Data
        for view in self.base_views:
            path = None
            tensor = None

            # Look for the tensor in nested batch
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
                # 1. Split [T, C, H, 2*W] -> [RGB | Skel]
                mid = tensor.shape[-1] // 2
                rgb = tensor[..., :mid]
                skel_3ch = tensor[..., mid:]

                # 2. Store RGB back (Still uint8, will be resized/normed by orig_transform)
                d = nested_batch
                for p in path[:-1]:
                    d = d[p]
                d[path[-1]] = rgb

                # 3. Process Skeleton (mean to 1-ch, float [0,1])
                skel = skel_3ch.float().mean(dim=1, keepdim=True) / 255.0
                # Resize to target img_size immediately
                skel = TF.resize(skel, [self.img_size, self.img_size], antialias=True)
                skeletons[view] = skel

        # B. Run Original Transforms (Resizing, Normalization, etc. on RGB)
        if self.orig_transform:
            nested_batch = self.orig_transform(nested_batch)

        # C. Post-Transform: Fuse Skeleton as 4th Channel
        for view, skel in skeletons.items():
            # Find the transformed RGB
            path = None
            rgb = None
            if view in nested_batch:
                rgb = nested_batch[view]
                path = [view]
            elif (
                "observation" in nested_batch
                and "images" in nested_batch["observation"]
                and view in nested_batch["observation"]["images"]
            ):
                rgb = nested_batch["observation"]["images"][view]
                path = ["observation", "images", view]

            if rgb is not None:
                # Standardize RGB to float if needed (usually it's float after transform)
                if rgb.dtype == torch.uint8:
                    rgb = rgb.float() / 255.0

                # Fuse! [T, 3, H, W] + [T, 1, H, W] -> [T, 4, H, W]
                fused = torch.cat([rgb, skel.to(rgb.device)], dim=1)

                d[path[-1]] = fused

        return nested_batch

    def __getitem__(self, idx):
        # Call base loader (which now calls tiled_transform_wrapper)
        batch = super().__getitem__(idx)

        # Handle multi-view stacking for the fused 4-channel pixels
        if self.use_multi_view:
            fused_views = []
            for vn in self.base_views:
                # Look for fused keys in flat batch (super().__getitem__ flattens it)
                # flattened keys from nest_dict: 'observation.images.world_center'
                key = f"observation.images.{vn}"
                if key in batch:
                    fused_views.append(batch[key])
                elif vn in batch:
                    fused_views.append(batch[vn])

            if fused_views:
                # Update pixels with fused 4-channel multi-view tensor [T, V, 4, H, W]
                batch["pixels"] = torch.stack(fused_views, dim=1)

        return batch
