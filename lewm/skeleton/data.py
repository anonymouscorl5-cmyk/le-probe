import torch
import torch.nn.functional as F
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
        for view in self.base_views:
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

        # A. Split Tiled Videos into RGB + Skeleton
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

                # 2. Resize RGB immediately on Worker (CPU)
                # This reduces tensor size from 480x480 -> 224x224 (4.5x reduction)
                # We use F.interpolate because it's generally faster for raw tensors
                if rgb_raw.shape[-2:] != (self.img_size, self.img_size):
                    # F.interpolate expects [B, C, H, W], we have [T, C, H, W]
                    rgb = torch.nn.functional.interpolate(
                        rgb_raw.float(),
                        size=(self.img_size, self.img_size),
                        mode="bilinear",
                        align_corners=False,
                    ).byte()
                else:
                    rgb = rgb_raw

                # 3. Store RGB back
                d = nested_batch
                for p in path[:-1]:
                    d = d[p]
                d[path[-1]] = rgb

                # 4. Store Raw Skeleton (Keep at original res or downsample?)
                # If we keep at 480, it's 3x more memory.
                # Let's downsample to 224 here too, since the GPU-side resizing
                # from 480->224 on the CPU is what's slow.
                # If we do it here, we don't need GPU resizing!
                if skel_raw.shape[-2:] != (self.img_size, self.img_size):
                    skeletons[view] = torch.nn.functional.interpolate(
                        skel_raw.float(),
                        size=(self.img_size, self.img_size),
                        mode="nearest",  # Use nearest for masks/skeletons
                    ).byte()
                else:
                    skeletons[view] = skel_raw
        # B. Run Original Transforms (Normalization, etc. - RESIZE will be a no-op now!)
        if self.orig_transform:
            nested_batch = self.orig_transform(nested_batch)

        # C. Post-Transform: Attach raw skeletons for GPU fusion
        for view, skel in skeletons.items():
            nested_batch[f"{view}_skel_raw"] = skel

        return nested_batch

    def __getitem__(self, idx):
        # 1. Base Loading (Video Decoding)
        batch = super().__getitem__(idx)

        # 2. Multi-view Stacking (Micro-optimized)
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

        return batch
