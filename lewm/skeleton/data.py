import time
import os
import csv
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
        for view in self.base_views:
            self.key_map[view] = f"observation.images.{view}_tiled"

        # 3. Wrap transform to handle splitting before standard transforms run
        self.orig_transform = self.transform
        self.transform = self.tiled_transform_wrapper

        # 4. Profiling Setup
        self.profile_csv = "dataloader_profile.csv"

        print(
            f"🚀 Tiled SkeletonDataPlugin initialized (Wrapped Transform) for views: {self.base_views}"
        )

    def _log_profile(self, idx, metrics_dict):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        file_exists = (
            os.path.exists(self.profile_csv) and os.path.getsize(self.profile_csv) > 0
        )

        with open(self.profile_csv, "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["timestamp", "worker_id", "idx"]
                + sorted(metrics_dict.keys()),
            )
            if not file_exists:
                writer.writeheader()

            row = {"timestamp": time.time(), "worker_id": worker_id, "idx": idx}
            row.update(metrics_dict)
            writer.writerow(row)

    def tiled_transform_wrapper(self, nested_batch):
        """
        Intercepts the transform call to split tiled videos and handle 4-channel fusion.
        """
        t0 = time.perf_counter()
        skeletons = {}

        # A. Split Tiled Videos into RGB + Skeleton
        t_split_s = time.perf_counter()
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
                rgb = tensor[..., :mid].clone()
                skeletons[view] = tensor[..., mid:].clone()  # Raw uint8 [T, 3, H, W]

                # 2. Store RGB back
                d = nested_batch
                for p in path[:-1]:
                    d = d[p]
                d[path[-1]] = rgb
        t_split_e = time.perf_counter()

        t_orig_s = time.perf_counter()
        # B. Run Original Transforms (Resizing, Normalization, etc. on RGB)
        if self.orig_transform:
            nested_batch = self.orig_transform(nested_batch)
        t_orig_e = time.perf_counter()

        # C. Post-Transform: Attach raw skeletons for GPU fusion
        t_attach_s = time.perf_counter()
        for view, skel in skeletons.items():
            nested_batch[f"{view}_skel_raw"] = skel
        t_attach_e = time.perf_counter()

        # Attach timing info to batch for __getitem__ to log
        if "_profile" not in nested_batch:
            nested_batch["_profile"] = {}
        nested_batch["_profile"]["split_tiled_ms"] = (t_split_e - t_split_s) * 1000
        nested_batch["_profile"]["orig_trans_ms"] = (t_orig_e - t_orig_s) * 1000
        nested_batch["_profile"]["attach_skel_ms"] = (t_attach_e - t_attach_s) * 1000

        return nested_batch

    def __getitem__(self, idx):
        start_time = time.perf_counter()

        # 1. Base Loading (Video Decoding)
        batch = super().__getitem__(idx)
        load_time = time.perf_counter()

        # 2. Multi-view Stacking
        stack_start = time.perf_counter()
        if self.use_multi_view:
            fused_views = []
            fused_skels = []
            for vn in self.base_views:
                # RGB Views
                key = f"observation.images.{vn}"
                if key in batch:
                    fused_views.append(batch[key])
                elif vn in batch:
                    fused_views.append(batch[vn])

                # Raw Skeleton Views
                skel_key = f"{vn}_skel_raw"
                if skel_key in batch:
                    fused_skels.append(batch[skel_key])
                    del batch[skel_key]

            if fused_views:
                batch["pixels"] = torch.stack(fused_views, dim=1)

            if fused_skels:
                batch["skeletons_raw"] = torch.stack(fused_skels, dim=1)
        stack_end = time.perf_counter()

        # 3. Logging
        total_time_ms = (time.perf_counter() - start_time) * 1000

        # Collect all metrics from nested _profile
        metrics = batch.get("_profile", {}).copy()
        metrics["super_getitem_total_ms"] = (load_time - start_time) * 1000
        metrics["multiview_stack_ms"] = (stack_end - stack_start) * 1000
        metrics["TOTAL_getitem_ms"] = total_time_ms

        self._log_profile(idx, metrics)

        # Cleanup internal profile keys
        if "_profile" in batch:
            del batch["_profile"]

        return batch
