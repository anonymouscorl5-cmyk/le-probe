"""
PyTorch-style dataset over workspace_probe_bundle.pt for IG / Neuronpedia graphs.

Each item is one static hull probe with temporal history formed by repeating the snapshot.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from interpretability.lewm_experiment import VIEW_NAMES, ExperimentConfig


class WorkspaceProbeDataset:
    def __init__(
        self,
        bundle_path: str | Path,
        cfg: ExperimentConfig,
        *,
        pose_labels: dict[int, str] | None = None,
    ):
        self.cfg = cfg
        self.bundle_path = Path(bundle_path)
        data = torch.load(self.bundle_path, map_location="cpu", weights_only=False)

        self.probe_ids = self._as_np(data["probe_ids"], np.int64)
        self.ee_xyz = self._as_np(data["ee_achieved_xyz"], np.float64)
        cube = data.get("cube_xyz", data.get("cube_xyz_m"))
        self.cube_xyz = self._as_np(cube, np.float64)
        self.rgb = data["rgb"]  # N, V, H, W, 3 uint8
        self.states = data["state_norm"].float()
        self.cam_names = list(data.get("cam_names", VIEW_NAMES))
        self.skeleton = data.get("skeleton")
        self.pose_labels = pose_labels or {}
        self.n = int(self.rgb.shape[0])

        self._pid_to_index = {int(pid): i for i, pid in enumerate(self.probe_ids)}

    @staticmethod
    def _as_np(x, dtype):
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        return np.asarray(x, dtype=dtype)

    def __len__(self) -> int:
        return self.n

    def index_for_probe_id(self, probe_id: int) -> int:
        if int(probe_id) not in self._pid_to_index:
            raise KeyError(f"probe_id {probe_id} not in bundle")
        return self._pid_to_index[int(probe_id)]

    def probe_id_at(self, index: int) -> int:
        return int(self.probe_ids[index])

    def _build_pixels(self, index: int) -> torch.Tensor:
        """Return pixels [T, V, C, H, W]."""
        T = self.cfg.history_size
        rgb = self.rgb[index]  # V, H, W, 3
        px = rgb.permute(0, 3, 1, 2).float() / 255.0  # V, C, H, W

        if not self.cfg.multi_view:
            px = px[:1]

        if self.cfg.use_skeleton and self.skeleton is not None:
            sk = self.skeleton[index].permute(0, 3, 1, 2).float() / 255.0
            if sk.shape[-3] != 1:
                sk = sk.mean(dim=-3, keepdim=True)
            if not self.cfg.multi_view:
                sk = sk[:1]
            if sk.shape[-2:] != px.shape[-2:]:
                Bv, Csk, H, W = sk.shape[0], sk.shape[-3], sk.shape[-2], sk.shape[-1]
                flat = sk.reshape(Bv * Csk, 1, H, W)
                flat = F.interpolate(flat, size=(224, 224), mode="nearest")
                sk = flat.view(Bv, 1, 224, 224)
            px = torch.cat([px[..., :224, :224], sk[..., :224, :224]], dim=-3)

        pixels = px.unsqueeze(0).repeat(T, 1, 1, 1, 1)
        return pixels

    def __getitem__(self, index: int) -> dict[str, Any]:
        T = self.cfg.history_size
        actions = self.states[index].unsqueeze(0).repeat(T, 1)
        batch: dict[str, Any] = {
            "pixels": self._build_pixels(index),
            "action": actions,
            "probe_id": int(self.probe_ids[index]),
            "bundle_index": int(index),
            "ee_xyz": self.ee_xyz[index].tolist(),
        }
        if self.cfg.use_dino:
            # Placeholder anchors so subgoal attribution can run; replace when bundle stores DINO.
            V = self.cfg.num_views if self.cfg.multi_view else 1
            batch["dino_anchor"] = torch.zeros(T, V, 384)
            batch["phase_idx"] = torch.zeros(T, 1)
            batch["is_checkpoint"] = torch.zeros(T, 1)
        return batch
