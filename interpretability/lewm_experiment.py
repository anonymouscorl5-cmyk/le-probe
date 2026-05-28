"""
Shared LeWM experiment flags and data/model helpers for the interpretability pipeline.

Mirrors inference flags from lewm/lewm_server.py:
  --multi_view, --use_skeleton, --use_dino
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lewm.goal_mapper import GoalMapper
from lewm.lewm_data_plugin import LEWMDataPlugin
from lewm.skeleton.data import SkeletonDataPlugin

VIEW_NAMES = [
    "world_center",
    "world_left",
    "world_right",
    "world_top",
    "world_wrist",
]

PATCHES_PER_FRAME = 257  # CLS + 16x16
DEFAULT_HISTORY_SIZE = 3


class TraceHook:
    """Capture activations on GPU; export to CPU after forward (avoids per-layer sync)."""

    def __init__(self):
        self.output = None

    def __call__(self, module, input, output):
        val = output[0] if isinstance(output, tuple) else output
        self.output = val.detach()


@dataclass
class ExperimentConfig:
    multi_view: bool = False
    use_skeleton: bool = False
    use_dino: bool = False
    num_views: int = 1
    history_size: int = DEFAULT_HISTORY_SIZE
    fusion_type: str = "linear"
    cls_only: bool = False
    attribution_target: str = "reward"

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "multi_view": self.multi_view,
            "use_skeleton": self.use_skeleton,
            "use_dino": self.use_dino,
            "num_views": self.num_views,
            "history_size": self.history_size,
            "fusion_type": self.fusion_type,
            "cls_only": self.cls_only,
            "attribution_target": self.attribution_target,
        }

    @classmethod
    def from_metadata(cls, meta: Dict[str, Any]) -> "ExperimentConfig":
        exp = meta.get("experiment", meta)
        num_views = int(exp.get("num_views", 5 if exp.get("multi_view") else 1))
        return cls(
            multi_view=bool(exp.get("multi_view", False)),
            use_skeleton=bool(exp.get("use_skeleton", False)),
            use_dino=bool(exp.get("use_dino", False)),
            num_views=num_views,
            history_size=int(exp.get("history_size", DEFAULT_HISTORY_SIZE)),
            fusion_type=str(exp.get("fusion_type", "linear")),
            cls_only=bool(exp.get("cls_only", False)),
            attribution_target=str(exp.get("attribution_target", "reward")),
        )

    def encoder_tokens_per_moment(self) -> int:
        tokens = self.history_size * self.num_views
        if self.cls_only:
            return tokens
        return tokens * PATCHES_PER_FRAME

    def predictor_tokens_per_moment(self) -> int:
        return self.history_size


def add_experiment_args(
    parser: argparse.ArgumentParser,
    *,
    include_cls_only: bool = True,
    include_attribution_target: bool = False,
) -> argparse.ArgumentParser:
    """Add the standard LeWM experiment flags (same as lewm_server / harvest_manifold)."""
    parser.add_argument(
        "--multi_view",
        action="store_true",
        default=False,
        help="Use 5-camera late-fusion encoder",
    )
    parser.add_argument(
        "--use_skeleton",
        action="store_true",
        default=False,
        help="4th-channel skeletal prior (RGB + skeleton)",
    )
    parser.add_argument(
        "--use_dino",
        action="store_true",
        default=False,
        help="DINOv3 waypoint heads (requires skeleton pipeline data)",
    )
    if include_cls_only:
        parser.add_argument(
            "--cls_only",
            action="store_true",
            default=False,
            help="Harvest encoder activations as CLS tokens only (smaller disk)",
        )
    if include_attribution_target:
        parser.add_argument(
            "--attribution_target",
            type=str,
            choices=("reward", "subgoal"),
            default="reward",
            help="IG target: reward_head or DINO subgoal head",
        )
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    num_views = 5 if getattr(args, "multi_view", False) else 1
    return ExperimentConfig(
        multi_view=getattr(args, "multi_view", False),
        use_skeleton=getattr(args, "use_skeleton", False),
        use_dino=getattr(args, "use_dino", False),
        num_views=num_views,
        history_size=getattr(args, "history_size", DEFAULT_HISTORY_SIZE),
        fusion_type=getattr(args, "fusion", "linear"),
        cls_only=getattr(args, "cls_only", False),
        attribution_target=getattr(args, "attribution_target", "reward"),
    )


def resolve_dataset_root(dataset_repo: str) -> str:
    try:
        ds = LeRobotDataset(dataset_repo)
        return str(ds.root)
    except Exception as exc:
        raise RuntimeError(
            f"Dataset '{dataset_repo}' is not available locally. "
            "HF fallback is disabled for submission mode."
        ) from exc


def build_goal_mapper(
    model_path: str,
    dataset_root: str,
    cfg: ExperimentConfig,
):
    return GoalMapper(
        model_path=model_path,
        dataset_root=dataset_root,
        use_multi_view=cfg.multi_view,
        num_views=cfg.num_views,
        use_skeleton=cfg.use_skeleton,
        use_dino=cfg.use_dino,
    )


def build_goal_mapper_for_probes(
    model_path: str,
    cfg: ExperimentConfig,
) -> GoalMapper:
    """
    GoalMapper for static workspace probes (IG / Neuronpedia).

    Does not require gr1_pickup_grasp: pixels come from workspace_probe_bundle.pt
    and preprocessing uses ImageNet transforms only (same as harvest_goals.py).
    """
    return GoalMapper(
        model_path=model_path,
        dataset_root=None,
        use_multi_view=cfg.multi_view,
        num_views=cfg.num_views,
        use_skeleton=cfg.use_skeleton,
        use_dino=cfg.use_dino,
    )


def data_keys_to_load(cfg: ExperimentConfig) -> list:
    keys = ["action"]
    if cfg.multi_view:
        keys.extend(VIEW_NAMES)
    else:
        keys.append("pixels")
    return keys


def build_data_plugin(dataset_repo: str, cfg: ExperimentConfig, num_steps: int):
    keys = data_keys_to_load(cfg)
    if cfg.use_skeleton:
        plugin = SkeletonDataPlugin(
            repo_id=dataset_repo,
            keys_to_load=keys,
            num_steps=num_steps,
            use_multi_view=cfg.multi_view,
        )
    else:
        plugin = LEWMDataPlugin(
            repo_id=dataset_repo,
            keys_to_load=keys,
            num_steps=num_steps,
            use_multi_view=cfg.multi_view,
        )
    plugin.clear_cache()
    return plugin


def prepare_pixels_6d(
    batch: Dict[str, torch.Tensor],
    mapper,
    device: torch.device,
    cfg: ExperimentConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a dataloader batch to (pixels, actions) with shape:
      pixels: (B, T, V, C, H, W)
      actions: (B, T, action_dim)
    """
    raw_pixels = batch["pixels"].to(device)
    actions = batch["action"].to(device)

    # Fused PT cache (cache_fused_dataset.py): pixels are already [..., 4, H, W] and
    # normalized in SkeletonDataPlugin.tiled_transform_wrapper — no skeletons_raw key.
    if cfg.use_skeleton and raw_pixels.shape[-3] == 4:
        if raw_pixels.ndim == 5:
            pixels = raw_pixels.unsqueeze(0)
        elif raw_pixels.ndim == 6:
            pixels = raw_pixels
        else:
            raise ValueError(
                f"Unexpected fused skeleton pixels shape: {tuple(raw_pixels.shape)}"
            )
        if pixels.dtype != torch.float32:
            pixels = pixels.float()
        if torch.isnan(actions).any():
            actions = torch.nan_to_num(actions, 0.0)
        return pixels, actions

    raw_skel_1ch = None
    if cfg.use_skeleton:
        raw_skel = batch["skeletons_raw"].to(device)
        raw_skel_1ch = raw_skel.float().mean(dim=-3, keepdim=True).byte()

    if raw_pixels.ndim == 5:
        if not cfg.multi_view:
            pixels_6d = raw_pixels.unsqueeze(2)
            if raw_skel_1ch is not None:
                skel_6d = raw_skel_1ch.unsqueeze(2)
        else:
            pixels_6d = raw_pixels.unsqueeze(1)
            if raw_skel_1ch is not None:
                skel_6d = raw_skel_1ch.unsqueeze(1)
    else:
        pixels_6d = raw_pixels
        if raw_skel_1ch is not None:
            skel_6d = raw_skel_1ch

    B, T, V, C, H, W = pixels_6d.shape
    raw_pixels_flat = pixels_6d.reshape(B * T * V, C, H, W)
    processed_pixels = mapper.transform({"pixels": raw_pixels_flat})["pixels"]
    pixels = processed_pixels.view(B, T, V, C, 224, 224)

    if cfg.use_skeleton:
        skel_flat = skel_6d.reshape(B * T * V, 1, H, W)
        if skel_flat.shape[-2:] != (224, 224):
            skel_flat_resized = F.interpolate(
                skel_flat.float(), size=(224, 224), mode="nearest"
            ).byte()
        else:
            skel_flat_resized = skel_flat
        skel_final = skel_flat_resized.view(B, T, V, 1, 224, 224)
        pixels = torch.cat([pixels, skel_final.to(pixels.dtype)], dim=-3)

    if torch.isnan(actions).any():
        actions = torch.nan_to_num(actions, 0.0)

    return pixels, actions


def forward_harvest(
    model,
    pixels: torch.Tensor,
    actions: torch.Tensor,
    cfg: ExperimentConfig,
    batch: Optional[Dict[str, torch.Tensor]] = None,
):
    """Run encode + predict (+ DINO path when enabled)."""
    info = model.encode({"pixels": pixels, "action": actions})
    model.predict(info["emb"], info["act_emb"])

    if cfg.use_dino and batch is not None and "dino_anchor" in batch:
        phi = batch["dino_anchor"].to(pixels.device)
        model.project_dino(phi)
        if "phase_idx" in batch:
            model.predict_subgoal(info["emb"], batch["phase_idx"].to(pixels.device))


def ghost_trace_batch(
    device: torch.device,
    cfg: ExperimentConfig,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
    """Synthetic batch for pre-flight hook verification."""
    B, T, V = 1, cfg.history_size, cfg.num_views
    C = 4 if cfg.use_skeleton else 3
    pixels = torch.randn(B, T, V, C, 224, 224, device=device)
    actions = torch.zeros(B, T, 32, device=device)
    extra: Dict[str, torch.Tensor] = {}
    if cfg.use_dino:
        extra["dino_anchor"] = torch.randn(B, T, V, 384, device=device)
        extra["phase_idx"] = torch.zeros(B, T, 1, device=device)
    return pixels, actions, extra if extra else None


def flatten_activation(
    acts,
    layer_id: str,
    cfg: ExperimentConfig,
) -> np.ndarray:
    """Flatten hook output; optional CLS-only for encoder layers."""
    if torch.is_tensor(acts):
        t = acts
        if t.ndim == 3 and cfg.cls_only and layer_id.startswith("encoder"):
            t = t[:, 0, :]
        elif t.ndim != 2:
            t = t.reshape(-1, t.shape[-1])
        return t.detach().cpu().numpy().astype(np.float16, copy=False)

    if acts.ndim == 2:
        flat = acts
    elif acts.ndim == 3:
        if cfg.cls_only and layer_id.startswith("encoder"):
            flat = acts[:, 0, :]
        else:
            flat = acts.reshape(-1, acts.shape[-1])
    else:
        flat = acts.reshape(-1, acts.shape[-1])
    return flat.astype(np.float16)


def discover_layer_ids(model) -> Dict[str, str]:
    """
    Map layer_id -> module path. Deduplicates by layer_id (prefers backbone path).
    """
    found: Dict[str, str] = {}
    for name, _module in model.named_modules():
        parts = name.split(".")
        if (
            len(parts) > 1
            and (parts[-2] == "layer" or parts[-2] == "layers")
            and parts[-1].isdigit()
        ):
            if "predictor" in name:
                component = "predictor"
            elif "backbone" in name or name.startswith("encoder."):
                component = "encoder"
            else:
                continue
            layer_id = f"{component}_L{parts[-1]}"
            if layer_id not in found or "backbone" in name:
                found[layer_id] = name
    return found


def resolve_layer_module(model, layer_id: str):
    """Resolve encoder_L* / predictor_L* to the hooked nn.Module."""
    component, idx_str = layer_id.split("_L")
    idx = int(idx_str.split("_")[0])
    if component == "encoder":
        enc = model.encoder
        if hasattr(enc, "backbone") and hasattr(enc.backbone, "encoder"):
            return enc.backbone.encoder.layer[idx]
        if hasattr(enc, "encoder"):
            return enc.encoder.layer[idx]
    elif component == "predictor":
        return model.predictor.transformer.layers[idx]
    return None


def decode_token_index(
    token_idx: int,
    cfg: ExperimentConfig,
) -> Tuple[int, int, int, int]:
    """
    Map a flat token index within one dataloader sample to:
      (frame_offset, view_idx, patch_token_idx)
    patch_token_idx: 0 = CLS, 1-256 = patches
    """
    block = cfg.num_views * PATCHES_PER_FRAME
    frame_offset = token_idx // block
    rem = token_idx % block
    view_idx = rem // PATCHES_PER_FRAME
    patch_token_idx = rem % PATCHES_PER_FRAME
    return frame_offset, view_idx, patch_token_idx


def sample_rgb_frame(
    sample: Dict[str, Any],
    cfg: ExperimentConfig,
    view_idx: int = 0,
    time_idx: int = 0,
) -> torch.Tensor:
    """Extract RGB [C,H,W] uint8/float tensor from a dataset sample."""
    if "pixels" in sample:
        px = sample["pixels"]
        if px.ndim == 5:
            # [T, V, C, H, W]
            img = px[time_idx, view_idx]
        elif px.ndim == 4:
            if px.shape[0] == cfg.num_views and cfg.multi_view:
                img = px[view_idx]
            else:
                img = px[time_idx]
        else:
            img = px
    else:
        key = f"observation.images.{VIEW_NAMES[view_idx]}"
        img = sample.get(key, sample.get("observation.images.world_center"))
        if img is None:
            raise KeyError("No image tensor in sample")
        if img.ndim == 4:
            img = img[time_idx]
    if img.shape[0] == 4:
        img = img[:3]
    return img
