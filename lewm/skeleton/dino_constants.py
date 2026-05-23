"""Shared DINO waypoint layout (must match SkeletonDataPlugin view order)."""

from __future__ import annotations

from typing import Optional

import torch

DINO_VIEW_KEYS = (
    "world_center",
    "world_left",
    "world_right",
    "world_top",
    "world_wrist",
)

# 0-based frame indices for phase-end checkpoints (frames 8, 16, 24, 32)
DINO_PHASE_CHECKPOINT_FRAMES = (7, 15, 23, 31)

DINO_DIM = 384
NUM_DINO_PHASES = 4


def num_dino_views() -> int:
    return len(DINO_VIEW_KEYS)


def dino_waypoints_shape(num_views: Optional[int] = None) -> tuple[int, int, int]:
    v = num_views if num_views is not None else num_dino_views()
    return (NUM_DINO_PHASES, v, DINO_DIM)


def validate_dino_waypoints(waypoints: torch.Tensor) -> torch.Tensor:
    """Require multi-view cache shape [4, V, 384]."""
    expected = dino_waypoints_shape()
    if waypoints.shape != expected:
        raise ValueError(
            f"Expected dino_waypoints shape {expected}, got {tuple(waypoints.shape)}. "
            "Re-run dataset/skeleton/generate_dino_priors.py."
        )
    return waypoints


def zeros_dino_waypoints(num_views: Optional[int] = None) -> torch.Tensor:
    return torch.zeros(dino_waypoints_shape(num_views))
