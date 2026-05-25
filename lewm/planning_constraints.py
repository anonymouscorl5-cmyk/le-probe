"""
MPC action feasibility checks applied **before** LeWM rollout.

- Default (no --task-workspace): CEM draws joints 16–19 inside the buffered norm
  envelope; ``right_arm_norm_feasible_mask`` remains a safety check before LeWM.
- With --task-workspace: unconstrained Gaussian on all dims; FK hull rejects infeasible.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from lewm.task_workspace import INFEASIBLE_COST, TaskWorkspaceMPCConstraint

# Hardcoded right arm and waist ranges
# Skeletal Priors: [-0.6, 0, -0.5, -0.5, -0.4, -0.3, -0.4], [0.2, 1, 0.5, 0.5, 0.4, 0.7, 0.4]
# DINO Waypoints: [-0.8, 0.4, -0.3, -0.4, -0.4, -0.2, -0.4], [-0.2, 1, 0.3, 0.4, 0.4, 0.5, 0.4]
RIGHT_ARM_NORM_SLICE = list(range(16, 20)) + list(range(29, 32))
RIGHT_ARM_NORM_MIN = np.array([-0.8, 0.4, -0.3, -0.4, -0.4, -0.2, -0.4], dtype=np.float64)
RIGHT_ARM_NORM_MAX = np.array([-0.2, 1, 0.3, 0.4, 0.4, 0.5, 0.4], dtype=np.float64)

# Extra CEM samples when filtering aggressively (no task-workspace gate).
CEM_NUM_SAMPLES_DEFAULT = 800
CEM_NUM_SAMPLES_HARD_ARM_GATE = 8000


def _right_arm_bounds(
    device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    lo = torch.as_tensor(RIGHT_ARM_NORM_MIN, device=device, dtype=dtype)
    hi = torch.as_tensor(RIGHT_ARM_NORM_MAX, device=device, dtype=dtype)
    return lo, hi


def constrain_right_arm_norm_actions(actions: torch.Tensor) -> torch.Tensor:
    """Clamp joints 16–19 into ``RIGHT_ARM_NORM_MIN`` / ``RIGHT_ARM_NORM_MAX``."""
    lo, hi = _right_arm_bounds(actions.device, actions.dtype)
    out = actions.clone()
    out[..., RIGHT_ARM_NORM_SLICE] = out[..., RIGHT_ARM_NORM_SLICE].clamp(
        min=lo, max=hi
    )
    return out


def constrain_right_arm_cem_mean(mean: torch.Tensor) -> torch.Tensor:
    """Keep CEM mean inside the right-arm envelope (after elite aggregation)."""
    return constrain_right_arm_norm_actions(mean)


def sample_cem_plan_candidates(
    batch_mean: torch.Tensor,
    batch_var: torch.Tensor,
    *,
    num_samples: int,
    generator: torch.Generator,
    constrain_right_arm: bool = True,
) -> torch.Tensor:
    """
    Gaussian CEM candidates: ``N(mean, var)`` with candidate 0 pinned to ``mean``.

    When ``constrain_right_arm`` (default, no task-workspace path), joints 16–19 are
    clamped into the norm envelope after each draw so the gate should not reject on
    right-arm bounds alone.
    """
    device = batch_mean.device
    dtype = batch_mean.dtype
    b, horizon, action_dim = batch_mean.shape
    candidates = torch.randn(
        b,
        num_samples,
        horizon,
        action_dim,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    candidates = candidates * batch_var.unsqueeze(1) + batch_mean.unsqueeze(1)
    candidates[:, 0] = batch_mean
    if constrain_right_arm:
        candidates = constrain_right_arm_norm_actions(candidates)
    return candidates


def freeze_and_clamp_actions(
    actions: torch.Tensor, frozen_pose: torch.Tensor
) -> torch.Tensor:
    """Freeze left arm + head (0–15); clamp active joints to [-1, 1]. No joint remap.

    ``frozen_pose`` may be ``(32,)`` (broadcast) or ``(N, 32)`` with ``N == actions.shape[0]``
    (one row per flattened CEM sample).
    """
    out = actions.clone()
    if frozen_pose.ndim == 1:
        out[..., 0:16] = frozen_pose[0:16]
    elif frozen_pose.ndim == 2:
        if frozen_pose.shape[0] != actions.shape[0]:
            raise ValueError(
                f"frozen_pose rows {frozen_pose.shape[0]} != action rows {actions.shape[0]}"
            )
        if actions.ndim == 3:
            out[:, :, 0:16] = frozen_pose[:, None, 0:16]
        else:
            out[..., 0:16] = frozen_pose[..., 0:16]
    else:
        raise ValueError(
            f"frozen_pose must be (32,) or (N, 32), got {tuple(frozen_pose.shape)}"
        )
    return torch.clamp(out, -1.0, 1.0)


def right_arm_norm_feasible_mask(plan_norm: np.ndarray) -> np.ndarray:
    """
    True when every plan step keeps joints 17–20 inside the buffered norm envelope.

    plan_norm: (N, T, 32) or (N, 32) — same normalized protocol space as CEM / LeWM.
    Bounds per joint are subsets of [-1, 1] derived from RIGHT_ARM_WIRE_* and
    gr1_config joint limits (see RIGHT_ARM_NORM_MIN / RIGHT_ARM_NORM_MAX).
    """
    plan_norm = np.asarray(plan_norm, dtype=np.float64)
    if plan_norm.ndim == 2:
        plan_norm = plan_norm[:, np.newaxis, :]

    arm = plan_norm[..., RIGHT_ARM_NORM_SLICE]  # (N, T, 4)
    in_range = (arm >= RIGHT_ARM_NORM_MIN) & (arm <= RIGHT_ARM_NORM_MAX)
    return in_range.all(axis=(1, 2))


# Back-compat alias
right_arm_wire_feasible_mask = right_arm_norm_feasible_mask


def task_workspace_feasible_mask(
    constraint: TaskWorkspaceMPCConstraint,
    wire32_rad: np.ndarray,
    plan_norm: np.ndarray,
    *,
    check_final_only: bool = True,
    cube_xyz: Optional[np.ndarray] = None,
    relaxed_eps_factor: float = 100.0,
) -> np.ndarray:
    """True when plan stays inside the fixed task hull (FK gate)."""
    plan_norm = np.asarray(plan_norm, dtype=np.float64)
    if plan_norm.ndim == 2:
        plan_norm = plan_norm[:, np.newaxis, :]

    feasible, violations = constraint.feasible_mask_batch(
        wire32_rad,
        plan_norm,
        check_all_steps=not check_final_only,
        cube_xyz=cube_xyz,
    )
    n_feas = int(feasible.sum())
    if n_feas == 0 and relaxed_eps_factor > 1.0:
        eps = constraint.feasibility_eps * relaxed_eps_factor
        feasible = violations <= eps
    return feasible


def scatter_infeasible_costs(
    total: int,
    feasible: np.ndarray,
    feasible_costs: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Fill INFEASIBLE_COST everywhere except feasible indices."""
    out = torch.full((total,), INFEASIBLE_COST, device=device, dtype=dtype)
    if feasible.any():
        idx = torch.from_numpy(np.nonzero(feasible)[0]).to(
            device=device, dtype=torch.long
        )
        out[idx] = feasible_costs.to(device=device, dtype=dtype)
    return out
