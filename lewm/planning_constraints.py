"""
MPC action feasibility checks applied **before** LeWM rollout.

- Default (no --task-workspace): reject CEM samples whose right-arm joints
  (indices 17–20) leave a buffered envelope in **normalized** [-1, 1] protocol space.
- With --task-workspace: reject samples whose FK fingertip leaves the fixed hull.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from lewm.task_workspace import INFEASIBLE_COST, TaskWorkspaceMPCConstraint

# Buffered right-arm norm ranges from pre-polytope LeWM runs (b1b95bf, May 2026).
RIGHT_ARM_NORM_SLICE = slice(16, 20)
RIGHT_ARM_NORM_MIN = np.array([-1, 0, -0.15, -0.5], dtype=np.float64)
RIGHT_ARM_NORM_MAX = np.array([0, 1, 0.15, 0], dtype=np.float64)

# Extra CEM samples when filtering aggressively (no task-workspace gate).
CEM_NUM_SAMPLES_DEFAULT = 800
CEM_NUM_SAMPLES_HARD_ARM_GATE = 8000


def freeze_and_clamp_actions(
    actions: torch.Tensor, frozen_pose: torch.Tensor
) -> torch.Tensor:
    """Freeze left arm + head (0–15); clamp active joints to [-1, 1]. No joint remap."""
    out = actions.clone()
    out[..., 0:16] = frozen_pose[..., 0:16]
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
