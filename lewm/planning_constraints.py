"""
MPC action feasibility checks applied **before** LeWM rollout.

- Default (no --task-workspace): reject CEM samples whose right-arm wire joints
  (indices 17–20) leave the buffered training envelope (no remap into that box).
- With --task-workspace: reject samples whose FK fingertip leaves the fixed hull.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from gr1_protocol import StandardScaler
from lewm.task_workspace import INFEASIBLE_COST, TaskWorkspaceMPCConstraint

# Buffered right-arm wire ranges from pre-polytope LeWM runs (b1b95bf, May 2026).
# Dataset stats + 20% margin; used for rejection only — never written into actions.
RIGHT_ARM_WIRE_SLICE = slice(17, 21)
RIGHT_ARM_WIRE_MIN = np.array([-0.312, -0.098, -0.156, -0.098], dtype=np.float64)
RIGHT_ARM_WIRE_MAX = np.array([1.172, 0.098, 0.236, 0.098], dtype=np.float64)

# Extra CEM samples when filtering aggressively (no task-workspace gate).
CEM_NUM_SAMPLES_DEFAULT = 800
CEM_NUM_SAMPLES_HARD_ARM_GATE = 1600


def freeze_and_clamp_actions(
    actions: torch.Tensor, frozen_pose: torch.Tensor
) -> torch.Tensor:
    """Freeze left arm + head (0–15); clamp active joints to [-1, 1]. No joint remap."""
    out = actions.clone()
    out[..., 0:16] = frozen_pose[..., 0:16]
    return torch.clamp(out, -1.0, 1.0)


def right_arm_wire_feasible_mask(
    plan_norm: np.ndarray,
    scaler: Optional[StandardScaler] = None,
) -> np.ndarray:
    """
    True when every plan step keeps wire joints 17–20 inside RIGHT_ARM_WIRE_*.

    plan_norm: (N, T, 32) or (N, 32) normalized protocol actions (post clamp).
    """
    plan_norm = np.asarray(plan_norm, dtype=np.float64)
    if plan_norm.ndim == 2:
        plan_norm = plan_norm[:, np.newaxis, :]

    scaler = scaler or StandardScaler()
    n = plan_norm.shape[0]
    feasible = np.ones(n, dtype=bool)

    for i in range(n):
        for t in range(plan_norm.shape[1]):
            wire = scaler.unscale_action(plan_norm[i, t])
            arm = wire[RIGHT_ARM_WIRE_SLICE]
            if np.any(arm < RIGHT_ARM_WIRE_MIN) or np.any(arm > RIGHT_ARM_WIRE_MAX):
                feasible[i] = False
                break

    return feasible


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
