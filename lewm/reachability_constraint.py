"""
Reachable-workspace constraints for LeWM CEM (task-space polytope, no 17–20 remap).
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import mujoco
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gr1_config import COMPACT_WIRE_JOINTS, SCENE_PATH
from gr1_protocol import StandardScaler
from dataset.gr1_reachability import (
    EE_BODY,
    GR1ReachabilityEngine,
    ReachabilityConfig,
    teleop_reachability_config,
)

# Penalty scale vs reward (~50×) and smoothness (100×) in GoalMapper.get_cost
DEFAULT_REACH_GAMMA = 75.0
# One FK pass per CEM sample (final plan step) keeps 800-sample plans tractable
PENALIZE_FINAL_STEP_ONLY = True


def ensure_halfspaces(poly) -> tuple[np.ndarray, np.ndarray]:
    if getattr(poly, "H", None) is None or getattr(poly, "d", None) is None:
        poly.find_halfplanes()
    H = np.asarray(poly.H, dtype=np.float64)
    d = np.asarray(poly.d, dtype=np.float64).reshape(-1)
    return H, d


def ee_halfspace_violation(ee: np.ndarray, H: np.ndarray, d: np.ndarray) -> float:
    """Squared slack outside polytope {x | Hx <= d}."""
    slack = H @ np.asarray(ee, dtype=np.float64).reshape(3) - d
    return float(np.sum(np.maximum(0.0, slack) ** 2))


def wire32_to_qpos(
    model: mujoco.MjModel, qpos: np.ndarray, wire32_rad: np.ndarray
) -> None:
    for i, name in enumerate(COMPACT_WIRE_JOINTS):
        j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j_id != -1:
            qpos[model.jnt_qposadr[j_id]] = float(wire32_rad[i])


class ReachabilityMPCConstraint:
    """Polytope from current pose + FK penalties on normalized CEM plans."""

    def __init__(
        self,
        scene_path: str = SCENE_PATH,
        cfg: Optional[ReachabilityConfig] = None,
        gamma: float = DEFAULT_REACH_GAMMA,
    ):
        self.cfg = cfg or teleop_reachability_config()
        self.gamma = gamma
        self.engine = GR1ReachabilityEngine(scene_path=scene_path)
        self.scaler = StandardScaler()
        self._poly = None
        self._H: Optional[np.ndarray] = None
        self._d: Optional[np.ndarray] = None
        self.ee_body_id = mujoco.mj_name2id(
            self.engine.model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY
        )

    def compute_polytope(self, wire32_rad: np.ndarray):
        self.engine.set_baseline_from_wire32(np.asarray(wire32_rad, dtype=np.float64))
        self._poly = self.engine.compute(cfg=self.cfg)
        self._H, self._d = ensure_halfspaces(self._poly)
        return self._poly

    def get_halfspaces(self) -> tuple[np.ndarray, np.ndarray]:
        if self._H is None or self._d is None:
            raise RuntimeError("Call compute_polytope() before get_halfspaces()")
        return self._H, self._d

    def _fk_ee_after_plan_step(
        self, qpos_baseline: np.ndarray, plan_norm_step: np.ndarray
    ) -> np.ndarray:
        q = np.array(qpos_baseline, dtype=np.float64, copy=True)
        wire_rad = self.scaler.unscale_action(plan_norm_step)
        wire32_to_qpos(self.engine.model, q, wire_rad)
        self.engine.data.qpos[:] = q
        mujoco.mj_forward(self.engine.model, self.engine.data)
        return self.engine.data.xpos[self.ee_body_id].copy()

    def plan_violation(
        self,
        wire32_rad: np.ndarray,
        plan_norm: np.ndarray,
        *,
        final_step_only: bool = PENALIZE_FINAL_STEP_ONLY,
    ) -> float:
        if self._H is None or self._d is None:
            return 0.0

        self.engine.set_baseline_from_wire32(np.asarray(wire32_rad, dtype=np.float64))
        q0 = self.engine._baseline_qpos.copy()

        plan_norm = np.asarray(plan_norm, dtype=np.float64)
        if plan_norm.ndim == 1:
            plan_norm = plan_norm.reshape(1, -1)

        total = 0.0
        steps = (
            [plan_norm.shape[0] - 1] if final_step_only else range(plan_norm.shape[0])
        )
        for t in steps:
            ee = self._fk_ee_after_plan_step(q0, plan_norm[t])
            total += ee_halfspace_violation(ee, self._H, self._d)
        return total

    def plan_penalty_batch(
        self,
        wire32_rad: np.ndarray,
        plan_norm_bs: np.ndarray,
        H: Optional[np.ndarray] = None,
        d: Optional[np.ndarray] = None,
        *,
        final_step_only: bool = PENALIZE_FINAL_STEP_ONLY,
    ) -> np.ndarray:
        """Per-sample violation costs (unnormalized); multiply by gamma in caller."""
        H_use = np.asarray(H if H is not None else self._H, dtype=np.float64)
        d_use = np.asarray(d if d is not None else self._d, dtype=np.float64).reshape(
            -1
        )
        plan_norm_bs = np.asarray(plan_norm_bs, dtype=np.float64)
        if plan_norm_bs.ndim == 2:
            plan_norm_bs = plan_norm_bs[:, np.newaxis, :]
        n = plan_norm_bs.shape[0]
        out = np.zeros(n, dtype=np.float64)
        old_H, old_d = self._H, self._d
        self._H, self._d = H_use, d_use
        try:
            for s in range(n):
                out[s] = self.plan_violation(
                    wire32_rad, plan_norm_bs[s], final_step_only=final_step_only
                )
        finally:
            self._H, self._d = old_H, old_d
        return out
