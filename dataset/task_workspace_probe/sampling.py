"""Sample workspace probes: joint-space (reachable) or hull rejection (EE targets)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from gr1_config import COMPACT_WIRE_JOINTS, JOINT_LIMITS_MAX, JOINT_LIMITS_MIN
from gr1_protocol import StandardScaler
from lewm.task_workspace import build_task_workspace_polytope

if TYPE_CHECKING:
    from dataset.task_workspace_probe.probe_sim import ProbeSimulator

# Matches ``wild_reset`` / grasp dataset: right arm + hand + waist (indices 16–31).
WILD_MOVABLE_INDICES = list(range(16, 32))

JointSampleMode = Literal["wild", "ik"]


def load_ik_whitelist_indices() -> list[int]:
    """Wire32 indices from ``ik_joints.txt`` (teleop default active sliders)."""
    ik_path = Path(REPO_DIR) / "ik_joints.txt"
    names: list[str] = []
    with ik_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            names.append(line.split("#")[0].strip())
    indices: list[int] = []
    for name in names:
        if name in COMPACT_WIRE_JOINTS:
            indices.append(COMPACT_WIRE_JOINTS.index(name))
    return sorted(indices)


def movable_indices_for_mode(mode: JointSampleMode) -> list[int]:
    if mode == "wild":
        return list(WILD_MOVABLE_INDICES)
    if mode == "ik":
        return load_ik_whitelist_indices()
    raise ValueError(f"Unknown joint sample mode: {mode!r}")


def is_inside_hull(
    p: np.ndarray,
    H: np.ndarray,
    d: np.ndarray,
    eps: float = 1e-6,
) -> bool:
    p = np.asarray(p, dtype=np.float64).reshape(3)
    slack = H @ p - d
    return bool(np.max(slack) <= eps)


def sample_points_in_hull(
    n: int,
    rng: np.random.Generator,
    *,
    eps: float = 1e-6,
    max_attempts_factor: int = 8,
) -> tuple[np.ndarray, dict]:
    """
    Rejection sample ``n`` points inside the task hull.

    Returns (n, 3) world-frame XYZ and stats dict.
    """
    poly = build_task_workspace_polytope()
    corners = poly.corner_points
    lo = corners.min(axis=0)
    hi = corners.max(axis=0)
    H, d = poly.H, poly.d

    accepted: list[np.ndarray] = []
    attempts = 0
    max_attempts = max(n * max_attempts_factor, n + 100)

    while len(accepted) < n and attempts < max_attempts:
        attempts += 1
        p = rng.uniform(lo, hi)
        if is_inside_hull(p, H, d, eps=eps):
            accepted.append(p)

    if len(accepted) < n:
        raise RuntimeError(
            f"Only sampled {len(accepted)}/{n} hull points in {attempts} attempts. "
            "Increase max_attempts_factor or check hull."
        )

    pts = np.stack(accepted, axis=0)
    stats = {
        "attempts": attempts,
        "accept_rate": len(accepted) / max(attempts, 1),
        "bbox_lo": lo.tolist(),
        "bbox_hi": hi.tolist(),
    }
    return pts, stats


def sample_joint_space_configs(
    n: int,
    rng: np.random.Generator,
    sim: ProbeSimulator,
    *,
    movable_indices: list[int],
    hull_filter: bool = True,
    hull_eps: float = 1e-4,
    max_attempts_factor: int = 32,
) -> tuple[list[dict], dict]:
    """
    Sample ``wire32`` on movable DoFs, FK to index tip; optional hull rejection.

    Every accepted sample is kinematically reachable (no IK).
    """
    sim.reset_probe_scene(lock_posture=True)
    base_wire32 = np.asarray(sim.qpos_to_action_32(sim.data.qpos), dtype=np.float64)
    scaler = StandardScaler()
    lmin = np.asarray(JOINT_LIMITS_MIN, dtype=np.float64)
    lmax = np.asarray(JOINT_LIMITS_MAX, dtype=np.float64)

    accepted: list[dict] = []
    attempts = 0
    hull_reject = 0
    max_attempts = max(n * max_attempts_factor, n + 200)

    while len(accepted) < n and attempts < max_attempts:
        attempts += 1
        wire32 = base_wire32.copy()
        for i in movable_indices:
            wire32[i] = float(rng.uniform(lmin[i], lmax[i]))

        sim.set_pose_from_wire32_rad(wire32)
        ee = sim.fingertip_xyz()
        if hull_filter and sim.hull_violation(ee) > hull_eps:
            hull_reject += 1
            continue

        state_norm = scaler.scale_state(wire32.astype(np.float32))
        accepted.append(
            {
                "wire32_rad": wire32.astype(np.float64).tolist(),
                "state_norm": state_norm.astype(np.float64).tolist(),
                "ee_xyz": ee.astype(np.float64).tolist(),
            }
        )

    if len(accepted) < n:
        raise RuntimeError(
            f"Only sampled {len(accepted)}/{n} joint configs in {attempts} attempts "
            f"(hull_reject={hull_reject}). "
            "Try --no-hull-filter or increase max_attempts_factor."
        )

    stats = {
        "attempts": attempts,
        "accept_rate": len(accepted) / max(attempts, 1),
        "hull_reject": hull_reject,
        "hull_filter": hull_filter,
        "movable_indices": movable_indices,
        "n_movable": len(movable_indices),
    }
    return accepted, stats
