"""
MPC diagnostics: console logging + JSON feasibility export (CLI).

Console (both off unless ``--verbose`` or ``LEWM_MPC_VERBOSE=1``):
  - ``mpc_log`` — costs, feasibility counts
  - ``mpc_shape_log`` — tensor shapes (debug only)

JSON export (no model; gate + CEM sampling only)::

  Single setting::
    python mpc_logging.py --gallery goal_gallery.pth \\
        --batch_indices 0 1 --num_samples 8000 --output_dir logs/mpc_debug

  Grid search (var_scale x horizon, compact JSON)::
    python mpc_logging.py --gallery goal_gallery.pth --grid_search \\
        --batch_indices 0 --output_dir logs/mpc_debug

  Right-arm sample histograms (feasible vs rejected, no model)::
    python mpc_logging.py --gallery goal_gallery.pth --plot_samples \\
        --horizon 4 --var_scale 0.1 --num_samples 8000 \\
        --batch_indices 0 1 2 --output_dir logs/mpc_debug
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# --- Path Stabilization ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# --------------------------

# Project paths
RESEARCH_DIR = Path(__file__).parent.absolute()
CORTEX_GR1 = RESEARCH_DIR.parent
sys.path.append(str(CORTEX_GR1))
sys.path.append(str(CORTEX_GR1 / "lewm/le_wm"))

from lewm.planning_constraints import (
    RIGHT_ARM_NORM_MAX,
    RIGHT_ARM_NORM_MIN,
    RIGHT_ARM_NORM_SLICE,
    constrain_right_arm_cem_mean,
    freeze_and_clamp_actions,
    right_arm_norm_feasible_mask,
    sample_cem_plan_candidates,
)

MPC_VERBOSE = os.environ.get("LEWM_MPC_VERBOSE", "").lower() in ("1", "true", "yes")

RIGHT_ARM_JOINT_LABELS = [
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow_pitch",
]


def set_mpc_verbose(enabled: bool) -> None:
    global MPC_VERBOSE
    MPC_VERBOSE = bool(enabled)


def mpc_log(msg: str) -> None:
    if MPC_VERBOSE:
        print(f"[MPC] {msg}")


def mpc_shape_log(where: str, **named_tensors) -> None:
    """Print tensor shapes when MPC verbose mode is on."""
    if not MPC_VERBOSE:
        return
    lines = [f"[MPC:shape] {where}"]
    for name, val in named_tensors.items():
        if val is None:
            lines.append(f"  {name}: None")
        elif hasattr(val, "shape"):
            lines.append(
                f"  {name}: shape={tuple(val.shape)} ndim={val.ndim} "
                f"dtype={getattr(val, 'dtype', '?')}"
            )
        elif isinstance(val, (tuple, list)):
            lines.append(f"  {name}: {val}")
        else:
            lines.append(f"  {name}: {val!r}")
    print("\n".join(lines))


def _per_joint_violation_rates(arm: np.ndarray) -> dict:
    n = arm.shape[0] * arm.shape[1]
    out = {}
    for j, label in enumerate(RIGHT_ARM_JOINT_LABELS):
        vals = arm[..., j].reshape(-1)
        out[label] = {
            "index": int(16 + j),
            "min_bound": float(RIGHT_ARM_NORM_MIN[j]),
            "max_bound": float(RIGHT_ARM_NORM_MAX[j]),
            "frac_below_min": float((vals < RIGHT_ARM_NORM_MIN[j]).sum()) / n,
            "frac_above_max": float((vals > RIGHT_ARM_NORM_MAX[j]).sum()) / n,
            "sample_min": float(vals.min()),
            "sample_max": float(vals.max()),
            "sample_mean": float(vals.mean()),
        }
    return out


def _arm_summary(arm: np.ndarray) -> dict:
    flat = arm.reshape(-1, 4)
    return {
        label: {
            "min": float(flat[:, j].min()),
            "max": float(flat[:, j].max()),
            "mean": float(flat[:, j].mean()),
            "p05": float(np.percentile(flat[:, j], 5)),
            "p95": float(np.percentile(flat[:, j], 95)),
        }
        for j, label in enumerate(RIGHT_ARM_JOINT_LABELS)
    }


def generate_cem_plans_for_env(
    *,
    frozen_pose: torch.Tensor,
    num_samples: int,
    horizon: int,
    var_scale: float,
    device: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    CEM-style Gaussian samples + freeze/clamp; same path as FeasibleEliteCEMSolver.

    Returns:
        plan_np: (num_samples, horizon, 32) normalized actions
        feasible: (num_samples,) bool — full-trajectory right-arm gate
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    mean = torch.zeros(1, horizon, 32, device=device)
    var = var_scale * torch.ones(1, horizon, 32, device=device)
    mean = constrain_right_arm_cem_mean(mean)

    candidates = sample_cem_plan_candidates(
        mean,
        var,
        num_samples=num_samples,
        generator=gen,
        constrain_right_arm=True,
    )

    flat = candidates.view(num_samples, horizon, 32)
    frozen_rows = frozen_pose.unsqueeze(0).expand(num_samples, -1)
    plan_np = freeze_and_clamp_actions(flat, frozen_rows).detach().cpu().numpy()
    feasible = right_arm_norm_feasible_mask(plan_np)
    return plan_np, feasible


def _right_arm_values_from_plans(
    plan_np: np.ndarray,
    feasible: np.ndarray,
    *,
    timestep: str = "all",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pool right-arm joint values into feasible / rejected 1D arrays per joint.

    timestep:
        ``all`` — every (candidate, t) for candidates tagged feasible/rejected
        ``first`` — t=0 only (first MPC step)
    """
    arm = plan_np[..., RIGHT_ARM_NORM_SLICE]  # (N, T, 4)
    if timestep == "first":
        arm = arm[:, :1, :]
    elif timestep != "all":
        raise ValueError(f"timestep must be 'all' or 'first', got {timestep!r}")

    feas_mask = feasible
    feas_arm = arm[feas_mask]  # (N_feas, T', 4)
    rej_arm = arm[~feas_mask]
    if feas_arm.size:
        feas_flat = feas_arm.reshape(-1, 4)
    else:
        feas_flat = np.empty((0, 4))
    if rej_arm.size:
        rej_flat = rej_arm.reshape(-1, 4)
    else:
        rej_flat = np.empty((0, 4))
    return feas_flat, rej_flat


def analyze_cem_feasibility_for_env(
    *,
    ep_id,
    frozen_pose: torch.Tensor,
    num_samples: int,
    horizon: int,
    var_scale: float,
    device: str,
    seed: int,
    detail: bool = True,
) -> dict:
    """One env row: CEM-style samples + right-arm gate (matches FeasibleEliteCEMSolver)."""
    plan_np, feasible = generate_cem_plans_for_env(
        frozen_pose=frozen_pose,
        num_samples=num_samples,
        horizon=horizon,
        var_scale=var_scale,
        device=device,
        seed=seed,
    )
    n_feas = int(feasible.sum())
    arm = plan_np[..., RIGHT_ARM_NORM_SLICE]
    feas_idx = np.nonzero(feasible)[0]
    infeas_idx = np.nonzero(~feasible)[0]

    report = {
        "episode_id": ep_id,
        "frozen_pose_right_arm": frozen_pose[16:20].detach().cpu().tolist(),
        "zero_plan_right_arm": plan_np[0, 0, 16:20].tolist(),
        "feasible_count": n_feas,
        "feasible_fraction": n_feas / num_samples,
        "candidate_0_feasible": bool(feasible[0]),
        "only_candidate_0_feasible": bool(
            n_feas == 1 and feasible[0] and not feasible[1:].any()
        ),
        "feasible_candidate_indices": feas_idx[:20].tolist(),
    }
    if detail:
        report["violation_rates_all_samples"] = _per_joint_violation_rates(arm)

    if detail and infeas_idx.size:
        ex = int(infeas_idx[0])
        report["example_infeasible"] = {
            "candidate_index": ex,
            "right_arm_per_timestep": plan_np[ex, :, 16:20].tolist(),
            "first_step_violations": {
                label: {
                    "value": float(plan_np[ex, 0, 16 + j]),
                    "below_min": bool(plan_np[ex, 0, 16 + j] < RIGHT_ARM_NORM_MIN[j]),
                    "above_max": bool(plan_np[ex, 0, 16 + j] > RIGHT_ARM_NORM_MAX[j]),
                }
                for j, label in enumerate(RIGHT_ARM_JOINT_LABELS)
            },
        }
        report["arm_stats_infeasible_only"] = _arm_summary(arm[~feasible])

    if detail and n_feas:
        report["arm_stats_feasible_only"] = _arm_summary(arm[feasible])

    return report


def _summarize_episode_reports(episodes: list[dict]) -> dict:
    if not episodes:
        return {
            "episode_count": 0,
            "mean_feasible_fraction": 0.0,
            "median_feasible_fraction": 0.0,
            "min_feasible_fraction": 0.0,
            "max_feasible_fraction": 0.0,
            "mean_feasible_count": 0.0,
            "episodes_only_candidate_0_feasible": 0,
        }
    fracs = [e["feasible_fraction"] for e in episodes]
    counts = [e["feasible_count"] for e in episodes]
    return {
        "episode_count": len(episodes),
        "mean_feasible_fraction": float(np.mean(fracs)),
        "median_feasible_fraction": float(np.median(fracs)),
        "min_feasible_fraction": float(np.min(fracs)),
        "max_feasible_fraction": float(np.max(fracs)),
        "mean_feasible_count": float(np.mean(counts)),
        "episodes_only_candidate_0_feasible": int(
            sum(1 for e in episodes if e["only_candidate_0_feasible"])
        ),
    }


def plot_right_arm_sample_distributions(
    *,
    feasible_by_joint: list[np.ndarray],
    rejected_by_joint: list[np.ndarray],
    out_path: Path,
    title: str,
    bins: int = 80,
) -> None:
    """Four-panel histogram: green=feasible, red=rejected; shaded band = envelope."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=False)
    axes = axes.ravel()
    n_feas = sum(len(v) for v in feasible_by_joint)
    n_rej = sum(len(v) for v in rejected_by_joint)

    for j, (ax, label) in enumerate(zip(axes, RIGHT_ARM_JOINT_LABELS)):
        lo, hi = float(RIGHT_ARM_NORM_MIN[j]), float(RIGHT_ARM_NORM_MAX[j])
        feas = feasible_by_joint[j]
        rej = rejected_by_joint[j]

        ax.axvspan(
            lo, hi, color="#4a90d9", alpha=0.18, label=f"envelope [{lo:g}, {hi:g}]"
        )
        if rej.size:
            ax.hist(
                rej,
                bins=bins,
                range=(-1.05, 1.05),
                density=True,
                alpha=0.55,
                color="#c44e52",
                label=f"rejected ({rej.size:,})",
            )
        if feas.size:
            ax.hist(
                feas,
                bins=bins,
                range=(-1.05, 1.05),
                density=True,
                alpha=0.55,
                color="#55a868",
                label=f"feasible ({feas.size:,})",
            )
        ax.axvline(lo, color="#2166ac", ls="--", lw=1)
        ax.axvline(hi, color="#2166ac", ls="--", lw=1)
        ax.set_xlim(-1.05, 1.05)
        ax.set_title(f"idx {16 + j}: {label}")
        ax.set_xlabel("normalized value")
        ax.set_ylabel("density")
        ax.legend(loc="upper right", fontsize=7)

    fig.suptitle(title, fontsize=11)
    fig.text(
        0.5,
        0.01,
        f"pooled values: feasible {n_feas:,} | rejected {n_rej:,}",
        ha="center",
        fontsize=9,
        color="#444",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def export_sample_distribution_plots(
    gallery_path: str | Path,
    *,
    batch_indices: list[int],
    batch_size: int = 10,
    num_samples: int = 8000,
    var_scale: float = 0.6,
    horizon: int = 15,
    output_dir: str | Path = "logs/mpc_debug",
    seed: int = 0,
    max_episodes: int | None = None,
    timestep: str = "all",
    plot_per_episode: bool = False,
    bins: int = 80,
) -> Path:
    """
    Pool CEM right-arm samples across episodes; write aggregate PNG + metadata JSON.

    Uses the same sampling + gate as diagnose / FeasibleEliteCEMSolver (no LeWM load).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gallery_path = Path(gallery_path)
    if not gallery_path.exists():
        raise FileNotFoundError(gallery_path)

    gallery = torch.load(gallery_path, map_location="cpu")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    episode_rows = _collect_episodes_for_batches(
        gallery,
        batch_indices=batch_indices,
        batch_size=batch_size,
        device=device,
    )
    if max_episodes is not None:
        episode_rows = episode_rows[:max_episodes]
    if not episode_rows:
        raise ValueError("No episodes found for the requested batch_indices")

    pooled_feas = [[] for _ in range(4)]
    pooled_rej = [[] for _ in range(4)]
    episode_summaries = []

    for i, (batch_idx, ep_id, frozen) in enumerate(episode_rows):
        plan_np, feasible = generate_cem_plans_for_env(
            frozen_pose=frozen,
            num_samples=num_samples,
            horizon=horizon,
            var_scale=var_scale,
            device=device,
            seed=seed + horizon * 10_000 + int(var_scale * 1000) + i,
        )
        feas_flat, rej_flat = _right_arm_values_from_plans(
            plan_np, feasible, timestep=timestep
        )
        for j in range(4):
            if feas_flat.size:
                pooled_feas[j].append(feas_flat[:, j])
            if rej_flat.size:
                pooled_rej[j].append(rej_flat[:, j])

        episode_summaries.append(
            {
                "episode_id": ep_id,
                "batch_index": batch_idx,
                "feasible_fraction": float(feasible.mean()),
                "feasible_count": int(feasible.sum()),
                "only_candidate_0_feasible": bool(
                    feasible.sum() == 1 and feasible[0] and not feasible[1:].any()
                ),
            }
        )

        if plot_per_episode:
            ep_feas = [
                feas_flat[:, j] if feas_flat.size else np.array([]) for j in range(4)
            ]
            ep_rej = [
                rej_flat[:, j] if rej_flat.size else np.array([]) for j in range(4)
            ]
            ep_path = out_dir / f"right_arm_dist_ep{ep_id}_h{horizon}_s{var_scale}.png"
            plot_right_arm_sample_distributions(
                feasible_by_joint=ep_feas,
                rejected_by_joint=ep_rej,
                out_path=ep_path,
                title=(
                    f"ep {ep_id} | h={horizon} σ={var_scale} | "
                    f"{num_samples} CEM samples | timestep={timestep}"
                ),
                bins=bins,
            )

    feas_arrays = [
        np.concatenate(chunks) if chunks else np.array([]) for chunks in pooled_feas
    ]
    rej_arrays = [
        np.concatenate(chunks) if chunks else np.array([]) for chunks in pooled_rej
    ]

    ts_tag = "t0" if timestep == "first" else "all_t"
    png_path = out_dir / f"right_arm_dist_h{horizon}_sigma{var_scale}_{ts_tag}.png"
    plot_right_arm_sample_distributions(
        feasible_by_joint=feas_arrays,
        rejected_by_joint=rej_arrays,
        out_path=png_path,
        title=(
            f"pooled {len(episode_rows)} episodes | h={horizon} σ={var_scale} | "
            f"{num_samples} samples/ep | timestep={timestep}"
        ),
        bins=bins,
    )

    meta = {
        "plot_path": str(png_path),
        "per_episode_plots": plot_per_episode,
        "envelope_right_arm_norm": {
            "indices": list(range(16, 20)),
            "labels": RIGHT_ARM_JOINT_LABELS,
            "min": RIGHT_ARM_NORM_MIN.tolist(),
            "max": RIGHT_ARM_NORM_MAX.tolist(),
        },
        "cem_sampling": {
            "num_samples": num_samples,
            "var_scale": var_scale,
            "horizon": horizon,
            "timestep_pooling": timestep,
        },
        "episode_count": len(episode_rows),
        "value_counts_per_joint": {
            RIGHT_ARM_JOINT_LABELS[j]: {
                "feasible": int(feas_arrays[j].size),
                "rejected": int(rej_arrays[j].size),
            }
            for j in range(4)
        },
        "mean_feasible_fraction": float(
            np.mean([e["feasible_fraction"] for e in episode_summaries])
        ),
        "episodes_only_candidate_0_feasible": int(
            sum(1 for e in episode_summaries if e["only_candidate_0_feasible"])
        ),
        "episodes": episode_summaries,
    }
    meta_path = out_dir / f"right_arm_dist_h{horizon}_sigma{var_scale}_{ts_tag}.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {meta_path}")
    return png_path


def _collect_episodes_for_batches(
    gallery: dict,
    *,
    batch_indices: list[int],
    batch_size: int,
    device: str,
) -> list[tuple[int, int, torch.Tensor]]:
    """(batch_idx, ep_id, frozen_pose) for all episodes in selected batches."""
    goal_ids = list(gallery["diagnostics"].keys())
    rows = []
    for batch_idx in batch_indices:
        start = batch_idx * batch_size
        for ep_id in goal_ids[start : start + batch_size]:
            frozen = gallery["diagnostics"][ep_id]["action"][-1].float().to(device)
            rows.append((batch_idx, ep_id, frozen))
    return rows


def export_feasibility_grid_search(
    gallery_path: str | Path,
    *,
    batch_indices: list[int],
    batch_size: int = 10,
    num_samples: int = 8000,
    var_scales: list[float] | None = None,
    horizons: list[int] | None = None,
    output_dir: str | Path = "logs/mpc_debug",
    seed: int = 0,
    include_per_episode: bool = False,
) -> Path:
    """
    Sweep ``var_scale`` x ``horizon``; write ``grid_search.json``.

    Default horizons include values below server horizon (4) and above diagnose (15).
    """
    if var_scales is None:
        var_scales = [0.05, 0.1, 0.15, 0.2, 0.3, 0.6]
    if horizons is None:
        horizons = [1, 2, 3, 4, 8, 15]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gallery_path = Path(gallery_path)
    if not gallery_path.exists():
        raise FileNotFoundError(gallery_path)

    gallery = torch.load(gallery_path, map_location="cpu")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    episode_rows = _collect_episodes_for_batches(
        gallery,
        batch_indices=batch_indices,
        batch_size=batch_size,
        device=device,
    )
    if not episode_rows:
        raise ValueError("No episodes found for the requested batch_indices")

    results = []
    total_cells = len(var_scales) * len(horizons)
    cell_idx = 0

    for horizon in horizons:
        for var_scale in var_scales:
            cell_idx += 1
            print(
                f"[grid {cell_idx}/{total_cells}] horizon={horizon} var_scale={var_scale} "
                f"({len(episode_rows)} episodes, {num_samples} samples)"
            )
            ep_reports = []
            for i, (batch_idx, ep_id, frozen) in enumerate(episode_rows):
                ep_reports.append(
                    analyze_cem_feasibility_for_env(
                        ep_id=ep_id,
                        frozen_pose=frozen,
                        num_samples=num_samples,
                        horizon=horizon,
                        var_scale=var_scale,
                        device=device,
                        seed=seed + horizon * 10_000 + int(var_scale * 1000) + i,
                        detail=False,
                    )
                )
            agg = _summarize_episode_reports(ep_reports)
            cell = {
                "horizon": horizon,
                "var_scale": var_scale,
                **agg,
            }
            if include_per_episode:
                cell["episodes"] = ep_reports
            results.append(cell)

    # Rank by exploration potential (more than just candidate 0)
    ranked = sorted(
        results,
        key=lambda r: (
            r["mean_feasible_fraction"],
            -r["episodes_only_candidate_0_feasible"],
            r["mean_feasible_count"],
        ),
        reverse=True,
    )
    collapsed = [
        r
        for r in results
        if r["episodes_only_candidate_0_feasible"] == r["episode_count"]
    ]

    payload = {
        "envelope_right_arm_norm": {
            "indices": list(range(16, 20)),
            "labels": RIGHT_ARM_JOINT_LABELS,
            "min": RIGHT_ARM_NORM_MIN.tolist(),
            "max": RIGHT_ARM_NORM_MAX.tolist(),
        },
        "grid_axes": {
            "var_scales": var_scales,
            "horizons": horizons,
            "num_samples": num_samples,
            "batch_indices": batch_indices,
            "batch_size": batch_size,
        },
        "episode_scope": {
            "episode_count": len(episode_rows),
            "episode_ids": [ep_id for _, ep_id, _ in episode_rows],
        },
        "results": results,
        "ranked_by_mean_feasible_fraction": [
            {
                "horizon": r["horizon"],
                "var_scale": r["var_scale"],
                "mean_feasible_fraction": r["mean_feasible_fraction"],
                "mean_feasible_count": r["mean_feasible_count"],
                "episodes_only_candidate_0_feasible": r[
                    "episodes_only_candidate_0_feasible"
                ],
                "episode_count": r["episode_count"],
            }
            for r in ranked[:10]
        ],
        "fully_collapsed_cells": [
            {
                "horizon": r["horizon"],
                "var_scale": r["var_scale"],
                "mean_feasible_fraction": r["mean_feasible_fraction"],
            }
            for r in collapsed
        ],
        "interpretation": (
            "mean_feasible_fraction << 0.01 with episodes_only_candidate_0_feasible == episode_count "
            "means CEM only explores the zero-mean anchor; raise feasible_fraction before trusting diagnose_mpc."
        ),
    }

    out_path = out_dir / "grid_search.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_path}")
    if ranked:
        best = ranked[0]
        print(
            f"Best cell: horizon={best['horizon']} var_scale={best['var_scale']} "
            f"mean_feasible_fraction={best['mean_feasible_fraction']:.6f} "
            f"only_c0={best['episodes_only_candidate_0_feasible']}/{best['episode_count']}"
        )
    return out_path


def export_feasibility_json(
    gallery_path: str | Path,
    *,
    batch_indices: list[int],
    batch_size: int = 10,
    num_samples: int = 8000,
    var_scale: float = 0.6,
    horizon: int = 15,
    output_dir: str | Path = "logs/mpc_debug",
    seed: int = 0,
) -> Path:
    """Write batch_XX_feasibility.json + summary.json; return summary path."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gallery_path = Path(gallery_path)
    if not gallery_path.exists():
        raise FileNotFoundError(gallery_path)

    gallery = torch.load(gallery_path, map_location="cpu")
    goal_ids = list(gallery["diagnostics"].keys())
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "envelope_right_arm_norm": {
            "indices": list(range(16, 20)),
            "labels": RIGHT_ARM_JOINT_LABELS,
            "min": RIGHT_ARM_NORM_MIN.tolist(),
            "max": RIGHT_ARM_NORM_MAX.tolist(),
        },
        "cem_sampling": {
            "num_samples": num_samples,
            "var_scale": var_scale,
            "horizon": horizon,
            "note": "N(0,var_scale) around zero mean; candidate 0 = mean.",
        },
        "batches": [],
    }

    for batch_idx in batch_indices:
        start = batch_idx * batch_size
        batch_ids = goal_ids[start : start + batch_size]
        if not batch_ids:
            continue

        batch_report = {
            "batch_index": batch_idx,
            "episode_ids": batch_ids,
            "episodes": [],
        }

        for i, ep_id in enumerate(batch_ids):
            frozen = gallery["diagnostics"][ep_id]["action"][-1].float().to(device)
            batch_report["episodes"].append(
                analyze_cem_feasibility_for_env(
                    ep_id=ep_id,
                    frozen_pose=frozen,
                    num_samples=num_samples,
                    horizon=horizon,
                    var_scale=var_scale,
                    device=device,
                    seed=seed + batch_idx * 1000 + i,
                )
            )

        n_only_zero = sum(
            1 for e in batch_report["episodes"] if e["only_candidate_0_feasible"]
        )
        batch_report["batch_summary"] = {
            "episodes": len(batch_report["episodes"]),
            "episodes_only_candidate_0_feasible": n_only_zero,
            "mean_feasible_fraction": float(
                np.mean([e["feasible_fraction"] for e in batch_report["episodes"]])
            ),
        }

        path = out_dir / f"batch_{batch_idx:02d}_feasibility.json"
        path.write_text(json.dumps(batch_report, indent=2))
        print(f"Wrote {path}")
        summary["batches"].append(
            {
                "batch_index": batch_idx,
                "path": str(path),
                **batch_report["batch_summary"],
            }
        )

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {summary_path}")
    return summary_path


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Export MPC right-arm gate feasibility JSON (no model load)"
    )
    parser.add_argument("--gallery", type=str, default="goal_gallery.pth")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--batch_indices", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--num_samples", type=int, default=8000)
    parser.add_argument("--var_scale", type=float, default=0.6)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--output_dir", type=str, default="logs/mpc_debug")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--grid_search",
        action="store_true",
        help="Sweep var_scale x horizon; writes grid_search.json (compact)",
    )
    parser.add_argument(
        "--var_scales",
        type=float,
        nargs="+",
        default=None,
        help="Grid axis (default: 0.05 0.1 0.15 0.2 0.3 0.6)",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=None,
        help="Grid axis incl. below server h=4 (default: 1 2 3 4 8 15)",
    )
    parser.add_argument(
        "--grid_include_episodes",
        action="store_true",
        help="Include per-episode rows in each grid cell (larger JSON)",
    )
    parser.add_argument(
        "--plot_samples",
        action="store_true",
        help="Histogram right-arm joint values (feasible vs rejected); no model",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Cap episodes for --plot_samples (default: all in selected batches)",
    )
    parser.add_argument(
        "--plot_per_episode",
        action="store_true",
        help="Also write one PNG per episode under output_dir",
    )
    parser.add_argument(
        "--plot_timestep",
        choices=("all", "first"),
        default="all",
        help="Pool all horizon steps or t=0 only (default: all)",
    )
    parser.add_argument(
        "--plot_bins",
        type=int,
        default=80,
        help="Histogram bins for --plot_samples",
    )
    args = parser.parse_args()

    if args.plot_samples:
        export_sample_distribution_plots(
            args.gallery,
            batch_indices=args.batch_indices,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            var_scale=args.var_scale,
            horizon=args.horizon,
            output_dir=args.output_dir,
            seed=args.seed,
            max_episodes=args.max_episodes,
            timestep=args.plot_timestep,
            plot_per_episode=args.plot_per_episode,
            bins=args.plot_bins,
        )
        return

    if args.grid_search:
        export_feasibility_grid_search(
            args.gallery,
            batch_indices=args.batch_indices,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            var_scales=args.var_scales,
            horizons=args.horizons,
            output_dir=args.output_dir,
            seed=args.seed,
            include_per_episode=args.grid_include_episodes,
        )
        return

    export_feasibility_json(
        args.gallery,
        batch_indices=args.batch_indices,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        var_scale=args.var_scale,
        horizon=args.horizon,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    _cli()
