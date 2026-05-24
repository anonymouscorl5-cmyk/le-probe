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
    freeze_and_clamp_actions,
    right_arm_norm_feasible_mask,
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
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    mean = torch.zeros(1, horizon, 32, device=device)
    var = var_scale * torch.ones(1, horizon, 32, device=device)

    candidates = torch.randn(1, num_samples, horizon, 32, generator=gen, device=device)
    candidates = candidates * var.unsqueeze(1) + mean.unsqueeze(1)
    candidates[:, 0] = mean

    flat = candidates.view(num_samples, horizon, 32)
    frozen_rows = frozen_pose.unsqueeze(0).expand(num_samples, -1)
    plan_np = freeze_and_clamp_actions(flat, frozen_rows).detach().cpu().numpy()

    feasible = right_arm_norm_feasible_mask(plan_np)
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
    args = parser.parse_args()

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
