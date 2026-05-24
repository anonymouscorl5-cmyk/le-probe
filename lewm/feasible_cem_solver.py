"""
CEM solver that averages only feasible elites (cost < INFEASIBLE_COST).

Upstream CEM takes top-k lowest costs; when fewer than k samples are feasible,
infeasible trajectories tie at INFEASIBLE_COST and can enter the elite mean.
This subclass ranks only feasible samples for the distribution update and
raises if an iteration has no feasible candidates at all.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from stable_worldmodel.solver.cem import CEMSolver

from lewm.mpc_logging import MPC_VERBOSE, mpc_log, mpc_shape_log
from lewm.planning_constraints import (
    constrain_right_arm_cem_mean,
    sample_cem_plan_candidates,
)
from lewm.task_workspace import INFEASIBLE_COST


class CEMNoFeasibleSamplesError(RuntimeError):
    """All CEM candidates were rejected by the planning feasibility gate."""


# Per-env metadata — not broadcast along CEM sample axis
_CEM_PASS_THROUGH_KEYS = frozenset(
    {
        "frozen_pose_per_env",
        "phase_idx",
        "task_workspace_wire32",
        "task_workspace_H",
        "task_workspace_d",
        "task_workspace_check_final_only",
        "task_workspace_cube_xyz",
    }
)


def _expand_obs_batch_for_cem(
    v_batch: torch.Tensor | np.ndarray,
    current_bs: int,
    num_samples: int,
) -> torch.Tensor | np.ndarray:
    """
    Broadcast env observations to CEM sample count.

    Pre-CEM layouts from diagnose / server (S=1 placeholder at dim 1)::
        pixels: (B, 1, T, V, C, H, W)  -> (B, num_samples, T, V, C, H, W)
        action: (B, 1, T_hist, 32)      -> (B, num_samples, T_hist, 32)

    Legacy (B, T, V, C, H, W) without S: ``unsqueeze(1)`` then expand.
    """
    if torch.is_tensor(v_batch):
        if v_batch.ndim in (4, 7) and v_batch.shape[1] == 1:
            expand_sizes = (current_bs, num_samples, *v_batch.shape[2:])
            mpc_shape_log(
                "_expand_obs_batch_for_cem (replace S=1)",
                tensor_in=v_batch,
                expand_sizes=expand_sizes,
            )
            try:
                return v_batch.expand(expand_sizes)
            except RuntimeError as exc:
                mpc_shape_log(
                    "_expand_obs_batch_for_cem FAILED",
                    error=str(exc),
                    tensor_in=v_batch,
                    expand_sizes=expand_sizes,
                )
                raise
        unsqueezed = v_batch.unsqueeze(1)
        expand_sizes = (current_bs, num_samples, *v_batch.shape[1:])
        mpc_shape_log(
            "_expand_obs_batch_for_cem (insert S)",
            tensor_in=v_batch,
            after_unsqueeze=unsqueezed,
            expand_sizes=expand_sizes,
        )
        try:
            return unsqueezed.expand(expand_sizes)
        except RuntimeError as exc:
            mpc_shape_log(
                "_expand_obs_batch_for_cem FAILED",
                error=str(exc),
                tensor_in=v_batch,
                after_unsqueeze=unsqueezed,
                expand_sizes=expand_sizes,
            )
            raise
    v_batch = np.asarray(v_batch)
    if v_batch.ndim in (4, 7) and v_batch.shape[1] == 1:
        reps = [1] * v_batch.ndim
        reps[1] = num_samples
        return np.tile(v_batch, reps)
    return np.repeat(v_batch[:, None, ...], num_samples, axis=1)


def _per_row_feasible_cost_stats(
    costs: torch.Tensor, *, infeasible_cost: float = INFEASIBLE_COST
) -> tuple[list[float], list[float], list[float], list[int]]:
    """Min / mean / max over feasible samples per batch row; feasible counts in topk pool."""
    batch_size = costs.shape[0]
    mins, means, maxs, n_feas = [], [], [], []
    for b in range(batch_size):
        row = costs[b]
        feas = row[row < infeasible_cost]
        n = int(feas.numel())
        n_feas.append(n)
        if n == 0:
            mins.append(float("inf"))
            means.append(float("inf"))
            maxs.append(float("inf"))
        else:
            mins.append(float(feas.min().item()))
            means.append(float(feas.mean().item()))
            maxs.append(float(feas.max().item()))
    return mins, means, maxs, n_feas


def aggregate_feasible_elites(
    costs: torch.Tensor,
    candidates: torch.Tensor,
    topk: int,
    *,
    infeasible_cost: float = INFEASIBLE_COST,
) -> tuple[torch.Tensor, torch.Tensor, list[float], list[float], list[int]]:
    """
    Build CEM mean/var from lowest-cost **feasible** samples only.

    Caller must ensure each batch row has at least one feasible sample.
    """
    if costs.ndim != 2:
        raise ValueError(f"costs must be (B, S), got {tuple(costs.shape)}")
    batch_size, num_samples = costs.shape
    k = min(topk, num_samples)
    device = costs.device

    rank_costs = torch.where(
        costs < infeasible_cost,
        costs,
        torch.full_like(costs, float("inf")),
    )
    topk_vals, topk_inds = torch.topk(rank_costs, k=k, dim=1, largest=False)
    batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, k)
    topk_candidates = candidates[batch_idx, topk_inds]  # (B, k, H, D)

    feasible = topk_vals < float("inf")  # (B, k)
    weights = feasible.unsqueeze(-1).unsqueeze(-1).to(dtype=candidates.dtype)
    denom = weights.sum(dim=1).clamp(min=1.0)  # (B, 1, 1)

    batch_mean = (topk_candidates * weights).sum(dim=1) / denom
    centered = topk_candidates - batch_mean.unsqueeze(1)
    batch_var = (
        ((weights * centered.square()).sum(dim=1) / denom).sqrt().clamp(min=1e-4)
    )

    feas_counts = feasible.sum(dim=1).clamp(min=1).to(costs.dtype)
    elite_mean_cost = (
        (topk_vals.masked_fill(~feasible, 0.0).sum(dim=1) / feas_counts).cpu().tolist()
    )
    elite_feasible_in_topk = feasible.sum(dim=1).cpu().tolist()
    min_feasible_cost, _, _, _ = _per_row_feasible_cost_stats(
        costs, infeasible_cost=infeasible_cost
    )

    return (
        batch_mean,
        batch_var,
        elite_mean_cost,
        min_feasible_cost,
        elite_feasible_in_topk,
    )


class FeasibleEliteCEMSolver(CEMSolver):
    """CEM with feasible-only elite mean/std; fails loudly if none are feasible."""

    def __init__(self, *args, verbose: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.verbose = verbose

    @torch.inference_mode()
    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        start_time = time.time()
        outputs: dict[str, Any] = {
            "costs": [],  # elite mean cost (last iter) — legacy field
            "elite_mean_costs": [],
            "min_feasible_costs": [],
            "mean": [],
            "var": [],
            "feasible_sample_counts": [],
            "cem_step_logs": [],
        }

        mean, var = self.init_action_distrib(init_action)
        mean = mean.to(self.device)
        var = var.to(self.device)
        constrain_right_arm = not getattr(self.model, "use_task_workspace", False)
        if constrain_right_arm:
            mean = constrain_right_arm_cem_mean(mean)

        for start_idx in range(0, self.n_envs, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.n_envs)
            current_bs = end_idx - start_idx

            batch_mean = mean[start_idx:end_idx]
            batch_var = var[start_idx:end_idx]

            mpc_shape_log(
                f"FeasibleEliteCEMSolver.solve batch env[{start_idx}:{end_idx}]",
                num_samples=self.num_samples,
                **{
                    k: info_dict[k][start_idx:end_idx]
                    for k in info_dict
                    if hasattr(info_dict[k], "shape")
                },
            )
            expanded_infos = {}
            for key, val in info_dict.items():
                v_batch = val[start_idx:end_idx]
                if key in _CEM_PASS_THROUGH_KEYS:
                    expanded_infos[key] = v_batch
                    mpc_shape_log(f"CEM pass-through key={key!r}", tensor=v_batch)
                else:
                    mpc_shape_log(f"CEM expanding key={key!r}", before=v_batch)
                    expanded_infos[key] = _expand_obs_batch_for_cem(
                        v_batch, current_bs, self.num_samples
                    )
                    mpc_shape_log(
                        f"CEM expanded key={key!r}", after=expanded_infos[key]
                    )

            final_elite_mean: list[float] | None = None
            final_min_feasible: list[float] | None = None
            last_feasible_counts: list[int] | None = None
            step_logs: list[dict[str, Any]] = []

            for step in range(self.n_steps):
                candidates = sample_cem_plan_candidates(
                    batch_mean,
                    batch_var,
                    num_samples=self.num_samples,
                    generator=self.torch_gen,
                    constrain_right_arm=constrain_right_arm,
                )

                costs = self.model.get_cost(expanded_infos, candidates)
                if not isinstance(costs, torch.Tensor):
                    raise TypeError(f"Expected Tensor costs, got {type(costs)}")
                if costs.shape != (current_bs, self.num_samples):
                    raise ValueError(
                        f"Expected costs ({current_bs}, {self.num_samples}), got {tuple(costs.shape)}"
                    )

                n_feasible = (costs < INFEASIBLE_COST).sum(dim=1)
                last_feasible_counts = [int(x) for x in n_feasible.cpu().tolist()]
                min_f, mean_f, max_f, _ = _per_row_feasible_cost_stats(costs)

                empty_rows = (n_feasible == 0).nonzero(as_tuple=True)[0]
                if empty_rows.numel() > 0:
                    global_rows = (empty_rows + start_idx).tolist()
                    raise CEMNoFeasibleSamplesError(
                        f"CEM step {step + 1}/{self.n_steps}: 0/{self.num_samples} "
                        f"feasible samples for env row(s) {global_rows}. "
                        "Every candidate violated the planning gate "
                        "(right-arm norm envelope or task workspace)."
                    )

                (
                    batch_mean,
                    batch_var,
                    elite_mean,
                    min_feas,
                    elite_in_topk,
                ) = aggregate_feasible_elites(costs, candidates, self.topk)
                if constrain_right_arm:
                    batch_mean = constrain_right_arm_cem_mean(batch_mean)
                final_elite_mean = elite_mean
                final_min_feasible = min_feas

                step_log = {
                    "step": step + 1,
                    "n_feasible": last_feasible_counts,
                    "min_feasible_cost": min_f,
                    "mean_feasible_cost": mean_f,
                    "max_feasible_cost": max_f,
                    "elite_mean_cost": elite_mean,
                    "elite_feasible_in_topk": elite_in_topk,
                    "batch_mean_abs": batch_mean.abs().mean().item(),
                    "batch_var_mean": batch_var.mean().item(),
                }
                step_logs.append(step_log)
                if self.verbose or MPC_VERBOSE:
                    mpc_log(
                        f"CEM env[{start_idx}:{end_idx}] step {step + 1}/{self.n_steps}: "
                        f"n_feas={last_feasible_counts} "
                        f"min_feas_cost={[round(x, 2) for x in min_f]} "
                        f"elite_mean_cost={[round(x, 2) for x in elite_mean]}"
                    )

            mean[start_idx:end_idx] = batch_mean
            var[start_idx:end_idx] = batch_var
            if final_elite_mean is not None:
                outputs["costs"].extend(final_elite_mean)
                outputs["elite_mean_costs"].extend(final_elite_mean)
            if final_min_feasible is not None:
                outputs["min_feasible_costs"].extend(final_min_feasible)
            if last_feasible_counts is not None:
                outputs["feasible_sample_counts"].append(last_feasible_counts)
            outputs["cem_step_logs"].append(step_logs)

        outputs["actions"] = mean.detach().cpu()
        outputs["mean"] = [mean.detach().cpu()]
        outputs["var"] = [var.detach().cpu()]
        print(
            f"CEM solve time: {time.time() - start_time:.4f} seconds (feasible elites only)"
        )
        return outputs
