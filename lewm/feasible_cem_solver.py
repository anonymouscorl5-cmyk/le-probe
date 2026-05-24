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

from lewm.task_workspace import INFEASIBLE_COST


class CEMNoFeasibleSamplesError(RuntimeError):
    """All CEM candidates were rejected by the planning feasibility gate."""


def _expand_obs_batch_for_cem(
    v_batch: torch.Tensor | np.ndarray,
    current_bs: int,
    num_samples: int,
) -> torch.Tensor | np.ndarray:
    """
    Add the CEM sample dimension: (B, ...) -> (B, num_samples, ...).

    Legacy layouts used a placeholder ``S=1`` axis: (B, 1, T, V, C, H, W).
    That extra dim must be stripped before expanding or ``expand`` rank-mismatches.
    """
    if torch.is_tensor(v_batch):
        if (
            v_batch.ndim >= 2
            and v_batch.shape[0] == current_bs
            and v_batch.shape[1] == 1
        ):
            # (B, 1, T, V, ...) or (B, 1, T, D) — drop placeholder sample axis
            if v_batch.ndim >= 6 or (v_batch.ndim == 4 and v_batch.shape[-1] == 32):
                v_batch = v_batch.squeeze(1)
        return v_batch.unsqueeze(1).expand(current_bs, num_samples, *v_batch.shape[1:])
    v_batch = np.asarray(v_batch)
    if v_batch.ndim >= 2 and v_batch.shape[0] == current_bs and v_batch.shape[1] == 1:
        if v_batch.ndim >= 6 or (v_batch.ndim == 4 and v_batch.shape[-1] == 32):
            v_batch = np.squeeze(v_batch, axis=1)
    return np.repeat(v_batch[:, None, ...], num_samples, axis=1)


def aggregate_feasible_elites(
    costs: torch.Tensor,
    candidates: torch.Tensor,
    topk: int,
    *,
    infeasible_cost: float = INFEASIBLE_COST,
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
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
    logged_cost = (
        (topk_vals.masked_fill(~feasible, 0.0).sum(dim=1) / feas_counts).cpu().tolist()
    )

    return batch_mean, batch_var, logged_cost


class FeasibleEliteCEMSolver(CEMSolver):
    """CEM with feasible-only elite mean/std; fails loudly if none are feasible."""

    @torch.inference_mode()
    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        start_time = time.time()
        outputs: dict[str, Any] = {
            "costs": [],
            "mean": [],
            "var": [],
            "feasible_elite_counts": [],
        }

        mean, var = self.init_action_distrib(init_action)
        mean = mean.to(self.device)
        var = var.to(self.device)

        for start_idx in range(0, self.n_envs, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.n_envs)
            current_bs = end_idx - start_idx

            batch_mean = mean[start_idx:end_idx]
            batch_var = var[start_idx:end_idx]

            expanded_infos = {}
            for key, val in info_dict.items():
                v_batch = val[start_idx:end_idx]
                expanded_infos[key] = _expand_obs_batch_for_cem(
                    v_batch, current_bs, self.num_samples
                )

            final_batch_cost = None
            last_feasible_counts = None

            for step in range(self.n_steps):
                candidates = torch.randn(
                    current_bs,
                    self.num_samples,
                    self.horizon,
                    self.action_dim,
                    generator=self.torch_gen,
                    device=self.device,
                )
                candidates = candidates * batch_var.unsqueeze(1) + batch_mean.unsqueeze(
                    1
                )
                candidates[:, 0] = batch_mean

                costs = self.model.get_cost(expanded_infos, candidates)
                if not isinstance(costs, torch.Tensor):
                    raise TypeError(f"Expected Tensor costs, got {type(costs)}")
                if costs.shape != (current_bs, self.num_samples):
                    raise ValueError(
                        f"Expected costs ({current_bs}, {self.num_samples}), got {tuple(costs.shape)}"
                    )

                n_feasible = (costs < INFEASIBLE_COST).sum(dim=1)
                last_feasible_counts = n_feasible.cpu().tolist()
                empty_rows = (n_feasible == 0).nonzero(as_tuple=True)[0]
                if empty_rows.numel() > 0:
                    global_rows = (empty_rows + start_idx).tolist()
                    raise CEMNoFeasibleSamplesError(
                        f"CEM step {step + 1}/{self.n_steps}: 0/{self.num_samples} "
                        f"feasible samples for env row(s) {global_rows}. "
                        "Every candidate violated the planning gate "
                        "(right-arm norm envelope or task workspace)."
                    )

                batch_mean, batch_var, final_batch_cost = aggregate_feasible_elites(
                    costs,
                    candidates,
                    self.topk,
                )

            mean[start_idx:end_idx] = batch_mean
            var[start_idx:end_idx] = batch_var
            outputs["costs"].extend(final_batch_cost)
            if last_feasible_counts is not None:
                outputs["feasible_elite_counts"].append(last_feasible_counts)

        outputs["actions"] = mean.detach().cpu()
        outputs["mean"] = [mean.detach().cpu()]
        outputs["var"] = [var.detach().cpu()]
        print(
            f"CEM solve time: {time.time() - start_time:.4f} seconds (feasible elites only)"
        )
        return outputs
