"""
FULL SPECTRUM DIAGNOSTIC SWEEP
Role: Audits the MPC planner across the entire 150-episode distilled gallery.
Mandatory: Requires goal_gallery.pth
"""

# --- Path Stabilization ---
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# --------------------------


import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Project paths
RESEARCH_DIR = Path(__file__).parent.absolute()
CORTEX_GR1 = RESEARCH_DIR.parent
sys.path.append(str(CORTEX_GR1))
sys.path.append(str(CORTEX_GR1 / "lewm/le_wm"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lewm.goal_mapper import GoalMapper
from lewm.feasible_cem_solver import CEMNoFeasibleSamplesError, FeasibleEliteCEMSolver
from lewm.mpc_logging import mpc_shape_log, set_mpc_verbose


class MockConfig:
    def __init__(self, horizon):
        self.horizon = horizon
        self.action_block = 1


class MockSpace:
    def __init__(self, shape):
        self.shape = shape
        self.low = -1.0
        self.high = 1.0


def run_diagnostic(
    model_path,
    gallery_path="goal_gallery.pth",
    batch_size=10,
    use_multi_view=False,
    use_skeleton=False,
    use_dino=False,
    dataset_root=".",
    verbose=False,
    num_samples=8000,
    var_scale=0.6,
    horizon=15,
):
    set_mpc_verbose(verbose)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔬 Running Full-Spectrum Diagnostic on {device}...")
    print(f"   - Multi-View: {use_multi_view}")
    print(f"   - Skeleton: {use_skeleton}")
    print(f"   - DINO: {use_dino}")
    print(f"   - CEM: samples={num_samples} var_scale={var_scale} horizon={horizon}")
    if use_dino:
        print(
            "   ⚠️  --use_dino: get_cost uses predict_subgoal (phase=0), "
            "NOT goal_gallery latents — gallery is only for encode-side goals."
        )

    if not Path(gallery_path).exists():
        print(f"❌ Error: Gallery not found at {gallery_path}")
        print(
            "💡 Please run 'python research/harvest_goals.py' first to generate the artifact."
        )
        return

    gallery = torch.load(gallery_path, map_location=device)
    goal_ids = list(gallery["diagnostics"].keys())
    num_episodes = len(goal_ids)
    print(f"📈 Found {num_episodes} episodes. Auditing in batches of {batch_size}...")

    mapper = GoalMapper(
        model_path,
        dataset_root=dataset_root,
        use_multi_view=use_multi_view,
        num_views=5 if use_multi_view else 1,
        use_skeleton=use_skeleton,
        use_dino=use_dino,
    )
    mapper.verbose_mpc = verbose
    mapper.frozen_pose = torch.zeros(
        32, device=device
    )  # fallback; batch uses frozen_pose_per_env

    solver = FeasibleEliteCEMSolver(
        model=mapper,
        num_samples=num_samples,
        var_scale=var_scale,
        n_steps=5,
        topk=100,
        device=device,
        verbose=verbose,
    )
    solver.configure(
        action_space=MockSpace(shape=(1, 32)),
        n_envs=batch_size,
        config=MockConfig(horizon=horizon),
    )

    improvements_elite = []
    improvements_min = []
    skipped_batches = 0

    for i in tqdm(range(0, num_episodes, batch_size), desc="Audit Batches"):
        batch_ids = goal_ids[i : i + batch_size]
        actual_batch_size = len(batch_ids)

        if actual_batch_size != batch_size:
            solver.configure(
                action_space=MockSpace(shape=(1, 32)),
                n_envs=actual_batch_size,
                config=MockConfig(horizon=horizon),
            )

        pixel_list = []
        latent_list = []
        frozen_pose_rows = []
        hist_action_rows = []
        for ep_idx, ep_id in enumerate(batch_ids):
            diag_entry = gallery["diagnostics"][ep_id]
            pixels = diag_entry["pixels"]  # (T_history, V, C, H, W)
            if ep_idx == 0:
                mpc_shape_log(
                    f"diagnose batch {i // batch_size} gallery raw ep_id={ep_id}",
                    gallery_pixels=pixels,
                    gallery_action=diag_entry["action"],
                )

            if use_multi_view and pixels.ndim == 4:
                pixels = pixels.unsqueeze(1).repeat(1, 5, 1, 1, 1)
            elif pixels.ndim == 4:
                pixels = pixels.unsqueeze(1)

            if ep_idx == 0:
                mpc_shape_log(
                    f"diagnose batch {i // batch_size} after per-ep unsqueeze(0)",
                    pixels_ep=pixels,
                )
            pixel_list.append(pixels)  # (T_history, V, C, H, W) per episode
            latent_list.append(gallery["goals"][ep_id])
            frozen_pose_rows.append(diag_entry["action"][-1].float())
            hist_action_rows.append(diag_entry["action"].float())

        stacked_pixels = torch.stack(pixel_list)
        stacked_action = torch.stack(hist_action_rows)
        mpc_shape_log(
            f"diagnose batch {i // batch_size} after stack (before .unsqueeze(1))",
            stacked_pixels=stacked_pixels,
            stacked_action=stacked_action,
        )
        # (B, 1, T, V, C, H, W) / (B, 1, T_hist, 32) — same layout as lewm_server pre-CEM
        info_dict = {
            "pixels": stacked_pixels.unsqueeze(1).to(device),
            "action": stacked_action.unsqueeze(1).to(device),
            "frozen_pose_per_env": torch.stack(frozen_pose_rows).to(device),
        }
        mapper.goal_latent = torch.stack(latent_list).squeeze(1).to(device)
        mpc_shape_log(
            f"diagnose batch {i // batch_size} info_dict (pre-CEM, expect B,1,T,...)",
            pixels=info_dict["pixels"],
            action=info_dict["action"],
            frozen_pose_per_env=info_dict["frozen_pose_per_env"],
            goal_latent=mapper.goal_latent,
        )

        zero_plan = torch.zeros(actual_batch_size, 1, horizon, 32).to(device)
        mpc_shape_log(
            f"diagnose batch {i // batch_size} zero_plan",
            zero_plan=zero_plan,
        )
        with torch.no_grad():
            initial_cost = mapper.get_cost(info_dict, zero_plan)

        try:
            outputs = solver.solve(info_dict, init_action=None)
        except CEMNoFeasibleSamplesError as exc:
            skipped_batches += 1
            print(f"⚠️ Batch {i // batch_size}: {exc}")
            continue

        elite_costs = torch.tensor(
            outputs.get("elite_mean_costs", outputs["costs"]), device=device
        ).view(actual_batch_size, -1)
        min_feas = torch.tensor(
            outputs.get("min_feasible_costs", outputs["costs"]), device=device
        ).view(actual_batch_size, -1)

        best_elite = elite_costs.min(dim=1).values
        best_min = min_feas.min(dim=1).values

        imp_elite = (initial_cost.view(-1) - best_elite).cpu().numpy()
        imp_min = (initial_cost.view(-1) - best_min).cpu().numpy()
        improvements_elite.extend(imp_elite.tolist())
        improvements_min.extend(imp_min.tolist())

        if verbose or i == 0:
            batch_idx = i // batch_size
            print(f"\n📊 Batch {batch_idx} (eps {batch_ids[0]}..{batch_ids[-1]}):")
            print(
                f"   frozen_pose_per_env right-arm |max|="
                f"{info_dict['frozen_pose_per_env'][:, 16:20].abs().max().item():.4f}"
            )
            print(
                f"   initial_cost (zero plan): "
                f"{[round(x, 2) for x in initial_cost.view(-1).cpu().tolist()]}"
            )
            print(
                f"   CEM elite_mean_cost: "
                f"{[round(x, 2) for x in best_elite.cpu().tolist()]}"
            )
            print(
                f"   CEM min_feasible_cost: "
                f"{[round(x, 2) for x in best_min.cpu().tolist()]}"
            )
            print(
                f"   improvement (elite mean): "
                f"{[round(x, 2) for x in imp_elite.tolist()]}"
            )
            print(
                f"   improvement (min feasible): "
                f"{[round(x, 2) for x in imp_min.tolist()]}"
            )
            if outputs.get("feasible_sample_counts"):
                print(
                    f"   n_feasible last CEM iter: {outputs['feasible_sample_counts'][-1]}"
                )

    avg_elite = float(np.mean(improvements_elite)) if improvements_elite else 0.0
    avg_min = float(np.mean(improvements_min)) if improvements_min else 0.0
    print(f"\n🏁 FULL SWEEP VERDICT:")
    print(f"   Episodes Audited: {len(improvements_elite)}")
    print(f"   Skipped batches (all infeasible): {skipped_batches}")
    print(f"   Avg improvement vs zero-plan (elite mean cost): {avg_elite:.4f}")
    print(f"   Avg improvement vs zero-plan (min feasible cost): {avg_min:.4f}")

    if avg_min > 50.0:
        print("🚀 VERDICT: MPC finds clearly better feasible plans vs zero baseline.")
    elif avg_min > 20.0:
        print("⚠️ VERDICT: Some improvement; check reward scale and frozen_pose.")
    elif avg_elite <= 0.0 and avg_min <= 0.0:
        print(
            "❌ VERDICT: No improvement. See [MPC] logs (--verbose): "
            "likely metric mismatch (use_dino vs gallery), zero frozen_pose, "
            "or CEM stuck at zero-mean init — not necessarily var_scale."
        )
    else:
        print("❌ VERDICT: Mixed / weak improvement — inspect per-batch logs above.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--gallery", type=str, default="goal_gallery.pth")
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--multi_view", action="store_true", default=False)
    parser.add_argument("--use_skeleton", action="store_true", default=False)
    parser.add_argument("--use_dino", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--num_samples", type=int, default=8000)
    parser.add_argument("--var_scale", type=float, default=0.6)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--dataset", type=str, default="gr1_pickup_grasp")
    args = parser.parse_args()
    from lewm.mpc_logging import MPC_VERBOSE

    mpc_verbose = args.verbose or MPC_VERBOSE

    try:
        ds = LeRobotDataset(args.dataset)
        resolved_root = ds.root
        print(f"📦 Local Dataset detected: {resolved_root}")
    except Exception as exc:
        raise RuntimeError(
            f"Dataset '{args.dataset}' is not available locally. "
            "HF fallback is disabled for submission mode."
        ) from exc

    run_diagnostic(
        args.model,
        args.gallery,
        args.batch,
        use_multi_view=args.multi_view,
        use_skeleton=args.use_skeleton,
        use_dino=args.use_dino,
        dataset_root=resolved_root,
        verbose=mpc_verbose,
        num_samples=args.num_samples,
        var_scale=args.var_scale,
        horizon=args.horizon,
    )
