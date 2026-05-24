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


import os
import sys
import argparse
import torch
import time
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
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔬 Running Full-Spectrum Diagnostic on {device}...")
    print(f"   - Multi-View: {use_multi_view}")
    print(f"   - Skeleton: {use_skeleton}")
    print(f"   - DINO: {use_dino}")

    if not Path(gallery_path).exists():
        print(f"❌ Error: Gallery not found at {gallery_path}")
        print(
            "💡 Please run 'python research/harvest_goals.py' first to generate the artifact."
        )
        return

    # 1. Load Gallery
    gallery = torch.load(gallery_path, map_location=device)
    goal_ids = list(gallery["diagnostics"].keys())
    num_episodes = len(goal_ids)
    print(f"📈 Found {num_episodes} episodes. Auditing in batches of {batch_size}...")

    # 2. Setup Vectorized Agent & Solver
    mapper = GoalMapper(
        model_path,
        dataset_root=dataset_root,
        use_multi_view=use_multi_view,
        num_views=5 if use_multi_view else 1,
        use_skeleton=use_skeleton,
        use_dino=use_dino,
    )

    # Initialize frozen_pose for manifold squashing (Offline diagnostic uses zero-pose)
    mapper.frozen_pose = torch.zeros(32, device=device)

    solver = FeasibleEliteCEMSolver(
        model=mapper,
        num_samples=8000,
        var_scale=0.6,
        n_steps=5,
        topk=100,
        device=device,
    )
    solver.configure(
        action_space=MockSpace(shape=(1, 32)),
        n_envs=batch_size,
        config=MockConfig(horizon=15),
    )

    improvements = []

    # 3. Batch Audit Loop
    for i in tqdm(range(0, num_episodes, batch_size), desc="Audit Batches"):
        batch_ids = goal_ids[i : i + batch_size]
        actual_batch_size = len(batch_ids)

        if actual_batch_size != batch_size:
            solver.configure(
                action_space=MockSpace(shape=(1, 32)),
                n_envs=actual_batch_size,
                config=MockConfig(horizon=15),
            )

        # A. Prepare Observation Batch
        pixel_list = []
        latent_list = []
        for ep_id in batch_ids:
            diag_entry = gallery["diagnostics"][ep_id]
            pixels = diag_entry["pixels"]  # (T_history, V, C, H, W)

            # 1. Handle Multi-View Geometry
            if use_multi_view and pixels.ndim == 4:
                pixels = pixels.unsqueeze(1).repeat(1, 5, 1, 1, 1)
            elif pixels.ndim == 4:
                pixels = pixels.unsqueeze(1)

            pixels = pixels.unsqueeze(0)  # (1, T_history, V, C, H, W)
            pixel_list.append(pixels)
            latent_list.append(gallery["goals"][ep_id])

        # Pixels: (B, 1, T_history, V, C, H, W), Actions: (B, 1, T_history, 32)
        info_dict = {
            "pixels": torch.stack(pixel_list).to(device),
            "action": torch.zeros(actual_batch_size, 1, 3, 32).to(device),
        }
        # Squeeze out the redundant (B, 1, 1, 192) --> (B, 1, 192)
        mapper.goal_latent = torch.stack(latent_list).squeeze(1).to(device)

        # B. Initial Cost (Current observations vs Goal)
        with torch.no_grad():
            initial_cost = mapper.get_cost(
                info_dict, torch.zeros(actual_batch_size, 1, 15, 32).to(device)
            )

        # C. Vectorized Planning
        try:
            outputs = solver.solve(info_dict, init_action=None)
        except CEMNoFeasibleSamplesError as exc:
            print(f"⚠️ Batch {i // batch_size}: {exc}")
            continue

        # solver.solve returns costs for all samples. We want the BEST per batch.
        # Shape: (B, S) -> find min over S -> (B,)
        raw_costs = (
            torch.tensor(outputs["costs"]).to(device).view(actual_batch_size, -1)
        )
        best_final_cost = raw_costs.min(dim=1).values

        # D. Improvement (B,)
        # Positive values mean the planner found a better path than staying still.
        imp = (initial_cost.view(-1) - best_final_cost).cpu().numpy()
        improvements.extend(imp.tolist())

        if i == 0:
            # Debug the first robot in the batch
            # initial_cost and best_final_cost are tensors, we need .item()
            print(
                f"   [DEBUG] Batch 0, Robot 0: Initial {initial_cost[0].item():.2f} -> Best Final {best_final_cost[0].item():.2f}"
            )
            print(f"   [DEBUG] Improvement: {imp[0].item():.2f}")

    # 4. Final Verdict
    avg_imp = np.mean(improvements)
    print(f"\n🏁 FULL SWEEP VERDICT:")
    print(f"   Episodes Audited: {len(improvements)}")
    print(f"   Avg Latent Improvement: {avg_imp:.4f}")

    if avg_imp > 50.0:
        print("🚀 VERDICT: MPC Parameters are robust and ready for simulator!")
    elif avg_imp > 20.0:
        print("⚠️ VERDICT: MPC is functional but cost sharpening may be needed.")
    else:
        print("❌ VERDICT: Planning failed to improve over initial state.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--gallery", type=str, default="goal_gallery.pth")
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--multi_view", action="store_true", default=False)
    parser.add_argument("--use_skeleton", action="store_true", default=False)
    parser.add_argument("--use_dino", action="store_true", default=False)
    parser.add_argument("--dataset", type=str, default="vedpatwardhan/gr1_pickup_grasp")
    args = parser.parse_args()

    # Resolve Dataset Root Dynamically
    try:
        ds = LeRobotDataset(args.dataset)
        resolved_root = ds.root
        print(f"📦 Local Dataset detected: {resolved_root}")
    except Exception:
        resolved_root = "."

    run_diagnostic(
        args.model,
        args.gallery,
        args.batch,
        use_multi_view=args.multi_view,
        use_skeleton=args.use_skeleton,
        use_dino=args.use_dino,
        dataset_root=resolved_root,
    )
