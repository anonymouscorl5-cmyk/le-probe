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

from lewm.goal_mapper import GoalMapper
from stable_worldmodel.solver import CEMSolver


class MockConfig:
    def __init__(self, horizon):
        self.horizon = horizon
        self.action_block = 1


class MockSpace:
    def __init__(self, shape):
        self.shape = shape
        self.low = -1.0
        self.high = 1.0


def run_diagnostic(model_path, gallery_path="goal_gallery.pth", batch_size=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔬 Running Full-Spectrum Diagnostic on {device}...")

    if not Path(gallery_path).exists():
        print(f"❌ Error: Gallery not found at {gallery_path}")
        print(
            "💡 Please run 'python research/harvest_goals.py' first to generate the artifact."
        )
    print(f"🔬 Running Vectorized Audit on {device}...")

    # 1. Load Gallery
    gallery = torch.load(gallery_path, map_location=device)
    goal_ids = list(gallery["diagnostics"].keys())
    num_episodes = len(goal_ids)
    print(f"📈 Found {num_episodes} episodes. Auditing in batches of {batch_size}...")

    # 2. Setup Vectorized Agent & Solver
    mapper = GoalMapper(model_path, dataset_root=".")

    # Initialize frozen_pose for manifold squashing (Offline diagnostic uses zero-pose)
    mapper.frozen_pose = torch.zeros(32, device=device)

    solver = CEMSolver(
        model=mapper,
        num_samples=8000,
        var_scale=3.0,
        n_steps=1,
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
        for eid in batch_ids:
            pixel_list.append(gallery["diagnostics"][eid]["pixels"])
            latent_list.append(gallery["goals"][eid])

        # Pixels: (B, 3, 3, 224, 224), Latents: (B, 1, 192)
        info_dict = {"pixels": torch.stack(pixel_list).to(device)}
        mapper.goal_latent = torch.stack(latent_list).to(device)

        # B. Initial Cost (Current observations vs Goal)
        with torch.no_grad():
            initial_cost = mapper.get_cost(
                info_dict, torch.zeros(actual_batch_size, 1, 15, 32).to(device)
            )

        # C. Vectorized Planning
        outputs = solver.solve(info_dict, init_action=None)
        final_cost = torch.tensor(outputs["costs"]).to(device)

        # D. Improvement (B, S) -> (B,)
        imp = (initial_cost.view(-1) - final_cost.view(-1)).cpu().numpy()
        improvements.extend(imp.tolist())

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
    args = parser.parse_args()
    run_diagnostic(args.model, args.gallery, args.batch)
