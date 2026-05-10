# --- Path Stabilization ---
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# --------------------------

import torch
import torch.nn as nn
import lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import wandb
import csv
import os
import time
from datetime import datetime


class MetricsCallback(pl.Callback):
    """
    Advanced interpretability metrics for world model training.
    Monitoring for:
    1. Latent Collapse (SoftRank)
    2. Grounding Linearity (Path Straightening)
    3. Manifold Geometry (PCA Visuals)
    """

    def __init__(
        self,
        log_every_n_steps=50,
        pca_n_samples=256,
        csv_path="manifold_diagnostics.csv",
    ):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.pca_n_samples = pca_n_samples
        self.csv_path = csv_path
        self._init_csv()

        # 🌟 Manifold Accumulation Buffers
        self.val_latents = []
        self.val_steps = []

    def _init_csv(self):
        """Initializes the CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.csv_path):
            headers = [
                "timestamp",
                "step",
                "epoch",
                "soft_rank",
                "participation_ratio",
                "action_mag",
                "emb_mag",
                "signal_ratio",
                "path_straightening",
            ]
            # Add headers for top 10 singular values
            headers += [f"s_{i}" for i in range(10)]
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def log_to_csv(self, data_dict):
        """Appends a row of data to the CSV file."""
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            # Ensure we follow the header order
            row = [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                data_dict.get("step", 0),
                data_dict.get("epoch", 0),
                f"{data_dict.get('soft_rank', 0):.4f}",
                f"{data_dict.get('participation_ratio', 0):.4f}",
                f"{data_dict.get('action_mag', 0):.6f}",
                f"{data_dict.get('emb_mag', 0):.6f}",
                f"{data_dict.get('signal_ratio', 0):.4f}",
                f"{data_dict.get('path_straightening', 0):.4f}",
            ]
            # singular values
            s_vals = data_dict.get("singular_values", [])
            for i in range(10):
                val = s_vals[i] if i < len(s_vals) else 0.0
                row.append(f"{val:.6f}")
            writer.writerow(row)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.log_every_n_steps != 0:
            return

        # 'outputs' contains the dict returned by lejepa_forward
        if not isinstance(outputs, dict) or "emb" not in outputs:
            return

        # 1. Latent Diagnostics (Rank and Variance)
        # emb shape: (B, T, D)
        z = outputs["emb"][:, 0, :]  # Use first timestep for analysis
        diagnostics = self.compute_latent_diagnostics(z)

        pl_module.log("research/soft_rank", diagnostics["soft_rank"])
        pl_module.log(
            "research/participation_ratio", diagnostics["participation_ratio"]
        )

        # 2. Path Straightening (Trajectory Linearity)
        linearity = 1.0
        if outputs["emb"].shape[1] > 1:
            linearity = self.compute_path_straightening(outputs["emb"])
            pl_module.log("research/path_straightening", linearity)

        # 3. Action Signal Metrics
        # batch['action'] shape: (B, T, 32)
        act_mag = batch["action"].abs().mean().item()
        emb_mag = z.abs().mean().item()
        sig_ratio = act_mag / (emb_mag + 1e-8)

        pl_module.log("research/action_mag", act_mag)
        pl_module.log("research/signal_ratio", sig_ratio)

        # 4. Persistent CSV Logging
        csv_data = {
            "step": trainer.global_step,
            "epoch": trainer.current_epoch,
            "soft_rank": diagnostics["soft_rank"],
            "participation_ratio": diagnostics["participation_ratio"],
            "action_mag": act_mag,
            "emb_mag": emb_mag,
            "signal_ratio": sig_ratio,
            "path_straightening": linearity,
            "singular_values": diagnostics["singular_values"],
        }
        self.log_to_csv(csv_data)

        # 5. PCA Visualization (Cloud Geometry)
        if batch_idx % (self.log_every_n_steps * 4) == 0:
            pca_start = time.time()
            self.log_pca_to_wandb(z, trainer.current_epoch, batch_idx)
            pca_duration = time.time() - pca_start
            if pca_duration > 0.5:
                print(f"  ⚠️  PCA Plotting took {pca_duration:.3f}s (Blocking GPU!)")

        # 6. Console Health Check (Immediate Feedback)
        if trainer.global_step == 1:
            print("\n" + "=" * 50)
            print("🩺 LEWM INITIAL HEALTH CHECK (STEP 1)")
            print(f"  - SoftRank:        {diagnostics['soft_rank']:.4f}")
            print(f"  - Participation:   {diagnostics['participation_ratio']:.4f}")
            print(f"  - Action Magnitude: {act_mag:.6f}")
            print(f"  - Embedding Mag:   {emb_mag:.6f}")
            print(f"  - Signal Ratio:    {sig_ratio:.4f}")
            print(
                f"  - Spectral Gap (S0/S1): {diagnostics['singular_values'][0]/(diagnostics['singular_values'][1]+1e-8):.2f}"
            )

            if (
                diagnostics["soft_rank"] < 1.05
                and diagnostics["singular_values"][1] < 1e-6
            ):
                print(
                    "🚨 ALERT: Model is CURRENTLY COLLAPSED. Monitor SigReg gradients."
                )
            else:
                print("✅ MANIFOLD IS BREATHING: Non-zero variance detected.")
            print("=" * 50 + "\n")

    @staticmethod
    def compute_latent_diagnostics(z):
        """
        Performs a single SVD and returns both SoftRank (Entropy)
        and Participation Ratio in a dictionary.
        """
        try:
            # emb: (B, D)
            if z.shape[0] < 2:
                return {
                    "soft_rank": 1.0,
                    "participation_ratio": 1.0,
                    "singular_values": [0.0] * 10,
                }

            # Center the latents (Force float32 for SVD stability)
            z_f32 = z.detach().float()
            z_centered = z_f32 - z_f32.mean(dim=0, keepdim=True)

            # SVD on the centered latent batch (B, D)
            singular_values = torch.linalg.svdvals(z_centered)

            # Avoid division by zero
            s_sum = singular_values.sum()
            if s_sum < 1e-16:
                return {
                    "soft_rank": 1.0,
                    "participation_ratio": 1.0,
                    "singular_values": [0.0] * 10,
                }

            # 1. Spectral Entropy Rank (Diversity focused)
            p = singular_values / s_sum
            entropy = -torch.sum(p * torch.log(p + 1e-12))
            soft_rank = torch.exp(entropy).item()

            # 2. Participation Ratio Rank (Variance focused)
            lambdas = singular_values**2
            pr_rank = (lambdas.sum() ** 2) / (lambdas**2).sum()

            return {
                "soft_rank": max(1.0, soft_rank),
                "participation_ratio": max(1.0, pr_rank.item()),
                "singular_values": singular_values[:10].tolist(),
            }
        except Exception as e:
            return {
                "soft_rank": 1.0,
                "participation_ratio": 1.0,
                "singular_values": [0.0] * 10,
            }

    @staticmethod
    def compute_path_straightening(emb):
        """
        Calculates the average cosine similarity between consecutive
        latent displacement vectors: (z_t - z_t-1) and (z_t+1 - z_t).
        Higher = Straighter/More Predictable trajectories.
        """
        # emb: (B, T, D)
        if emb.shape[1] < 3:
            return 1.0  # Not enough history to compute straightening

        v1 = emb[:, 1] - emb[:, 0]
        v2 = emb[:, 2] - emb[:, 1]

        cos_sim = torch.nn.functional.cosine_similarity(v1, v2, dim=-1)
        return cos_sim.mean().item()

    def log_pca_to_wandb(self, z, epoch, step):
        """
        Generates and logs a 2D PCA cloud of the latents.
        Uses GPU-accelerated PCA to avoid CPU/Bus bottlenecks.
        """
        # z shape: (B, D)
        if z.shape[0] < 2:
            return

        with torch.no_grad():
            # 1. GPU-Native PCA Calculation
            z_f32 = z.detach().float()
            # Mean center the data
            z_centered = z_f32 - z_f32.mean(dim=0, keepdim=True)

            # Use torch's low-rank PCA implementation (Runs on GPU)
            # q=2 for 2 principal components
            try:
                # V is the right-singular vectors (D x 2)
                # S is the singular values
                _, S, V = torch.pca_lowrank(z_centered, q=2)

                # Project the data onto the first 2 principal components
                z_2d_gpu = torch.matmul(z_centered, V[:, :2])

                # 2. Minimum Data Transfer: Only move the 2D coordinates to CPU
                z_2d = z_2d_gpu.cpu().numpy()

                # Calculate explained variance for the labels
                total_var = torch.var(z_centered, dim=0).sum()
                # singular values to variance: s^2 / (n-1)
                varexp = (S**2) / (z.shape[0] - 1)
                ratios = varexp / (total_var + 1e-8)
                ratio_1, ratio_2 = ratios[0].item(), ratios[1].item()

            except Exception as e:
                print(f"  ⚠️  GPU PCA failed: {e}. Falling back to simple plot.")
                z_2d = z_f32[:, :2].cpu().numpy()
                ratio_1, ratio_2 = 0.0, 0.0

        # 3. Local MATPLOTLIB plotting (still on CPU, but input is tiny now)
        plt.figure(figsize=(6, 6))
        plt.scatter(z_2d[:, 0], z_2d[:, 1], alpha=0.6, c="blue")
        plt.title(f"Latent PCA Cloud (Epoch {epoch}, Step {step})")
        plt.xlabel(f"PC1 (Var: {ratio_1:.2%})")
        plt.ylabel(f"PC2 (Var: {ratio_2:.2%})")
        plt.grid(True)

        wandb.log({"visuals/latent_pca": wandb.Image(plt)})
        plt.close()

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Accumulates a small subset of validation latents for manifold plotting."""
        if len(self.val_latents) < self.pca_n_samples:
            # outputs is the result of lejepa_forward
            if "emb" in outputs:
                z = outputs["emb"].detach().float()  # (B, T, D)
                b, t, d = z.shape

                # Fetch temporal steps from frame_indices if available
                # Fallback to range(T) if not found
                steps = torch.arange(t).repeat(b, 1).to(z.device)

                self.val_latents.append(z.cpu())
                self.val_steps.append(steps.cpu())

    def on_validation_epoch_end(self, trainer, pl_module):
        """Generates the 3D Manifold Plot at the end of the validation epoch."""
        if not self.val_latents:
            return

        try:
            # 1. Prepare Data
            z = torch.cat(self.val_latents, dim=0)  # (N, T, D)
            steps = torch.cat(self.val_steps, dim=0)  # (N, T)

            n, t, d = z.shape
            z_flat = z.view(-1, d)
            steps_flat = steps.view(-1)

            # Subsample for plotting efficiency
            if z_flat.shape[0] > 2048:
                indices = torch.randperm(z_flat.shape[0])[:2048]
                z_flat = z_flat[indices]
                steps_flat = steps_flat[indices]

            # 2. 3D PCA Projection
            z_centered = z_flat - z_flat.mean(dim=0, keepdim=True)
            U, S, V = torch.pca_lowrank(z_centered, q=3)
            z_3d = torch.matmul(z_centered, V[:, :3]).numpy()

            # 4. Topology Health Report (Calibrated to GR-1 Baselines)
            # Baseline (Single-View) SoftRank is ~45.
            # We expect Multi-View to push this higher as it resolves geometric ambiguities.
            avg_soft_rank = self.compute_latent_diagnostics(z_flat)["soft_rank"]

            print(
                f"\n📊 --- TOPOLOGY HEALTH REPORT (Epoch {trainer.current_epoch}) ---"
            )
            print(f"    - Avg SoftRank:  {avg_soft_rank:.4f}")
            if avg_soft_rank < 30.0:
                print("    - Status:        🚨 COLLAPSED (Significant Variance Loss)")
            elif avg_soft_rank < 55.0:
                print("    - Status:        🟡 STAGNANT (Baseline Performance)")
            else:
                print("    - Status:        ✅ UNRAVELING (High Geometric Diversity)")
            print("    ---------------------------------------------\n")

            # 3. Create 3D Plot
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")

            scatter = ax.scatter(
                z_3d[:, 0],
                z_3d[:, 1],
                z_3d[:, 2],
                c=steps_flat.numpy(),
                cmap="viridis",
                alpha=0.6,
                s=10,
            )

            plt.colorbar(scatter, label="Temporal Step (0-31)")
            ax.set_title(f"LeWM Manifold Topology (Epoch {trainer.current_epoch})")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_zlabel("PC3")

            # Log to WandB
            wandb.log(
                {
                    "visuals/manifold_3d": wandb.Image(plt),
                    "research/epoch": trainer.current_epoch,
                }
            )
            plt.close()

            print(f"📊 Manifold Plot generated for Epoch {trainer.current_epoch}")

        except Exception as e:
            print(f"⚠️ Manifold Plotting failed: {e}")

        # Clear buffers for next epoch
        self.val_latents = []
        self.val_steps = []
