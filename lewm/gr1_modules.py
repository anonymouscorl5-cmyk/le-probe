# --- Path Stabilization ---
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# --------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jepa import JEPA
from lewm.skeleton.dino_constants import DINO_DIM, num_dino_views


class MultiViewJEPA(JEPA):
    """
    Improved JEPA for Multi-View and Spatiotemporal Tubelets.
    Overrides encode() to pass (B, T, V, C, H, W) to the encoder
    instead of flattening T into the batch dimension.
    """

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        embed_dim=192,
        dino_dim=384,
        use_dino: bool = True,
        fusion_type: str = "mean",
        num_views: int | None = None,
    ):
        super().__init__(encoder, predictor, action_encoder, projector, pred_proj)
        self.use_dino = use_dino
        self.dino_dim = dino_dim
        self.fusion_type = fusion_type

        if use_dino:
            n_views = num_views if num_views is not None else num_dino_views()
            # 1. High-Level Waypoint Latent Predictor (Vector Reward Head)
            # Input: current latent z_t (embed_dim) + phase one-hot (4)
            self.high_level_predictor = HWMPredictor(
                input_dim=embed_dim + 4,
                output_dim=embed_dim,
                hidden_dim=512,
            )

            # 2. Trainable DINO Latent Projection MLP
            # Input: frozen DINOv3 feature dimension (dino_dim)
            self.dino_projector = DINOProjector(
                input_dim=dino_dim,
                output_dim=embed_dim,
                hidden_dim=512,
            )

            # Separate view fusion for DINO subgoals (not shared with encoder weights)
            if fusion_type == "linear":
                self.dino_fusion_layer = nn.Linear(embed_dim * n_views, embed_dim)
            elif fusion_type == "attention":
                self.dino_fusion_query = nn.Parameter(torch.randn(1, 1, embed_dim))
                self.dino_fusion_attn = nn.MultiheadAttention(
                    embed_dim, num_heads=3, batch_first=True
                )

    def predict_subgoal(self, z_t, phase_idx):
        """
        Predicts the 192-dimensional latent subgoal target for the next phase checkpoint.
        z_t: (B, D) or (B, T, D)
        phase_idx: (B, 1) or (B, T, 1)
        """
        if not self.use_dino:
            raise RuntimeError(
                "predict_subgoal requires use_dino=True (use --use_dino)"
            )
        device = z_t.device

        # Keep dims clean if sequence dimension is present
        is_seq = z_t.dim() == 3
        if is_seq:
            B, T, D = z_t.shape
            z_t_flat = rearrange(z_t, "b t d -> (b t) d")
            phase_flat = rearrange(phase_idx, "b t 1 -> (b t) 1")
        else:
            z_t_flat = z_t
            phase_flat = phase_idx

        # Convert phase index to 4-dimensional one-hot tensor
        phase_onehot = (
            F.one_hot(phase_flat.squeeze(-1).long(), num_classes=4).float().to(device)
        )

        # Predict macro subgoal coordinate
        mlp_input = torch.cat([z_t_flat, phase_onehot], dim=-1)
        subgoal_flat = self.high_level_predictor(mlp_input)

        if is_seq:
            return rearrange(subgoal_flat, "(b t) d -> b t d", b=B, t=T)
        return subgoal_flat

    def project_dino(self, phi_dino, aggregate_views: bool = True):
        """
        Projects frozen DINOv3 embeddings into latent space.

        Input must be (..., V, 384) with V = num_dino_views(); per-view projection
        then fusion_type (mean/linear/attention) when aggregate_views=True.
        """
        if not self.use_dino:
            raise RuntimeError("project_dino requires use_dino=True (use --use_dino)")

        n_views = num_dino_views()
        if phi_dino.shape[-1] != self.dino_dim or phi_dino.shape[-2] != n_views:
            raise ValueError(
                f"Expected dino_anchor shape (..., {n_views}, {self.dino_dim}), "
                f"got {tuple(phi_dino.shape)}"
            )

        lead = phi_dino.shape[:-2]
        flat = rearrange(phi_dino, "... v d -> (...) d")
        proj = self.dino_projector(flat)
        proj = proj.view(*lead, n_views, -1)
        if aggregate_views:
            return self._fuse_dino_views(proj)
        return proj

    def _fuse_dino_views(self, proj: torch.Tensor) -> torch.Tensor:
        """Fuse per-view projected DINO latents (..., V, D) -> (..., D)."""
        if self.fusion_type == "mean":
            return proj.mean(dim=-2)

        lead = proj.shape[:-2]
        v, d = proj.shape[-2], proj.shape[-1]

        if self.fusion_type == "linear":
            flat = rearrange(proj, "... v d -> (...) (v d)")
            fused = self.dino_fusion_layer(flat)
            return fused.view(*lead, d)

        if self.fusion_type == "attention":
            flat = rearrange(proj, "... v d -> (...) v d")
            n = flat.shape[0]
            query = self.dino_fusion_query.expand(n, 1, -1)
            fused, _ = self.dino_fusion_attn(query, flat, flat)
            return fused.squeeze(1).view(*lead, d)

        raise ValueError(f"Unknown fusion type: {self.fusion_type}")

    def encode(self, info):
        """
        Encode multi-view observations.
        info['pixels']: (B, T, V, C, H, W)
        """
        pixels = info["pixels"].float()
        b, t = pixels.shape[:2]

        # Pass the full (B, T, V, C, H, W) to the encoder
        # The encoder should handle the spatiotemporal tokenization
        output = self.encoder(pixels, interpolate_pos_encoding=True)

        # The encoder should return (B, T, D) for the predictor
        # or (B*T, D) which we then rearrange.
        emb = output.last_hidden_state
        if emb.dim() == 2:  # (B*T, D)
            emb = rearrange(emb, "(b t) d -> b t d", b=b, t=t)

        # Project if needed
        pixels_emb = rearrange(emb, "b t d -> (b t) d")
        pixels_emb = self.projector(pixels_emb)
        info["emb"] = rearrange(pixels_emb, "(b t) d -> b t d", b=b, t=t)

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info


class HWMPredictor(nn.Module):
    """
    High-Level Latent MLP Predictor for HWM Waypoints.
    Utilizes LayerNorm and GELU for batch-size-independent planning stability.
    """

    def __init__(self, input_dim, output_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class DINOProjector(nn.Module):
    """
    Trainable DINO Latent Projection Head.
    Maps frozen DINOv3 visual representations to the world model's latent space.
    """

    def __init__(self, input_dim=384, output_dim=192, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class GR1Embedder(nn.Module):
    """
    Robust action encoder with residual connections for GR-1.
    Designed to prevent the 32-DoF signal from being 'crushed'
    by high-magnitude vision embeddings.
    """

    def __init__(
        self,
        input_dim=10,
        smoothed_dim=256,
        emb_dim=10,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, smoothed_dim)

        # Residual Block
        self.residual_net = nn.Sequential(
            nn.LayerNorm(smoothed_dim),
            nn.Linear(smoothed_dim, smoothed_dim * 2),
            nn.GELU(),
            nn.Linear(smoothed_dim * 2, smoothed_dim),
            nn.Dropout(0.05),
        )

        self.output_proj = nn.Linear(smoothed_dim, emb_dim)

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        # Project raw actions to hidden space
        h = self.input_proj(x)

        # Apply residual transformation
        h = h + self.residual_net(h)

        # Project to final embedding dimension
        return self.output_proj(h)


class GR1MLP(nn.Module):
    """
    Standard MLP for GR-1 Projector.
    Matches the quentinll/lewm-cube architecture (Linear -> BN -> GELU -> Linear).
    """

    def __init__(self, input_dim, output_dim, hidden_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D) or (B, D)
        """
        return self.net(x)
