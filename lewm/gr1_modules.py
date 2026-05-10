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


class MultiViewJEPA(JEPA):
    """
    Improved JEPA for Multi-View and Spatiotemporal Tubelets.
    Overrides encode() to pass (B, T, V, C, H, W) to the encoder
    instead of flattening T into the batch dimension.
    """

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
