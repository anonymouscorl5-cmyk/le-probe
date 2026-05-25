import torch
import torch.nn as nn
from einops import rearrange
import stable_pretraining as spt


class _EncoderOutput:
    def __init__(self, last_hidden_state: torch.Tensor):
        self.last_hidden_state = last_hidden_state


class LegacySingleViewEncoder(nn.Module):
    """
    Plain HF ViT for May-2026 single-view JEPA checkpoints (``encoder.*`` keys).

    Accepts ``(B, T, V, C, H, W)`` with ``V=1`` — same contract as ``LateFusionEncoder``.
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.config = backbone.config

    def forward(self, x, interpolate_pos_encoding=True):
        b, t, v, c, h, w = x.shape
        if v != 1:
            raise ValueError(
                f"LegacySingleViewEncoder expects V=1, got V={v} "
                "(use LateFusion for multi-view checkpoints)"
            )
        flat = rearrange(x, "b t v c h w -> (b t v) c h w")
        output = self.backbone(flat, interpolate_pos_encoding=interpolate_pos_encoding)
        cls = output.last_hidden_state[:, 0]
        return _EncoderOutput(rearrange(cls, "(b t) d -> b t d", b=b, t=t))


def infer_checkpoint_encoder_style(state_dict: dict) -> str:
    """``legacy_vit`` (encoder.embeddings.*) vs ``late_fusion`` (encoder.backbone.*)."""
    keys = []
    for k in state_dict:
        keys.append(k.replace("model.", "", 1) if k.startswith("model.") else k)
    if any(k.startswith("encoder.backbone.") for k in keys):
        return "late_fusion"
    if any(k.startswith("encoder.embeddings.") for k in keys) and not any(
        k.startswith("encoder.fusion_layer.") for k in keys
    ):
        return "legacy_vit"
    return "late_fusion"


def build_legacy_single_view_encoder(cfg):
    """ViT backbone only — matches pre–late-fusion single-view ``gr1_reward_tuned_v2``."""
    print("📷 INITIALIZING LEGACY SINGLE-VIEW ViT (plain encoder.* checkpoint)...")
    backbone = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )
    return LegacySingleViewEncoder(backbone)


class LateFusionEncoder(nn.Module):
    """
    Shared Encoder + Late Fusion for Multi-View Robotic Manipulation.
    Processes each view independently using a standard 2D ViT and then fuses them.
    Leverages 100% of pretrained 2D weights to prevent manifold collapse.
    """

    def __init__(self, backbone, embed_dim=192, fusion="mean", num_views=5):
        super().__init__()
        self.backbone = backbone
        self.fusion = fusion
        self.embed_dim = embed_dim
        self.num_views = num_views

        # To match HF ViT API for the predictor
        self.config = backbone.config

        if self.fusion == "linear":
            self.fusion_layer = nn.Linear(embed_dim * self.num_views, embed_dim)
        elif self.fusion == "attention":
            self.fusion_query = nn.Parameter(torch.randn(1, 1, embed_dim))
            self.fusion_attn = nn.MultiheadAttention(
                embed_dim, num_heads=3, batch_first=True
            )

    def forward(self, x, interpolate_pos_encoding=True):
        """
        x: (B, T, V, C, H, W)
        """
        b, t, v, c, h, w = x.shape

        # 1. Fold views into batch dimension for parallel processing
        # (B, T, V, C, H, W) -> (B*T*V, C, H, W)
        x = rearrange(x, "b t v c h w -> (b t v) c h w")

        # 2. Pass through shared 2D Encoder
        output = self.backbone(x, interpolate_pos_encoding=interpolate_pos_encoding)
        # Extract CLS token: (B*T*V, D)
        z = output.last_hidden_state[:, 0]

        # 3. Unfold
        z = rearrange(z, "(b t v) d -> b t v d", b=b, t=t, v=v)

        # 4. Fusion
        if self.fusion == "mean":
            fused = z.mean(dim=2)  # (B, T, D)
        elif self.fusion == "linear":
            fused = rearrange(z, "b t v d -> (b t) (v d)")
            fused = self.fusion_layer(fused)
            fused = rearrange(fused, "(b t) d -> b t d", b=b, t=t)
        elif self.fusion == "attention":
            # z: (B, T, V, D)
            b_size, t_size, v_size, d_size = z.shape
            z_flat = rearrange(z, "b t v d -> (b t) v d")
            query = self.fusion_query.expand(b_size * t_size, 1, -1)
            # Use learned query to attend over view tokens
            fused, _ = self.fusion_attn(query, z_flat, z_flat)
            fused = rearrange(fused, "(b t) 1 d -> b t d", b=b, t=t)
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion}")

        return _EncoderOutput(fused)


def get_multi_view_encoder(cfg):
    """Factory function for creating the Late Fusion encoder."""
    print("⚓ INITIALIZING LATE FUSION (Shared 2D Encoder)...")

    # Use standard single-view backbone
    backbone = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,  # Weights will be loaded by the main train script
        use_mask_token=False,
    )

    # Determine number of views
    if not cfg.get("use_multi_view", True):
        num_views = 1
    else:
        # Default to 5 for standard GR1 multi-view if not specified
        num_views = cfg.get("num_views", 5)

    return LateFusionEncoder(
        backbone,
        embed_dim=backbone.config.hidden_size,
        fusion=cfg.get("fusion_type", "mean"),
        num_views=num_views,
    )
