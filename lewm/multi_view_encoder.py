import torch
import torch.nn as nn
from einops import rearrange
import stable_pretraining as spt


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

        if self.fusion == "learned":
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
        elif self.fusion == "learned":
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

        # Wrap in Output class for JEPA compatibility
        class Output:
            def __init__(self, last_hidden_state):
                self.last_hidden_state = last_hidden_state

        return Output(fused)


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

    return LateFusionEncoder(
        backbone,
        embed_dim=backbone.config.hidden_size,
        fusion=cfg.get("fusion_type", "mean"),
        num_views=len(cfg.data.dataset.get("keys", ["world_center"])),
    )
