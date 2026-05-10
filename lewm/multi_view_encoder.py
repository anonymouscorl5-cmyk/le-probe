import torch
import torch.nn as nn
from einops import rearrange, repeat
from timm.models.vision_transformer import Block


class MultiViewTubeletEncoder(nn.Module):
    """
    Spatiotemporal Tubelet Encoder for Multi-View Robotic Manipulation.
    Implements 3D RoPE and Tubelet Tokenization to fuse multiple camera views.

    Expected input shape: (B, T, V, C, H, W) where T is history and V is the number of views.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=14,
        in_chans=3,
        num_views=5,
        num_frames=3,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_views = num_views
        self.num_frames = num_frames
        self.embed_dim = embed_dim

        # 1. 4D Tubelet Tokenization (Time x Views x Space)
        # We treat each (T, V) pair as depth in a 3D Conv
        # This encapsulates spatiotemporal info into tokens
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size),
        )

        self.num_patches = (img_size // patch_size) ** 2
        # Tokens are per (B, T). Each T gets its own V*N tokens.
        # Or we can merge T into tokens as well.
        # JEPA usually wants (B, T, D) as output.
        # To get (B, T, D), we process each T independently but use multi-view info.
        # For full 4D, we would output one token for the whole sequence.
        # Let's stick to (B, T, V, N) tokens and then pool to (B, T, D).

        self.tot_tokens = num_views * self.num_patches

        # 2. CLS Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 3. 3D Learned Positional Embeddings (Fallback for RoPE for now)
        # We use learned embeddings for View, X, and Y dimensions
        self.pos_embed = nn.Parameter(torch.zeros(1, self.tot_tokens + 1, embed_dim))

        # 4. Transformer Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

        # Match HF ViT API
        class Config:
            def __init__(self, hidden_size):
                self.hidden_size = hidden_size

        self.config = Config(embed_dim)

        # Initialization
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, interpolate_pos_encoding=True):
        """
        x: (B, T, V, C, H, W)
        """
        b, t, v, c, h, w = x.shape

        # Flatten B, T to process each time step's multi-view set
        x = rearrange(x, "b t v c h w -> (b t) c v h w")

        # Project to tokens
        x = self.proj(x)  # (B*T, D, V, H', W')
        x = rearrange(x, "bt d v h w -> bt (v h w) d")

        # Append CLS token
        cls_tokens = repeat(self.cls_token, "() n d -> bt n d", bt=x.shape[0])
        x = torch.cat((cls_tokens, x), dim=1)

        # Add Positional Embeddings
        x = x + self.pos_embed

        # Transformer Blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Extract CLS token as the latent for each time step
        # Output shape: (B*T, D)
        cls_out = x[:, 0]

        # Wrap in Output class
        class Output:
            def __init__(self, last_hidden_state):
                self.last_hidden_state = last_hidden_state

        return Output(cls_out)


def get_multi_view_encoder(cfg):
    """Factory function for creating the encoder from config."""
    # Mapping scale names to dims/depths (matching ViT tiny/small/base)
    scales = {
        "tiny": {"dim": 192, "depth": 12, "heads": 3},
        "small": {"dim": 384, "depth": 12, "heads": 6},
        "base": {"dim": 768, "depth": 12, "heads": 12},
    }

    params = scales.get(cfg.encoder_scale, scales["tiny"])

    return MultiViewTubeletEncoder(
        img_size=cfg.img_size,
        patch_size=cfg.patch_size,
        num_views=5,  # Standard for GR-1 in this setup
        num_frames=cfg.wm.history_size,
        embed_dim=params["dim"],
        depth=params["depth"],
        num_heads=params["heads"],
    )
