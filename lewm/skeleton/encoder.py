import torch
import torch.nn as nn
import stable_pretraining as spt
from lewm.multi_view_encoder import LateFusionEncoder


def patch_vit_for_skeleton(backbone):
    """
    Assertive expansion of the ViT backbone to support 4 channels.
    Targets the stable_pretraining / HuggingFace ViT architecture directly.
    """
    # 1. Expand the projection layer
    old_proj = backbone.embeddings.patch_embeddings.projection

    new_proj = nn.Conv2d(
        in_channels=4,
        out_channels=old_proj.out_channels,
        kernel_size=old_proj.kernel_size,
        stride=old_proj.stride,
        padding=old_proj.padding,
        bias=(old_proj.bias is not None),
    )

    with torch.no_grad():
        new_proj.weight[:, :3, :, :] = old_proj.weight.clone()
        new_proj.weight[:, 3:, :, :] = 0.0  # Zero-Init for stability
        if old_proj.bias is not None:
            new_proj.bias.copy_(old_proj.bias)

    # 2. Swap the physical layer
    backbone.embeddings.patch_embeddings.projection = new_proj

    # 3. Update internal configuration attributes to bypass shape validation
    backbone.config.num_channels = 4
    backbone.embeddings.patch_embeddings.num_channels = 4

    return backbone


class SkeletonEncoder(LateFusionEncoder):
    """
    A 4-channel version of the LateFusionEncoder that incorporates
    skeletal priors as an auxiliary input stream.
    """

    def __init__(self, backbone, embed_dim=192, fusion="linear", num_views=5):
        patched_backbone = patch_vit_for_skeleton(backbone)
        super().__init__(
            patched_backbone, embed_dim=embed_dim, fusion=fusion, num_views=num_views
        )
        print(f"🦾 SkeletonEncoder initialized with 4-channel backbone.")


def get_skeleton_encoder(cfg):
    """Factory function for creating the 4-channel Skeleton encoder."""
    print("🦴 INITIALIZING SKELETON-PRIOR ENCODER...")

    backbone = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )

    return SkeletonEncoder(
        backbone,
        embed_dim=backbone.config.hidden_size,
        fusion=cfg.get("fusion_type", "linear"),
        num_views=cfg.get("num_views", 5),
    )
