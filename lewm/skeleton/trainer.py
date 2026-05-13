import os
import sys
import torch
import numpy as np
import hydra
import lightning as pl
import stable_pretraining as spt
from functools import partial
from pathlib import Path
from omegaconf import OmegaConf, open_dict
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

# --- Path Stabilization (Robust Repo Root Targeting) ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Add the 'lewm' directory itself to allow direct imports like train_lewm.py does
LEWM_DIR = os.path.join(REPO_ROOT, "lewm")
if LEWM_DIR not in sys.path:
    sys.path.append(LEWM_DIR)

# Also add the le_wm submodule for its internal direct imports (utils, module)
LEWM_ROOT = os.path.join(LEWM_DIR, "le_wm")
if LEWM_ROOT not in sys.path:
    sys.path.append(LEWM_ROOT)
# -------------------------------------------------------

# Project Imports
from lewm.train_lewm import lejepa_forward, RewardPredictor, SIGReg
from lewm.skeleton.encoder import get_skeleton_encoder
from lewm.skeleton.data import SkeletonDataPlugin
from lewm.gr1_modules import GR1Embedder, GR1MLP, MultiViewJEPA
from metrics import MetricsCallback
from utils import get_img_preprocessor, ModelObjectCallBack
from stable_pretraining.optim.lr_scheduler import LinearWarmupCosineAnnealingLR

# Submodule direct imports
from module import ARPredictor, SIGReg


class SkeletonImportanceCallback(pl.Callback):
    """
    Logs the relative importance of the 4th channel (Skeleton)
    compared to the RGB channels during training.
    """

    def on_train_epoch_end(self, trainer, pl_module):
        # Target the patched projection layer
        # path: model.encoder.backbone.embeddings.patch_embeddings.projection.weight
        try:
            # Navigate to the backbone (handles LateFusion wrapper nesting)
            backbone = pl_module.model.encoder.backbone
            weight = backbone.embeddings.patch_embeddings.projection.weight

            # Calculate Mean Absolute Weights for comparison
            rgb_weight_norm = weight[:, :3, :, :].abs().mean()
            skel_weight_norm = weight[:, 3:, :, :].abs().mean()

            importance_ratio = skel_weight_norm / (rgb_weight_norm + 1e-8)

            # Log to WandB/Logger for real-time manifold monitoring
            pl_module.log_dict(
                {
                    "skeleton/weight_norm": skel_weight_norm,
                    "skeleton/rgb_norm_ratio": importance_ratio,
                },
                sync_dist=True,
            )

            print(f"\n📊 [SKELETON AUDIT] Epoch {trainer.current_epoch}:")
            print(f"   - Skeleton Weight Norm: {skel_weight_norm:.6f}")
            print(f"   - Relative Importance:   {importance_ratio*100:.2f}% of RGB")
        except Exception:
            pass


def lejepa_forward_bips(self, batch, stage, cfg):
    """
    Augmented JEPA forward pass with Bi-Directional Perceptual Shaping (BiPS).
    """
    pixels = batch["pixels"]  # [B, T, V, 4, H, W]

    if stage == "train":
        rand = torch.rand(1).item()

        # 1. Skeletal Dropout (10%): Force reliance on geometry
        # By zeroing out RGB, we force the latent manifold to ground itself in the 4th channel.
        if rand < 0.10:
            pixels[:, :, :, :3, :, :] = 0.0

        # 2. Structural Reliance (5%): Force hallucination of interactions
        # We mask RGB patches only where the skeleton exists, forcing texture reconstruction from priors.
        elif rand < 0.15:
            skeleton_mask = (pixels[:, :, :, 3:, :, :] > 0.1).float()
            pixels[:, :, :, :3, :, :] *= 1.0 - skeleton_mask

    batch["pixels"] = pixels
    return lejepa_forward(self, batch, stage, cfg)


@hydra.main(version_base=None, config_path="../config", config_name="lewm")
def run(cfg):
    print("🦾 Starting Skeleton-Prior Augmented Training (BiPS)...")

    # 1. Seed Everything
    pl.seed_everything(cfg.get("seed", 3072), workers=True)

    # 2. Data Ingestion & Transformation setup
    repo_id = cfg.data.dataset.get("repo_id", "vedpatwardhan/gr1_pickup_grasp")
    keys_to_load = [
        "observation.state",
        "action",
        "world_center",
        "world_left",
        "world_right",
        "world_top",
        "world_wrist",
    ]

    dataset = SkeletonDataPlugin(
        repo_id=repo_id,
        keys_to_load=keys_to_load,
        num_steps=cfg.wm.history_size + cfg.wm.num_preds,
        use_virtual_actions=cfg.data.get("use_virtual_actions", True),
        use_multi_view=True,
        img_size=cfg.img_size,
    )

    # Apply Image Preprocessors & Standard Normalization (Z-Score)
    transforms = []
    with open_dict(cfg):
        for col in keys_to_load:
            # A. Image Preprocessing (Includes skeleton keys if present)
            if any(k in col for k in ["pixels", "images", "world_"]):
                transforms.append(
                    get_img_preprocessor(source=col, target=col, img_size=cfg.img_size)
                )
                # Handle the generated skeleton stream as well
                if "world_" in col:
                    transforms.append(
                        get_img_preprocessor(
                            source=f"{col}_skeleton",
                            target=f"{col}_skeleton",
                            img_size=cfg.img_size,
                        )
                    )
            else:
                # B. State/Action Z-Score Normalization
                col_data = dataset.get_col_data(col)
                data_tensor = torch.from_numpy(np.array(col_data))
                data_tensor = data_tensor[~torch.isnan(data_tensor).any(dim=1)]
                mean = data_tensor.mean(0, keepdim=True).clone()
                std = data_tensor.std(0, keepdim=True).clone()

                def norm_fn(x, m=mean, s=std):
                    return ((x - m) / (s + 1e-8)).float()

                transforms.append(
                    spt.data.transforms.WrapTorchTransform(
                        norm_fn, source=col, target=col
                    )
                )

                # C. DYNAMIC DIMENSION DETECTION
                col_dim = dataset.get_dim(col)
                clean_name = col.split(".")[-1]
                setattr(cfg.wm, f"{clean_name}_dim", col_dim)
                print(f"📊 Auto-detected {col} dimension ({clean_name}_dim): {col_dim}")

    dataset.transform = spt.data.transforms.Compose(*transforms)

    # 2. Architecture Initialization (4-channel expanded backbone)
    encoder = get_skeleton_encoder(cfg)
    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    world_model = MultiViewJEPA(
        encoder=encoder,
        predictor=ARPredictor(
            num_frames=cfg.wm.history_size,
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            **cfg.predictor,
        ),
        action_encoder=GR1Embedder(input_dim=effective_act_dim, emb_dim=embed_dim),
        projector=GR1MLP(input_dim=hidden_dim, output_dim=embed_dim, hidden_dim=2048),
        pred_proj=GR1MLP(input_dim=hidden_dim, output_dim=embed_dim, hidden_dim=2048),
    )
    world_model.reward_head = RewardPredictor(input_dim=embed_dim, hidden_dim=512)

    # 4. Training Module setup with BiPS Forward
    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": lambda optimizer, module: LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_steps=max(
                    1,
                    int(
                        0.01
                        * getattr(module.trainer, "estimated_stepping_batches", 100)
                    ),
                ),
                max_steps=getattr(module.trainer, "estimated_stepping_batches", 1000),
                warmup_start_lr=1e-5,
            ),
            "interval": "epoch",
        },
    }

    world_model_module = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward_bips, cfg=cfg),
        optim=optimizers,
    )

    # 4. Logger & Callbacks
    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)

    # 5. Data Loading (90/10 Split Parity)
    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=cfg.loader.batch_size,
        num_workers=cfg.loader.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.loader.num_workers > 0,
        generator=rnd_gen,
    )

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=cfg.loader.batch_size,
        num_workers=cfg.loader.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        persistent_workers=cfg.loader.num_workers > 0,
    )

    run_id = cfg.get("subdir") or "gr1_skeleton_official"
    run_dir = Path("./outputs", run_id).absolute()
    run_dir.mkdir(parents=True, exist_ok=True)

    # 6. Lightning Launch
    trainer = pl.Trainer(
        **cfg.trainer,
        default_root_dir=run_dir,
        logger=logger,
        callbacks=[
            SkeletonImportanceCallback(),
            ModelObjectCallBack(
                dirpath=run_dir,
                filename="skeleton_lewm",
                epoch_interval=cfg.get("save_interval", 1),
            ),
            MetricsCallback(log_every_n_steps=1),
            ModelCheckpoint(
                dirpath=run_dir / "checkpoints",
                every_n_epochs=cfg.get("save_interval", 1),
            ),
        ],
    )

    print(f"🚀 Launching BiPS Training Loop (Batch Size: {cfg.loader.batch_size})...")
    trainer.fit(
        model=world_model_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )


if __name__ == "__main__":
    run()
