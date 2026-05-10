# --- Path Stabilization ---
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# --------------------------

import os
import sys
import time
from functools import partial
from pathlib import Path
import numpy as np
import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from huggingface_hub import hf_hub_download
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf, open_dict
from stable_pretraining.optim.lr_scheduler import LinearWarmupCosineAnnealingLR
import pkg_resources

# Ensure we can import from the le_wm submodule
LEWM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "le_wm"))
sys.path.append(LEWM_ROOT)

# Import official LeWM components
from jepa import JEPA
from module import ARPredictor, SIGReg
from gr1_modules import GR1Embedder, GR1MLP, MultiViewJEPA
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack
from lewm_data_plugin import LEWMDataPlugin
from metrics import MetricsCallback
from multi_view_encoder import get_multi_view_encoder


class RewardPredictor(torch.nn.Module):
    """RA-LeWM MLP Head for predicting future rewards from World Model latents."""

    def __init__(self, input_dim, hidden_dim=512):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.mlp(x)


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""
    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds

    # --- ENHANCED DIAGNOSTIC PROBE ---
    # PREPARATION
    # Pixels are already normalized by the dataloader transforms [0, 1] -> ImageNet norm
    pixels = batch["pixels"]
    actions = torch.nan_to_num(batch["action"], 0.0)

    # Forward pass through model
    info = {"pixels": pixels, "action": actions}
    output = self.model.encode(info)
    emb = output["emb"]  # (B, T, D)
    self.last_z = emb.detach()

    if self.trainer.global_step == 0:
        print(f"\n🩺 [STEP 0] DATA HEALTH CHECK:")
        print(f"  - Pixel Shape:    {pixels.shape}")
        if cfg.get("use_multi_view", True):
            print(f"  - Multi-View:     Detected {pixels.shape[2]} views.")
        print(f"  - Pixel Range:    [{pixels.min():.2f}, {pixels.max():.2f}]")
        print(f"  - Pixel Mean/Var: {pixels.mean():.4f} / {pixels.var():.8f}")

        # BATCH UNIQUENESS CHECK
        if pixels.shape[0] > 1:
            px_diff = (pixels[0] - pixels[1]).abs().var()
            act_diff = (batch["action"][0] - batch["action"][1]).abs().var()
            print(f"  - Batch Variance (Sample 0 vs 1):")
            print(f"    - Pixel Diff Var:  {px_diff:.8f}")
            print(f"    - Action Diff Var: {act_diff:.8f}")
            if px_diff < 1e-8:
                print("🚨 CRITICAL: BATCH IS CLONED! Sample 0 and 1 are identical.")

        emb_diff = (emb[0] - emb[1]).abs().var() if emb.shape[0] > 1 else 0.0
        print(f"  - Latent Variance:  {emb.var():.8f}")
        print(f"  - Latent Diff Var:  {emb_diff:.8f}")
        if emb_diff < 1e-8:
            print("🚨 ALERT: Latent manifold has zero batch variance.")
        print("---------------------------------\n")

    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]

    tgt_emb = emb[:, n_preds:]  # label
    pred_emb = self.model.predict(ctx_emb, ctx_act)  # pred

    # LeWM loss (Force SIGReg to float32 for SVD stability)
    raw_pred_loss = (pred_emb - tgt_emb).pow(2).mean(dim=-1)

    if "progress" in batch:
        R = batch["progress"].to(pred_emb.dtype)
        # Reward at start of prediction vs targets
        R_t = R[:, ctx_len - 1].unsqueeze(1)
        R_tk = R[:, ctx_len:]

        # Exponential Mask Weighting
        delta_R = R_tk - R_t
        kappa = cfg.get("rabc_kappa", 0.01)
        w_i = torch.exp(kappa * delta_R)
        output["pred_loss"] = (raw_pred_loss * w_i).mean()

        # Reward Supervision (Value Predictor)
        # Align reward prediction with the future latents (steps ctx_len:)
        # Since pred_emb is [1, 2, 3] and ctx_len is 3, we take the last steps
        future_pred_emb = pred_emb[:, (ctx_len - n_preds) :]
        pred_reward = self.model.reward_head(future_pred_emb).squeeze(-1)

        # Ensure target shape matches prediction
        target_reward = R_tk.to(pred_reward.dtype)
        output["reward_loss"] = torch.nn.functional.mse_loss(pred_reward, target_reward)
    else:
        output["pred_loss"] = raw_pred_loss.mean()

    output["sigreg_loss"] = self.sigreg(emb.float().transpose(0, 1))

    # Combined Loss with weighting
    reward_weight = cfg.loss.get("reward", {}).get("weight", 0.1)
    sigreg_weight = cfg.loss.sigreg.weight

    output["loss"] = output["pred_loss"] + sigreg_weight * output["sigreg_loss"].to(
        output["pred_loss"].dtype
    )

    if "reward_loss" in output:
        output["loss"] = output["loss"] + reward_weight * output["reward_loss"]

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    self._step_end_time = time.time()
    return output


@hydra.main(version_base=None, config_path="./config", config_name="lewm")
def run(cfg):
    print("🏗️  Initializing GR-1 Implementation of LeWorldModel...")

    #########################
    ##       dataset       ##
    #########################

    # 1. Initialize the Data Plugin (LeRobot -> LeWorldModel Shim)
    # We prioritize the 'processed' dataset for significant training speedups.
    # Note: Even if a checkpoint reloads an old repo_id, we override it here for performance.
    repo_id = cfg.data.dataset.get("repo_id", "vedpatwardhan/gr1_pickup_grasp")
    print(f"📦 Initializing Data Plugin for: {repo_id}")

    # Standard keys to load if not specified in config
    default_keys = [
        "observation.state",
        "observation.images.world_center",
        "action",
    ]
    # Check both data and data.dataset for keys_to_load
    keys_to_load = (
        cfg.data.get("keys_to_load")
        or cfg.data.dataset.get("keys_to_load")
        or default_keys
    )

    dataset = LEWMDataPlugin(
        repo_id=repo_id,
        keys_to_load=keys_to_load,
        num_steps=cfg.wm.history_size + cfg.wm.num_preds,
        use_virtual_actions=cfg.data.get("use_virtual_actions", True),
        use_multi_view=cfg.get("use_multi_view", True),
        img_size=cfg.img_size,
    )

    # 2. Data Integrity Guard (Sanity Check)
    print("\n🔍 DATA INTEGRITY GUARD: Inspecting raw pixels...")
    raw_sample = dataset[0]
    raw_pixels = raw_sample["pixels"]  # (T, C, H, W)

    p_max = raw_pixels.max().item()
    p_min = raw_pixels.min().item()
    p_dtype = raw_pixels.dtype
    # Calculate channel means to verify RGB signature (R=104, B=90)
    # Handle both (T, C, H, W) and (T, V, C, H, W)
    if raw_pixels.ndim == 5:
        p_means = raw_pixels.float().mean(dim=(0, 1, 3, 4))
    else:
        p_means = raw_pixels.float().mean(dim=(0, 2, 3))

    print(f"  - Dtype: {p_dtype}")
    print(f"  - Range: [{p_min}, {p_max}]")
    print(
        f"  - Channel Means (R,G,B): [{p_means[0].item():.2f}, {p_means[1].item():.2f}, {p_means[2].item():.2f}]"
    )

    if p_max > 255:
        raise RuntimeError(
            f"🚨 DATA CORRUPTION DETECTED: Pixel value {p_max} exceeds 255. Overflow fix is missing!"
        )
    if p_means[2] > p_means[0]:
        print("  ✅ RGB Signature Confirmed (Cool-toned/Blue > Red).")
    else:
        print(
            "  ⚠️ NOTE: Red channel is higher than Blue. (Unexpected for this cool-toned dataset, verify visual parity)."
        )
    print("-------------------------------------------\n")

    # Release file handles before forking workers
    dataset.clear_cache()

    # 3. Rescale & Normalize Pixels
    transforms = []
    for col in keys_to_load:
        if "pixels" in col or "images" in col or col.startswith("world_"):
            transforms.append(
                get_img_preprocessor(source=col, target=col, img_size=cfg.img_size)
            )

    # 3. Standardize States/Actions (Z-Score)
    with open_dict(cfg):
        for col in keys_to_load:
            if "pixels" in col or "images" in col or col.startswith("world_"):
                continue

            # SAFE NORMALIZATION (Injected locally to avoid submodule edits)
            # This fix includes a 1e-8 epsilon to prevent NaNs on stationary joints.
            col_data = dataset.get_col_data(col)
            data = torch.from_numpy(np.array(col_data))
            data = data[~torch.isnan(data).any(dim=1)]
            mean = data.mean(0, keepdim=True).clone()
            std = data.std(0, keepdim=True).clone()

            def norm_fn(x, m=mean, s=std):
                return ((x - m) / (s + 1e-8)).float()

            normalizer = spt.data.transforms.WrapTorchTransform(
                norm_fn, source=col, target=col
            )
            transforms.append(normalizer)

            # Update WM dims for the predictor
            col_dim = dataset.get_dim(col)

            # Clean name for the config: observation.state -> state_dim, action -> action_dim
            clean_name = col.split(".")[-1]
            setattr(cfg.wm, f"{clean_name}_dim", col_dim)
            print(f"📊 Auto-detected {col} dimension ({clean_name}_dim): {col_dim}")

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    # Configure loaders robustly for Colab (handle num_workers=0 case)
    loader_kwargs = dict(cfg.loader)
    if loader_kwargs.get("num_workers", 0) == 0:
        loader_kwargs.pop("prefetch_factor", None)
        loader_kwargs.pop("persistent_workers", None)

    train = torch.utils.data.DataLoader(
        train_set, **loader_kwargs, shuffle=True, drop_last=True, generator=rnd_gen
    )
    val = torch.utils.data.DataLoader(
        val_set,
        batch_size=loader_kwargs.get("batch_size", 1),
        num_workers=loader_kwargs.get("num_workers", 0),
        shuffle=False,
        drop_last=False,
    )

    ##############################
    ##       model / optim      ##
    ##############################

    # --- WORLD MODEL INITIALIZATION ---
    if cfg.get("use_multi_view", True):
        # Replaces standard ViT with Tubelet Tokenization + 3D RoPE (learned for now)
        encoder = get_multi_view_encoder(cfg)
    else:
        # Standard Single-View Baseline
        encoder = spt.backbone.utils.vit_hf(
            cfg.encoder_scale,
            patch_size=cfg.patch_size,
            image_size=cfg.img_size,
            pretrained=False,
            use_mask_token=False,
        )

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )

    action_encoder = GR1Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)

    projector = GR1MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
    )

    predictor_proj = GR1MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
    )

    # --- UPGRADED WORLD MODEL ---
    # Uses MultiViewJEPA to handle spatiotemporal tubelets
    WM_CLASS = MultiViewJEPA if cfg.get("use_multi_view", True) else JEPA
    world_model = WM_CLASS(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
    )

    # 🌟 RA-LeWM Reward Head 🌟
    reward_head = RewardPredictor(input_dim=embed_dim, hidden_dim=512)
    world_model.reward_head = reward_head

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

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    # 0. Global Seeding
    pl.seed_everything(cfg.get("seed", 42), workers=True)

    # 📁 LOCAL PERSISTENCE: We save to ./outputs instead of the system cache
    # to ensure checkpoints are visible in the user's workspace.
    run_id = cfg.get("subdir") or "gr1_official"
    # Force absolute paths to prevent libraries from re-routing to /root/.stable_worldmodel
    run_dir = Path("./outputs", run_id).absolute()
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Run Directory: {run_dir}")
    print(f"💾 Checkpoint Directory: {checkpoint_dir}")

    logger = None
    if cfg.wandb.enabled:
        # Override wandb name for GR-1
        cfg.wandb.config.name = f"gr1-lewm-{cfg.data.dataset.repo_id.split('/')[-1]}"
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir,
        filename=cfg.output_model_name,
        epoch_interval=cfg.get("save_interval", 1),
    )

    # AUTO-BALANCE VALIDATION: If we are in a debug run (limited train batches),
    # automatically limit validation batches to prevent long hangs.
    with open_dict(cfg):
        if cfg.trainer.get("limit_train_batches"):
            # Debug Mode: Set val batches to 1/2 of train batches
            balanced_val = max(2, int(cfg.trainer.limit_train_batches * 0.5))
        else:
            # Full Run Mode: Use a safety cap of 50 batches unless overridden in config
            # This prevents validation from taking 10x longer than training.
            balanced_val = cfg.trainer.get("limit_val_batches", 50)

        cfg.trainer.limit_val_batches = balanced_val
        print(f"⚖️  Validation capped at {balanced_val} batches per epoch.")

    metrics_callback = MetricsCallback(log_every_n_steps=1)

    # 💾 CHECKPOINT PERSISTENCE & SAFE TRANSFER LOGIC
    ckpt_path_str = cfg.get("ckpt_path")
    if ckpt_path_str:
        ckpt_path_str = ckpt_path_str.strip("\"'")
        checkpoint_dir = str(Path(ckpt_path_str).parent)
        print(f"📊 PERSISTENCE: Checkpoints will be saved to: {checkpoint_dir}")

        # --- SAFE TRANSFER (Check for Shape Mismatches) ---
        print(f"🧬 SAFE TRANSFER: Loading weights from {ckpt_path_str}...")
        try:
            checkpoint = torch.load(ckpt_path_str, map_location="cpu")
            # PyTorch Lightning stores weights in "state_dict", naked models store them directly
            state_dict = checkpoint.get("state_dict", checkpoint)

            # --- KEY MAPPING BRIDGE (Handle Late Fusion Nesting) ---
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("model.", "") if k.startswith("model.") else k

                # If we are using Multi-View, map 'encoder.*' to 'encoder.backbone.*' (Late Fusion)
                if cfg.get("use_multi_view"):
                    if new_key.startswith("encoder.") and not new_key.startswith(
                        "encoder.backbone."
                    ):
                        new_key = new_key.replace("encoder.", "encoder.backbone.", 1)

                new_state_dict[new_key] = v

            model_dict = world_model.model.state_dict()
            filtered_dict = {
                k: v
                for k, v in new_state_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }

            msg = world_model.model.load_state_dict(filtered_dict, strict=False)
            print(f"✅ Safe Transfer Results: Loaded {len(filtered_dict)} layers.")
            if msg.missing_keys:
                print(
                    f"⚠️ Re-initialized layers (Mismatched/Missing): {[k for k in msg.missing_keys if 'action_encoder' in k]}"
                )

            # Since we have manually loaded the weights (partial load),
            # we tell Lightning to start a FRESH run from this state
            # rather than a strict resume (which would fail on shape mismatch).
            ckpt_path_str = None
        except Exception as e:
            print(f"❌ Safe Transfer Failed: {e}. Falling back to standard resume.")
    else:
        checkpoint_dir = str(run_dir / "checkpoints")

    # 2. Configure versioned checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="gr1-epoch={epoch:02d}-step={step:06d}",
        save_top_k=-1,  # Research mode: Keep all historical saves
        every_n_epochs=cfg.get("save_interval", 1),
        save_on_train_epoch_end=True,
        auto_insert_metric_name=False,
    )

    # 3. 🔍 RESTORE RESEARCH FEATURES
    # We manually load the 10+ research callbacks that spt.Manager usually adds.
    # This ensures WandB resumption and CPU offloading work correctly.
    spt_callbacks = []
    # Selective Filtering: We only keep the metadata and logging callbacks.
    # We discard SklearnCheckpoint, WandbCheckpoint, and CPUOffloadCallback to stop disk spam and -v1 duplicates.
    print("🔖 Research Features: Loading essential library callbacks...")
    for entry_point in pkg_resources.iter_entry_points("stablepretraining_callbacks"):
        if entry_point.name in [
            "ModuleRegistryCallback",
            "LoggingCallback",
            "EnvironmentDumpCallback",
            "TrainerInfo",
            "ModuleSummary",
            "LogUnusedParametersOnce",
        ]:
            try:
                cb_cls = entry_point.load()
                spt_callbacks.append(cb_cls())
                print(f"  ✓ Attached (Essential): {entry_point.name}")
            except Exception:
                pass
        else:
            print(f"  ○ Skipping (Redundant/Aggressive): {entry_point.name}")

    # Combine Library Callbacks + Our Custom Callbacks
    all_callbacks = spt_callbacks + [
        object_dump_callback,
        metrics_callback,
        checkpoint_callback,
    ]

    # 4. Instantiate Trainer directly (Safest way to avoid OmegaConf errors)
    trainer = pl.Trainer(
        **cfg.trainer,
        default_root_dir=run_dir,
        callbacks=all_callbacks,
        num_sanity_val_steps=1,
        logger=logger,
        log_every_n_steps=1,
        enable_checkpointing=True,
    )

    print(
        f"📣 Ready to start training. Resuming from: {ckpt_path_str or 'FRESH START'}"
    )

    # 🔗 Warm-start from Pretrained Weights (HF: quentinll/lewm-cube)
    # This seeds the Vision Encoder and Predictor with manipulation "common sense"
    # while allowing the action_encoder to re-initialize for the 32-DoF GR-1.
    # SKIPPED if resuming from a checkpoint to avoid overwriting robot-specific weights.
    if cfg.get("use_pretrained_cube") and not cfg.get("ckpt_path"):
        print("📥 Downloading pretrained cube manipulation weights from HF...")
        weights_path = hf_hub_download(
            repo_id="quentinll/lewm-cube", filename="weights.pt"
        )
        state_dict = torch.load(weights_path, map_location="cpu")

        print("🧠 Loading weights into World Model (Warm-start)...")
        model_dict = world_model.model.state_dict()

        # --- KEY MAPPING BRIDGE (Pretrained Cube) ---
        new_state_dict = {}
        is_multi_view = cfg.get("use_multi_view", True)

        for k, v in state_dict.items():
            new_key = k
            if (
                is_multi_view
                and k.startswith("encoder.")
                and not k.startswith("encoder.backbone.")
            ):
                new_key = k.replace("encoder.", "encoder.backbone.", 1)
            new_state_dict[new_key] = v

        filtered_dict = {
            k: v
            for k, v in new_state_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }

        # DIAGNOSTIC: Check if patch embeddings are being skipped
        patch_key = "encoder.embeddings.patch_embeddings.projection.weight"
        if is_multi_view:
            patch_key = "encoder.backbone.embeddings.patch_embeddings.projection.weight"

        if patch_key not in filtered_dict:
            print(f"⚠️  WARNING: {patch_key} NOT FOUND OR SHAPE MISMATCH.")
            if is_multi_view and patch_key in new_sd:
                print(f"   - Found in state_dict (mapped): {new_sd[patch_key].shape}")
            elif not is_multi_view and patch_key in state_dict:
                print(f"   - Found in state_dict (raw): {state_dict[patch_key].shape}")

            if patch_key in model_dict:
                print(f"   - Expected in model: {model_dict[patch_key].shape}")

        msg = world_model.model.load_state_dict(filtered_dict, strict=False)
        print(
            f"✅ Weights loaded. Transferred: {len(filtered_dict)} layers. "
            f"Skipped: {len(model_dict) - len(filtered_dict)} "
            "(due to configuration mismatch)."
        )

    print("🚀 Launching GR-1 Official Training Loop...")
    # We use the standard Trainer.fit() instead of the library's Manager
    # to maintain full control over the training loop and checkpoint frequency.
    trainer.fit(
        model=world_model,
        datamodule=data_module,
        ckpt_path=ckpt_path_str,
    )


if __name__ == "__main__":
    run()
