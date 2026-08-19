import json
import sys
import time
from pathlib import Path

import torch


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))


# ============================================================
# Project imports
# ============================================================

from detection.losses import BCEDiceLoss
from detection.snunet_cd import SNUNetCD

from training.config import TrainingConfig
from training.dataloaders import (
    create_train_dataloader,
    create_eval_dataloader,
)
from training.engine import TrainingEngine
from training.reproducibility import set_seed


def main():
    print("=" * 70)
    print("MODEL 3 — SNUNET-CD TRAINING ENGINE SMOKE TEST")
    print("=" * 70)

    # ========================================================
    # Configuration
    # ========================================================

    config = TrainingConfig()

    # Force CPU for this development-machine smoke test.
    config.device = "cpu"

    # Minimal smoke-test settings.
    config.epochs = 1
    config.batch_size = 2
    config.num_workers = 0

    # Locked Model 3 training configuration.
    config.model_name = "snunet_cd"

    config.learning_rate = 1e-4
    config.weight_decay = 1e-4

    config.bce_weight = 0.5
    config.dice_weight = 0.5
    config.pos_weight = 1.0

    config.scheduler_factor = 0.5
    config.scheduler_patience = 3
    config.scheduler_min_lr = 1e-6

    config.early_stopping_patience = 12

    config.threshold = 0.5

    config.use_amp = True
    config.gradient_clip_norm = 1.0

    # ========================================================
    # Run directory
    # ========================================================

    timestamp = time.strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    run_dir = (
        PROJECT_ROOT
        / "runs"
        / f"{timestamp}_snunet_cd_engine_smoke"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config.run_dir = str(run_dir)
    config.checkpoint_dir = str(
        run_dir / "checkpoints"
    )

    config.experiment_name = (
        "snunet_cd_engine_smoke"
    )

    config.log_level = "normal"

    # ========================================================
    # Reproducibility
    # ========================================================

    set_seed(config.seed)

    print()
    print(f"Device:        {config.get_device()}")
    print(f"Run directory: {run_dir}")
    print(f"Batch size:    {config.batch_size}")
    print(f"Epochs:        {config.epochs}")
    print(f"Seed:          {config.seed}")

    # ========================================================
    # Dataset
    # ========================================================

    dataset_dir = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "LEVIR-CD"
    )

    if not dataset_dir.exists():
        raise FileNotFoundError(
            "LEVIR-CD dataset not found at:\n"
            f"{dataset_dir}"
        )

    print()
    print("Creating training DataLoader...")

    train_loader = create_train_dataloader(
        dataset_dir=dataset_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        patch_size=config.patch_size,
        stride=config.stride,
    )

    print(
        f"Training patches: "
        f"{len(train_loader.dataset):,}"
    )

    print()
    print("Creating validation DataLoader...")

    val_loader = create_eval_dataloader(
        dataset_dir=dataset_dir,
        split="val",
        batch_size=1,
        num_workers=config.num_workers,
    )

    print(
        f"Validation scenes: "
        f"{len(val_loader.dataset):,}"
    )

    # ========================================================
    # Model 3
    # ========================================================

    print()
    print("Creating SNUNet-CD...")

    model = SNUNetCD(
        in_channels=3,
        num_classes=1,
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Total parameters:     "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    # ========================================================
    # Loss
    # ========================================================

    criterion = BCEDiceLoss(
        bce_weight=config.bce_weight,
        dice_weight=config.dice_weight,
        pos_weight=config.pos_weight,
    )

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # ========================================================
    # Scheduler
    # ========================================================

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.scheduler_min_lr,
    )

    print()
    print("Scheduler:")
    print("  ReduceLROnPlateau")
    print("  mode=max")
    print(
        f"  factor={config.scheduler_factor}"
    )
    print(
        f"  patience={config.scheduler_patience}"
    )
    print(
        f"  min_lr={config.scheduler_min_lr}"
    )

    # ========================================================
    # Save config
    # ========================================================

    with open(
        run_dir / "config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            vars(config),
            file,
            indent=2,
        )

    # ========================================================
    # Actual TrainingEngine
    # ========================================================

    print()
    print("Creating actual TrainingEngine...")

    engine = TrainingEngine(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )

    print(
        f"AMP enabled by engine: "
        f"{engine.use_amp}"
    )

    print(
        f"Gradient clip norm: "
        f"{engine.gradient_clip_norm}"
    )

    # ========================================================
    # Smoke test
    # ========================================================

    print()
    print("=" * 70)
    print("RUNNING MODEL 3 ENGINE SMOKE TEST")
    print("Training batches: 2")
    print("Validation batches: 1")
    print("Batch size: 2")
    print("=" * 70)

    start_time = time.time()

    history = engine.fit(
        max_train_batches=2,
        max_val_batches=1,
    )

    duration = time.time() - start_time

    # ========================================================
    # Results
    # ========================================================

    print()
    print("=" * 70)
    print("MODEL 3 ENGINE SMOKE TEST COMPLETE")
    print("=" * 70)

    print(
        f"Duration: {duration:.2f} seconds"
    )

    print(
        f"History entries: {len(history)}"
    )

    if not history:
        raise RuntimeError(
            "TrainingEngine returned no history."
        )

    result = history[-1]

    print()
    print("Epoch result:")

    print(
        f"  Train loss: "
        f"{result['train_loss']:.6f}"
    )

    print(
        f"  Val loss:   "
        f"{result['val_loss']:.6f}"
    )

    print(
        f"  Precision:  "
        f"{result['precision']:.6f}"
    )

    print(
        f"  Recall:     "
        f"{result['recall']:.6f}"
    )

    print(
        f"  F1:         "
        f"{result['f1']:.6f}"
    )

    print(
        f"  IoU:        "
        f"{result['iou']:.6f}"
    )

    print(
        f"  Accuracy:   "
        f"{result['accuracy']:.6f}"
    )

    print(
        f"  LR:         "
        f"{result['learning_rate']:.8f}"
    )

    # ========================================================
    # Check checkpoints
    # ========================================================

    best_checkpoint = (
        run_dir
        / "checkpoints"
        / "best.pt"
    )

    last_checkpoint = (
        run_dir
        / "checkpoints"
        / "last.pt"
    )

    print()
    print("Checkpoint verification:")

    if not last_checkpoint.exists():
        raise RuntimeError(
            "last.pt was not created."
        )

    print(
        f"  ✓ last.pt created: "
        f"{last_checkpoint}"
    )

    if not best_checkpoint.exists():
        raise RuntimeError(
            "best.pt was not created."
        )

    print(
        f"  ✓ best.pt created: "
        f"{best_checkpoint}"
    )

    # ========================================================
    # Final success
    # ========================================================

    print()
    print("=" * 70)
    print("MODEL 3 TRAINING ENGINE SMOKE TEST PASSED")
    print("=" * 70)

    print()
    print("Verified:")
    print("  ✓ Real LEVIR-CD DataLoader")
    print("  ✓ SNUNet-CD")
    print("  ✓ BCE + Dice")
    print("  ✓ AdamW")
    print("  ✓ ReduceLROnPlateau mode=max")
    print("  ✓ Actual TrainingEngine")
    print("  ✓ Forward pass")
    print("  ✓ Backward pass")
    print("  ✓ Gradient clipping")
    print("  ✓ Validation")
    print("  ✓ F1 calculation")
    print("  ✓ Checkpoint creation")
    print("  ✓ Reporting")

    print()
    print(
        f"Smoke-test artifacts: {run_dir}"
    )


if __name__ == "__main__":
    main()