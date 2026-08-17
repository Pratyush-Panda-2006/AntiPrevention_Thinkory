from pathlib import Path
import argparse
import sys
import torch

# Allow imports from src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from training.config import TrainingConfig
from training.dataloaders import (
    create_train_dataloader,
    create_eval_dataloader,
)
from training.engine import TrainingEngine
from training.optimizers import create_training_components
from training.reproducibility import set_seed


def main():

    # =========================================================
    # Command-line arguments
    # =========================================================

    parser = argparse.ArgumentParser(
        description="Train the LEVIR-CD Siamese U-Net."
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a limited training sanity test.",
    )

    parser.add_argument(
        "--train-batches",
        type=int,
        default=2,
        help="Number of training batches for smoke test.",
    )

    parser.add_argument(
        "--val-batches",
        type=int,
        default=1,
        help="Number of validation batches for smoke test.",
    )

    args = parser.parse_args()

    # =========================================================
    # Validate arguments
    # =========================================================

    if args.train_batches < 1:
        raise ValueError(
            "--train-batches must be at least 1."
        )

    if args.val_batches < 1:
        raise ValueError(
            "--val-batches must be at least 1."
        )

    # =========================================================
    # Configuration
    # =========================================================

    config = TrainingConfig()

    if args.smoke_test:

        config.epochs = 1

        config.checkpoint_dir = (
            "checkpoints/smoke_test_final"
        )

    # =========================================================
    # Paths
    # =========================================================

    dataset_dir = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "LEVIR-CD"
    )

    config.checkpoint_dir = str(
        PROJECT_ROOT
        / config.checkpoint_dir
    )

    # =========================================================
    # Reproducibility
    # =========================================================

    set_seed(config.seed)

    # =========================================================
    # Device
    # =========================================================

    device = config.get_device()

    print("=" * 70)
    print("LEVIR-CD SIAMESE U-NET TRAINING")
    print("=" * 70)

    print(
        f"\nMode: "
        f"{'SMOKE TEST' if args.smoke_test else 'FULL TRAINING'}"
    )

    print(
        f"Device: {device}"
    )

    if device.type == "cuda":
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"Seed: {config.seed}"
    )

    # =========================================================
    # Dataset
    # =========================================================

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"LEVIR-CD dataset not found at:\n"
            f"{dataset_dir}"
        )

    print(
        f"\nDataset: {dataset_dir}"
    )

    # =========================================================
    # Training DataLoader
    # =========================================================

    print(
        "\nCreating training DataLoader..."
    )

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

    print(
        f"Batch size: "
        f"{config.batch_size}"
    )

    # =========================================================
    # Validation DataLoader
    # =========================================================

    print(
        "\nCreating validation DataLoader..."
    )

    val_loader = create_eval_dataloader(
        dataset_dir=dataset_dir,
        split="val",
        batch_size=1,
        num_workers=config.num_workers,
    )

    print(
        f"Validation scenes: "
        f"{len(val_loader.dataset)}"
    )

    # =========================================================
    # Model / Loss / Optimizer / Scheduler
    # =========================================================

    print(
        "\nCreating training components..."
    )

    (
        model,
        criterion,
        optimizer,
        scheduler,
    ) = create_training_components(
        config
    )

    # =========================================================
    # Training Engine
    # =========================================================

    engine = TrainingEngine(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )

    # =========================================================
    # Configuration summary
    # =========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "TRAINING CONFIGURATION"
    )

    print(
        "=" * 70
    )

    print(
        f"Epochs:              {config.epochs}"
    )

    print(
        f"Batch size:          {config.batch_size}"
    )

    print(
        f"Patch size:          {config.patch_size}"
    )

    print(
        f"Stride:              {config.stride}"
    )

    print(
        f"Learning rate:       {config.learning_rate}"
    )

    print(
        f"Weight decay:        {config.weight_decay}"
    )

    print(
        f"BCE weight:          {config.bce_weight}"
    )

    print(
        f"Dice weight:         {config.dice_weight}"
    )

    print(
        f"Positive weight:     {config.pos_weight}"
    )

    print(
        f"Threshold:           {config.threshold}"
    )

    print(
        f"Checkpoint directory:"
        f" {config.checkpoint_dir}"
    )

    if args.smoke_test:

        print(
            f"Smoke train batches: "
            f"{args.train_batches}"
        )

        print(
            f"Smoke validation batches: "
            f"{args.val_batches}"
        )

    print(
        "=" * 70
    )

    # =========================================================
    # Run training
    # =========================================================

    if args.smoke_test:

        print(
            "\nRunning smoke test..."
        )

        engine.fit(
            max_train_batches=args.train_batches,
            max_val_batches=args.val_batches,
        )

    else:

        print(
            "\nStarting FULL TRAINING..."
        )

        engine.fit()

    print(
        "\nTraining command completed successfully."
    )


if __name__ == "__main__":
    main()