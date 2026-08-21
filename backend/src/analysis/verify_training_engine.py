import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Add src to Python path
sys.path.append(str(PROJECT_ROOT / "src"))

from training.config import TrainingConfig
from training.dataloaders import (
    create_eval_dataloader,
    create_train_dataloader,
)
from training.engine import TrainingEngine
from training.optimizers import (
    create_training_components,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "LEVIR-CD"
)


def main():

    print("=" * 60)
    print("TRAINING ENGINE SMOKE TEST")
    print("=" * 60)

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    config = TrainingConfig()

    # Keep smoke-test checkpoints separate.
    config.checkpoint_dir = (
        "checkpoints/smoke_test"
    )

    # Only one epoch for the smoke test.
    config.epochs = 1

    # ---------------------------------------------------------
    # DataLoaders
    # ---------------------------------------------------------

    train_loader = create_train_dataloader(
        dataset_dir=DATASET_DIR,
        batch_size=2,
        num_workers=0,
        patch_size=config.patch_size,
        stride=config.stride,
    )

    val_loader = create_eval_dataloader(
        dataset_dir=DATASET_DIR,
        split="val",
        batch_size=1,
        num_workers=0,
    )

    # ---------------------------------------------------------
    # Model + loss + optimizer + scheduler
    # ---------------------------------------------------------

    (
        model,
        criterion,
        optimizer,
        scheduler,
    ) = create_training_components(
        config
    )

    # ---------------------------------------------------------
    # Training engine
    # ---------------------------------------------------------

    engine = TrainingEngine(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )

    # ---------------------------------------------------------
    # Smoke test
    #
    # Only:
    #   2 training batches
    #   1 validation scene
    # ---------------------------------------------------------

    history = engine.fit(
        max_train_batches=2,
        max_val_batches=1,
    )

    # ---------------------------------------------------------
    # Final checks
    # ---------------------------------------------------------

    assert len(history) == 1

    checkpoint_dir = Path(
        config.checkpoint_dir
    )

    assert (
        checkpoint_dir / "last.pt"
    ).exists()

    assert (
        checkpoint_dir / "best.pt"
    ).exists()

    assert (
        checkpoint_dir / "training_history.json"
    ).exists()

    print(
        "\n✓ One training epoch completed"
    )

    print(
        "✓ Validation completed"
    )

    print(
        "✓ Checkpoint saved"
    )

    print(
        "✓ Best checkpoint saved"
    )

    print(
        "✓ Training history saved"
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "TRAINING ENGINE SMOKE TEST PASSED"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()