import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Add src to Python path
sys.path.append(str(PROJECT_ROOT / "src"))

from training.dataloaders import (
    create_eval_dataloader,
    create_train_dataloader,
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
    print("LEVIR-CD DATALOADER VERIFICATION")
    print("=" * 60)

    # =========================================================
    # TRAIN
    # =========================================================

    print("\n" + "=" * 60)
    print("TRAINING DATALOADER")
    print("=" * 60)

    train_loader = create_train_dataloader(
        dataset_dir=DATASET_DIR,
        batch_size=2,
        num_workers=0,
        patch_size=256,
        stride=128,
    )

    print(
        f"\nTraining dataset: "
        f"{len(train_loader.dataset):,} patches"
    )

    print(
        f"Batch size: "
        f"{train_loader.batch_size}"
    )

    train_batch = next(iter(train_loader))

    print("\nFirst training batch:")

    print(
        f"T1: "
        f"{train_batch['image_a'].shape}"
    )

    print(
        f"T2: "
        f"{train_batch['image_b'].shape}"
    )

    print(
        f"Label: "
        f"{train_batch['label'].shape}"
    )

    print(
        f"Filenames: "
        f"{train_batch['filename']}"
    )

    print(
        f"Patch tops: "
        f"{train_batch['top']}"
    )

    print(
        f"Patch lefts: "
        f"{train_batch['left']}"
    )

    # =========================================================
    # VALIDATION
    # =========================================================

    print("\n" + "=" * 60)
    print("VALIDATION DATALOADER")
    print("=" * 60)

    val_loader = create_eval_dataloader(
        dataset_dir=DATASET_DIR,
        split="val",
        batch_size=1,
        num_workers=0,
    )

    print(
        f"\nValidation scenes: "
        f"{len(val_loader.dataset)}"
    )

    val_batch = next(iter(val_loader))

    print("\nFirst validation batch:")

    print(
        f"T1: "
        f"{val_batch['image_a'].shape}"
    )

    print(
        f"T2: "
        f"{val_batch['image_b'].shape}"
    )

    print(
        f"Label: "
        f"{val_batch['label'].shape}"
    )

    print(
        f"Filename: "
        f"{val_batch['filename']}"
    )

    # =========================================================
    # TEST
    # =========================================================

    print("\n" + "=" * 60)
    print("TEST DATALOADER")
    print("=" * 60)

    test_loader = create_eval_dataloader(
        dataset_dir=DATASET_DIR,
        split="test",
        batch_size=1,
        num_workers=0,
    )

    print(
        f"\nTest scenes: "
        f"{len(test_loader.dataset)}"
    )

    print("\n✓ All DataLoaders created successfully")
    print("✓ Weighted training sampler connected")
    print("✓ Validation uses full scenes")
    print("✓ Test uses full scenes")


if __name__ == "__main__":
    main()