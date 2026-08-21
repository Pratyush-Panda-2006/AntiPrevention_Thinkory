from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from training.sar_dataloaders import create_sar_dataloader


def inspect_loader(name, loader):
    print()
    print("=" * 70)
    print(f"{name}")
    print("=" * 70)

    print("Dataset size:", len(loader.dataset))

    batch = next(iter(loader))

    print("Batch keys:", sorted(batch.keys()))

    print("image_a:", tuple(batch["image_a"].shape))
    print("image_b:", tuple(batch["image_b"].shape))
    print("label:", tuple(batch["label"].shape))
    print("valid_mask:", tuple(batch["valid_mask"].shape))

    assert batch["image_a"].shape[1:] == (2, 256, 256)
    assert batch["image_b"].shape[1:] == (2, 256, 256)
    assert batch["label"].shape[1:] == (1, 256, 256)
    assert batch["valid_mask"].shape[1:] == (1, 256, 256)

    assert batch["image_a"].dtype == torch.float32
    assert batch["image_b"].dtype == torch.float32
    assert batch["label"].dtype == torch.float32
    assert batch["valid_mask"].dtype == torch.bool

    assert torch.isfinite(batch["image_a"]).all()
    assert torch.isfinite(batch["image_b"]).all()
    assert torch.isfinite(batch["label"]).all()

    assert (
        batch["image_a"].min() >= 0.0
        and batch["image_a"].max() <= 1.0 + 1e-6
    )

    assert (
        batch["image_b"].min() >= 0.0
        and batch["image_b"].max() <= 1.0 + 1e-6
    )

    print("Train/validation batch shape: PASS")
    print("dtype checks: PASS")
    print("finite checks: PASS")
    print("normalization range: PASS")


def main():
    patch_index = (
        PROJECT_ROOT
        / "data"
        / "sar"
        / "tum_oscd"
        / "sar_patch_index.json"
    )

    train_loader = create_sar_dataloader(
        patch_index_path=patch_index,
        split="train",
        batch_size=2,
        num_workers=0,
        pin_memory=True,
        shuffle=True,
    )

    val_loader = create_sar_dataloader(
        patch_index_path=patch_index,
        split="validation",
        batch_size=2,
        num_workers=0,
        pin_memory=True,
        shuffle=False,
    )

    inspect_loader(
        "SAR TRAIN LOADER",
        train_loader,
    )

    inspect_loader(
        "SAR VALIDATION LOADER",
        val_loader,
    )

    print()
    print("=" * 70)
    print("SAR DATALOADER PREFLIGHT PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()