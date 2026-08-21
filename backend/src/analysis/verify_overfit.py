import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Add src to Python path
sys.path.append(str(PROJECT_ROOT / "src"))

import torch
from torch.utils.data import DataLoader, Subset

from data.levir_patch_dataset import LEVIRCDPatchDataset
from detection.losses import BCEDiceLoss
from detection.siamese_unet import SiameseUNet
from preprocessing.transforms import LEVIRCDTrainTransform
from training.reproducibility import set_seed


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "LEVIR-CD"
)


def main():

    print("=" * 60)
    print("SIAMESE U-NET SMALL-DATA OVERFIT TEST")
    print("=" * 60)

    set_seed(42)

    device = torch.device("cpu")

    print(
        f"\nDevice: {device}"
    )

    # =========================================================
    # Dataset
    # =========================================================

    dataset = LEVIRCDPatchDataset(
        root_dir=DATASET_DIR,
        split="train",
        patch_size=256,
        stride=128,
        transform=LEVIRCDTrainTransform(),
        change_sampling=True,
        change_threshold=0.01,
    )

    # ---------------------------------------------------------
    # Select patches that contain meaningful change.
    # Weight 3.0 means the patch passed our change threshold.
    # ---------------------------------------------------------

    positive_indices = [
        index
        for index, weight
        in enumerate(dataset.patch_weights)
        if float(weight.item()) == 3.0
    ]

    if len(positive_indices) < 2:
        raise RuntimeError(
            "Not enough change-containing patches "
            "for the overfit test."
        )

    # Use only a tiny fixed subset.
    selected_indices = positive_indices[:2]

    print(
        f"\nTotal training patches: "
        f"{len(dataset):,}"
    )

    print(
        f"Change-containing patches available: "
        f"{len(positive_indices):,}"
    )

    print(
        f"Selected patches for overfit test: "
        f"{len(selected_indices)}"
    )

    subset = Subset(
        dataset,
        selected_indices,
    )

    loader = DataLoader(
        subset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    # =========================================================
    # Model
    # =========================================================

    model = SiameseUNet(
        in_channels=3,
        num_classes=1,
    ).to(device)

    model.train()

    # =========================================================
    # Loss
    # =========================================================

    criterion = BCEDiceLoss(
        bce_weight=0.5,
        dice_weight=0.5,
        pos_weight=5.0,
    )

    # =========================================================
    # Optimizer
    # =========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )

    print(
        "\nStarting optimization..."
    )

    # =========================================================
    # Overfit loop
    # =========================================================

    initial_loss = None
    final_loss = None

    for step in range(1, 21):

        for batch in loader:

            image_a = batch["image_a"].to(
                device
            )

            image_b = batch["image_b"].to(
                device
            )

            targets = batch["label"].to(
                device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                image_a,
                image_b,
            )

            loss = criterion(
                logits,
                targets,
            )

            loss.backward()

            optimizer.step()

            loss_value = loss.item()

            if initial_loss is None:
                initial_loss = loss_value

            final_loss = loss_value

        print(
            f"Step {step:02d}/20 | "
            f"Loss: {loss_value:.6f}"
        )

    # =========================================================
    # Final evaluation on same patches
    # =========================================================

    model.eval()

    true_positive = 0
    false_positive = 0
    false_negative = 0

    with torch.no_grad():

        for batch in loader:

            image_a = batch["image_a"].to(
                device
            )

            image_b = batch["image_b"].to(
                device
            )

            targets = batch["label"].to(
                device
            )

            logits = model(
                image_a,
                image_b,
            )

            predictions = (
                torch.sigmoid(logits)
                >= 0.5
            )

            target_binary = (
                targets >= 0.5
            )

            true_positive += int(
                (
                    predictions
                    & target_binary
                ).sum().item()
            )

            false_positive += int(
                (
                    predictions
                    & (~target_binary)
                ).sum().item()
            )

            false_negative += int(
                (
                    (~predictions)
                    & target_binary
                ).sum().item()
            )

    precision_denominator = (
        true_positive
        + false_positive
    )

    recall_denominator = (
        true_positive
        + false_negative
    )

    precision = (
        true_positive
        / precision_denominator
        if precision_denominator > 0
        else 0.0
    )

    recall = (
        true_positive
        / recall_denominator
        if recall_denominator > 0
        else 0.0
    )

    f1_denominator = (
        precision + recall
    )

    f1 = (
        2.0
        * precision
        * recall
        / f1_denominator
        if f1_denominator > 0
        else 0.0
    )

    # =========================================================
    # Results
    # =========================================================

    print("\n" + "=" * 60)
    print("OVERFIT TEST RESULTS")
    print("=" * 60)

    print(
        f"\nInitial loss: "
        f"{initial_loss:.6f}"
    )

    print(
        f"Final loss:   "
        f"{final_loss:.6f}"
    )

    print(
        f"Loss reduction:"
        f" {initial_loss - final_loss:.6f}"
    )

    print(
        f"\nTP: {true_positive:,}"
    )

    print(
        f"FP: {false_positive:,}"
    )

    print(
        f"FN: {false_negative:,}"
    )

    print(
        f"\nPrecision: "
        f"{precision:.4f}"
    )

    print(
        f"Recall: "
        f"{recall:.4f}"
    )

    print(
        f"F1: "
        f"{f1:.4f}"
    )

    # =========================================================
    # Basic sanity assertions
    # =========================================================

    assert final_loss < initial_loss, (
        "Loss did not decrease during the overfit test."
    )

    print(
        "\n✓ Loss decreased"
    )

    print(
        "✓ Model learned from the selected patches"
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "OVERFIT SANITY TEST COMPLETE"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()