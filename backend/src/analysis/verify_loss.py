import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Add src to Python path
sys.path.append(str(PROJECT_ROOT / "src"))

from data.levir_patch_dataset import LEVIRCDPatchDataset
from detection.losses import BCEDiceLoss
from detection.siamese_unet import SiameseUNet
from preprocessing.transforms import LEVIRCDTrainTransform

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "LEVIR-CD"
)


def main():

    print("=" * 60)
    print("LOSS + BACKPROPAGATION VERIFICATION")
    print("=" * 60)

    device = torch.device("cpu")

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    transform = LEVIRCDTrainTransform()

    dataset = LEVIRCDPatchDataset(
        root_dir=DATASET_DIR,
        split="train",
        patch_size=256,
        stride=128,
        transform=transform,
        change_sampling=True,
    )

    sample = dataset[0]

    image_a = (
        sample["image_a"]
        .unsqueeze(0)
        .to(device)
    )

    image_b = (
        sample["image_b"]
        .unsqueeze(0)
        .to(device)
    )

    target = (
        sample["label"]
        .unsqueeze(0)
        .to(device)
    )

    print("\nInput:")
    print(f"T1:     {image_a.shape}")
    print(f"T2:     {image_b.shape}")
    print(f"Target: {target.shape}")

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = SiameseUNet(
        in_channels=3,
        num_classes=1,
    ).to(device)

    model.train()

    # ---------------------------------------------------------
    # Loss
    # ---------------------------------------------------------

    criterion = BCEDiceLoss(
        bce_weight=0.5,
        dice_weight=0.5,
    )

    # ---------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------

    prediction = model(
        image_a,
        image_b,
    )

    print(
        f"\nPrediction: "
        f"{prediction.shape}"
    )

    # ---------------------------------------------------------
    # Calculate loss
    # ---------------------------------------------------------

    loss = criterion(
        prediction,
        target,
    )

    print(
        f"Loss: "
        f"{loss.item():.6f}"
    )

    # ---------------------------------------------------------
    # Backpropagation
    # ---------------------------------------------------------

    model.zero_grad()

    loss.backward()

    # Check whether gradients were produced.
    parameters_with_gradients = 0
    parameters_without_gradients = 0

    for parameter in model.parameters():

        if parameter.requires_grad:

            if parameter.grad is None:
                parameters_without_gradients += 1
            else:
                parameters_with_gradients += 1

    print(
        f"\nParameters with gradients: "
        f"{parameters_with_gradients}"
    )

    print(
        f"Parameters without gradients: "
        f"{parameters_without_gradients}"
    )

    # ---------------------------------------------------------
    # Assertions
    # ---------------------------------------------------------

    assert torch.isfinite(loss), (
        "Loss is NaN or infinite."
    )

    assert loss.item() >= 0, (
        "Loss should be non-negative."
    )

    assert parameters_with_gradients > 0, (
        "No model parameters received gradients."
    )

    assert parameters_without_gradients == 0, (
        "Some trainable model parameters did not "
        "receive gradients."
    )

    print("\n✓ Loss calculation successful")
    print("✓ Loss is finite")
    print("✓ Backpropagation successful")
    print("✓ Gradients reached the model")


if __name__ == "__main__":
    main()