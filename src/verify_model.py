import sys
from pathlib import Path

# Add src to Python path so it can find the 'detection' module
sys.path.append(str(Path(__file__).resolve().parent))

import torch

from detection.siamese_unet import SiameseUNet


def main():

    print("=" * 60)
    print("SIAMESE U-NET FORWARD PASS TEST")
    print("=" * 60)

    device = torch.device("cpu")

    model = SiameseUNet(
        in_channels=3,
        num_classes=1,
    ).to(device)

    model.eval()

    # One synthetic batch.
    image_a = torch.randn(
        1,
        3,
        256,
        256,
        device=device,
    )

    image_b = torch.randn(
        1,
        3,
        256,
        256,
        device=device,
    )

    print("\nInput:")
    print(f"T1: {image_a.shape}")
    print(f"T2: {image_b.shape}")

    with torch.no_grad():
        output = model(
            image_a,
            image_b,
        )

    print("\nOutput:")
    print(f"Prediction: {output.shape}")

    print(
        f"\nModel parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    expected = (
        1,
        1,
        256,
        256,
    )

    assert output.shape == expected, (
        f"Expected {expected}, "
        f"got {tuple(output.shape)}"
    )

    print("\n✓ Forward pass successful")
    print("✓ Output shape is correct")


if __name__ == "__main__":
    main()