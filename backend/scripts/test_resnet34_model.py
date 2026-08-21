import torch

from detection.siamese_resnet34_unet import (
    SiameseResNet34UNet,
)


def main():
    device = torch.device("cpu")

    print("=" * 60)
    print("MODEL 2 RESNET-34 ARCHITECTURE TEST")
    print("=" * 60)

    model = SiameseResNet34UNet(
        in_channels=3,
        num_classes=1,
    ).to(device)

    model.eval()

    # ---------------------------------------------------------
    # Parameter count
    # ---------------------------------------------------------

    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # ---------------------------------------------------------
    # Dummy 256x256 temporal pair
    # ---------------------------------------------------------

    image_a = torch.randn(
        2,
        3,
        256,
        256,
        device=device,
    )

    image_b = torch.randn(
        2,
        3,
        256,
        256,
        device=device,
    )

    # ---------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------

    with torch.no_grad():
        output = model(
            image_a,
            image_b,
        )

    print(f"Input shape:  {tuple(image_a.shape)}")
    print(f"Output shape: {tuple(output.shape)}")

    # ---------------------------------------------------------
    # Assertions
    # ---------------------------------------------------------

    assert output.shape == (
        2,
        1,
        256,
        256,
    ), (
        "Unexpected output shape: "
        f"{tuple(output.shape)}"
    )

    print()
    print("✓ Model instantiated successfully")
    print("✓ Forward pass successful")
    print("✓ Output is 256x256")
    print("✓ Output has 1 change-detection channel")
    print()
    print("MODEL 2 ARCHITECTURE TEST PASSED")


if __name__ == "__main__":
    main()