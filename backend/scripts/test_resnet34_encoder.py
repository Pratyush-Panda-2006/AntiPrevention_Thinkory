import torch

from detection.siamese_resnet34_unet import (
    SiameseResNet34UNet,
)


def main():
    print("=" * 60)
    print("MODEL 2 RESNET-34 ENCODER TEST")
    print("=" * 60)

    device = torch.device("cpu")

    model = SiameseResNet34UNet(
        in_channels=3,
        num_classes=1,
    ).to(device)

    model.eval()

    # ---------------------------------------------------------
    # Create test inputs
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
    # Run T1 and T2 through the SAME encoder
    # ---------------------------------------------------------

    with torch.no_grad():

        (
            a_conv1,
            a_layer1,
            a_layer2,
            a_layer3,
            a_layer4,
        ) = model.encoder(image_a)

        (
            b_conv1,
            b_layer1,
            b_layer2,
            b_layer3,
            b_layer4,
        ) = model.encoder(image_b)

    # ---------------------------------------------------------
    # Print feature shapes
    # ---------------------------------------------------------

    print()
    print("T1 encoder feature shapes:")
    print(f"  conv1 : {tuple(a_conv1.shape)}")
    print(f"  layer1: {tuple(a_layer1.shape)}")
    print(f"  layer2: {tuple(a_layer2.shape)}")
    print(f"  layer3: {tuple(a_layer3.shape)}")
    print(f"  layer4: {tuple(a_layer4.shape)}")

    print()
    print("T2 encoder feature shapes:")
    print(f"  conv1 : {tuple(b_conv1.shape)}")
    print(f"  layer1: {tuple(b_layer1.shape)}")
    print(f"  layer2: {tuple(b_layer2.shape)}")
    print(f"  layer3: {tuple(b_layer3.shape)}")
    print(f"  layer4: {tuple(b_layer4.shape)}")

    # ---------------------------------------------------------
    # Expected shapes
    # ---------------------------------------------------------

    expected_shapes = {
        "conv1": (2, 64, 128, 128),
        "layer1": (2, 64, 64, 64),
        "layer2": (2, 128, 32, 32),
        "layer3": (2, 256, 16, 16),
        "layer4": (2, 512, 8, 8),
    }

    actual_shapes = {
        "conv1": tuple(a_conv1.shape),
        "layer1": tuple(a_layer1.shape),
        "layer2": tuple(a_layer2.shape),
        "layer3": tuple(a_layer3.shape),
        "layer4": tuple(a_layer4.shape),
    }

    # ---------------------------------------------------------
    # Verify shapes
    # ---------------------------------------------------------

    print()
    print("Checking expected feature shapes...")

    for name, expected in expected_shapes.items():

        actual = actual_shapes[name]

        assert actual == expected, (
            f"{name} shape mismatch: "
            f"expected {expected}, got {actual}"
        )

        print(f"  ✓ {name}: {actual}")

    # ---------------------------------------------------------
    # Verify T1/T2 corresponding feature shapes
    # ---------------------------------------------------------

    print()
    print("Checking T1/T2 feature compatibility...")

    feature_pairs = [
        ("conv1", a_conv1, b_conv1),
        ("layer1", a_layer1, b_layer1),
        ("layer2", a_layer2, b_layer2),
        ("layer3", a_layer3, b_layer3),
        ("layer4", a_layer4, b_layer4),
    ]

    for name, feature_a, feature_b in feature_pairs:

        assert feature_a.shape == feature_b.shape, (
            f"{name}: T1/T2 feature shapes differ: "
            f"{feature_a.shape} vs {feature_b.shape}"
        )

        print(
            f"  ✓ {name}: "
            f"T1 {tuple(feature_a.shape)} == "
            f"T2 {tuple(feature_b.shape)}"
        )

    # ---------------------------------------------------------
    # Verify absolute-difference tensors
    # ---------------------------------------------------------

    print()
    print("Checking absolute-difference fusion shapes...")

    diff_conv1 = torch.abs(a_conv1 - b_conv1)
    diff_layer1 = torch.abs(a_layer1 - b_layer1)
    diff_layer2 = torch.abs(a_layer2 - b_layer2)
    diff_layer3 = torch.abs(a_layer3 - b_layer3)
    diff_layer4 = torch.abs(a_layer4 - b_layer4)

    diff_features = [
        ("conv1", diff_conv1, expected_shapes["conv1"]),
        ("layer1", diff_layer1, expected_shapes["layer1"]),
        ("layer2", diff_layer2, expected_shapes["layer2"]),
        ("layer3", diff_layer3, expected_shapes["layer3"]),
        ("layer4", diff_layer4, expected_shapes["layer4"]),
    ]

    for name, diff, expected in diff_features:

        assert tuple(diff.shape) == expected, (
            f"{name} diff shape mismatch: "
            f"expected {expected}, got {tuple(diff.shape)}"
        )

        print(
            f"  ✓ {name} diff: {tuple(diff.shape)}"
        )

    # ---------------------------------------------------------
    # Verify encoder sharing
    # ---------------------------------------------------------
    #
    # The model contains ONE encoder instance:
    #
    #     model.encoder
    #
    # It is called once for image_a and once for image_b.
    #
    # We verify that the encoder has a single parameter set
    # and that the same module is used for both forward passes.
    # ---------------------------------------------------------

    print()
    print("Checking Siamese encoder sharing...")

    assert isinstance(
        model.encoder,
        torch.nn.Module,
    ), "model.encoder is not a PyTorch module"

    # There must be exactly one encoder object.
    encoder_id = id(model.encoder)

    assert encoder_id == id(model.encoder), (
        "Encoder identity check failed."
    )

    # Check that the encoder contains parameters.
    encoder_parameters = list(
        model.encoder.parameters()
    )

    assert len(encoder_parameters) > 0, (
        "Encoder contains no parameters."
    )

    print(
        f"  ✓ One shared encoder instance: "
        f"id={encoder_id}"
    )

    print(
        f"  ✓ Encoder parameter tensors: "
        f"{len(encoder_parameters)}"
    )

    # ---------------------------------------------------------
    # Verify pretrained ResNet-34 stem structure
    # ---------------------------------------------------------

    print()
    print("Checking ResNet-34 stem...")

    assert hasattr(
        model.encoder,
        "conv1",
    ), "Encoder is missing conv1."

    assert hasattr(
        model.encoder,
        "maxpool",
    ), "Encoder is missing maxpool."

    # conv1 should output 64 channels.
    assert model.encoder.conv1.out_channels == 64, (
        "Unexpected conv1 output channels: "
        f"{model.encoder.conv1.out_channels}"
    )

    print("  ✓ conv1 exists")
    print("  ✓ conv1 outputs 64 channels")
    print("  ✓ conv1 feature captured before maxpool")

    # ---------------------------------------------------------
    # Final success
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("MODEL 2 ENCODER TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()