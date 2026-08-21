import torch

from detection.losses import BCEDiceLoss
from detection.siamese_resnet34_unet import (
    SiameseResNet34UNet,
)


def main():
    print("=" * 60)
    print("MODEL 2 LOSS + BACKWARD TEST")
    print("=" * 60)

    device = torch.device("cpu")

    # ---------------------------------------------------------
    # Create model
    # ---------------------------------------------------------

    model = SiameseResNet34UNet(
        in_channels=3,
        num_classes=1,
    ).to(device)

    model.train()

    # ---------------------------------------------------------
    # Create existing project loss
    # ---------------------------------------------------------

    criterion = BCEDiceLoss(
        bce_weight=0.5,
        dice_weight=0.5,
        pos_weight=1.0,
    )

    # ---------------------------------------------------------
    # Dummy temporal images
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

    # Binary change mask.
    #
    # Keep a mixture of changed and unchanged pixels so the
    # BCE + Dice loss receives a meaningful target.
    target = torch.zeros(
        2,
        1,
        256,
        256,
        device=device,
    )

    target[:, :, 64:128, 64:128] = 1.0

    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )

    # ---------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------

    print()
    print("Running forward pass...")

    optimizer.zero_grad(set_to_none=True)

    logits = model(
        image_a,
        image_b,
    )

    print(f"Logits shape: {tuple(logits.shape)}")

    assert logits.shape == (
        2,
        1,
        256,
        256,
    ), (
        "Unexpected logits shape: "
        f"{tuple(logits.shape)}"
    )

    # ---------------------------------------------------------
    # Loss
    # ---------------------------------------------------------

    loss = criterion(
        logits,
        target,
    )

    print(f"Loss: {loss.item():.6f}")

    assert torch.isfinite(loss), (
        f"Loss is not finite: {loss.item()}"
    )

    # ---------------------------------------------------------
    # Backward pass
    # ---------------------------------------------------------

    print()
    print("Running backward pass...")

    loss.backward()

    print("✓ Backward pass completed")

    # ---------------------------------------------------------
    # Check gradients
    # ---------------------------------------------------------

    print()
    print("Checking gradients...")

    encoder_gradient_found = False
    decoder_gradient_found = False

    encoder_gradient_max = 0.0
    decoder_gradient_max = 0.0

    for name, parameter in model.named_parameters():

        if parameter.grad is None:
            continue

        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError(
                f"Non-finite gradient detected in: {name}"
            )

        gradient_max = (
            parameter.grad.detach()
            .abs()
            .max()
            .item()
        )

        if name.startswith("encoder."):
            encoder_gradient_found = True
            encoder_gradient_max = max(
                encoder_gradient_max,
                gradient_max,
            )

        elif name.startswith("decoder"):
            decoder_gradient_found = True
            decoder_gradient_max = max(
                decoder_gradient_max,
                gradient_max,
            )

    assert encoder_gradient_found, (
        "No gradients reached the ResNet-34 encoder."
    )

    assert decoder_gradient_found, (
        "No gradients reached the decoder."
    )

    print(
        "✓ Encoder gradients detected"
    )

    print(
        f"  Maximum encoder gradient: "
        f"{encoder_gradient_max:.6e}"
    )

    print(
        "✓ Decoder gradients detected"
    )

    print(
        f"  Maximum decoder gradient: "
        f"{decoder_gradient_max:.6e}"
    )

    # ---------------------------------------------------------
    # Optimizer update
    # ---------------------------------------------------------

    print()
    print("Running optimizer step...")

    optimizer.step()

    print("✓ Optimizer step completed")

    # ---------------------------------------------------------
    # Final checks
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("MODEL 2 LOSS + BACKWARD TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()