import sys
from pathlib import Path

import torch


# -------------------------------------------------------------
# Project paths
# -------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))


from detection.losses import BCEDiceLoss
from detection.snunet_cd import SNUNetCD


def main():
    print("=" * 70)
    print("MODEL 3 — SNUNET-CD LOSS + BACKWARD TEST")
    print("=" * 70)

    device = torch.device("cpu")

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = SNUNetCD(
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
        pos_weight=1.0,
    )

    # ---------------------------------------------------------
    # Dummy bi-temporal inputs
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
    # Dummy binary target
    #
    # Include both unchanged and changed pixels.
    # ---------------------------------------------------------

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
    # Forward
    # ---------------------------------------------------------

    print()
    print("Running forward pass...")

    optimizer.zero_grad(
        set_to_none=True
    )

    logits = model(
        image_a,
        image_b,
    )

    print(
        f"Logits shape: "
        f"{tuple(logits.shape)}"
    )

    assert logits.shape == (
        2,
        1,
        256,
        256,
    ), (
        "Unexpected logits shape: "
        f"{tuple(logits.shape)}"
    )

    assert torch.isfinite(
        logits
    ).all(), (
        "Logits contain NaN or Inf."
    )

    print("✓ Logits are finite")

    # ---------------------------------------------------------
    # Loss
    # ---------------------------------------------------------

    print()
    print("Calculating BCE + Dice loss...")

    loss = criterion(
        logits,
        target,
    )

    print(
        f"Loss: {loss.item():.6f}"
    )

    assert torch.isfinite(loss), (
        f"Loss is not finite: "
        f"{loss.item()}"
    )

    print("✓ Loss is finite")

    # ---------------------------------------------------------
    # Backward
    # ---------------------------------------------------------

    print()
    print("Running backward pass...")

    loss.backward()

    print("✓ Backward pass completed")

    # ---------------------------------------------------------
    # Gradient inspection
    # ---------------------------------------------------------

    print()
    print("Checking gradient flow...")

    encoder_grad_found = False
    nested_grad_found = False
    ecam_grad_found = False

    encoder_max_grad = 0.0
    nested_max_grad = 0.0
    ecam_max_grad = 0.0

    nonfinite_gradients = []

    for name, parameter in model.named_parameters():

        if parameter.grad is None:
            continue

        gradient = parameter.grad.detach()

        if not torch.isfinite(
            gradient
        ).all():
            nonfinite_gradients.append(name)
            continue

        max_grad = (
            gradient
            .abs()
            .max()
            .item()
        )

        # Shared encoder blocks
        if name.startswith(
            (
                "conv0_0.",
                "conv1_0.",
                "conv2_0.",
                "conv3_0.",
                "conv4_0.",
            )
        ):
            encoder_grad_found = True
            encoder_max_grad = max(
                encoder_max_grad,
                max_grad,
            )

        # Nested decoder blocks
        elif name.startswith(
            "conv"
        ):
            nested_grad_found = True
            nested_max_grad = max(
                nested_max_grad,
                max_grad,
            )

        # ECAM / attention modules
        elif name.startswith(
            (
                "ca.",
                "ca1.",
            )
        ):
            ecam_grad_found = True
            ecam_max_grad = max(
                ecam_max_grad,
                max_grad,
            )

    if nonfinite_gradients:
        raise RuntimeError(
            "Non-finite gradients found in:\n"
            + "\n".join(
                f"  - {name}"
                for name in nonfinite_gradients
            )
        )

    assert encoder_grad_found, (
        "No gradients reached the shared encoder."
    )

    assert nested_grad_found, (
        "No gradients reached the nested decoder."
    )

    assert ecam_grad_found, (
        "No gradients reached ECAM."
    )

    print(
        "✓ Encoder gradients detected"
    )
    print(
        f"  Maximum encoder gradient: "
        f"{encoder_max_grad:.6e}"
    )

    print(
        "✓ Nested decoder gradients detected"
    )
    print(
        f"  Maximum nested gradient: "
        f"{nested_max_grad:.6e}"
    )

    print(
        "✓ ECAM gradients detected"
    )
    print(
        f"  Maximum ECAM gradient: "
        f"{ecam_max_grad:.6e}"
    )

    # ---------------------------------------------------------
    # Optimizer update
    # ---------------------------------------------------------

    print()
    print("Running optimizer step...")

    optimizer.step()

    print("✓ Optimizer step completed")

    # ---------------------------------------------------------
    # Final
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("MODEL 3 SNUNET-CD BACKWARD TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()