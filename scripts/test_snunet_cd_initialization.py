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


def describe_tensor(name, tensor):
    tensor = tensor.detach().float()

    print(
        f"{name}: "
        f"min={tensor.min().item():.6f}, "
        f"max={tensor.max().item():.6f}, "
        f"mean={tensor.mean().item():.6f}, "
        f"std={tensor.std().item():.6f}"
    )


def main():
    print("=" * 70)
    print("MODEL 3 — SNUNET-CD INITIALIZATION / LOSS SANITY TEST")
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
    # Optimizer
    # ---------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )

    # ---------------------------------------------------------
    # Deterministic-ish test inputs
    # ---------------------------------------------------------

    torch.manual_seed(42)

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

    target = torch.zeros(
        2,
        1,
        256,
        256,
        device=device,
    )

    # Give the target some changed pixels.
    target[:, :, 64:128, 64:128] = 1.0

    # ---------------------------------------------------------
    # Initial forward
    # ---------------------------------------------------------

    print()
    print("INITIAL MODEL OUTPUT")

    optimizer.zero_grad(set_to_none=True)

    logits_before = model(
        image_a,
        image_b,
    )

    probabilities_before = torch.sigmoid(
        logits_before.float()
    )

    describe_tensor(
        "Initial logits",
        logits_before,
    )

    describe_tensor(
        "Initial probabilities",
        probabilities_before,
    )

    # ---------------------------------------------------------
    # BCE and Dice components
    #
    # Prefer the public criterion interface. If the loss module
    # exposes separate helpers, this test also reports them.
    # ---------------------------------------------------------

    total_loss_before = criterion(
        logits_before,
        target,
    )

    print(
        f"\nInitial total loss: "
        f"{total_loss_before.item():.6f}"
    )

    assert torch.isfinite(
        total_loss_before
    ), "Initial loss is not finite."

    # ---------------------------------------------------------
    # Basic probability sanity
    # ---------------------------------------------------------

    assert (
        probabilities_before.min().item() >= 0.0
    )

    assert (
        probabilities_before.max().item() <= 1.0
    )

    print(
        "✓ Initial probabilities are in [0, 1]"
    )

    # ---------------------------------------------------------
    # Backward
    # ---------------------------------------------------------

    total_loss_before.backward()

    gradient_values = []

    for parameter in model.parameters():
        if parameter.grad is not None:
            gradient_values.append(
                parameter.grad.detach()
                .abs()
                .max()
                .item()
            )

    assert gradient_values, (
        "No gradients were produced."
    )

    max_gradient = max(
        gradient_values
    )

    print(
        f"Maximum initial gradient: "
        f"{max_gradient:.6e}"
    )

    assert torch.isfinite(
        torch.tensor(max_gradient)
    ), "Maximum gradient is not finite."

    print(
        "✓ Initial backward pass is finite"
    )

    # ---------------------------------------------------------
    # One optimizer update
    # ---------------------------------------------------------

    optimizer.step()

    print(
        "✓ One optimizer step completed"
    )

    # ---------------------------------------------------------
    # Second forward after one update
    # ---------------------------------------------------------

    optimizer.zero_grad(
        set_to_none=True
    )

    logits_after = model(
        image_a,
        image_b,
    )

    probabilities_after = torch.sigmoid(
        logits_after.float()
    )

    total_loss_after = criterion(
        logits_after,
        target,
    )

    print()
    print("AFTER ONE OPTIMIZER UPDATE")

    describe_tensor(
        "Updated logits",
        logits_after,
    )

    describe_tensor(
        "Updated probabilities",
        probabilities_after,
    )

    print(
        f"Updated total loss: "
        f"{total_loss_after.item():.6f}"
    )

    assert torch.isfinite(
        total_loss_after
    ), "Updated loss is not finite."

    # ---------------------------------------------------------
    # Compare
    # ---------------------------------------------------------

    print()
    print("LOSS COMPARISON")

    print(
        f"Before update: "
        f"{total_loss_before.item():.6f}"
    )

    print(
        f"After update:  "
        f"{total_loss_after.item():.6f}"
    )

    loss_delta = (
        total_loss_after.item()
        - total_loss_before.item()
    )

    print(
        f"Delta:         "
        f"{loss_delta:+.6f}"
    )

    # A single optimizer step is NOT expected to guarantee
    # lower loss on arbitrary data, so we do not assert that
    # the loss must decrease.

    print()
    print(
        "NOTE: One optimizer step is a sanity check only. "
        "The loss is not required to decrease on a random "
        "synthetic batch."
    )

    # ---------------------------------------------------------
    # Final
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "MODEL 3 INITIALIZATION / LOSS SANITY TEST PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()