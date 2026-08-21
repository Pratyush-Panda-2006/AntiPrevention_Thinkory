from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from detection.losses import BCEDiceLoss


def main():
    print("=" * 70)
    print("SAR MASKED BCE + DICE LOSS TEST")
    print("=" * 70)

    torch.manual_seed(42)

    logits = torch.randn(
        2, 1, 8, 8,
        requires_grad=True,
    )

    targets = torch.randint(
        0,
        2,
        (2, 1, 8, 8),
    ).float()

    criterion = BCEDiceLoss(
        bce_weight=0.5,
        dice_weight=0.5,
        pos_weight=1.0,
    )

    # =========================================================
    # 1. All-valid mask should match legacy loss
    # =========================================================

    all_valid = torch.ones_like(
        targets,
        dtype=torch.bool,
    )

    legacy_loss = criterion(
        logits,
        targets,
    )

    masked_loss = criterion(
        logits,
        targets,
        valid_mask=all_valid,
    )

    print()
    print("ALL-VALID MASK")
    print(f"Legacy loss: {legacy_loss.item():.8f}")
    print(f"Masked loss: {masked_loss.item():.8f}")

    assert torch.allclose(
        legacy_loss,
        masked_loss,
        atol=1e-6,
        rtol=1e-6,
    )

    # =========================================================
    # 2. Ignore a region
    # =========================================================

    partial_mask = torch.ones_like(
        targets,
        dtype=torch.bool,
    )

    partial_mask[:, :, :4, :4] = False

    loss_a = criterion(
        logits,
        targets,
        valid_mask=partial_mask,
    )

    # Change ONLY ignored pixels.
    modified_targets = targets.clone()
    modified_targets[:, :, :4, :4] = (
        1.0 - modified_targets[:, :, :4, :4]
    )

    loss_b = criterion(
        logits,
        modified_targets,
        valid_mask=partial_mask,
    )

    print()
    print("IGNORED-REGION INVARIANCE")
    print(f"Original loss:  {loss_a.item():.8f}")
    print(f"Modified loss:  {loss_b.item():.8f}")

    assert torch.allclose(
        loss_a,
        loss_b,
        atol=1e-6,
        rtol=1e-6,
    )

    # =========================================================
    # 3. Backward / gradient test
    # =========================================================

    logits.grad = None

    loss_a.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()

    # Ignored logits should receive zero gradient from the loss.
    ignored_grad = logits.grad[
        :,
        :,
        :4,
        :4,
    ]

    print()
    print("GRADIENT TEST")
    print(
        "Maximum ignored-region gradient:",
        float(ignored_grad.abs().max()),
    )

    assert torch.allclose(
        ignored_grad,
        torch.zeros_like(ignored_grad),
        atol=1e-7,
    )

    # =========================================================
    # 4. All-invalid mask
    # =========================================================

    all_invalid = torch.zeros_like(
        targets,
        dtype=torch.bool,
    )

    try:
        criterion(
            logits.detach(),
            targets,
            valid_mask=all_invalid,
        )

    except ValueError:
        print("All-invalid mask correctly rejected.")
    else:
        raise AssertionError(
            "Expected ValueError for all-invalid mask."
        )

    # =========================================================
    # Final
    # =========================================================

    print()
    print("=" * 70)
    print("SAR MASKED LOSS TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()