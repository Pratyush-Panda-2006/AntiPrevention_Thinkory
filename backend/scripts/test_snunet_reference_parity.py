import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
REFERENCE_DIR = PROJECT_ROOT / "Siam-NestedUNet"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(REFERENCE_DIR))

from detection.snunet_cd import SNUNetCD
from models.Models import SNUNet_ECAM


def main():
    print("=" * 70)
    print("SNUNET-CD EXACT REFERENCE/PPORT PARITY TEST")
    print("=" * 70)

    torch.manual_seed(42)

    # ---------------------------------------------------------
    # Create both models
    # ---------------------------------------------------------

    reference_model = SNUNet_ECAM(
        in_ch=3,
        out_ch=1,
    )

    ported_model = SNUNetCD(
        in_channels=3,
        num_classes=1,
    )

    reference_model.eval()
    ported_model.eval()

    # ---------------------------------------------------------
    # Copy EXACT reference state into the ported model.
    #
    # State-dict names and tensor shapes were already verified
    # to match exactly.
    # ---------------------------------------------------------

    reference_state = reference_model.state_dict()

    ported_model.load_state_dict(
        reference_state,
        strict=True,
    )

    print()
    print("✓ Reference state loaded into ported model")
    print("✓ strict=True state-dict load succeeded")

    # ---------------------------------------------------------
    # Same fixed input pair
    # ---------------------------------------------------------

    torch.manual_seed(12345)

    image_a = torch.randn(
        2,
        3,
        256,
        256,
    )

    image_b = torch.randn(
        2,
        3,
        256,
        256,
    )

    # ---------------------------------------------------------
    # Reference forward
    #
    # The archived reference returns a tuple.
    # The ECAM implementation returns the final output in
    # the first/only tuple position.
    # ---------------------------------------------------------

    with torch.no_grad():

        reference_result = reference_model(
            image_a,
            image_b,
        )

        if isinstance(
            reference_result,
            (tuple, list),
        ):
            reference_logits = reference_result[-1]
        else:
            reference_logits = reference_result

    # ---------------------------------------------------------
    # Ported forward
    # ---------------------------------------------------------

    with torch.no_grad():

        ported_logits = ported_model(
            image_a,
            image_b,
        )

    # ---------------------------------------------------------
    # Basic shape checks
    # ---------------------------------------------------------

    print()
    print(
        f"Reference output shape: "
        f"{tuple(reference_logits.shape)}"
    )

    print(
        f"Ported output shape:    "
        f"{tuple(ported_logits.shape)}"
    )

    assert (
        reference_logits.shape
        == ported_logits.shape
    ), (
        "Output shape mismatch."
    )

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    reference_float = (
        reference_logits
        .detach()
        .float()
    )

    ported_float = (
        ported_logits
        .detach()
        .float()
    )

    print()
    print("REFERENCE OUTPUT")
    print(
        f"min  = {reference_float.min().item():.8f}"
    )
    print(
        f"max  = {reference_float.max().item():.8f}"
    )
    print(
        f"mean = {reference_float.mean().item():.8f}"
    )
    print(
        f"std  = {reference_float.std().item():.8f}"
    )

    print()
    print("PORTED OUTPUT")
    print(
        f"min  = {ported_float.min().item():.8f}"
    )
    print(
        f"max  = {ported_float.max().item():.8f}"
    )
    print(
        f"mean = {ported_float.mean().item():.8f}"
    )
    print(
        f"std  = {ported_float.std().item():.8f}"
    )

    # ---------------------------------------------------------
    # Difference
    # ---------------------------------------------------------

    absolute_difference = (
        reference_float
        - ported_float
    ).abs()

    max_absolute_difference = (
        absolute_difference.max().item()
    )

    mean_absolute_difference = (
        absolute_difference.mean().item()
    )

    print()
    print("OUTPUT DIFFERENCE")
    print(
        f"Max absolute difference:  "
        f"{max_absolute_difference:.8e}"
    )

    print(
        f"Mean absolute difference: "
        f"{mean_absolute_difference:.8e}"
    )

    # ---------------------------------------------------------
    # Final assessment
    #
    # We do not require bit-for-bit equality because the same
    # weights should produce extremely close outputs here,
    # but a gross mismatch is a strong signal of a porting
    # difference.
    # ---------------------------------------------------------

    if max_absolute_difference < 1e-5:

        print()
        print(
            "✓ Reference and ported outputs are "
            "numerically equivalent."
        )

    elif max_absolute_difference < 1e-3:

        print()
        print(
            "✓ Reference and ported outputs are "
            "very closely aligned."
        )

    else:

        print()
        print(
            "⚠ Reference and ported outputs differ "
            "materially."
        )

        print(
            "This requires investigation before "
            "production training."
        )

        raise RuntimeError(
            "SNUNet-CD reference parity check failed."
        )

    # ---------------------------------------------------------
    # Final
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("SNUNET-CD REFERENCE PARITY TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
    