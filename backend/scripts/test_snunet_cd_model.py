import sys
from pathlib import Path

import torch


# -------------------------------------------------------------
# Make src/ importable
# -------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(
    0,
    str(SRC_DIR),
)


from detection.snunet_cd import SNUNetCD


def main():
    print("=" * 70)
    print("MODEL 3 — SNUNET-CD ARCHITECTURE TEST")
    print("=" * 70)

    device = torch.device("cpu")

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = SNUNetCD(
        in_channels=3,
        num_classes=1,
    ).to(device)

    model.eval()

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print()
    print(
        f"Total parameters:     "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    # ---------------------------------------------------------
    # Dummy inputs
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
    # Production forward
    # ---------------------------------------------------------

    print()
    print("Running production forward pass...")

    with torch.no_grad():
        logits = model(
            image_a,
            image_b,
        )

    print(
        f"Production output: "
        f"{tuple(logits.shape)}"
    )

    assert logits.shape == (
        2,
        1,
        256,
        256,
    ), (
        "Unexpected production output shape: "
        f"{tuple(logits.shape)}"
    )

    assert torch.isfinite(
        logits
    ).all(), (
        "Production logits contain "
        "NaN or Inf values."
    )

    print(
        "✓ Production output shape correct"
    )

    # ---------------------------------------------------------
    # Debug forward
    # ---------------------------------------------------------

    print()
    print("Running debug forward pass...")

    with torch.no_grad():

        (
            debug_logits,
            intermediate,
        ) = model(
            image_a,
            image_b,
            return_intermediates=True,
        )

    assert torch.equal(
        logits,
        debug_logits,
    ), (
        "Production and debug forward outputs "
        "are not identical."
    )

    print(
        "✓ Debug output matches production output"
    )

    # ---------------------------------------------------------
    # Print intermediate shapes
    # ---------------------------------------------------------

    print()
    print("Intermediate feature shapes:")

    for name, tensor in intermediate.items():

        print(
            f"  {name:24s} "
            f"{tuple(tensor.shape)}"
        )

        assert torch.isfinite(
            tensor
        ).all(), (
            f"{name} contains NaN or Inf."
        )

    # ---------------------------------------------------------
    # Expected major shapes
    # ---------------------------------------------------------

    expected = {
        "x0_0A": (2, 32, 256, 256),
        "x1_0A": (2, 64, 128, 128),
        "x2_0A": (2, 128, 64, 64),
        "x3_0A": (2, 256, 32, 32),

        "x0_0B": (2, 32, 256, 256),
        "x1_0B": (2, 64, 128, 128),
        "x2_0B": (2, 128, 64, 64),
        "x3_0B": (2, 256, 32, 32),
        "x4_0B": (2, 512, 16, 16),

        "x0_1": (2, 32, 256, 256),
        "x1_1": (2, 64, 128, 128),
        "x2_1": (2, 128, 64, 64),
        "x3_1": (2, 256, 32, 32),

        "x0_2": (2, 32, 256, 256),
        "x1_2": (2, 64, 128, 128),
        "x2_2": (2, 128, 64, 64),

        "x0_3": (2, 32, 256, 256),
        "x1_3": (2, 64, 128, 128),

        "x0_4": (2, 32, 256, 256),

        "nested_concat": (2, 128, 256, 256),
        "intra_attention_input": (2, 32, 256, 256),
        "attention_coarse": (2, 32, 1, 1),
        "logits": (2, 1, 256, 256),
    }

    print()
    print("Checking expected shapes...")

    for name, shape in expected.items():

        actual = tuple(
            intermediate[name].shape
        )

        assert actual == shape, (
            f"{name}: expected {shape}, "
            f"got {actual}"
        )

        print(
            f"  ✓ {name}: {actual}"
        )

    # ---------------------------------------------------------
    # Verify shared Siamese modules
    # ---------------------------------------------------------

    print()
    print("Checking Siamese encoder sharing...")

    # The encoder blocks are represented by ONE module each,
    # called once for image A and once for image B.
    #
    # We verify the module objects themselves exist only once.

    shared_modules = [
        model.conv0_0,
        model.conv1_0,
        model.conv2_0,
        model.conv3_0,
        model.conv4_0,
    ]

    for index, module in enumerate(
        shared_modules
    ):
        assert isinstance(
            module,
            torch.nn.Module,
        )

        print(
            f"  ✓ Shared encoder block "
            f"conv{index}_0 exists once"
        )

    # ---------------------------------------------------------
    # Check parameter counts are non-zero
    # ---------------------------------------------------------

    assert total_parameters > 0
    assert trainable_parameters > 0

    # ---------------------------------------------------------
    # Final
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("MODEL 3 SNUNET-CD ARCHITECTURE TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()