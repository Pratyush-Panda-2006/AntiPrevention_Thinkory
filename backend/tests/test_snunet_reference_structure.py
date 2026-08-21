from pathlib import Path
import sys

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
    print("SNUNET-CD REFERENCE STRUCTURE COMPARISON")
    print("=" * 70)

    reference_model = SNUNet_ECAM(
        in_ch=3,
        out_ch=1,
    )

    ported_model = SNUNetCD(
        in_channels=3,
        num_classes=1,
    )

    reference_state = reference_model.state_dict()
    ported_state = ported_model.state_dict()

    print()
    print(
        f"Reference parameters: "
        f"{sum(p.numel() for p in reference_model.parameters()):,}"
    )

    print(
        f"Ported parameters:    "
        f"{sum(p.numel() for p in ported_model.parameters()):,}"
    )

    print()
    print(
        f"Reference tensors: "
        f"{len(reference_state)}"
    )

    print(
        f"Ported tensors:    "
        f"{len(ported_state)}"
    )

    reference_names = list(
        reference_state.keys()
    )

    ported_names = list(
        ported_state.keys()
    )

    # ---------------------------------------------------------
    # Exact name comparison
    # ---------------------------------------------------------

    reference_name_set = set(
        reference_names
    )

    ported_name_set = set(
        ported_names
    )

    only_reference = sorted(
        reference_name_set - ported_name_set
    )

    only_ported = sorted(
        ported_name_set - reference_name_set
    )

    print()
    print("Names only in reference:")

    if only_reference:
        for name in only_reference:
            print(f"  - {name}")
    else:
        print("  None")

    print()
    print("Names only in ported model:")

    if only_ported:
        for name in only_ported:
            print(f"  - {name}")
    else:
        print("  None")

    # ---------------------------------------------------------
    # Shape comparison for shared names
    # ---------------------------------------------------------

    shared_names = sorted(
        reference_name_set & ported_name_set
    )

    shape_mismatches = []

    for name in shared_names:

        reference_shape = tuple(
            reference_state[name].shape
        )

        ported_shape = tuple(
            ported_state[name].shape
        )

        if reference_shape != ported_shape:
            shape_mismatches.append(
                (
                    name,
                    reference_shape,
                    ported_shape,
                )
            )

    print()
    print(
        f"Shared tensor names: "
        f"{len(shared_names)}"
    )

    print()
    print("Shape mismatches:")

    if shape_mismatches:
        for (
            name,
            reference_shape,
            ported_shape,
        ) in shape_mismatches:
            print(
                f"  - {name}: "
                f"reference={reference_shape}, "
                f"ported={ported_shape}"
            )
    else:
        print("  None")

    # ---------------------------------------------------------
    # Positional/order comparison
    # ---------------------------------------------------------

    common_length = min(
        len(reference_names),
        len(ported_names),
    )

    order_mismatches = []

    for index in range(common_length):

        if (
            reference_names[index]
            != ported_names[index]
        ):
            order_mismatches.append(
                (
                    index,
                    reference_names[index],
                    ported_names[index],
                )
            )

    print()
    print("State-dict ordering mismatches:")

    if order_mismatches:
        for (
            index,
            reference_name,
            ported_name,
        ) in order_mismatches[:30]:

            print(
                f"  - index {index}: "
                f"reference={reference_name} | "
                f"ported={ported_name}"
            )

        if len(order_mismatches) > 30:
            print(
                f"  ... and "
                f"{len(order_mismatches) - 30} more"
            )
    else:
        print("  None")

    # ---------------------------------------------------------
    # Final assessment
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("REFERENCE STRUCTURE SUMMARY")
    print("=" * 70)

    print(
        f"Parameter count equal: "
        f"{sum(p.numel() for p in reference_model.parameters()) == sum(p.numel() for p in ported_model.parameters())}"
    )

    print(
        f"Tensor count equal: "
        f"{len(reference_state) == len(ported_state)}"
    )

    print(
        f"No reference-only names: "
        f"{not only_reference}"
    )

    print(
        f"No ported-only names: "
        f"{not only_ported}"
    )

    print(
        f"No shape mismatches: "
        f"{not shape_mismatches}"
    )

    print()
    print(
        "NOTE: Parameter-name identity is useful evidence, "
        "but different module naming can still represent an "
        "equivalent architecture. Any mismatch must be reviewed "
        "before concluding that the port is wrong."
    )


if __name__ == "__main__":
    main()