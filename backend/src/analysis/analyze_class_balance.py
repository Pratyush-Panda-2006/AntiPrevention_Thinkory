from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "LEVIR-CD"
)

LABEL_DIR = DATASET_DIR / "label"


def main():

    print("=" * 60)
    print("LEVIR-CD TRAINING CLASS BALANCE ANALYSIS")
    print("=" * 60)

    split = "train"
    prefix = f"{split}_"

    files = sorted(
        file
        for file in LABEL_DIR.iterdir()
        if file.is_file()
        and file.suffix.lower() in {
            ".png",
            ".jpg",
            ".jpeg",
        }
        and file.name.startswith(prefix)
    )

    if not files:
        raise RuntimeError(
            f"No training labels found in {LABEL_DIR}"
        )

    total_pixels = 0
    changed_pixels = 0

    print(f"\nTraining scenes: {len(files)}")

    for index, label_path in enumerate(files, start=1):

        label = np.asarray(
            Image.open(label_path).convert("L")
        )

        changed = np.count_nonzero(label > 0)
        total = label.size

        changed_pixels += changed
        total_pixels += total

        if index % 50 == 0 or index == len(files):
            print(
                f"Processed: {index}/{len(files)}"
            )

    unchanged_pixels = (
        total_pixels - changed_pixels
    )

    change_percentage = (
        changed_pixels / total_pixels * 100
    )

    unchanged_percentage = (
        unchanged_pixels / total_pixels * 100
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(
        f"\nTotal pixels: "
        f"{total_pixels:,}"
    )

    print(
        f"Changed pixels: "
        f"{changed_pixels:,}"
    )

    print(
        f"Unchanged pixels: "
        f"{unchanged_pixels:,}"
    )

    print(
        f"\nChanged: "
        f"{change_percentage:.4f}%"
    )

    print(
        f"Unchanged: "
        f"{unchanged_percentage:.4f}%"
    )

    # ---------------------------------------------------------
    # Simple positive-class weighting estimate
    # ---------------------------------------------------------

    if changed_pixels > 0:

        positive_ratio = (
            changed_pixels / total_pixels
        )

        negative_ratio = (
            unchanged_pixels / total_pixels
        )

        suggested_pos_weight = (
            negative_ratio / positive_ratio
        )

        print(
            f"\nPixel-frequency pos_weight estimate: "
            f"{suggested_pos_weight:.4f}"
        )

    else:
        print(
            "\nNo changed pixels found. "
            "Cannot calculate pos_weight."
        )

    print("\n✓ Class balance analysis complete")


if __name__ == "__main__":
    main()