import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Add src to Python path
sys.path.append(str(PROJECT_ROOT / "src"))

from data.levir_patch_dataset import LEVIRCDPatchDataset
from data.sampler import (
    create_weighted_sampler,
    summarize_sampler_weights,
)
from preprocessing.transforms import LEVIRCDTrainTransform


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "LEVIR-CD"
)


def main():

    print("=" * 60)
    print("LEVIR-CD WEIGHTED SAMPLER VERIFICATION")
    print("=" * 60)

    dataset = LEVIRCDPatchDataset(
        root_dir=DATASET_DIR,
        split="train",
        patch_size=256,
        stride=128,
        transform=LEVIRCDTrainTransform(),
        change_sampling=True,
        change_threshold=0.01,
    )

    print(
        f"\nTraining patches: "
        f"{len(dataset):,}"
    )

    summary = summarize_sampler_weights(dataset)

    print("\nPatch weight distribution:")

    for weight, count in sorted(summary.items()):
        percentage = (
            count / len(dataset) * 100
        )

        print(
            f"Weight {weight:.1f}: "
            f"{count:,} patches "
            f"({percentage:.2f}%)"
        )

    sampler = create_weighted_sampler(
        dataset
    )

    print(
        f"\nSamples drawn per epoch: "
        f"{len(sampler):,}"
    )

    print(
        "\nReplacement sampling: "
        f"{sampler.replacement}"
    )

    print("\n✓ Weighted sampler created successfully")


if __name__ == "__main__":
    main()