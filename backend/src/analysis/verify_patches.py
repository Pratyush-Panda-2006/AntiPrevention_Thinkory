import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Add src to Python path to allow direct execution
sys.path.append(str(PROJECT_ROOT / "src"))

from data.levir_dataset import LEVIRCDPatchDataset


DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "LEVIR-CD"
)


def main():

    print("=" * 60)
    print("LEVIR-CD PATCH DATASET VERIFICATION")
    print("=" * 60)

    dataset = LEVIRCDPatchDataset(
        root_dir=DATASET_DIR,
        split="train",
        patch_size=256,
        stride=128,
        change_sampling=True,
    )

    print(f"\nScenes: {len(dataset.files)}")
    print(f"Patches: {len(dataset)}")

    print(
        f"\nPatch size: "
        f"{dataset.patch_size}x{dataset.patch_size}"
    )

    print(
        f"Stride: "
        f"{dataset.stride}"
    )

    sample = dataset[0]

    print("\nFirst patch:")
    print(f"Filename: {sample['filename']}")
    print(f"Top: {sample['top']}")
    print(f"Left: {sample['left']}")

    print(f"T1 shape: {sample['image_a'].shape}")
    print(f"T2 shape: {sample['image_b'].shape}")
    print(f"Label shape: {sample['label'].shape}")

    print(
        f"\nLabel values: "
        f"{sample['label'].unique().tolist()}"
    )

    print("\nChange-aware sampling weights:")

    unique_weights = (
        dataset.patch_weights.unique().tolist()
    )

    print(unique_weights)


if __name__ == "__main__":
    main()
