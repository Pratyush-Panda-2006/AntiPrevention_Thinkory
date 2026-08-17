import sys
from pathlib import Path

import matplotlib.pyplot as plt

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add src to Python path to allow direct execution
sys.path.append(str(PROJECT_ROOT / "src"))

from data.levir_dataset import LEVIRCDDataset

# LEVIR-CD dataset location
DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "LEVIR-CD"


def main():
    print("=" * 60)
    print("LEVIR-CD DATASET VERIFICATION")
    print("=" * 60)

    print(f"\nDataset path:")
    print(DATASET_DIR)

    # Load datasets for all splits
    train_dataset = LEVIRCDDataset(DATASET_DIR, split="train")
    val_dataset = LEVIRCDDataset(DATASET_DIR, split="val")
    test_dataset = LEVIRCDDataset(DATASET_DIR, split="test")

    print(f"\nDatasets loaded successfully!")
    print(f"Train dataset: {len(train_dataset)}")
    print(f"Validation dataset: {len(val_dataset)}")
    print(f"Test dataset: {len(test_dataset)}")

    # Load first sample from train
    sample = train_dataset[0]

    image_a = sample["image_a"]
    image_b = sample["image_b"]
    label = sample["label"]
    filename = sample["filename"]

    print("\nFirst sample:")
    print(f"Filename: {filename}")
    print(f"T1 shape: {image_a.shape}")
    print(f"T2 shape: {image_b.shape}")
    print(f"Label shape: {label.shape}")

    print(f"\nT1 value range: {image_a.min():.3f} -> {image_a.max():.3f}")
    print(f"T2 value range: {image_b.min():.3f} -> {image_b.max():.3f}")

    print(f"Label unique values: {label.unique().tolist()}")

    # Convert tensors to numpy for visualization
    image_a_np = image_a.permute(1, 2, 0).numpy()
    image_b_np = image_b.permute(1, 2, 0).numpy()
    label_np = label.squeeze(0).numpy()

    # Display sample
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image_a_np)
    axes[0].set_title("T1 - Image A")
    axes[0].axis("off")

    axes[1].imshow(image_b_np)
    axes[1].set_title("T2 - Image B")
    axes[1].axis("off")

    axes[2].imshow(label_np, cmap="gray")
    axes[2].set_title("Ground Truth Change Mask")
    axes[2].axis("off")

    plt.suptitle(f"LEVIR-CD Sample: {filename}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()