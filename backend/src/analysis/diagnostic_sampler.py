import sys
from pathlib import Path
import torch

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.dataloaders import create_train_dataloader
from training.config import TrainingConfig

def main():
    print("=" * 60)
    print("SAMPLER DIAGNOSTIC: REAL TRAIN DATALOADER")
    print("=" * 60)

    dataset_dir = PROJECT_ROOT / "data" / "raw" / "LEVIR-CD"
    
    if not dataset_dir.exists():
        print(f"Error: Dataset not found at {dataset_dir}")
        return

    config = TrainingConfig()
    
    # Create the dataloader exactly as it is used in train.py
    train_loader = create_train_dataloader(
        dataset_dir=dataset_dir,
        batch_size=4,
        num_workers=config.num_workers,
        patch_size=config.patch_size,
        stride=config.stride,
    )
    
    dataset = train_loader.dataset
    
    print("Calculating unweighted base rate (might take a minute)...")
    if hasattr(dataset, 'patch_weights'):
        high_weight_mask = dataset.patch_weights > 1.0
        total_patches = len(dataset.patch_weights)
        changed_patches_base = high_weight_mask.sum().item()
        base_rate = changed_patches_base / total_patches
        print(f"Total patches in dataset: {total_patches:,}")
        print(f"Patches with >= 1% change (base): {changed_patches_base:,}")
        print(f"Base Rate (Unweighted): {base_rate:.2%}")
    else:
        print("Warning: patch_weights not found on dataset. Change-aware sampling might be OFF.")
        return
        
    print("\nIterating 100 batches from DataLoader to check sampled rate...")
    sampled_changed = 0
    total_sampled = 0
    
    MAX_BATCHES = 100 
    
    for idx, batch in enumerate(train_loader):
        if idx >= MAX_BATCHES:
            break
            
        labels = batch["label"] # Shape: [B, 1, H, W]
        # Calculate change percentage for each patch in batch
        for i in range(labels.shape[0]):
            patch_label = labels[i]
            change_ratio = patch_label.mean().item()
            if change_ratio > 0.01: # > 1% change
                sampled_changed += 1
            total_sampled += 1
            
        if (idx + 1) % 25 == 0:
            print(f"Processed {idx + 1} batches...")

    sampled_rate = sampled_changed / total_sampled
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Base Rate (Unweighted): {base_rate:.2%}")
    print(f"Sampled Rate (from DataLoader): {sampled_rate:.2%}")
    
    if sampled_rate > (base_rate * 1.5):
        print("\n✅ PASS: Sampler is successfully shifting the distribution!")
    else:
        print("\n❌ FAIL: Sampled rate matches base rate. Sampler is NOT wired in correctly.")

if __name__ == "__main__":
    main()
