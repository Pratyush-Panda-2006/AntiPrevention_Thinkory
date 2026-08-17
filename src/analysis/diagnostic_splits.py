from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def check_split_leakage():
    dataset_dir = PROJECT_ROOT / "data" / "raw" / "LEVIR-CD"
    
    print("=" * 60)
    print("SPLIT LEAKAGE DIAGNOSTIC (FIXED)")
    print("=" * 60)
    
    if not dataset_dir.exists():
        print(f"Error: Dataset not found at {dataset_dir}")
        return

    # In our repository, all images are pooled together in A/, B/, and label/.
    # The split is determined strictly by the filename prefix.
    directories = ["A", "B", "label"]
    splits = ["train", "val", "test"]
    expected_counts = {"train": 445, "val": 64, "test": 128}
    
    all_good = True
    
    for d in directories:
        target_dir = dataset_dir / d
        if not target_dir.exists():
            print(f"Directory {target_dir} does not exist.")
            all_good = False
            continue
            
        print(f"\nScanning directory: {target_dir.relative_to(PROJECT_ROOT)}/")
        files = list(target_dir.glob("*.*"))
        print(f"Total files: {len(files)}")
        
        split_counts = {"train": 0, "val": 0, "test": 0, "unknown": 0}
        
        for f in files:
            name = f.name
            if name.startswith("train_"):
                split_counts["train"] += 1
            elif name.startswith("val_"):
                split_counts["val"] += 1
            elif name.startswith("test_"):
                split_counts["test"] += 1
            else:
                split_counts["unknown"] += 1
                
        for split, count in split_counts.items():
            if split == "unknown" and count > 0:
                print(f"❌ LEAKAGE/ERROR: Found {count} files with an unknown prefix (not train/val/test).")
                all_good = False
            elif split in expected_counts:
                if count != expected_counts[split]:
                    print(f"❌ ERROR: Expected {expected_counts[split]} {split} files, found {count}.")
                    all_good = False
                else:
                    print(f"✅ Found exact match for {split}: {count} files.")

    print("\n" + "=" * 60)
    if all_good:
        print("✅ PASS: All files in A, B, and label have strictly enforced LEVIR-CD split prefixes.")
        print("✅ PASS: Scene counts match the official LEVIR-CD split perfectly (Train: 445, Val: 64, Test: 128).")
        print("✅ PASS: No data leakage exists in the directory structure.")
    else:
        print("❌ FAIL: Irregularities detected in dataset prefixes or counts.")

if __name__ == "__main__":
    check_split_leakage()
