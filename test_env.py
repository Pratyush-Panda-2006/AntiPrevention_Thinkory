import torch
import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src"))

from data.levir_patch_dataset import LEVIRCDPatchDataset
from data.levir_fullscene_dataset import LEVIRCDFullSceneDataset

print("--- 2. CUDA STATUS ---")
cuda = torch.cuda.is_available()
print(f"CUDA Available: {cuda}")
if cuda:
    print(f"Device: {torch.cuda.get_device_name(0)}")

print("\n--- 3. DATASET STATUS ---")
try:
    dataset_dir = PROJECT_ROOT / "data" / "raw" / "LEVIR-CD"
    train_ds = LEVIRCDPatchDataset(dataset_dir, split="train")
    val_ds = LEVIRCDFullSceneDataset(dataset_dir, split="val")
    print(f"Train patches found: {len(train_ds)}")
    print(f"Validation scenes found: {len(val_ds)}")
except Exception as e:
    print(f"Dataset verification failed: {e}")
