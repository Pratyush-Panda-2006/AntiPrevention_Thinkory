import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy.ndimage

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.config import TrainingConfig
from training.optimizers import create_training_components
from data.levir_fullscene_dataset import LEVIRCDFullSceneDataset

def get_scene_categories(dataset_dir):
    label_dir = dataset_dir / "label"
    files = sorted([f.name for f in label_dir.iterdir() if f.name.startswith("test_")])
    
    no_change = []
    normal = []
    dense = []
    
    for f in files:
        lbl = np.array(Image.open(label_dir / f).convert("L")) > 0
        s = lbl.sum()
        if s == 0:
            no_change.append(f)
        elif s > 0:
            labeled, num_features = scipy.ndimage.label(lbl)
            if num_features > 50:
                dense.append((f, num_features))
            else:
                normal.append(f)
                
    dense = sorted(dense, key=lambda x: x[1], reverse=True)
    dense_files = [x[0] for x in dense]
    
    return no_change, dense_files, normal

def main():
    dataset_dir = PROJECT_ROOT / "data" / "raw" / "LEVIR-CD"
    checkpoint_path = PROJECT_ROOT / "runs" / "2026-08-17_20-42-38_siamese_unet_baseline" / "checkpoints" / "best.pt"
    out_dir = Path(r"C:\Users\Shaurya Deep Rai\.gemini\antigravity-ide\brain\bf7174b5-c186-41e5-9bc7-3ca96b84654b\scratch\spotcheck")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = TrainingConfig()
    for k, v in checkpoint.get("config", {}).items():
        if hasattr(config, k):
            setattr(config, k, v)
            
    model, _, _, _ = create_training_components(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    print("Scanning test set for categories...")
    no_change, dense, normal = get_scene_categories(dataset_dir)
    
    selected_files = []
    if no_change: selected_files.append(("no_change", no_change[0]))
    if dense: selected_files.append(("dense", dense[0]))
    # Pick 4 normal images spread across the normal set
    indices = np.linspace(0, len(normal) - 1, 4, dtype=int)
    for i in indices:
        selected_files.append((f"normal_{i}", normal[i]))
        
    print(f"Selected files: {selected_files}")
    
    dataset = LEVIRCDFullSceneDataset(root_dir=dataset_dir, split="test")
    file_to_idx = {f: i for i, f in enumerate(dataset.files)}
    
    for cat, fname in selected_files:
        idx = file_to_idx[fname]
        sample = dataset[idx]
        
        image_a = sample["image_a"].unsqueeze(0).to(device)
        image_b = sample["image_b"].unsqueeze(0).to(device)
        gt = sample["label"].squeeze().cpu().numpy()
        
        with torch.no_grad():
            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                logits = model(image_a, image_b)
                probs = torch.sigmoid(logits.float())
                pred = (probs >= 0.5).squeeze().cpu().numpy()
                
        # Convert T1/T2 back to images for display
        img_a_disp = image_a.squeeze().cpu().numpy().transpose(1, 2, 0)
        img_b_disp = image_b.squeeze().cpu().numpy().transpose(1, 2, 0)
        
        # Denormalize if they were normalized?
        # LEVIRCDFullSceneDataset just does img / 255.0 so they are already in [0, 1] range!
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(img_a_disp)
        axes[0].set_title(f"T1 ({fname})")
        axes[0].axis('off')
        
        axes[1].imshow(img_b_disp)
        axes[1].set_title("T2")
        axes[1].axis('off')
        
        axes[2].imshow(gt, cmap='gray')
        axes[2].set_title("Ground Truth")
        axes[2].axis('off')
        
        axes[3].imshow(pred, cmap='gray')
        axes[3].set_title("Prediction (thr=0.5)")
        axes[3].axis('off')
        
        plt.tight_layout()
        out_path = out_dir / f"{cat}_{fname}"
        plt.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"Saved {out_path}")

if __name__ == '__main__':
    main()
