import sys
from pathlib import Path
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

# Setup paths
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir / "backend/src"))

from detection.snunet_cd import SNUNetCD
from detection.siamese_resnet34_unet import SiameseResNet34UNet
from training.sar_config import SARTrainingConfig
from data.sar_patch_dataset import TUMSARChangeDetectionDataset
from torch.utils.data import DataLoader

def calculate_metrics(preds, labels, threshold):
    preds_bin = (preds >= threshold).float()
    
    tp = (preds_bin * labels).sum().item()
    fp = (preds_bin * (1 - labels)).sum().item()
    fn = ((1 - preds_bin) * labels).sum().item()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    
    return {
        "threshold": round(threshold, 2),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "tp": tp,
        "fp": fp,
        "fn": fn
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model", choices=["snunet", "resnet34"], default="snunet")
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    checkpoint_path = run_dir / "checkpoints/best.pt"
    
    config = SARTrainingConfig()
    device = config.get_device()
    
    # 1. Dataloader
    dataset = TUMSARChangeDetectionDataset(
        patch_index_path=root_dir / config.patch_index_path,
        split=config.validation_split,
        root_dir=root_dir
    )
    
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0
    )
    
    # 2. Model
    if args.model == "resnet34":
        model = SiameseResNet34UNet(in_channels=2, num_classes=1, sar_init_mode="average")
    else:
        model = SNUNetCD(in_channels=2, num_classes=1)
    
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
        
    model = model.to(device)
    model.eval()
    
    # 3. Collect Predictions
    all_preds = []
    all_labels = []
    
    print("Running inference on validation set...")
    with torch.no_grad():
        for batch in tqdm(loader):
            img_a = batch['image_a'].to(device)
            img_b = batch['image_b'].to(device)
            labels = batch['label'].to(device)
            valid_mask = batch['valid_mask'].to(device)
            
            outputs = model(img_a, img_b)
            # Apply sigmoid
            probs = torch.sigmoid(outputs)
            
            # Mask out invalid padding
            probs = probs[valid_mask]
            labels = labels[valid_mask]
            
            all_preds.append(probs.cpu())
            all_labels.append(labels.cpu())
            
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    # 4. Sweep
    thresholds = np.arange(0.10, 0.91, 0.05)
    results = []
    
    print("Sweeping thresholds...")
    for t in thresholds:
        metrics = calculate_metrics(all_preds, all_labels, float(t))
        results.append(metrics)
        
    # 5. Output Results
    # Find best F1 and IoU
    best_f1_result = max(results, key=lambda x: x['f1'])
    best_iou_result = max(results, key=lambda x: x['iou'])
    result_05 = next((r for r in results if r['threshold'] == 0.5), None)
    
    # Save JSON
    with open(run_dir / "threshold_sweep.json", "w") as f:
        json.dump({
            "sweep": results,
            "best_f1": best_f1_result,
            "best_iou": best_iou_result,
            "baseline_0.5": result_05
        }, f, indent=2)
        
    # Plot
    th = [r['threshold'] for r in results]
    f1 = [r['f1'] for r in results]
    iou = [r['iou'] for r in results]
    prec = [r['precision'] for r in results]
    rec = [r['recall'] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(th, f1, label='F1 Score', color='green', marker='o')
    plt.plot(th, iou, label='IoU', color='purple', marker='s')
    plt.plot(th, prec, label='Precision', color='blue', linestyle='--')
    plt.plot(th, rec, label='Recall', color='red', linestyle='--')
    plt.axvline(x=best_f1_result['threshold'], color='gray', linestyle=':', label=f"Best F1 Thr: {best_f1_result['threshold']}")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Metrics vs Decision Threshold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_dir / "threshold_sweep.png")
    plt.close()
    
    # Markdown
    md = [
        "# Validation Threshold Sweep",
        "",
        "## Best Thresholds",
        f"- **Best F1 Threshold**: {best_f1_result['threshold']:.2f}",
        f"  - **F1**: {best_f1_result['f1']:.4f}",
        f"  - **IoU**: {best_f1_result['iou']:.4f}",
        f"  - **Precision**: {best_f1_result['precision']:.4f}",
        f"  - **Recall**: {best_f1_result['recall']:.4f}",
        "",
        f"- **Best IoU Threshold**: {best_iou_result['threshold']:.2f}",
        f"  - **IoU**: {best_iou_result['iou']:.4f}",
        "",
        "## Baseline (0.50 Threshold)",
        f"- **F1**: {result_05['f1']:.4f}",
        f"- **IoU**: {result_05['iou']:.4f}",
        f"- **Precision**: {result_05['precision']:.4f}",
        f"- **Recall**: {result_05['recall']:.4f}",
        "",
        "## Sweep Results",
        "| Threshold | Precision | Recall | F1 | IoU |",
        "| :---: | :---: | :---: | :---: | :---: |"
    ]
    
    for r in results:
        md.append(f"| {r['threshold']:.2f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['iou']:.4f} |")
        
    with open(run_dir / "threshold_sweep.md", "w") as f:
        f.write("\n".join(md))
        
    print("Done. Generated threshold_sweep.json, threshold_sweep.md, and threshold_sweep.png.")

if __name__ == "__main__":
    main()
