from pathlib import Path
import argparse
import sys
import json
import torch
from tqdm import tqdm

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# Allow imports from src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from training.config import TrainingConfig
from training.dataloaders import create_eval_dataloader
from training.optimizers import create_training_components


def sweep_thresholds(
    model, 
    val_loader, 
    device, 
    thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
):
    print("Running forward passes over validation set...")
    model.eval()

    all_probs = []
    all_targets = []

    with torch.no_grad():
        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
            for batch in tqdm(val_loader, desc="Forward Passes"):
                image_a = batch["image_a"].to(device, non_blocking=True)
                image_b = batch["image_b"].to(device, non_blocking=True)
                targets = batch["label"]

                logits = model(image_a, image_b)
                probs = torch.sigmoid(logits.float()).cpu()

                # Flatten to save memory and prepare for vectorized metric calc
                all_probs.append(probs.view(-1))
                all_targets.append(targets.view(-1))

    print("Concatenating results...")
    probs_tensor = torch.cat(all_probs)
    targets_tensor = torch.cat(all_targets).bool()

    print(f"Total pixels evaluated: {probs_tensor.shape[0]:,}")

    results = []

    for t in tqdm(thresholds, desc="Evaluating Thresholds"):
        predictions = probs_tensor >= t

        tp = (predictions & targets_tensor).sum().item()
        tn = ((~predictions) & (~targets_tensor)).sum().item()
        fp = (predictions & (~targets_tensor)).sum().item()
        fn = ((~predictions) & targets_tensor).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

        results.append({
            "threshold": t,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
            "accuracy": accuracy,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Sweep threshold on validation set.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--batch-size", type=int, default=1, help="Eval batch size")
    parser.add_argument("--num-workers", type=int, default=8, help="Dataloader workers")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config_dict = checkpoint.get("config", {})
    
    config = TrainingConfig()
    for k, v in config_dict.items():
        if hasattr(config, k):
            setattr(config, k, v)
            
    config.batch_size = args.batch_size
    config.num_workers = args.num_workers

    # Create dataset and loader
    dataset_dir = PROJECT_ROOT / "data" / "raw" / "LEVIR-CD"
    val_loader = create_eval_dataloader(
        dataset_dir=dataset_dir,
        split="val",
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    # Initialize model
    model, _, _, _ = create_training_components(config)
    model = model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    thresholds = [x / 100.0 for x in range(5, 96, 5)] # 0.05 to 0.95
    results = sweep_thresholds(model, val_loader, device, thresholds)

    best_result = max(results, key=lambda x: x["f1"])
    
    print("\n" + "=" * 60)
    print("THRESHOLD SWEEP RESULTS")
    print("=" * 60)
    print(f"{'Threshold':<10} | {'F1':<10} | {'IoU':<10} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 60)
    for r in results:
        marker = " *" if r["threshold"] == best_result["threshold"] else ""
        print(f"{r['threshold']:<10.2f} | {r['f1']:<10.4f} | {r['iou']:<10.4f} | {r['precision']:<10.4f} | {r['recall']:<10.4f}{marker}")
    
    print("=" * 60)
    print(f"Optimal F1 Threshold: {best_result['threshold']:.2f}")
    print(f"Best F1 Score: {best_result['f1']:.4f}")
    print(f"Best IoU Score: {best_result['iou']:.4f}")
    
    # Save results alongside checkpoint
    run_dir = checkpoint_path.parent.parent
    save_path = run_dir / "threshold_sweep.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved sweep results to {save_path}")

    # Plot
    if plt is not None:
        ts = [r["threshold"] for r in results]
        f1s = [r["f1"] for r in results]
        ious = [r["iou"] for r in results]
        
        plt.figure(figsize=(8, 6))
        plt.plot(ts, f1s, label="F1 Score", marker="o", color="green")
        plt.plot(ts, ious, label="IoU", marker="o", color="purple")
        plt.axvline(best_result["threshold"], color="red", linestyle="--", label=f"Best ({best_result['threshold']:.2f})")
        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.title("Validation Metrics vs Threshold")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(run_dir / "plots" / "threshold_sweep.png")


if __name__ == "__main__":
    main()
