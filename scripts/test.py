from pathlib import Path
import argparse
import sys
import json
import time
import torch
from tqdm import tqdm

# Allow imports from src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from training.config import TrainingConfig
from training.dataloaders import create_eval_dataloader
from training.optimizers import create_training_components
from evaluation.metrics import empty_counts, accumulate_counts, calculate_metrics_from_counts


def evaluate_test_set(model, test_loader, device, threshold=0.5):
    print(f"Evaluating on Test Set with threshold {threshold}...")
    model.eval()

    counts = empty_counts()
    start_time = time.time()

    with torch.no_grad():
        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
            for batch in tqdm(test_loader, desc="Testing"):
                image_a = batch["image_a"].to(device, non_blocking=True)
                image_b = batch["image_b"].to(device, non_blocking=True)
                targets = batch["label"].to(device, non_blocking=True)

                logits = model(image_a, image_b)
                probs = torch.sigmoid(logits.float())
                
                predictions = probs >= threshold
                target_binary = targets >= 0.5

                batch_counts = {
                    "tp": int((predictions & target_binary).sum().item()),
                    "tn": int(((~predictions) & (~target_binary)).sum().item()),
                    "fp": int((predictions & (~target_binary)).sum().item()),
                    "fn": int(((~predictions) & target_binary).sum().item()),
                }
                accumulate_counts(counts, batch_counts)

    duration = time.time() - start_time
    metrics = calculate_metrics_from_counts(counts)
    metrics["duration"] = duration
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate the model on the unseen Test Set.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--batch-size", type=int, default=1, help="Eval batch size")
    parser.add_argument("--num-workers", type=int, default=2, help="Dataloader workers")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
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
    
    test_loader = create_eval_dataloader(
        dataset_dir=dataset_dir,
        split="test",
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    print(f"Found {len(test_loader.dataset)} test scenes.")

    # Initialize model
    model, _, _, _ = create_training_components(config)
    model = model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics = evaluate_test_set(model, test_loader, device, threshold=args.threshold)

    print("\n" + "=" * 60)
    print("FINAL TEST SET METRICS")
    print("=" * 60)
    print(f"F1 Score:    {metrics['f1']:.4f}")
    print(f"IoU Score:   {metrics['iou']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print("-" * 60)
    print(f"True Pos:    {metrics.get('tp', 0):,}")
    print(f"True Neg:    {metrics.get('tn', 0):,}")
    print(f"False Pos:   {metrics.get('fp', 0):,}")
    print(f"False Neg:   {metrics.get('fn', 0):,}")
    print(f"Time Taken:  {metrics['duration']:.1f}s")
    print("=" * 60)
    
    # Save results
    run_dir = checkpoint_path.parent.parent
    save_path = run_dir / "test_metrics.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved test results to {save_path}")


if __name__ == "__main__":
    main()
