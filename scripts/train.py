from pathlib import Path
import argparse
import sys
import time
import json
import traceback

import torch

# Allow imports from src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from training.config import TrainingConfig
from training.dataloaders import (
    create_train_dataloader,
    create_eval_dataloader,
)
from training.engine import TrainingEngine
from training.optimizers import create_training_components
from training.reproducibility import set_seed
from training.telemetry import SystemTelemetry


def main():

    # =========================================================
    # Command-line arguments
    # =========================================================

    parser = argparse.ArgumentParser(
        description="Train the LEVIR-CD Siamese U-Net."
    )

    parser.add_argument("--smoke-test", action="store_true", help="Run a limited training sanity test.")
    parser.add_argument("--train-batches", type=int, default=2, help="Number of training batches for smoke test.")
    parser.add_argument("--val-batches", type=int, default=1, help="Number of validation batches for smoke test.")
    
    parser.add_argument("--experiment-name", type=str, default="siamese_unet_baseline", help="Name of the experiment run.")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from.")
    parser.add_argument("--log-level", type=str, choices=["minimal", "normal", "detailed"], default="normal", help="Logging verbosity.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override number of DataLoader workers.")
    parser.add_argument("--epochs", type=int, default=None, help="Override total epochs.")

    args = parser.parse_args()

    # =========================================================
    # Validate arguments
    # =========================================================

    if args.train_batches < 1:
        raise ValueError("--train-batches must be at least 1.")
    if args.val_batches < 1:
        raise ValueError("--val-batches must be at least 1.")

    # =========================================================
    # Configuration Setup
    # =========================================================

    config = TrainingConfig()
    config.experiment_name = args.experiment_name
    config.log_level = args.log_level

    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.epochs is not None:
        config.epochs = args.epochs

    if args.smoke_test:
        config.epochs = 1
        config.experiment_name += "_smoke"

    # =========================================================
    # Paths & Run Directory
    # =========================================================

    dataset_dir = PROJECT_ROOT / "data" / "raw" / "LEVIR-CD"

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"{timestamp}_{config.experiment_name}"
    run_dir = PROJECT_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    config.run_dir = str(run_dir)
    config.checkpoint_dir = str(run_dir / "checkpoints")

    # =========================================================
    # Reproducibility
    # =========================================================

    set_seed(config.seed)

    # =========================================================
    # Environment & System Info
    # =========================================================

    env_info = SystemTelemetry.get_environment_info()
    
    with open(run_dir / "environment.json", "w", encoding="utf-8") as f:
        json.dump(env_info, f, indent=2)

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(config), f, indent=2)

    print("=" * 70)
    print("LEVIR-CD SIAMESE U-NET TRAINING")
    print("=" * 70)
    print(f"Run Directory: {run_dir}")
    print(f"Mode: {'SMOKE TEST' if args.smoke_test else 'FULL TRAINING'}")

    # =========================================================
    # Dataset Preparation & Verification
    # =========================================================

    if not dataset_dir.exists():
        raise FileNotFoundError(f"LEVIR-CD dataset not found at:\n{dataset_dir}")

    train_loader = create_train_dataloader(
        dataset_dir=dataset_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        patch_size=config.patch_size,
        stride=config.stride,
    )

    val_loader = create_eval_dataloader(
        dataset_dir=dataset_dir,
        split="val",
        batch_size=1,
        num_workers=config.num_workers,
    )
    
    dataset_info = {
        "dataset_name": "LEVIR-CD",
        "dataset_root": str(dataset_dir),
        "train_patch_count": len(train_loader.dataset),
        "val_scene_count": len(val_loader.dataset),
        "patch_size": config.patch_size,
        "stride": config.stride
    }
    with open(run_dir / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2)

    # =========================================================
    # Training Components
    # =========================================================

    model, criterion, optimizer, scheduler = create_training_components(config)

    engine = TrainingEngine(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )

    # Resume checkpoint if provided
    if args.resume:
        engine.load_checkpoint(args.resume)

    # =========================================================
    # Execution & Crash Handling
    # =========================================================

    run_info = {
        "run_id": run_id,
        "experiment_name": config.experiment_name,
        "timestamp": timestamp,
    }

    start_time = time.time()
    status = "completed"
    exception_msg = None

    try:
        if args.smoke_test:
            engine.logger.log_info("\nRunning smoke test...")
            engine.fit(
                max_train_batches=args.train_batches,
                max_val_batches=args.val_batches,
            )
        else:
            engine.logger.log_info("\nStarting FULL TRAINING...")
            engine.fit()
            
    except KeyboardInterrupt:
        status = "interrupted"
        exception_msg = "Training was manually interrupted via KeyboardInterrupt (Ctrl+C)."
        print("\n\n[!] Training Interrupted by User")
        
    except Exception as e:
        if "CUDA out of memory" in str(e):
            status = "failed_oom"
        else:
            status = "failed"
            
        exception_msg = traceback.format_exc()
        print(f"\n\n[!] Training Failed:\n{exception_msg}")
        
    finally:
        # Guarantee report generation
        run_info["duration"] = time.time() - start_time
        engine.reporter.generate_report(run_info, status, exception_msg)
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE" if status == "completed" else f"TRAINING {status.upper()}")
        print("=" * 70)
        print(f"Run Directory: {run_dir}")
        print(f"Report: {run_dir / 'report.md'}")
        print("=" * 70)

        if status in ["failed", "failed_oom"]:
            sys.exit(1)


if __name__ == "__main__":
    main()