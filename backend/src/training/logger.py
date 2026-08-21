import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


class TrainingLogger:
    """
    Handles terminal progress display and batch-level JSONL logging.
    """

    def __init__(self, run_dir: str, level: str = "normal"):
        self.run_dir = Path(run_dir)
        self.level = level
        self.pbar = None
        
        self.batch_metrics_file = None
        if self.level == "detailed":
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.batch_metrics_file = open(
                self.run_dir / "batch_metrics.jsonl", 
                "a", 
                encoding="utf-8"
            )

    def __del__(self):
        if self.batch_metrics_file:
            self.batch_metrics_file.close()

    def start_epoch(self, epoch: int, total_epochs: int, total_batches: int, mode: str = "Train"):
        """
        Initializes the tqdm progress bar for the epoch.
        """
        if tqdm is None:
            print(f"\n{mode} Epoch {epoch}/{total_epochs} started. (tqdm not installed)")
            return

        desc = f"Epoch {epoch:02d}/{total_epochs:02d} {mode}"
        self.pbar = tqdm(
            total=total_batches,
            desc=desc,
            file=sys.stdout,
            dynamic_ncols=True,
            leave=True,
        )

    def log_batch(self, metrics: Dict[str, Any]):
        """
        Updates the progress bar and logs to file if detailed.
        """
        # 1. Update Progress Bar
        if self.pbar is not None:
            postfix = {}
            
            if "loss" in metrics:
                postfix["loss"] = f"{metrics['loss']:.4f}"
            
            if "lr" in metrics:
                postfix["lr"] = f"{metrics['lr']:.2e}"
                
            if "samples_per_sec" in metrics:
                postfix["samples/s"] = f"{metrics['samples_per_sec']:.1f}"
                
            if "gpu_allocated_gb" in metrics and "gpu_total_memory_gb" in metrics:
                postfix["GPU"] = f"{metrics['gpu_allocated_gb']:.1f}/{metrics['gpu_total_memory_gb']:.1f}GB"
            
            if "epoch_eta" in metrics:
                postfix["eta"] = metrics["epoch_eta"]

            self.pbar.set_postfix(postfix)
            self.pbar.update(1)

        # 2. Detailed File Logging
        if self.batch_metrics_file is not None:
            # We only record batch logs if detailed
            self.batch_metrics_file.write(json.dumps(metrics) + "\n")
            # Don't flush every batch to save IO, rely on close or OS buffering.

    def end_epoch(self):
        """
        Closes the progress bar.
        """
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None
            
        if self.batch_metrics_file is not None:
            self.batch_metrics_file.flush()

    def log_info(self, message: str):
        """
        Log a general message to stdout (bypassing pbar if active).
        """
        if self.pbar is not None:
            self.pbar.write(message)
        else:
            print(message)
