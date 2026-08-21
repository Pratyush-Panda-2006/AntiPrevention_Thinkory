from pathlib import Path
import json
import time
from typing import Optional

import torch

from evaluation.metrics import (
    accumulate_counts,
    calculate_metrics_from_counts,
    empty_counts,
)

from training.telemetry import TimingTelemetry, GPUTelemetry
from training.logger import TrainingLogger
from training.reporting import ExperimentReporter


class TrainingEngine:
    """
    Training engine for the baseline Siamese U-Net.

    Responsibilities:
        - training epochs
        - validation
        - loss calculation
        - metrics
        - AMP mixed-precision training
        - gradient clipping
        - scheduler updates
        - checkpoint saving
        - checkpoint resume
        - early stopping
        - training history
        - observable telemetry and logging
    """

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        config,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.config = config

        self.device = config.get_device()

        self.model = self.model.to(self.device)

        # ---------------------------------------------------------
        # Observability Modules
        # ---------------------------------------------------------
        self.run_dir = Path(config.run_dir) if getattr(config, "run_dir", "") else Path("runs/default")
        self.logger = TrainingLogger(run_dir=self.run_dir, level=getattr(config, "log_level", "normal"))
        self.reporter = ExperimentReporter(run_dir=self.run_dir)
        
        self.timing = TimingTelemetry(smoothing=0.1)
        self.gpu_stats = GPUTelemetry(self.device)

        # ---------------------------------------------------------
        # AMP
        # ---------------------------------------------------------

        self.use_amp = (
            bool(getattr(config, "use_amp", True))
            and self.device.type == "cuda"
        )

        self.gradient_clip_norm = float(
            getattr(
                config,
                "gradient_clip_norm",
                1.0,
            )
        )

        if self.use_amp:
            if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
                self.scaler = torch.amp.GradScaler("cuda", enabled=True)
            else:
                self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        else:
            if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
                self.scaler = torch.amp.GradScaler("cuda", enabled=False)
            else:
                self.scaler = torch.cuda.amp.GradScaler(enabled=False)

        # ---------------------------------------------------------
        # Checkpoint directory
        # ---------------------------------------------------------

        self.checkpoint_dir = Path(getattr(config, "checkpoint_dir", self.run_dir / "checkpoints"))
        if self.checkpoint_dir.name != "checkpoints" and str(self.checkpoint_dir) != getattr(config, "checkpoint_dir", ""):
            self.checkpoint_dir = self.run_dir / "checkpoints"
            
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # ---------------------------------------------------------
        # Training state
        # ---------------------------------------------------------

        self.start_epoch = 1
        self.best_f1 = -float("inf")
        self.epochs_without_improvement = 0
        self.history = []
        self.global_step = 0

    # =============================================================
    # AMP helper
    # =============================================================

    def _autocast_context(self):
        if self.use_amp:
            return torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=True,
            )

        return torch.autocast(
            device_type="cpu",
            enabled=False,
        )

    # =============================================================
    # Training
    # =============================================================

    def train_one_epoch(
        self,
        epoch: int,
        max_batches=None,
    ):
        self.model.train()

        total_loss = 0.0
        samples_processed = 0
        batches_processed = 0
        
        total_batches = len(self.train_loader) if max_batches is None else min(len(self.train_loader), max_batches)
        
        self.logger.start_epoch(epoch, self.config.epochs, total_batches, mode="Train")
        self.timing.start_epoch()

        for batch_index, batch in enumerate(self.train_loader):
            if max_batches is not None and batch_index >= max_batches:
                break

            self.timing.start_batch()
            data_time = time.time() - self.timing.batch_start_time # Approximated here as very small, real data time occurs during iterator next()

            image_a = batch["image_a"].to(self.device, non_blocking=True)
            image_b = batch["image_b"].to(self.device, non_blocking=True)
            targets = batch["label"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with self._autocast_context():
                logits = self.model(image_a, image_b)
                loss = self.criterion(logits, targets)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                
                # Compute gradient norm for tracking if possible
                total_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_norm,
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                total_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_norm,
                )
                self.optimizer.step()

            batch_size = targets.shape[0]
            loss_val = loss.detach().item()
            total_loss += loss_val * batch_size

            samples_processed += batch_size
            batches_processed += 1
            self.global_step += 1

            batch_duration = self.timing.end_batch()
            
            # Telemetry logic
            current_lr = self.optimizer.param_groups[0]["lr"]
            gpu_metrics = self.gpu_stats.get_metrics()
            
            remaining_batches_epoch = total_batches - batches_processed
            remaining_epochs = self.config.epochs - epoch
            
            eta_seconds = self.timing.estimate_run_eta(
                remaining_batches_epoch, remaining_epochs, total_batches
            )
            
            samples_per_sec = batch_size / batch_duration if batch_duration > 0 else 0.0

            log_dict = {
                "timestamp": time.time(),
                "epoch": epoch,
                "batch": batch_index,
                "global_step": self.global_step,
                "loss": loss_val,
                "lr": current_lr,
                "batch_time_sec": batch_duration,
                "samples_per_sec": samples_per_sec,
                "gradient_norm": total_norm.item() if isinstance(total_norm, torch.Tensor) else float(total_norm),
                "epoch_eta": self.timing.format_time(self.timing.estimate_epoch_eta(remaining_batches_epoch)),
                "run_eta": self.timing.format_time(eta_seconds),
                **gpu_metrics
            }
            
            self.logger.log_batch(log_dict)

        if batches_processed == 0:
            raise RuntimeError("No training batches were processed.")

        average_loss = total_loss / samples_processed
        epoch_duration = self.timing.get_epoch_elapsed()

        self.logger.end_epoch()

        return {
            "loss": average_loss,
            "batches": batches_processed,
            "samples": samples_processed,
            "duration": epoch_duration,
            "throughput": samples_processed / epoch_duration if epoch_duration > 0 else 0.0
        }

    # =============================================================
    # Validation
    # =============================================================

    @torch.no_grad()
    def validate(
        self,
        epoch: int,
        max_batches=None,
    ):
        self.model.eval()

        total_loss = 0.0
        samples_processed = 0
        batches_processed = 0

        counts = empty_counts()
        
        total_batches = len(self.val_loader) if max_batches is None else min(len(self.val_loader), max_batches)
        self.logger.start_epoch(epoch, self.config.epochs, total_batches, mode="Val")
        
        val_start_time = time.time()

        for batch_index, batch in enumerate(self.val_loader):
            if max_batches is not None and batch_index >= max_batches:
                break

            b_start = time.time()

            image_a = batch["image_a"].to(self.device, non_blocking=True)
            image_b = batch["image_b"].to(self.device, non_blocking=True)
            targets = batch["label"].to(self.device, non_blocking=True)

            with self._autocast_context():
                logits = self.model(image_a, image_b)
                loss = self.criterion(logits, targets)

            probabilities = torch.sigmoid(logits.float())
            predictions = probabilities >= self.config.threshold
            target_binary = targets >= 0.5

            batch_counts = {
                "tp": int((predictions & target_binary).sum().item()),
                "tn": int(((~predictions) & (~target_binary)).sum().item()),
                "fp": int((predictions & (~target_binary)).sum().item()),
                "fn": int(((~predictions) & target_binary).sum().item()),
            }
            accumulate_counts(counts, batch_counts)

            batch_size = targets.shape[0]
            loss_val = loss.detach().item()
            total_loss += loss_val * batch_size

            samples_processed += batch_size
            batches_processed += 1
            
            b_dur = time.time() - b_start
            self.logger.log_batch({
                "loss": loss_val,
                "samples_per_sec": batch_size / b_dur if b_dur > 0 else 0.0
            })

        if batches_processed == 0:
            raise RuntimeError("No validation batches were processed.")

        metrics = calculate_metrics_from_counts(counts)
        metrics["loss"] = total_loss / samples_processed
        metrics["batches"] = batches_processed
        metrics["samples"] = samples_processed
        metrics["duration"] = time.time() - val_start_time

        self.logger.end_epoch()

        return metrics

    # =============================================================
    # Checkpoint
    # =============================================================

    def save_checkpoint(
        self,
        epoch,
        filename="last.pt",
    ):
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_f1": self.best_f1,
            "epochs_without_improvement": self.epochs_without_improvement,
            "history": self.history,
            "config": vars(self.config),
            "accumulated_training_seconds": self.timing.accumulated_training_seconds + (time.time() - self.timing.global_start_time)
        }

        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    # =============================================================
    # Resume
    # =============================================================

    def load_checkpoint(
        self,
        checkpoint_path,
    ):
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        scaler_state = checkpoint.get("scaler_state_dict")
        if scaler_state:
            self.scaler.load_state_dict(scaler_state)

        self.start_epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint.get("global_step", 0)
        self.best_f1 = checkpoint.get("best_f1", -float("inf"))
        self.epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)
        self.history = checkpoint.get("history", [])
        
        self.timing.accumulated_training_seconds = checkpoint.get("accumulated_training_seconds", 0.0)

        self.logger.log_info(f"Resumed from epoch {checkpoint['epoch']}")

    # =============================================================
    # Full training
    # =============================================================

    def fit(
        self,
        max_train_batches=None,
        max_val_batches=None,
    ):
        self.timing.global_start_time = time.time()

        for epoch in range(self.start_epoch, self.config.epochs + 1):
            
            # Training
            train_result = self.train_one_epoch(
                epoch=epoch,
                max_batches=max_train_batches
            )

            # Validation
            val_result = self.validate(
                epoch=epoch,
                max_batches=max_val_batches
            )

            # Scheduler step using validation F1 (higher is better)
            old_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_result["f1"])
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Record history
            epoch_result = {
                "epoch": epoch,
                "train_loss": train_result["loss"],
                "val_loss": val_result["loss"],
                "precision": val_result["precision"],
                "recall": val_result["recall"],
                "f1": val_result["f1"],
                "iou": val_result["iou"],
                "accuracy": val_result["accuracy"],
                "tp": val_result["tp"],
                "tn": val_result["tn"],
                "fp": val_result["fp"],
                "fn": val_result["fn"],
                "learning_rate": current_lr,
                "train_batches": train_result["batches"],
                "train_samples": train_result["samples"],
                "val_batches": val_result["batches"],
                "val_samples": val_result["samples"],
                "epoch_seconds": train_result["duration"] + val_result["duration"],
                "train_seconds": train_result["duration"],
                "validation_seconds": val_result["duration"],
                "samples_per_second": train_result["throughput"],
                "lr_reduced": current_lr < old_lr
            }

            self.save_checkpoint(epoch, filename="last.pt")

            is_new_best = False
            if val_result["f1"] > self.best_f1:
                self.best_f1 = val_result["f1"]
                self.epochs_without_improvement = 0
                is_new_best = True
                self.save_checkpoint(epoch, filename="best.pt")
                self.logger.log_info(f"*** New best validation F1: {self.best_f1:.4f} ***")
            else:
                self.epochs_without_improvement += 1
                
            epoch_result["new_best"] = is_new_best
            epoch_result["epochs_without_improvement"] = self.epochs_without_improvement
            
            self.history.append(epoch_result)
            self.reporter.append_epoch_metric(epoch_result)

            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                self.logger.log_info(f"Early stopping triggered at epoch {epoch}")
                break

        return self.history
