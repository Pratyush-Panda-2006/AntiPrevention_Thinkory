from pathlib import Path
import json

import torch

from evaluation.metrics import (
    accumulate_counts,
    calculate_metrics_from_counts,
    empty_counts,
)


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
            # Use the modern torch.amp API when available.
            if hasattr(torch, "amp") and hasattr(
                torch.amp,
                "GradScaler",
            ):
                self.scaler = torch.amp.GradScaler(
                    "cuda",
                    enabled=True,
                )
            else:
                # Compatibility fallback for older PyTorch.
                self.scaler = torch.cuda.amp.GradScaler(
                    enabled=True,
                )
        else:
            # A disabled scaler keeps the training code simple
            # and makes CPU smoke tests work normally.
            if hasattr(torch, "amp") and hasattr(
                torch.amp,
                "GradScaler",
            ):
                self.scaler = torch.amp.GradScaler(
                    "cuda",
                    enabled=False,
                )
            else:
                self.scaler = torch.cuda.amp.GradScaler(
                    enabled=False,
                )

        # ---------------------------------------------------------
        # Checkpoint directory
        # ---------------------------------------------------------

        self.checkpoint_dir = Path(
            config.checkpoint_dir
        )

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ---------------------------------------------------------
        # Training state
        # ---------------------------------------------------------

        self.start_epoch = 1
        self.best_f1 = -float("inf")
        self.epochs_without_improvement = 0
        self.history = []

    # =============================================================
    # AMP helper
    # =============================================================

    def _autocast_context(self):
        """
        Return an autocast context.

        AMP is enabled only for CUDA. CPU training remains
        full precision.
        """

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
        max_batches=None,
    ):
        """
        Train the model for one epoch.

        Args:
            max_batches:
                Optional limit used for smoke tests.
                None = complete training loader.
        """

        self.model.train()

        total_loss = 0.0
        samples_processed = 0
        batches_processed = 0

        for batch_index, batch in enumerate(
            self.train_loader
        ):

            if (
                max_batches is not None
                and batch_index >= max_batches
            ):
                break

            image_a = batch["image_a"].to(
                self.device,
                non_blocking=True,
            )

            image_b = batch["image_b"].to(
                self.device,
                non_blocking=True,
            )

            targets = batch["label"].to(
                self.device,
                non_blocking=True,
            )

            self.optimizer.zero_grad(
                set_to_none=True
            )

            # -----------------------------------------------------
            # Forward + loss
            # -----------------------------------------------------

            with self._autocast_context():

                logits = self.model(
                    image_a,
                    image_b,
                )

                loss = self.criterion(
                    logits,
                    targets,
                )

            # -----------------------------------------------------
            # Backward
            # -----------------------------------------------------

            if self.use_amp:

                self.scaler.scale(
                    loss
                ).backward()

                # Gradients must be unscaled before clipping.
                self.scaler.unscale_(
                    self.optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_norm,
                )

                self.scaler.step(
                    self.optimizer
                )

                self.scaler.update()

            else:

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_norm,
                )

                self.optimizer.step()

            # -----------------------------------------------------
            # Loss accounting
            # -----------------------------------------------------

            batch_size = targets.shape[0]

            total_loss += (
                loss.detach().item()
                * batch_size
            )

            samples_processed += batch_size
            batches_processed += 1

        if batches_processed == 0:
            raise RuntimeError(
                "No training batches were processed."
            )

        average_loss = (
            total_loss
            / samples_processed
        )

        return {
            "loss": average_loss,
            "batches": batches_processed,
            "samples": samples_processed,
        }

    # =============================================================
    # Validation
    # =============================================================

    @torch.no_grad()
    def validate(
        self,
        max_batches=None,
    ):
        """
        Validate the model over complete validation scenes.

        Metrics are calculated from accumulated TP/TN/FP/FN
        counts over all processed pixels.
        """

        self.model.eval()

        total_loss = 0.0
        samples_processed = 0
        batches_processed = 0

        counts = empty_counts()

        for batch_index, batch in enumerate(
            self.val_loader
        ):

            if (
                max_batches is not None
                and batch_index >= max_batches
            ):
                break

            image_a = batch["image_a"].to(
                self.device,
                non_blocking=True,
            )

            image_b = batch["image_b"].to(
                self.device,
                non_blocking=True,
            )

            targets = batch["label"].to(
                self.device,
                non_blocking=True,
            )

            # -----------------------------------------------------
            # Forward + loss
            # -----------------------------------------------------

            with self._autocast_context():

                logits = self.model(
                    image_a,
                    image_b,
                )

                loss = self.criterion(
                    logits,
                    targets,
                )

            # -----------------------------------------------------
            # Metrics
            # -----------------------------------------------------

            probabilities = torch.sigmoid(
                logits.float()
            )

            predictions = (
                probabilities
                >= self.config.threshold
            )

            target_binary = (
                targets >= 0.5
            )

            batch_counts = {
                "tp": int(
                    (
                        predictions
                        & target_binary
                    ).sum().item()
                ),
                "tn": int(
                    (
                        (~predictions)
                        & (~target_binary)
                    ).sum().item()
                ),
                "fp": int(
                    (
                        predictions
                        & (~target_binary)
                    ).sum().item()
                ),
                "fn": int(
                    (
                        (~predictions)
                        & target_binary
                    ).sum().item()
                ),
            }

            accumulate_counts(
                counts,
                batch_counts,
            )

            batch_size = targets.shape[0]

            total_loss += (
                loss.detach().item()
                * batch_size
            )

            samples_processed += batch_size
            batches_processed += 1

        if batches_processed == 0:
            raise RuntimeError(
                "No validation batches were processed."
            )

        metrics = calculate_metrics_from_counts(
            counts
        )

        metrics["loss"] = (
            total_loss
            / samples_processed
        )

        metrics["batches"] = batches_processed
        metrics["samples"] = samples_processed

        return metrics

    # =============================================================
    # Checkpoint
    # =============================================================

    def save_checkpoint(
        self,
        epoch,
        filename="last.pt",
    ):
        """
        Save complete training state.
        """

        checkpoint_path = (
            self.checkpoint_dir
            / filename
        )

        checkpoint = {
            "epoch": epoch,

            "model_state_dict":
                self.model.state_dict(),

            "optimizer_state_dict":
                self.optimizer.state_dict(),

            "scheduler_state_dict":
                self.scheduler.state_dict(),

            "scaler_state_dict":
                self.scaler.state_dict(),

            "best_f1":
                self.best_f1,

            "epochs_without_improvement":
                self.epochs_without_improvement,

            "history":
                self.history,

            "config":
                vars(self.config),
        }

        torch.save(
            checkpoint,
            checkpoint_path,
        )

        return checkpoint_path

    # =============================================================
    # Resume
    # =============================================================

    def load_checkpoint(
        self,
        checkpoint_path,
    ):
        """
        Resume training from a checkpoint.
        """

        checkpoint_path = Path(
            checkpoint_path
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                checkpoint_path
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        self.scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

        scaler_state = checkpoint.get(
            "scaler_state_dict"
        )

        if scaler_state:
            self.scaler.load_state_dict(
                scaler_state
            )

        self.start_epoch = (
            checkpoint["epoch"] + 1
        )

        self.best_f1 = checkpoint.get(
            "best_f1",
            -float("inf"),
        )

        self.epochs_without_improvement = (
            checkpoint.get(
                "epochs_without_improvement",
                0,
            )
        )

        self.history = checkpoint.get(
            "history",
            [],
        )

        print(
            f"Resumed from epoch "
            f"{checkpoint['epoch']}"
        )

    # =============================================================
    # History
    # =============================================================

    def save_history(self):
        """
        Save training history as JSON.
        """

        history_path = (
            self.checkpoint_dir
            / "training_history.json"
        )

        with open(
            history_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                self.history,
                file,
                indent=2,
            )

        return history_path

    # =============================================================
    # Full training
    # =============================================================

    def fit(
        self,
        max_train_batches=None,
        max_val_batches=None,
    ):
        """
        Run the training process.

        max_train_batches and max_val_batches are intended
        for smoke testing.

        For the real run, both should remain None.
        """

        print("=" * 60)
        print("SIAMESE U-NET TRAINING")
        print("=" * 60)

        print(
            f"\nDevice: {self.device}"
        )

        print(
            f"AMP enabled: "
            f"{self.use_amp}"
        )

        print(
            f"Gradient clip norm: "
            f"{self.gradient_clip_norm}"
        )

        print(
            f"Starting epoch: "
            f"{self.start_epoch}"
        )

        print(
            f"Total epochs: "
            f"{self.config.epochs}"
        )

        for epoch in range(
            self.start_epoch,
            self.config.epochs + 1,
        ):

            print(
                "\n" + "=" * 60
            )

            print(
                f"EPOCH {epoch}/"
                f"{self.config.epochs}"
            )

            print(
                "=" * 60
            )

            # -----------------------------------------------------
            # Training
            # -----------------------------------------------------

            train_result = (
                self.train_one_epoch(
                    max_batches=max_train_batches
                )
            )

            # -----------------------------------------------------
            # Validation
            # -----------------------------------------------------

            val_result = self.validate(
                max_batches=max_val_batches
            )

            # -----------------------------------------------------
            # Scheduler
            # -----------------------------------------------------

            # IMPORTANT:
            # Higher F1 is better, therefore the scheduler must be
            # constructed with mode="max".
            self.scheduler.step(
                val_result["f1"]
            )

            current_lr = (
                self.optimizer
                .param_groups[0]["lr"]
            )

            # -----------------------------------------------------
            # Record history
            # -----------------------------------------------------

            epoch_result = {
                "epoch": epoch,

                "train_loss":
                    train_result["loss"],

                "val_loss":
                    val_result["loss"],

                "precision":
                    val_result["precision"],

                "recall":
                    val_result["recall"],

                "f1":
                    val_result["f1"],

                "iou":
                    val_result["iou"],

                "accuracy":
                    val_result["accuracy"],

                "tp":
                    val_result["tp"],

                "tn":
                    val_result["tn"],

                "fp":
                    val_result["fp"],

                "fn":
                    val_result["fn"],

                "learning_rate":
                    current_lr,

                "train_batches":
                    train_result["batches"],

                "train_samples":
                    train_result["samples"],

                "val_batches":
                    val_result["batches"],

                "val_samples":
                    val_result["samples"],
            }

            self.history.append(
                epoch_result
            )

            # -----------------------------------------------------
            # Print results
            # -----------------------------------------------------

            print(
                f"\nTrain loss: "
                f"{train_result['loss']:.6f}"
            )

            print(
                f"Val loss: "
                f"{val_result['loss']:.6f}"
            )

            print(
                f"Precision: "
                f"{val_result['precision']:.4f}"
            )

            print(
                f"Recall: "
                f"{val_result['recall']:.4f}"
            )

            print(
                f"F1: "
                f"{val_result['f1']:.4f}"
            )

            print(
                f"IoU: "
                f"{val_result['iou']:.4f}"
            )

            print(
                f"Accuracy: "
                f"{val_result['accuracy']:.4f}"
            )

            # -----------------------------------------------------
            # Confusion counts
            # -----------------------------------------------------

            print(
                "\nConfusion counts:"
            )

            print(
                f"TP: {val_result['tp']:,}"
            )

            print(
                f"TN: {val_result['tn']:,}"
            )

            print(
                f"FP: {val_result['fp']:,}"
            )

            print(
                f"FN: {val_result['fn']:,}"
            )

            print(
                f"\nLearning rate: "
                f"{current_lr:.8f}"
            )

            # -----------------------------------------------------
            # Save latest checkpoint
            # -----------------------------------------------------

            self.save_checkpoint(
                epoch,
                filename="last.pt",
            )

            # -----------------------------------------------------
            # Best checkpoint
            # -----------------------------------------------------

            if val_result["f1"] > self.best_f1:

                self.best_f1 = (
                    val_result["f1"]
                )

                self.epochs_without_improvement = 0

                best_path = (
                    self.save_checkpoint(
                        epoch,
                        filename="best.pt",
                    )
                )

                print(
                    f"\n* New best F1: "
                    f"{self.best_f1:.4f}"
                )

                print(
                    f"* Best checkpoint: "
                    f"{best_path}"
                )

            else:

                self.epochs_without_improvement += 1

                print(
                    "\nNo F1 improvement."
                )

                print(
                    f"Epochs without improvement: "
                    f"{self.epochs_without_improvement}"
                )

            # -----------------------------------------------------
            # Save history
            # -----------------------------------------------------

            self.save_history()

            # -----------------------------------------------------
            # Early stopping
            # -----------------------------------------------------

            if (
                self.epochs_without_improvement
                >= self.config.early_stopping_patience
            ):

                print(
                    "\nEarly stopping triggered."
                )

                break

        print(
            "\n" + "=" * 60
        )

        print(
            "TRAINING COMPLETE"
        )

        print(
            "=" * 60
        )

        print(
            f"\nBest validation F1: "
            f"{self.best_f1:.4f}"
        )

        print(
            f"Checkpoint directory: "
            f"{self.checkpoint_dir}"
        )

        return self.history
