from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """
    Configuration for the baseline Siamese U-Net training run.
    """

    # =========================================================
    # Reproducibility
    # =========================================================

    seed: int = 42

    # =========================================================
    # Data
    # =========================================================

    patch_size: int = 256
    stride: int = 128

    batch_size: int = 4
    num_workers: int = 0

    change_threshold: float = 0.01

    # =========================================================
    # Loss
    # =========================================================

    bce_weight: float = 0.5
    dice_weight: float = 0.5

    # Initial baseline value.
    # We will benchmark this later.
    pos_weight: float = 5.0

    # =========================================================
    # Optimizer
    # =========================================================

    learning_rate: float = 1e-4
    weight_decay: float = 1e-4

    # =========================================================
    # Scheduler
    # =========================================================

    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    scheduler_min_lr: float = 1e-6

    # =========================================================
    # Training
    # =========================================================

    epochs: int = 30

    # Number of epochs without validation improvement
    # before early stopping.
    early_stopping_patience: int = 7

    # =========================================================
    # Checkpointing
    # =========================================================

    checkpoint_dir: str = "checkpoints"

    save_best_only: bool = False

    # =========================================================
    # Evaluation
    # =========================================================

    threshold: float = 0.5

    # =========================================================
    # Device
    # =========================================================

    device: str = "auto"

    def get_device(self):
        """
        Select CUDA when available, otherwise CPU.
        """

        import torch

        if self.device == "auto":
            return torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        return torch.device(self.device)