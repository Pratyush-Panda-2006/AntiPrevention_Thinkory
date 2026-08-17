from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """
    Configuration for the baseline Siamese U-Net training run.
    """

    # =========================================================
    # Observability / Run Tracking
    # =========================================================

    experiment_name: str = "siamese_unet_baseline"
    log_level: str = "normal"  # "minimal", "normal", "detailed"
    run_dir: str = ""  # Populated dynamically at runtime

    # =========================================================
    # Reproducibility
    # =========================================================

    seed: int = 42

    # =========================================================
    # Data
    # =========================================================

    patch_size: int = 256
    stride: int = 128

    # Start with 16.
    # Increase to 32 only if GPU memory allows.
    batch_size: int = 16

    # Adjust according to friend's CPU.
    num_workers: int = 8

    change_threshold: float = 0.01

    # =========================================================
    # Loss
    # =========================================================

    bce_weight: float = 0.5
    dice_weight: float = 0.5

    # Baseline deliberately avoids stacking strong
    # pixel-level positive weighting on top of
    # change-aware patch sampling.
    pos_weight: float = 1.0

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

    # Give the scheduler time to reduce LR and
    # allow the model to improve afterward.
    early_stopping_patience: int = 12

    # =========================================================
    # Stability / performance
    # =========================================================

    use_amp: bool = True
    gradient_clip_norm: float = 1.0

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