from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """
    Configuration for LEVIR-CD change-detection training.

    Supported models:
        - siamese_unet
        - resnet34_unet
        - snunet_cd
    """

    # =========================================================
    # Observability / Run Tracking
    # =========================================================

    experiment_name: str = "siamese_unet"
    log_level: str = "normal"
    run_dir: str = ""

    # =========================================================
    # Model Selection
    # =========================================================

    # Available:
    #   siamese_unet
    #   resnet34_unet
    #   snunet_cd
    #
    # Model 1:
    #   From-scratch Siamese U-Net
    #
    # Model 2:
    #   ImageNet-pretrained ResNet-34 Siamese U-Net
    #
    # Model 3:
    #   SNUNet-CD architecture
    #   (final-output-only training)

    model_name: str = "siamese_unet"

    # =========================================================
    # Reproducibility
    # =========================================================

    seed: int = 42

    # =========================================================
    # Data
    # =========================================================

    patch_size: int = 256
    stride: int = 128

    # Production defaults are intended for the GPU machine.
    # Local CPU smoke tests should override these from CLI.
    batch_size: int = 16
    num_workers: int = 8

    change_threshold: float = 0.01

    # =========================================================
    # Loss
    # =========================================================

    bce_weight: float = 0.5
    dice_weight: float = 0.5

    # Avoid stacking strong pixel-level positive weighting
    # on top of change-aware patch sampling.
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
    early_stopping_patience: int = 12

    # =========================================================
    # Stability / Performance
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