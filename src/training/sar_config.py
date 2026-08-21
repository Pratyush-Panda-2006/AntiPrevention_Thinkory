from dataclasses import dataclass


@dataclass
class SARTrainingConfig:
    """
    Configuration for Sentinel-1 SAR change-detection training.

    The configuration is intentionally separate from the optical
    LEVIR-CD training configuration.
    """

    # =========================================================
    # Experiment
    # =========================================================

    experiment_name: str = "tum_oscd_sar"

    seed: int = 42

    # =========================================================
    # Dataset
    # =========================================================

    patch_index_path: str = (
        "data/sar/tum_oscd/sar_patch_index.json"
    )

    train_split: str = "train"
    validation_split: str = "validation"

    patch_size: int = 256
    stride: int = 128

    # =========================================================
    # SAR channels
    # =========================================================

    in_channels: int = 2
    num_classes: int = 1

    # =========================================================
    # Locked normalization
    # =========================================================

    vv_min_db: float = -22.98
    vv_max_db: float = 5.63

    vh_min_db: float = -32.33
    vh_max_db: float = -2.53

    # =========================================================
    # Loss
    # =========================================================

    bce_weight: float = 0.5
    dice_weight: float = 0.5
    pos_weight: float = 1.0

    # =========================================================
    # Data loading
    # =========================================================

    batch_size: int = 2
    num_workers: int = 0

    pin_memory: bool = True

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
    # Stability
    # =========================================================

    use_amp: bool = True
    gradient_clip_norm: float = 1.0

    # =========================================================
    # Evaluation
    # =========================================================

    threshold: float = 0.5

    # =========================================================
    # Device
    # =========================================================

    device: str = "auto"

    def get_device(self):
        import torch

        if self.device == "auto":
            return torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        return torch.device(self.device)