import torch

from detection.losses import BCEDiceLoss
from detection.siamese_unet import SiameseUNet
from training.config import TrainingConfig


def create_model():
    """
    Create the baseline Siamese U-Net.
    """

    return SiameseUNet(
        in_channels=3,
        num_classes=1,
    )


def create_loss(config):
    """
    Create BCE + Dice loss using the training configuration.
    """

    return BCEDiceLoss(
        bce_weight=config.bce_weight,
        dice_weight=config.dice_weight,
        pos_weight=config.pos_weight,
    )


def create_optimizer(model, config):
    """
    Create AdamW optimizer.
    """

    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def create_scheduler(optimizer, config):
    """
    Create ReduceLROnPlateau scheduler.

    Validation F1 will be used as the monitored metric
    by the training engine.
    """

    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.scheduler_min_lr,
    )


def create_training_components(config):
    """
    Create all components required for training.

    Returns:
        model
        criterion
        optimizer
        scheduler
    """

    model = create_model()

    criterion = create_loss(config)

    optimizer = create_optimizer(
        model,
        config,
    )

    scheduler = create_scheduler(
        optimizer,
        config,
    )

    return (
        model,
        criterion,
        optimizer,
        scheduler,
    )