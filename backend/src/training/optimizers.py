import torch

from detection.losses import BCEDiceLoss
from detection.siamese_unet import SiameseUNet
from detection.siamese_resnet34_unet import (
    SiameseResNet34UNet,
)
from detection.snunet_cd import SNUNetCD


def create_model(config):
    """
    Create the selected change-detection model.

    Supported:
        siamese_unet
        resnet34_unet
        snunet_cd
    """

    if config.model_name == "siamese_unet":

        return SiameseUNet(
            in_channels=3,
            num_classes=1,
        )

    if config.model_name == "resnet34_unet":

        return SiameseResNet34UNet(
            in_channels=3,
            num_classes=1,
        )

    if config.model_name == "snunet_cd":

        return SNUNetCD(
            in_channels=3,
            num_classes=1,
        )

    raise ValueError(
        f"Unknown model: {config.model_name}\n"
        "Available models: "
        "'siamese_unet', "
        "'resnet34_unet', "
        "'snunet_cd'"
    )


def create_loss(config):
    """
    Create BCE + Dice loss.
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

    Validation F1 is the monitored metric.
    Higher F1 is better, therefore mode='max'.
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
    Create the complete training stack.

    Returns:
        model
        criterion
        optimizer
        scheduler
    """

    model = create_model(config)

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