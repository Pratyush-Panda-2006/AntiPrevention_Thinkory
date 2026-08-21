import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice loss for binary segmentation.

    Dice loss is useful for change detection because changed
    pixels can represent a relatively small portion of the image.
    """

    def __init__(self, smooth=1.0):
        super().__init__()

        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits:
                Raw model outputs.
                Shape: [B, 1, H, W]

            targets:
                Binary ground-truth masks.
                Shape: [B, 1, H, W]

        Returns:
            Scalar Dice loss.
        """

        probabilities = torch.sigmoid(logits)

        probabilities = probabilities.contiguous().view(
            probabilities.shape[0],
            -1,
        )

        targets = targets.contiguous().view(
            targets.shape[0],
            -1,
        )

        intersection = (
            probabilities * targets
        ).sum(dim=1)

        denominator = (
            probabilities.sum(dim=1)
            + targets.sum(dim=1)
        )

        dice = (
            2.0 * intersection
            + self.smooth
        ) / (
            denominator
            + self.smooth
        )

        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross-Entropy + Dice loss.

    BCE provides stable pixel-level supervision.
    Dice directly encourages overlap between predicted
    and ground-truth change regions.

    Total loss:

        loss = bce_weight * BCE
             + dice_weight * Dice
    """

    def __init__(
        self,
        bce_weight=0.5,
        dice_weight=0.5,
        pos_weight=None,
    ):
        super().__init__()

        if bce_weight < 0 or dice_weight < 0:
            raise ValueError(
                "Loss weights must be non-negative."
            )

        if bce_weight == 0 and dice_weight == 0:
            raise ValueError(
                "At least one loss weight must be greater than zero."
            )

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        self.dice_loss = DiceLoss()

        if pos_weight is not None:
            pos_weight = torch.as_tensor(
                pos_weight,
                dtype=torch.float32,
            )

        self.register_buffer(
            "pos_weight",
            pos_weight,
            persistent=False,
        )

    def forward(self, logits, targets):

        if logits.shape != targets.shape:
            raise ValueError(
                "Logits and targets must have identical shapes. "
                f"Got logits={logits.shape}, "
                f"targets={targets.shape}"
            )

        if targets.min() < 0 or targets.max() > 1:
            raise ValueError(
                "Targets must contain values in [0, 1]."
            )

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
        )

        dice = self.dice_loss(
            logits,
            targets,
        )

        total = (
            self.bce_weight * bce
            + self.dice_weight * dice
        )

        return total


def build_loss(
    bce_weight=0.5,
    dice_weight=0.5,
    pos_weight=None,
):
    """
    Factory function used by the training configuration.
    """

    return BCEDiceLoss(
        bce_weight=bce_weight,
        dice_weight=dice_weight,
        pos_weight=pos_weight,
    )

