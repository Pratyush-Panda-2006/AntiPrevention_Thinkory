"""
src/detection/losses.py
=======================

Binary segmentation losses used by the optical and SAR change-detection
pipelines.

Default behavior remains unchanged for existing optical models:

    BCEDiceLoss(logits, targets)

SAR can additionally provide:

    BCEDiceLoss(logits, targets, valid_mask=mask)

where invalid/padded pixels contribute zero loss and zero gradient.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice loss for binary segmentation.

    Supports optional per-pixel validity masking.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()

        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            logits:
                Raw model outputs.
                Shape: [B, 1, H, W]

            targets:
                Binary masks.
                Shape: [B, 1, H, W]

            valid_mask:
                Optional boolean/float mask.
                Shape: [B, 1, H, W]
                1/True = valid pixel
                0/False = ignored pixel

        Returns:
            Scalar Dice loss.
        """

        probabilities = torch.sigmoid(logits)

        if valid_mask is not None:
            if valid_mask.shape != logits.shape:
                raise ValueError(
                    "valid_mask must have the same shape as logits. "
                    f"Got logits={logits.shape}, "
                    f"valid_mask={valid_mask.shape}"
                )

            valid_mask = valid_mask.to(
                dtype=probabilities.dtype,
                device=probabilities.device,
            )

            probabilities = probabilities * valid_mask
            targets = targets * valid_mask

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
    Combined BCE + Dice loss.

    Existing optical usage remains:

        loss = criterion(logits, targets)

    SAR usage can provide:

        loss = criterion(
            logits,
            targets,
            valid_mask=valid_mask,
        )

    Total loss:

        loss = bce_weight * BCE
             + dice_weight * Dice
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pos_weight: float | None = None,
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

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            logits:
                Shape [B, 1, H, W]

            targets:
                Shape [B, 1, H, W]

            valid_mask:
                Optional shape [B, 1, H, W].
                Invalid pixels contribute zero loss/gradient.

        Returns:
            Scalar combined BCE + Dice loss.
        """

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

        if valid_mask is not None:
            if valid_mask.shape != logits.shape:
                raise ValueError(
                    "valid_mask must have the same shape as logits. "
                    f"Got logits={logits.shape}, "
                    f"valid_mask={valid_mask.shape}"
                )

            if valid_mask.dtype != torch.bool:
                valid_mask = valid_mask > 0

            valid_count = int(valid_mask.sum().item())

            if valid_count == 0:
                raise ValueError(
                    "valid_mask contains no valid pixels."
                )

            mask = valid_mask.to(
                dtype=logits.dtype,
                device=logits.device,
            )

            # -----------------------------------------------------
            # Masked BCE
            # -----------------------------------------------------

            bce_map = F.binary_cross_entropy_with_logits(
                logits,
                targets,
                pos_weight=self.pos_weight,
                reduction="none",
            )

            masked_bce = (
                bce_map * mask
            ).sum() / mask.sum()

        else:
            # Preserve original optical behavior exactly.
            masked_bce = F.binary_cross_entropy_with_logits(
                logits,
                targets,
                pos_weight=self.pos_weight,
            )

        # ---------------------------------------------------------
        # Dice
        # ---------------------------------------------------------

        dice = self.dice_loss(
            logits,
            targets,
            valid_mask=valid_mask,
        )

        total = (
            self.bce_weight * masked_bce
            + self.dice_weight * dice
        )

        return total


def build_loss(
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
    pos_weight: float | None = None,
):
    """
    Factory function used by the training configuration.
    """

    return BCEDiceLoss(
        bce_weight=bce_weight,
        dice_weight=dice_weight,
        pos_weight=pos_weight,
    )