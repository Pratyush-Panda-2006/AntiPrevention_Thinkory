import random

import numpy as np
import torch
from torchvision.transforms import functional as TF


class LEVIRCDTrainTransform:
    """
    Training augmentation for an already extracted 256x256 patch.

    Spatial transformations are synchronized across:
        - T1
        - T2
        - label

    Augmentations:
        - Random horizontal flip
        - Random vertical flip
        - Random 90-degree rotation
        - Random temporal swap

    Images are converted to [0, 1].
    Labels are converted to binary float32 tensors.
    """

    def __init__(self):
        pass

    def __call__(self, image_a, image_b, label):

        # Random horizontal flip
        if random.random() < 0.5:
            image_a = TF.hflip(image_a)
            image_b = TF.hflip(image_b)
            label = TF.hflip(label)

        # Random vertical flip
        if random.random() < 0.5:
            image_a = TF.vflip(image_a)
            image_b = TF.vflip(image_b)
            label = TF.vflip(label)

        # Random 90-degree rotation
        k = random.randint(0, 3)

        if k != 0:
            angle = 90 * k

            image_a = TF.rotate(
                image_a,
                angle,
            )

            image_b = TF.rotate(
                image_b,
                angle,
            )

            label = TF.rotate(
                label,
                angle,
            )

        # Temporal swap.
        # The binary change mask is symmetric:
        # change(A, B) == change(B, A)
        if random.random() < 0.5:
            image_a, image_b = image_b, image_a

        # Convert images to [C, H, W], [0, 1]
        image_a = TF.to_tensor(image_a)
        image_b = TF.to_tensor(image_b)

        # Convert label to binary [1, H, W]
        label = np.asarray(
            label,
            dtype=np.uint8,
        )

        label = (
            label > 0
        ).astype(np.float32)

        label = torch.from_numpy(
            label
        ).unsqueeze(0)

        return (
            image_a,
            image_b,
            label,
        )


class LEVIRCDEvalTransform:
    """
    Deterministic preprocessing for validation/test.

    Default:
        Full image is preserved.

    No random augmentation is applied.
    """

    def __init__(
        self,
        crop_size=None,
    ):
        self.crop_size = crop_size

    def __call__(
        self,
        image_a,
        image_b,
        label,
    ):

        if self.crop_size is not None:

            width, height = image_a.size

            if (
                width < self.crop_size
                or height < self.crop_size
            ):
                raise ValueError(
                    f"Image size "
                    f"({width}, {height}) is smaller than "
                    f"crop size "
                    f"({self.crop_size}, {self.crop_size})."
                )

            top = (
                height - self.crop_size
            ) // 2

            left = (
                width - self.crop_size
            ) // 2

            image_a = TF.crop(
                image_a,
                top,
                left,
                self.crop_size,
                self.crop_size,
            )

            image_b = TF.crop(
                image_b,
                top,
                left,
                self.crop_size,
                self.crop_size,
            )

            label = TF.crop(
                label,
                top,
                left,
                self.crop_size,
                self.crop_size,
            )

        image_a = TF.to_tensor(image_a)
        image_b = TF.to_tensor(image_b)

        label = np.asarray(
            label,
            dtype=np.uint8,
        )

        label = (
            label > 0
        ).astype(np.float32)

        label = torch.from_numpy(
            label
        ).unsqueeze(0)

        return (
            image_a,
            image_b,
            label,
        )