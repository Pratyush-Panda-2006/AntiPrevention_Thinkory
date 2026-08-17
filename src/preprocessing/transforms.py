import random

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF


class LEVIRCDTrainTransform:
    """
    Training augmentation applied to an already extracted
    256x256 patch.

    T1, T2 and label always receive identical spatial
    transformations.
    """

    def __init__(self):
        pass

    def __call__(self, image_a, image_b, label):

        if random.random() < 0.5:
            image_a = TF.hflip(image_a)
            image_b = TF.hflip(image_b)
            label = TF.hflip(label)

        if random.random() < 0.5:
            image_a = TF.vflip(image_a)
            image_b = TF.vflip(image_b)
            label = TF.vflip(label)

        image_a = TF.to_tensor(image_a)
        image_b = TF.to_tensor(image_b)

        label = np.array(label, dtype=np.uint8)
        label = (label > 0).astype(np.float32)
        label = torch.from_numpy(label).unsqueeze(0)

        return image_a, image_b, label


class LEVIRCDEvalTransform:
    """
    Deterministic preprocessing for validation/test.

    Can optionally center crop, but defaults to full image 
    since Fully Convolutional Networks (like Siamese U-Net) 
    can evaluate on full 1024x1024 images if GPU memory permits.
    """

    def __init__(self, crop_size=None):
        self.crop_size = crop_size

    def __call__(self, image_a, image_b, label):
        if self.crop_size is not None:
            width, height = image_a.size

            if width < self.crop_size or height < self.crop_size:
                raise ValueError(
                    f"Image size ({width}, {height}) is smaller than "
                    f"crop size ({self.crop_size}, {self.crop_size})."
                )

            # ---------------------------------------------------------
            # Center crop
            # ---------------------------------------------------------
            top = (height - self.crop_size) // 2
            left = (width - self.crop_size) // 2

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

        # ---------------------------------------------------------
        # Convert images
        # ---------------------------------------------------------
        image_a = TF.to_tensor(image_a)
        image_b = TF.to_tensor(image_b)

        # ---------------------------------------------------------
        # Convert mask
        # ---------------------------------------------------------
        label = np.array(label, dtype=np.uint8)
        label = (label > 0).astype(np.float32)

        label = torch.from_numpy(label).unsqueeze(0)

        return image_a, image_b, label