from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class LEVIRCDFullSceneDataset(Dataset):
    """
    Full-scene LEVIR-CD dataset for validation and testing.

    Unlike LEVIRCDPatchDataset, this class does NOT crop the
    1024x1024 scene into patches.

    Each sample returns:

        image_a -> T1 image
        image_b -> T2 image
        label   -> ground-truth change mask

    Expected structure:

        LEVIR-CD/
        ├── A/
        ├── B/
        └── label/

    Files are selected according to the requested split:

        train_*
        val_*
        test_*
    """

    VALID_SPLITS = {"train", "val", "test"}

    def __init__(self, root_dir, split="val"):
        if split not in self.VALID_SPLITS:
            raise ValueError(
                f"Invalid split '{split}'. "
                f"Expected one of {sorted(self.VALID_SPLITS)}."
            )

        self.root_dir = Path(root_dir)
        self.split = split

        self.image_a_dir = self.root_dir / "A"
        self.image_b_dir = self.root_dir / "B"
        self.label_dir = self.root_dir / "label"

        # ---------------------------------------------------------
        # Validate directories
        # ---------------------------------------------------------

        for directory in (
            self.image_a_dir,
            self.image_b_dir,
            self.label_dir,
        ):
            if not directory.exists():
                raise FileNotFoundError(
                    f"Required directory not found: {directory}"
                )

        # ---------------------------------------------------------
        # Find files belonging to the requested split
        # ---------------------------------------------------------

        prefix = f"{split}_"

        self.files = sorted(
            file.name
            for file in self.image_a_dir.iterdir()
            if file.is_file()
            and file.suffix.lower() in {
                ".png",
                ".jpg",
                ".jpeg",
            }
            and file.name.startswith(prefix)
        )

        if not self.files:
            raise RuntimeError(
                f"No {split} images found in {self.image_a_dir}"
            )

        # ---------------------------------------------------------
        # Verify A / B / label correspondence
        # ---------------------------------------------------------

        for filename in self.files:
            image_b_path = self.image_b_dir / filename
            label_path = self.label_dir / filename

            if not image_b_path.exists():
                raise RuntimeError(
                    f"Missing B image for {filename}: "
                    f"{image_b_path}"
                )

            if not label_path.exists():
                raise RuntimeError(
                    f"Missing label for {filename}: "
                    f"{label_path}"
                )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        filename = self.files[index]

        image_a_path = self.image_a_dir / filename
        image_b_path = self.image_b_dir / filename
        label_path = self.label_dir / filename

        # ---------------------------------------------------------
        # Load complete 1024x1024 scene
        # ---------------------------------------------------------

        image_a = Image.open(image_a_path).convert("RGB")
        image_b = Image.open(image_b_path).convert("RGB")
        label = Image.open(label_path).convert("L")

        # ---------------------------------------------------------
        # Size verification
        # ---------------------------------------------------------

        expected_size = (1024, 1024)

        if image_a.size != expected_size:
            raise RuntimeError(
                f"Unexpected T1 size for {filename}: "
                f"{image_a.size}, expected {expected_size}"
            )

        if image_b.size != expected_size:
            raise RuntimeError(
                f"Unexpected T2 size for {filename}: "
                f"{image_b.size}, expected {expected_size}"
            )

        if label.size != expected_size:
            raise RuntimeError(
                f"Unexpected label size for {filename}: "
                f"{label.size}, expected {expected_size}"
            )

        # ---------------------------------------------------------
        # Convert images to [C, H, W] tensors in [0, 1]
        # ---------------------------------------------------------

        image_a = self._image_to_tensor(image_a)
        image_b = self._image_to_tensor(image_b)

        # ---------------------------------------------------------
        # Convert label to binary [1, H, W] tensor
        # ---------------------------------------------------------

        label = np.asarray(label, dtype=np.uint8)

        label = (label > 0).astype(np.float32)

        label = torch.from_numpy(label).unsqueeze(0)

        # ---------------------------------------------------------
        # Sanity checks
        # ---------------------------------------------------------

        if image_a.shape[-2:] != image_b.shape[-2:]:
            raise RuntimeError(
                f"T1/T2 spatial dimensions do not match for "
                f"{filename}: "
                f"{image_a.shape} vs {image_b.shape}"
            )

        if image_a.shape[-2:] != label.shape[-2:]:
            raise RuntimeError(
                f"Image/label dimensions do not match for "
                f"{filename}: "
                f"{image_a.shape} vs {label.shape}"
            )

        return {
            "image_a": image_a,
            "image_b": image_b,
            "label": label,
            "filename": filename,
        }

    @staticmethod
    def _image_to_tensor(image):
        """
        Convert PIL RGB image to float tensor.

        Input:
            H x W x 3

        Output:
            3 x H x W

        Values:
            [0, 1]
        """

        image = np.asarray(
            image,
            dtype=np.float32,
        ) / 255.0

        image = np.transpose(
            image,
            (2, 0, 1),
        )

        return torch.from_numpy(image)
