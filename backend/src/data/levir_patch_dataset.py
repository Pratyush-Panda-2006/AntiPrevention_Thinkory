from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class LEVIRCDPatchDataset(Dataset):
    """
    Coverage-aware patch dataset for LEVIR-CD.

    Each 1024x1024 scene is divided into overlapping patches.

    Default:
        patch_size = 256
        stride = 128

    This gives complete scene coverage while providing overlap
    around patch boundaries.

    For training, patches containing meaningful change can be
    sampled more frequently using change-aware sampling.
    """

    def __init__(
        self,
        root_dir,
        split="train",
        patch_size=256,
        stride=128,
        transform=None,
        change_sampling=True,
        change_threshold=0.01,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.change_sampling = change_sampling
        self.change_threshold = change_threshold

        self.image_a_dir = self.root_dir / "A"
        self.image_b_dir = self.root_dir / "B"
        self.label_dir = self.root_dir / "label"

        if not self.image_a_dir.exists():
            raise FileNotFoundError(self.image_a_dir)

        if not self.image_b_dir.exists():
            raise FileNotFoundError(self.image_b_dir)

        if not self.label_dir.exists():
            raise FileNotFoundError(self.label_dir)

        # ---------------------------------------------------------
        # Find files belonging to requested split
        # ---------------------------------------------------------
        prefix = f"{split}_"

        self.files = sorted(
            file.name
            for file in self.image_a_dir.iterdir()
            if file.is_file()
            and file.suffix.lower() in [".png", ".jpg", ".jpeg"]
            and file.name.startswith(prefix)
        )

        if not self.files:
            raise RuntimeError(
                f"No {split} images found in {self.image_a_dir}"
            )

        # ---------------------------------------------------------
        # Verify matching B and label files
        # ---------------------------------------------------------
        for filename in self.files:
            if not (self.image_b_dir / filename).exists():
                raise RuntimeError(
                    f"Missing B image: {filename}"
                )

            if not (self.label_dir / filename).exists():
                raise RuntimeError(
                    f"Missing label: {filename}"
                )

        # ---------------------------------------------------------
        # Build complete patch index
        # ---------------------------------------------------------
        self.patch_index = self._build_patch_index()

        # Optional change-aware weights
        self.patch_weights = None

        if self.change_sampling:
            self.patch_weights = self._calculate_patch_weights()

    # =============================================================
    # Patch generation
    # =============================================================

    def _get_positions(self, image_size):
        """
        Generate patch start positions while guaranteeing that
        the final patch reaches the image boundary.
        """

        positions = list(
            range(
                0,
                image_size - self.patch_size + 1,
                self.stride,
            )
        )

        last_position = image_size - self.patch_size

        if positions[-1] != last_position:
            positions.append(last_position)

        return positions

    def _build_patch_index(self):
        """
        Create:

            (filename, top, left)

        for every patch covering every image.
        """

        # LEVIR-CD images are 1024x1024.
        # We read the first image to avoid hardcoding dimensions.
        first_image = Image.open(
            self.image_a_dir / self.files[0]
        )

        width, height = first_image.size

        if width < self.patch_size or height < self.patch_size:
            raise ValueError(
                f"Patch size {self.patch_size} is larger than "
                f"image size {width}x{height}"
            )

        rows = self._get_positions(height)
        cols = self._get_positions(width)

        patches = []

        for filename in self.files:
            for top in rows:
                for left in cols:
                    patches.append(
                        (filename, top, left)
                    )

        return patches

    # =============================================================
    # Change-aware sampling
    # =============================================================

    def _calculate_patch_weights(self):
        """
        Give patches containing change higher sampling weight.

        This does NOT delete unchanged patches.

        It simply makes change-containing patches more likely to
        appear during training.
        """

        weights = []
        current_filename = None
        current_label = None

        for filename, top, left in self.patch_index:
            if filename != current_filename:
                label_path = self.label_dir / filename
                current_label = np.asarray(Image.open(label_path).convert("L"))
                current_filename = filename

            patch = current_label[
                top:top + self.patch_size,
                left:left + self.patch_size,
            ]

            change_ratio = np.mean(patch > 0)

            if change_ratio >= self.change_threshold:
                # Higher probability for change-containing patches
                weight = 3.0
            else:
                weight = 1.0

            weights.append(weight)

        return torch.tensor(
            weights,
            dtype=torch.double,
        )

    # =============================================================
    # Dataset interface
    # =============================================================

    def __len__(self):
        return len(self.patch_index)

    def __getitem__(self, index):

        filename, top, left = self.patch_index[index]

        image_a = Image.open(
            self.image_a_dir / filename
        ).convert("RGB")

        image_b = Image.open(
            self.image_b_dir / filename
        ).convert("RGB")

        label = Image.open(
            self.label_dir / filename
        ).convert("L")

        # ---------------------------------------------------------
        # Same spatial crop for T1, T2 and label
        # ---------------------------------------------------------

        image_a = image_a.crop(
            (
                left,
                top,
                left + self.patch_size,
                top + self.patch_size,
            )
        )

        image_b = image_b.crop(
            (
                left,
                top,
                left + self.patch_size,
                top + self.patch_size,
            )
        )

        label = label.crop(
            (
                left,
                top,
                left + self.patch_size,
                top + self.patch_size,
            )
        )

        # ---------------------------------------------------------
        # Apply synchronized training/evaluation transforms
        # ---------------------------------------------------------

        if self.transform is not None:
            image_a, image_b, label = self.transform(
                image_a,
                image_b,
                label,
            )

        else:
            image_a = self._image_to_tensor(image_a)
            image_b = self._image_to_tensor(image_b)

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

        return {
            "image_a": image_a,
            "image_b": image_b,
            "label": label,
            "filename": filename,
            "top": top,
            "left": left,
        }

    # =============================================================
    # Utilities
    # =============================================================

    @staticmethod
    def _image_to_tensor(image):
        image = np.asarray(
            image,
            dtype=np.float32,
        ) / 255.0

        image = np.transpose(
            image,
            (2, 0, 1),
        )

        return torch.from_numpy(image)