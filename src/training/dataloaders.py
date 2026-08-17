from pathlib import Path

from torch.utils.data import DataLoader

from data.levir_patch_dataset import LEVIRCDPatchDataset
from data.levir_fullscene_dataset import LEVIRCDFullSceneDataset
from data.sampler import create_weighted_sampler
from preprocessing.transforms import (
    LEVIRCDTrainTransform,
)


def create_train_dataloader(
    dataset_dir,
    batch_size=4,
    num_workers=0,
    patch_size=256,
    stride=128,
):
    """
    Create the training DataLoader.

    Training uses:
        - overlapping 256x256 patches
        - change-aware weighted sampling
        - synchronized augmentation
    """

    dataset = LEVIRCDPatchDataset(
        root_dir=dataset_dir,
        split="train",
        patch_size=patch_size,
        stride=stride,
        transform=LEVIRCDTrainTransform(),
        change_sampling=True,
        change_threshold=0.01,
    )

    sampler = create_weighted_sampler(
        dataset
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )

    return loader


def create_eval_dataloader(
    dataset_dir,
    split,
    batch_size=1,
    num_workers=0,
):
    """
    Create a validation/test DataLoader.

    Evaluation uses complete scenes rather than the
    training patch sampler.

    Args:
        dataset_dir:
            Path to LEVIR-CD.

        split:
            "val" or "test".
    """

    if split not in {"val", "test"}:
        raise ValueError(
            "Evaluation split must be 'val' or 'test'."
        )

    dataset = LEVIRCDFullSceneDataset(
        root_dir=dataset_dir,
        split=split,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return loader