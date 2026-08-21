from torch.utils.data import DataLoader

from data.levir_patch_dataset import LEVIRCDPatchDataset
from data.levir_fullscene_dataset import LEVIRCDFullSceneDataset
from data.sampler import create_weighted_sampler
from preprocessing.transforms import (
    LEVIRCDTrainTransform,
)


def create_train_dataloader(
    dataset_dir,
    batch_size=16,
    num_workers=8,
    patch_size=256,
    stride=128,
):
    """
    Create the training DataLoader.

    Training uses:
        - overlapping 256x256 patches
        - change-aware weighted sampling
        - synchronized augmentation
        - multi-worker loading
        - pinned memory for CUDA
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

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": True,
        "drop_last": True,
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    return DataLoader(
        **loader_kwargs
    )


def create_eval_dataloader(
    dataset_dir,
    split,
    batch_size=1,
    num_workers=4,
):
    """
    Create validation/test DataLoader.

    Evaluation uses complete scenes.
    """

    if split not in {"val", "test"}:
        raise ValueError(
            "Evaluation split must be 'val' or 'test'."
        )

    dataset = LEVIRCDFullSceneDataset(
        root_dir=dataset_dir,
        split=split,
    )

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": True,
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    return DataLoader(
        **loader_kwargs
    )