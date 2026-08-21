import torch
from torch.utils.data import WeightedRandomSampler


def create_weighted_sampler(
    dataset,
    num_samples=None,
):
    """
    Create a WeightedRandomSampler from the patch weights
    calculated by LEVIRCDPatchDataset.

    Changed-containing patches currently have a higher weight
    than unchanged patches.

    Args:
        dataset:
            LEVIRCDPatchDataset with change_sampling=True.

        num_samples:
            Number of samples drawn per epoch.
            Defaults to len(dataset).

    Returns:
        WeightedRandomSampler
    """

    if not hasattr(dataset, "patch_weights"):
        raise AttributeError(
            "Dataset does not contain patch_weights."
        )

    if dataset.patch_weights is None:
        raise ValueError(
            "dataset.patch_weights is None. "
            "Create the dataset with change_sampling=True."
        )

    if len(dataset.patch_weights) != len(dataset):
        raise ValueError(
            "Number of patch weights does not match "
            "dataset length."
        )

    if num_samples is None:
        num_samples = len(dataset)

    return WeightedRandomSampler(
        weights=dataset.patch_weights,
        num_samples=num_samples,
        replacement=True,
    )


def summarize_sampler_weights(dataset):
    """
    Return basic statistics about the patch sampling weights.
    """

    weights = dataset.patch_weights

    if weights is None:
        raise ValueError(
            "Dataset does not contain patch weights."
        )

    unique, counts = torch.unique(
        weights,
        return_counts=True,
    )

    summary = {}

    for weight, count in zip(unique, counts):
        summary[float(weight.item())] = int(count.item())

    return summary