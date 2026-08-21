from pathlib import Path

from torch.utils.data import DataLoader

from data.sar_patch_dataset import TUMSARChangeDetectionDataset


def create_sar_dataloader(
    patch_index_path,
    split,
    batch_size,
    num_workers=0,
    pin_memory=True,
    shuffle=False,
):
    """
    Create a DataLoader for the TUM/OSCD SAR patch dataset.
    """

    dataset = TUMSARChangeDetectionDataset(
        patch_index_path=Path(patch_index_path),
        split=split,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return loader