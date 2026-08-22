from pathlib import Path
import argparse
import json
import sys
import time

import torch

# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
SRC_DIR = BACKEND_DIR / "src"

sys.path.insert(0, str(SRC_DIR))

# ============================================================
# Project imports
# ============================================================

from detection.snunet_cd import SNUNetCD
from detection.losses import BCEDiceLoss

from training.sar_config import SARTrainingConfig
from training.engine import TrainingEngine
from training.reproducibility import set_seed


from data.sar_patch_dataset import TUMSARChangeDetectionDataset

from torch.utils.data import DataLoader


# ============================================================
# Checkpoint utilities
# ============================================================

def load_checkpoint_state(checkpoint_path):
    """
    Load a checkpoint and extract its model state dictionary.

    Supports:
        model_state_dict
        state_dict
        raw state dictionaries
    """

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]

        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]

        # Raw state_dict
        if all(
            isinstance(k, str)
            for k in checkpoint.keys()
        ):
            return checkpoint

    raise ValueError(
        f"Could not find model state dictionary in:\n"
        f"{checkpoint_path}"
    )


def load_transfer_weights(
    model,
    checkpoint_path,
):
    """
    Load the previously created RGB -> SAR transfer checkpoint.

    The checkpoint already contains the special 2-channel
    initialization for conv0_0.conv1.weight.

    Therefore no RGB->SAR conversion is performed here.
    """

    state_dict = load_checkpoint_state(
        checkpoint_path
    )

    model_state = model.state_dict()

    compatible = {}
    skipped = {}

    for key, value in state_dict.items():

        if key not in model_state:
            skipped[key] = "unexpected"
            continue

        if model_state[key].shape != value.shape:
            skipped[key] = (
                f"shape mismatch: "
                f"checkpoint={tuple(value.shape)} "
                f"model={tuple(model_state[key].shape)}"
            )
            continue

        compatible[key] = value

    model_state.update(compatible)
    model.load_state_dict(model_state)

    print(
        f"Transfer tensors loaded : {len(compatible)}"
    )

    if skipped:
        print(
            f"Skipped tensors         : {len(skipped)}"
        )

        for key, reason in skipped.items():
            print(
                f"  {key}: {reason}"
            )

    return model


# ============================================================
# Data loaders
# ============================================================

def create_sar_dataloader(
    patch_index_path,
    split,
    batch_size,
    num_workers,
    pin_memory,
):
    dataset = TUMSARChangeDetectionDataset(
        patch_index_path=patch_index_path,
        split=split,
        root_dir=PROJECT_ROOT,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return loader


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train SNUNet-CD on TUM/OSCD Sentinel-1 SAR "
            "using the project's TrainingEngine."
        )
    )

    parser.add_argument(
        "--init",
        choices=["scratch", "transfer"],
        default="scratch",
        help=(
            "Model initialization strategy. "
            "'scratch' uses random initialization; "
            "'transfer' loads the RGB->SAR transfer checkpoint."
        ),
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a very short training sanity test.",
    )

    parser.add_argument(
        "--train-batches",
        type=int,
        default=2,
        help="Maximum training batches for smoke test.",
    )

    parser.add_argument(
        "--val-batches",
        type=int,
        default=1,
        help="Maximum validation batches for smoke test.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override DataLoader workers.",
    )

    args = parser.parse_args()

    if args.train_batches < 1:
        raise ValueError(
            "--train-batches must be >= 1."
        )

    if args.val_batches < 1:
        raise ValueError(
            "--val-batches must be >= 1."
        )

    # ========================================================
    # Configuration
    # ========================================================

    config = SARTrainingConfig()

    if args.epochs is not None:
        config.epochs = args.epochs

    if args.batch_size is not None:
        config.batch_size = args.batch_size

    if args.num_workers is not None:
        config.num_workers = args.num_workers

    if args.smoke_test:
        config.epochs = 1

    config.experiment_name = (
        f"tum_oscd_sar_{args.init}"
        + ("_smoke" if args.smoke_test else "")
    )

    # ========================================================
    # Device
    # ========================================================

    device = config.get_device()

    # ========================================================
    # Run directory
    # ========================================================

    timestamp = time.strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    run_id = (
        f"{timestamp}_"
        f"{config.experiment_name}"
    )

    run_dir = (
        BACKEND_DIR
        / "runs"
        / run_id
    )

    checkpoint_dir = (
        run_dir / "checkpoints"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # The existing TrainingEngine expects these
    # attributes on its config object.
    config.run_dir = str(run_dir)
    config.checkpoint_dir = str(
        checkpoint_dir
    )

    # ========================================================
    # Reproducibility
    # ========================================================

    set_seed(config.seed)

    # ========================================================
    # Header
    # ========================================================

    print("=" * 70)
    print("SAR SNUNET-CD — TRAINING ENGINE")
    print("=" * 70)
    print(f"Initialization : {args.init}")
    print(f"Device         : {device}")
    print(f"Input channels : {config.in_channels}")
    print(f"Batch size     : {config.batch_size}")
    print(f"Epochs         : {config.epochs}")
    print(f"Run directory  : {run_dir}")

    # ========================================================
    # Dataset
    # ========================================================

    patch_index_path = (
        PROJECT_ROOT
        / config.patch_index_path
    )

    if not patch_index_path.exists():
        raise FileNotFoundError(
            "SAR patch index not found:\n"
            f"{patch_index_path}"
        )

    train_loader = create_sar_dataloader(
        patch_index_path=patch_index_path,
        split=config.train_split,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    val_loader = create_sar_dataloader(
        patch_index_path=patch_index_path,
        split=config.validation_split,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    print(
        f"Train patches : "
        f"{len(train_loader.dataset)}"
    )

    print(
        f"Val patches   : "
        f"{len(val_loader.dataset)}"
    )

    # ========================================================
    # Model
    # ========================================================

    model = SNUNetCD(
        in_channels=config.in_channels,
        num_classes=config.num_classes,
    )

    print(
        "Model created: "
        "SNUNet-CD 2-channel SAR"
    )

    # ========================================================
    # Initialization
    # ========================================================

    if args.init == "transfer":

        transfer_checkpoint = (
            BACKEND_DIR
            / "runs"
            / "sar_transfer_init"
            / "snunet_cd_sar_transfer_init.pt"
        )

        if not transfer_checkpoint.exists():
            raise FileNotFoundError(
                "SAR transfer checkpoint not found:\n"
                f"{transfer_checkpoint}"
            )

        print(
            "Loading SAR transfer checkpoint:"
        )
        print(
            transfer_checkpoint
        )

        model = load_transfer_weights(
            model,
            transfer_checkpoint,
        )

        print(
            "Transfer initialization loaded."
        )

    else:

        print(
            "Using random initialization."
        )

    # ========================================================
    # Loss
    # ========================================================

    criterion = BCEDiceLoss(
        bce_weight=config.bce_weight,
        dice_weight=config.dice_weight,
        pos_weight=config.pos_weight,
    )

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # ========================================================
    # Scheduler
    # ========================================================

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.scheduler_min_lr,
    )



    # ========================================================
    # Engine
    # ========================================================

    engine = TrainingEngine(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )



    # ========================================================
    # Dataset metadata
    # ========================================================

    dataset_info = {
        "dataset_name": "TUM OSCD",
        "dataset_root": str(
            PROJECT_ROOT
            / "data"
            / "sar"
            / "tum_oscd"
        ),
        "patch_index": str(
            patch_index_path
        ),
        "train_patch_count": len(
            train_loader.dataset
        ),
        "validation_patch_count": len(
            val_loader.dataset
        ),
        "patch_size": config.patch_size,
        "stride": config.stride,
        "input_channels": config.in_channels,
        "initialization": args.init,
    }

    with open(
        run_dir / "dataset_info.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            dataset_info,
            f,
            indent=2,
        )

    with open(
        run_dir / "config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            vars(config),
            f,
            indent=2,
            default=str,
        )

    # ========================================================
    # Training
    # ========================================================

    print()
    print("=" * 70)

    if args.smoke_test:

        print(
            "RUNNING SAR ENGINE SMOKE TEST"
        )

        engine.fit(
            max_train_batches=args.train_batches,
            max_val_batches=args.val_batches,
        )

    else:

        print(
            "STARTING FULL SAR TRAINING"
        )

        engine.fit()

    # ========================================================
    # Completion
    # ========================================================

    print()
    print("=" * 70)
    print("SAR TRAINING COMPLETE")
    print("=" * 70)
    print(
        f"Run directory: {run_dir}"
    )

    if hasattr(engine, "best_f1"):
        print(
            f"Best F1: {engine.best_f1:.4f}"
        )


if __name__ == "__main__":
    main()