import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Add src to Python path
sys.path.append(str(PROJECT_ROOT / "src"))

import torch
from torch.utils.data import DataLoader, Subset

from data.levir_patch_dataset import LEVIRCDPatchDataset
from detection.losses import BCEDiceLoss
from detection.siamese_unet import SiameseUNet
from preprocessing.transforms import LEVIRCDTrainTransform
from training.reproducibility import set_seed


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "LEVIR-CD"
)

DEVICE = torch.device("cpu")

STEPS = 20

NUM_PATCHES = 8


def calculate_metrics(logits, targets):

    predictions = (
        torch.sigmoid(logits) >= 0.5
    )

    target_binary = (
        targets >= 0.5
    )

    tp = int(
        (predictions & target_binary)
        .sum()
        .item()
    )

    fp = int(
        (predictions & (~target_binary))
        .sum()
        .item()
    )

    fn = int(
        ((~predictions) & target_binary)
        .sum()
        .item()
    )

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def run_experiment(
    dataset,
    selected_indices,
    change_sampling,
    pos_weight,
):

    set_seed(42)

    subset = Subset(
        dataset,
        selected_indices,
    )

    loader = DataLoader(
        subset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    model = SiameseUNet(
        in_channels=3,
        num_classes=1,
    ).to(DEVICE)

    model.train()

    criterion = BCEDiceLoss(
        bce_weight=0.5,
        dice_weight=0.5,
        pos_weight=pos_weight,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )

    initial_loss = None
    final_loss = None

    for step in range(STEPS):

        for batch in loader:

            image_a = batch["image_a"].to(
                DEVICE
            )

            image_b = batch["image_b"].to(
                DEVICE
            )

            targets = batch["label"].to(
                DEVICE
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                image_a,
                image_b,
            )

            loss = criterion(
                logits,
                targets,
            )

            loss.backward()

            optimizer.step()

            loss_value = loss.item()

            if initial_loss is None:
                initial_loss = loss_value

            final_loss = loss_value

    # ---------------------------------------------------------
    # Evaluate on the same fixed patches
    # ---------------------------------------------------------

    model.eval()

    all_logits = []
    all_targets = []

    with torch.no_grad():

        for batch in loader:

            image_a = batch["image_a"].to(
                DEVICE
            )

            image_b = batch["image_b"].to(
                DEVICE
            )

            targets = batch["label"].to(
                DEVICE
            )

            logits = model(
                image_a,
                image_b,
            )

            all_logits.append(logits)
            all_targets.append(targets)

    logits = torch.cat(
        all_logits,
        dim=0,
    )

    targets = torch.cat(
        all_targets,
        dim=0,
    )

    metrics = calculate_metrics(
        logits,
        targets,
    )

    return {
        "sampling": change_sampling,
        "pos_weight": pos_weight,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_reduction": (
            initial_loss - final_loss
        ),
        **metrics,
    }


def main():

    print("=" * 70)
    print("SAMPLING + POS_WEIGHT CONTROLLED EXPERIMENT")
    print("=" * 70)

    print(
        f"\nDevice: {DEVICE}"
    )

    print(
        f"Steps per experiment: {STEPS}"
    )

    print(
        f"Patches per experiment: {NUM_PATCHES}"
    )

    # =========================================================
    # Dataset
    # =========================================================

    set_seed(42)

    dataset = LEVIRCDPatchDataset(
        root_dir=DATASET_DIR,
        split="train",
        patch_size=256,
        stride=128,
        transform=LEVIRCDTrainTransform(),
        change_sampling=True,
        change_threshold=0.01,
    )

    positive_indices = [
        index
        for index, weight
        in enumerate(dataset.patch_weights)
        if float(weight.item()) == 3.0
    ]

    if len(positive_indices) < NUM_PATCHES:
        raise RuntimeError(
            "Not enough change-containing patches."
        )

    # Same patches for every experiment.
    selected_indices = positive_indices[
        :NUM_PATCHES
    ]

    print(
        f"\nTotal training patches: "
        f"{len(dataset):,}"
    )

    print(
        f"Change-containing patches: "
        f"{len(positive_indices):,}"
    )

    print(
        f"Selected patches: "
        f"{selected_indices}"
    )

    # =========================================================
    # Four experiments
    # =========================================================

    experiments = [
        {
            "sampling": False,
            "pos_weight": 1.0,
        },
        {
            "sampling": False,
            "pos_weight": 5.0,
        },
        {
            "sampling": True,
            "pos_weight": 1.0,
        },
        {
            "sampling": True,
            "pos_weight": 5.0,
        },
    ]

    results = []

    for experiment in experiments:

        sampling = experiment["sampling"]
        pos_weight = experiment["pos_weight"]

        print(
            "\n" + "-" * 70
        )

        print(
            f"Change-aware sampling: "
            f"{sampling}"
        )

        print(
            f"pos_weight: "
            f"{pos_weight}"
        )

        print(
            "-" * 70
        )

        result = run_experiment(
            dataset=dataset,
            selected_indices=selected_indices,
            change_sampling=sampling,
            pos_weight=pos_weight,
        )

        results.append(result)

        print(
            f"Initial loss: "
            f"{result['initial_loss']:.6f}"
        )

        print(
            f"Final loss: "
            f"{result['final_loss']:.6f}"
        )

        print(
            f"Loss reduction: "
            f"{result['loss_reduction']:.6f}"
        )

        print(
            f"TP: {result['tp']:,}"
        )

        print(
            f"FP: {result['fp']:,}"
        )

        print(
            f"FN: {result['fn']:,}"
        )

        print(
            f"Precision: "
            f"{result['precision']:.4f}"
        )

        print(
            f"Recall: "
            f"{result['recall']:.4f}"
        )

        print(
            f"F1: "
            f"{result['f1']:.4f}"
        )

    # =========================================================
    # Comparison
    # =========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "EXPERIMENT COMPARISON"
    )

    print(
        "=" * 70
    )

    print(
        "\nSampling | pos_weight | "
        "Loss       | Precision | Recall | F1"
    )

    print(
        "-" * 70
    )

    for result in results:

        print(
            f"{str(result['sampling']):8} | "
            f"{result['pos_weight']:10.1f} | "
            f"{result['final_loss']:.6f} | "
            f"{result['precision']:.4f}    | "
            f"{result['recall']:.4f} | "
            f"{result['f1']:.4f}"
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "CONTROLLED EXPERIMENT COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()