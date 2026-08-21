# Evaluation metrics
import torch


def _validate_inputs(logits, targets):
    """
    Validate prediction and target tensors.
    """

    if logits.shape != targets.shape:
        raise ValueError(
            "Logits and targets must have the same shape. "
            f"Got logits={logits.shape}, "
            f"targets={targets.shape}"
        )

    if logits.ndim != 4:
        raise ValueError(
            "Expected tensors with shape [B, 1, H, W]. "
            f"Got {logits.shape}"
        )

    if logits.shape[1] != 1:
        raise ValueError(
            "This metric implementation expects a binary "
            f"segmentation output with 1 channel. "
            f"Got {logits.shape[1]} channels."
        )

    if targets.min() < 0 or targets.max() > 1:
        raise ValueError(
            "Targets must contain values in [0, 1]."
        )


def calculate_confusion_counts(
    logits,
    targets,
    threshold=0.5,
):
    """
    Calculate binary segmentation confusion counts.

    Args:
        logits:
            Raw model outputs [B, 1, H, W].

        targets:
            Binary ground truth [B, 1, H, W].

        threshold:
            Probability threshold used to classify a pixel
            as changed.

    Returns:
        Dictionary containing TP, TN, FP and FN.
    """

    _validate_inputs(logits, targets)

    if not 0.0 < threshold < 1.0:
        raise ValueError(
            "Threshold must be between 0 and 1."
        )

    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities >= threshold
    )

    targets_binary = (
        targets >= 0.5
    )

    true_positive = (
        predictions & targets_binary
    ).sum().item()

    true_negative = (
        (~predictions) & (~targets_binary)
    ).sum().item()

    false_positive = (
        predictions & (~targets_binary)
    ).sum().item()

    false_negative = (
        (~predictions) & targets_binary
    ).sum().item()

    return {
        "tp": int(true_positive),
        "tn": int(true_negative),
        "fp": int(false_positive),
        "fn": int(false_negative),
    }


def calculate_metrics_from_counts(counts):
    """
    Calculate segmentation metrics from confusion counts.

    Handles zero-denominator cases safely.
    """

    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]

    # ---------------------------------------------------------
    # Precision
    # ---------------------------------------------------------

    precision_denominator = tp + fp

    if precision_denominator > 0:
        precision = tp / precision_denominator
    else:
        precision = 0.0

    # ---------------------------------------------------------
    # Recall
    # ---------------------------------------------------------

    recall_denominator = tp + fn

    if recall_denominator > 0:
        recall = tp / recall_denominator
    else:
        recall = 0.0

    # ---------------------------------------------------------
    # F1
    # ---------------------------------------------------------

    f1_denominator = precision + recall

    if f1_denominator > 0:
        f1 = (
            2.0
            * precision
            * recall
            / f1_denominator
        )
    else:
        f1 = 0.0

    # ---------------------------------------------------------
    # IoU
    # ---------------------------------------------------------

    iou_denominator = tp + fp + fn

    if iou_denominator > 0:
        iou = tp / iou_denominator
    else:
        iou = 0.0

    # ---------------------------------------------------------
    # Accuracy
    # ---------------------------------------------------------

    total = tp + tn + fp + fn

    if total > 0:
        accuracy = (
            tp + tn
        ) / total
    else:
        accuracy = 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "accuracy": accuracy,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def calculate_metrics(
    logits,
    targets,
    threshold=0.5,
):
    """
    Calculate all binary change-detection metrics.

    Returns:
        Dictionary containing:

        precision
        recall
        f1
        iou
        accuracy
        tp
        tn
        fp
        fn
    """

    counts = calculate_confusion_counts(
        logits,
        targets,
        threshold=threshold,
    )

    return calculate_metrics_from_counts(
        counts
    )


def accumulate_counts(
    total_counts,
    batch_counts,
):
    """
    Add confusion counts from one batch into running totals.

    Useful for calculating validation metrics over an entire
    validation dataset rather than averaging batch metrics.
    """

    for key in (
        "tp",
        "tn",
        "fp",
        "fn",
    ):
        total_counts[key] += batch_counts[key]

    return total_counts


def empty_counts():
    """
    Create an empty confusion-count dictionary.
    """

    return {
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0,
    }