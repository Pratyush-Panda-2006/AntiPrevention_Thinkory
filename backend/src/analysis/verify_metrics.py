# Verify metrics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Add src to Python path
sys.path.append(str(PROJECT_ROOT / "src"))

import torch

from evaluation.metrics import (
    calculate_metrics,
    calculate_metrics_from_counts,
)


def test_perfect_prediction():

    # Ground truth:
    #
    # 1 0
    # 0 1
    #
    targets = torch.tensor(
        [
            [
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ]
            ]
        ]
    )

    # Very confident correct predictions.
    logits = torch.tensor(
        [
            [
                [
                    [10.0, -10.0],
                    [-10.0, 10.0],
                ]
            ]
        ]
    )

    metrics = calculate_metrics(
        logits,
        targets,
    )

    assert metrics["tp"] == 2
    assert metrics["tn"] == 2
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["iou"] == 1.0

    print("✓ Perfect prediction test passed")


def test_all_negative_prediction():

    targets = torch.tensor(
        [
            [
                [
                    [1.0, 0.0],
                    [0.0, 0.0],
                ]
            ]
        ]
    )

    # All predictions are negative.
    logits = torch.tensor(
        [
            [
                [
                    [-10.0, -10.0],
                    [-10.0, -10.0],
                ]
            ]
        ]
    )

    metrics = calculate_metrics(
        logits,
        targets,
    )

    assert metrics["tp"] == 0
    assert metrics["tn"] == 3
    assert metrics["fp"] == 0
    assert metrics["fn"] == 1

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["iou"] == 0.0

    print("✓ All-negative prediction test passed")


def test_all_positive_prediction():

    targets = torch.tensor(
        [
            [
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ]
            ]
        ]
    )

    # All predictions are positive.
    logits = torch.tensor(
        [
            [
                [
                    [10.0, 10.0],
                    [10.0, 10.0],
                ]
            ]
        ]
    )

    metrics = calculate_metrics(
        logits,
        targets,
    )

    assert metrics["tp"] == 2
    assert metrics["tn"] == 0
    assert metrics["fp"] == 2
    assert metrics["fn"] == 0

    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 1.0

    print("✓ All-positive prediction test passed")


def test_count_accumulation():

    counts_a = {
        "tp": 10,
        "tn": 20,
        "fp": 5,
        "fn": 3,
    }

    counts_b = {
        "tp": 7,
        "tn": 10,
        "fp": 2,
        "fn": 4,
    }

    total = {
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0,
    }

    for key in total:
        total[key] += counts_a[key]
        total[key] += counts_b[key]

    assert total == {
        "tp": 17,
        "tn": 30,
        "fp": 7,
        "fn": 7,
    }

    metrics = calculate_metrics_from_counts(
        total
    )

    assert metrics["tp"] == 17
    assert metrics["tn"] == 30
    assert metrics["fp"] == 7
    assert metrics["fn"] == 7

    print("✓ Count accumulation test passed")


def main():

    print("=" * 60)
    print("METRICS VERIFICATION")
    print("=" * 60)

    test_perfect_prediction()

    test_all_negative_prediction()

    test_all_positive_prediction()

    test_count_accumulation()

    print("\n" + "=" * 60)
    print("ALL METRIC TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()