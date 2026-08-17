import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Add src to Python path
sys.path.append(str(PROJECT_ROOT / "src"))

import torch

from training.config import TrainingConfig
from training.optimizers import (
    create_training_components,
)


def main():

    print("=" * 60)
    print("OPTIMIZER + SCHEDULER VERIFICATION")
    print("=" * 60)

    config = TrainingConfig()

    model, criterion, optimizer, scheduler = (
        create_training_components(config)
    )

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    device = config.get_device()

    model = model.to(device)

    print(
        f"\nDevice: {device}"
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"Model parameters: "
        f"{parameter_count:,}"
    )

    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------

    print(
        f"\nOptimizer: "
        f"{optimizer.__class__.__name__}"
    )

    print(
        f"Learning rate: "
        f"{config.learning_rate}"
    )

    print(
        f"Weight decay: "
        f"{config.weight_decay}"
    )

    # ---------------------------------------------------------
    # Scheduler
    # ---------------------------------------------------------

    print(
        f"\nScheduler: "
        f"{scheduler.__class__.__name__}"
    )

    print(
        f"Scheduler factor: "
        f"{config.scheduler_factor}"
    )

    print(
        f"Scheduler patience: "
        f"{config.scheduler_patience}"
    )

    # ---------------------------------------------------------
    # Loss
    # ---------------------------------------------------------

    print(
        f"\nLoss: "
        f"{criterion.__class__.__name__}"
    )

    print(
        f"BCE weight: "
        f"{config.bce_weight}"
    )

    print(
        f"Dice weight: "
        f"{config.dice_weight}"
    )

    print(
        f"Positive weight: "
        f"{config.pos_weight}"
    )

    # ---------------------------------------------------------
    # Test optimizer update
    # ---------------------------------------------------------

    image_a = torch.randn(
        1,
        3,
        256,
        256,
        device=device,
    )

    image_b = torch.randn(
        1,
        3,
        256,
        256,
        device=device,
    )

    target = torch.randint(
        0,
        2,
        (
            1,
            1,
            256,
            256,
        ),
        device=device,
    ).float()

    model.train()

    optimizer.zero_grad()

    prediction = model(
        image_a,
        image_b,
    )

    loss = criterion(
        prediction,
        target,
    )

    loss.backward()

    optimizer.step()

    # ---------------------------------------------------------
    # Scheduler test
    # ---------------------------------------------------------

    scheduler.step(
        0.5
    )

    print(
        f"\nTest loss: "
        f"{loss.item():.6f}"
    )

    print(
        f"Current learning rate: "
        f"{optimizer.param_groups[0]['lr']}"
    )

    print(
        "\n✓ Forward pass successful"
    )

    print(
        "✓ Loss calculation successful"
    )

    print(
        "✓ Backward pass successful"
    )

    print(
        "✓ Optimizer update successful"
    )

    print(
        "✓ Scheduler step successful"
    )

    print("\n" + "=" * 60)
    print("OPTIMIZER PIPELINE VERIFIED")
    print("=" * 60)


if __name__ == "__main__":
    main()