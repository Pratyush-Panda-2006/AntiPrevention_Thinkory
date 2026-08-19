import sys
from pathlib import Path

import torch


# -------------------------------------------------------------
# Project paths
# -------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))


from detection.losses import BCEDiceLoss
from detection.snunet_cd import SNUNetCD
from training.config import TrainingConfig
from training.engine import TrainingEngine


def build_training_stack(run_dir):
    config = TrainingConfig()

    config.model_name = "snunet_cd"
    config.device = "cpu"

    config.batch_size = 2
    config.num_workers = 0
    config.epochs = 1

    config.use_amp = False
    config.gradient_clip_norm = 1.0

    config.learning_rate = 1e-4
    config.weight_decay = 1e-4

    config.bce_weight = 0.5
    config.dice_weight = 0.5
    config.pos_weight = 1.0

    config.scheduler_factor = 0.5
    config.scheduler_patience = 3
    config.scheduler_min_lr = 1e-6

    config.early_stopping_patience = 12
    config.threshold = 0.5

    config.run_dir = str(run_dir)
    config.checkpoint_dir = str(
        Path(run_dir) / "checkpoints"
    )

    model = SNUNetCD(
        in_channels=3,
        num_classes=1,
    )

    criterion = BCEDiceLoss(
        bce_weight=config.bce_weight,
        dice_weight=config.dice_weight,
        pos_weight=config.pos_weight,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.scheduler_min_lr,
    )

    return (
        config,
        model,
        criterion,
        optimizer,
        scheduler,
    )


def main():
    print("=" * 70)
    print("MODEL 3 — CHECKPOINT SAVE / LOAD / RESUME TEST")
    print("=" * 70)

    run_dir = (
        PROJECT_ROOT
        / "runs"
        / "snunet_cd_checkpoint_test"
    )

    checkpoint_dir = (
        run_dir
        / "checkpoints"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================================================
    # Build first training stack
    # =========================================================

    (
        config_a,
        model_a,
        criterion_a,
        optimizer_a,
        scheduler_a,
    ) = build_training_stack(run_dir)

    engine_a = TrainingEngine(
        model=model_a,
        criterion=criterion_a,
        optimizer=optimizer_a,
        scheduler=scheduler_a,
        train_loader=[],
        val_loader=[],
        config=config_a,
    )

    # =========================================================
    # Perform a real optimizer update
    # =========================================================

    torch.manual_seed(42)

    image_a = torch.randn(
        2,
        3,
        256,
        256,
    )

    image_b = torch.randn(
        2,
        3,
        256,
        256,
    )

    target = torch.zeros(
        2,
        1,
        256,
        256,
    )

    target[:, :, 64:128, 64:128] = 1.0

    model_a.train()

    optimizer_a.zero_grad(
        set_to_none=True
    )

    logits = model_a(
        image_a,
        image_b,
    )

    loss = criterion_a(
        logits,
        target,
    )

    assert torch.isfinite(loss), (
        "Initial loss is not finite."
    )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model_a.parameters(),
        1.0,
    )

    optimizer_a.step()

    # Advance scheduler once so scheduler state is non-default.
    scheduler_a.step(0.25)

    # =========================================================
    # Populate engine state
    # =========================================================

    engine_a.history = [
        {
            "epoch": 1,
            "train_loss": float(loss.item()),
            "val_loss": 1.0,
            "precision": 0.1,
            "recall": 0.2,
            "f1": 0.15,
            "iou": 0.08,
            "accuracy": 0.9,
            "learning_rate": 1e-4,
        }
    ]

    engine_a.start_epoch = 2
    engine_a.global_step = 7
    engine_a.best_f1 = 0.15
    engine_a.epochs_without_improvement = 2

    # =========================================================
    # Save checkpoint using the ACTUAL engine API
    # =========================================================

    checkpoint_path = engine_a.save_checkpoint(
        epoch=1,
        filename="checkpoint_test.pt",
    )

    checkpoint_path = Path(
        checkpoint_path
    )

    assert checkpoint_path.exists(), (
        f"Checkpoint was not created: "
        f"{checkpoint_path}"
    )

    print()
    print(
        f"✓ Checkpoint created: {checkpoint_path}"
    )

    # =========================================================
    # Capture saved states
    # =========================================================

    saved_model_state = {
        key: value.detach().clone()
        for key, value in model_a.state_dict().items()
    }

    saved_optimizer_state = optimizer_a.state_dict()
    saved_scheduler_state = scheduler_a.state_dict()
    saved_history = list(engine_a.history)
    saved_best_f1 = engine_a.best_f1
    saved_global_step = engine_a.global_step
    saved_epochs_without_improvement = (
        engine_a.epochs_without_improvement
    )

    # =========================================================
    # Build a completely fresh stack
    # =========================================================

    (
        config_b,
        model_b,
        criterion_b,
        optimizer_b,
        scheduler_b,
    ) = build_training_stack(run_dir)

    engine_b = TrainingEngine(
        model=model_b,
        criterion=criterion_b,
        optimizer=optimizer_b,
        scheduler=scheduler_b,
        train_loader=[],
        val_loader=[],
        config=config_b,
    )

    # =========================================================
    # Load checkpoint
    # =========================================================

    engine_b.load_checkpoint(
        checkpoint_path
    )

    print(
        "✓ Checkpoint loaded successfully"
    )

    # =========================================================
    # Verify model parameters
    # =========================================================

    for key, expected in saved_model_state.items():

        actual = model_b.state_dict()[key]

        if not torch.equal(
            actual,
            expected,
        ):
            raise AssertionError(
                f"Model parameter mismatch "
                f"after resume: {key}"
            )

    print(
        "✓ Model parameters restored exactly"
    )

    # =========================================================
    # Verify optimizer state
    # =========================================================

    restored_optimizer_state = (
        optimizer_b.state_dict()
    )

    assert (
        restored_optimizer_state["param_groups"]
        == saved_optimizer_state["param_groups"]
    ), "Optimizer param-group state mismatch."

    print(
        "✓ Optimizer state restored"
    )

    # =========================================================
    # Verify scheduler state
    # =========================================================

    restored_scheduler_state = (
        scheduler_b.state_dict()
    )

    assert (
        restored_scheduler_state
        == saved_scheduler_state
    ), "Scheduler state mismatch."

    print(
        "✓ Scheduler state restored"
    )

    # =========================================================
    # Verify engine counters
    #
    # Note: the engine converts checkpoint epoch N into
    # start_epoch = N + 1 when resuming.
    # =========================================================

    assert engine_b.start_epoch == 2, (
        f"Unexpected start_epoch: "
        f"{engine_b.start_epoch}"
    )

    assert (
        engine_b.global_step
        == saved_global_step
    ), "global_step mismatch."

    assert (
        engine_b.best_f1
        == saved_best_f1
    ), "best_f1 mismatch."

    assert (
        engine_b.epochs_without_improvement
        == saved_epochs_without_improvement
    ), "early-stopping counter mismatch."

    print(
        "✓ Epoch / step / best-F1 / patience state restored"
    )

    # =========================================================
    # Verify history
    # =========================================================

    assert (
        engine_b.history
        == saved_history
    ), "Training history mismatch."

    print(
        "✓ Training history restored"
    )

    # =========================================================
    # Verify resumed model output
    # =========================================================

    model_a.eval()
    model_b.eval()

    with torch.no_grad():

        output_a = model_a(
            image_a,
            image_b,
        )

        output_b = model_b(
            image_a,
            image_b,
        )

    max_difference = (
        output_a
        - output_b
    ).abs().max().item()

    print()
    print(
        f"Maximum resumed-output difference: "
        f"{max_difference:.8e}"
    )

    assert max_difference == 0.0, (
        "Resumed model output does not exactly "
        "match the saved model output."
    )

    print(
        "✓ Resumed model produces identical output"
    )

    # =========================================================
    # Final
    # =========================================================

    print()
    print("=" * 70)
    print("MODEL 3 CHECKPOINT / RESUME TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()