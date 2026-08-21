from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from data.sar_patch_dataset import TUMSARChangeDetectionDataset
from detection.losses import BCEDiceLoss
from detection.siamese_unet import SiameseUNet
from detection.siamese_resnet34_unet import SiameseResNet34UNet
from detection.snunet_cd import SNUNetCD


def move_batch_to_device(batch, device):
    image_a = batch["image_a"].unsqueeze(0).to(device)
    image_b = batch["image_b"].unsqueeze(0).to(device)
    target = batch["label"].unsqueeze(0).to(device)
    valid_mask = batch["valid_mask"].unsqueeze(0).to(device)

    return image_a, image_b, target, valid_mask


def test_model(model_name, model, batch, criterion, device):
    print()
    print("-" * 70)
    print(f"TESTING: {model_name}")
    print("-" * 70)

    model = model.to(device)
    model.train()

    image_a, image_b, target, valid_mask = move_batch_to_device(
        batch,
        device,
    )

    print("Input A:", tuple(image_a.shape))
    print("Input B:", tuple(image_b.shape))
    print("Target :", tuple(target.shape))
    print("Mask   :", tuple(valid_mask.shape))

    assert image_a.shape == (1, 2, 256, 256)
    assert image_b.shape == (1, 2, 256, 256)
    assert target.shape == (1, 1, 256, 256)
    assert valid_mask.shape == (1, 1, 256, 256)

    assert image_a.dtype == torch.float32
    assert image_b.dtype == torch.float32
    assert target.dtype == torch.float32
    assert valid_mask.dtype == torch.bool

    assert torch.isfinite(image_a).all()
    assert torch.isfinite(image_b).all()
    assert torch.isfinite(target).all()

    valid_pixels = int(valid_mask.sum().item())

    print("Valid pixels:", valid_pixels)

    assert valid_pixels > 0

    model.zero_grad(set_to_none=True)

    logits = model(
        image_a,
        image_b,
    )

    print("Logits:", tuple(logits.shape))

    assert logits.shape == (1, 1, 256, 256)
    assert torch.isfinite(logits).all()

    loss = criterion(
        logits,
        target,
        valid_mask=valid_mask,
    )

    print(f"Loss: {loss.item():.6f}")

    assert torch.isfinite(loss)

    loss.backward()

    max_grad = 0.0
    gradient_tensors = 0

    for parameter in model.parameters():
        if parameter.grad is not None:
            gradient_tensors += 1
            max_grad = max(
                max_grad,
                float(parameter.grad.detach().abs().max().item()),
            )

    print("Gradient tensors:", gradient_tensors)
    print(f"Maximum gradient: {max_grad:.6e}")

    assert gradient_tensors > 0
    assert max_grad > 0.0

    print(f"✓ {model_name} real-data integration passed")


def main():
    print("=" * 70)
    print("REAL TUM SAR → MODEL INTEGRATION TEST")
    print("=" * 70)

    device = torch.device("cpu")

    print("Device:", device)

    dataset = TUMSARChangeDetectionDataset(
        patch_index_path=(
            PROJECT_ROOT
            / "data"
            / "sar"
            / "tum_oscd"
            / "sar_patch_index.json"
        ),
        split="train",
    )

    print("Train dataset patches:", len(dataset))

    assert len(dataset) > 0

    # Use the first deterministic patch.
    batch = dataset[0]

    print()
    print("Selected patch:")
    print("  City:", batch["city"])
    print("  X:", batch["x"])
    print("  Y:", batch["y"])

    print("  Image A:", tuple(batch["image_a"].shape))
    print("  Image B:", tuple(batch["image_b"].shape))
    print("  Target :", tuple(batch["label"].shape))
    print("  Mask   :", tuple(batch["valid_mask"].shape))

    criterion = BCEDiceLoss(
        bce_weight=0.5,
        dice_weight=0.5,
        pos_weight=1.0,
    )

    # ---------------------------------------------------------
    # Model 1
    # ---------------------------------------------------------

    model1 = SiameseUNet(
        in_channels=2,
        num_classes=1,
    )

    test_model(
        "Siamese U-Net (2-channel)",
        model1,
        batch,
        criterion,
        device,
    )

    # ---------------------------------------------------------
    # Model 2
    # ---------------------------------------------------------

    model2 = SiameseResNet34UNet(
        in_channels=2,
        num_classes=1,
        sar_init_mode="average",
    )

    test_model(
        "ResNet-34 Siamese U-Net (2-channel)",
        model2,
        batch,
        criterion,
        device,
    )

    # ---------------------------------------------------------
    # Model 3
    # ---------------------------------------------------------

    model3 = SNUNetCD(
        in_channels=2,
        num_classes=1,
    )

    test_model(
        "SNUNet-CD (2-channel)",
        model3,
        batch,
        criterion,
        device,
    )

    print()
    print("=" * 70)
    print("REAL TUM SAR → MODEL INTEGRATION TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()