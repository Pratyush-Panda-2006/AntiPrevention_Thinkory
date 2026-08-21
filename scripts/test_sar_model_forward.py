import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from detection.snunet_cd import SNUNetCD
from detection.siamese_resnet34_unet import SiameseResNet34UNet
from detection.siamese_unet import SiameseUNet
from detection.losses import BCEDiceLoss

def test_model(model_class, name, sar_init_kwarg=False):
    print("=" * 60)
    print(f"TESTING {name} WITH 2-CHANNEL INPUT")
    print("=" * 60)
    
    device = torch.device("cpu")
    
    kwargs = {"in_channels": 2, "num_classes": 1}
    if sar_init_kwarg:
        kwargs["sar_init_mode"] = "average"
        
    model = model_class(**kwargs).to(device)
    model.train()
    
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5, pos_weight=1.0)
    
    # [2, 2, 256, 256] -> Batch size 2, 2 channels
    image_a = torch.randn(2, 2, 256, 256, device=device)
    image_b = torch.randn(2, 2, 256, 256, device=device)
    
    # Binary change mask, with 0s in the padded regions.
    target = torch.zeros(2, 1, 256, 256, device=device)
    target[:, :, 64:128, 64:128] = 1.0
    
    # Valid mask (e.g. 1.0 for valid pixels, 0.0 for padded)
    valid_mask = torch.ones(2, 1, 256, 256, device=device)
    valid_mask[:, :, 200:, 200:] = 0.0 # Some padded regions
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    print("\nRunning forward pass...")
    optimizer.zero_grad(set_to_none=True)
    
    logits = model(image_a, image_b)
    
    print(f"Logits shape: {tuple(logits.shape)}")
    assert logits.shape == (2, 1, 256, 256)
    
    loss = criterion(logits, target, valid_mask=valid_mask)
    print(f"Loss (Masked): {loss.item():.6f}")
    assert torch.isfinite(loss)
    
    print("\nRunning backward pass...")
    loss.backward()
    print("✓ Backward pass completed")
    
    print("\nRunning optimizer step...")
    optimizer.step()
    print("✓ Optimizer step completed")
    
    print(f"\n✓ {name} supports 2-channel inputs seamlessly.")
    print("=" * 60)
    print()

def main():
    test_model(SiameseUNet, "Siamese U-Net (Model 1)")
    test_model(SiameseResNet34UNet, "Siamese ResNet34 U-Net (Model 2)", sar_init_kwarg=True)
    test_model(SNUNetCD, "SNUNet-CD (Model 3)")
    print("ALL 2-CHANNEL SYNTHETIC TESTS PASSED!")

if __name__ == "__main__":
    main()
