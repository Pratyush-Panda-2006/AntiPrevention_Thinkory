# MODEL 1 / BASELINE V1 - Freeze Record

## 1. Status
**STATUS:** FROZEN (Do not modify or overwrite)

## 2. Artifact Preservation
All training artifacts for this baseline are preserved and untouched in the local directory:
`runs/2026-08-17_20-42-38_siamese_unet_baseline/`

This includes:
- `checkpoints/best.pt` (Epoch 25 weights)
- `checkpoints/last.pt` (Epoch 30 weights)
- `training_history.csv`
- `report.md`
- `config.json`
- `environment.json`
- `test_metrics.json`

## 3. Final Benchmark (Unseen Test Set)
- **Test F1 Score:** 0.9080
- **Test IoU Score:** 0.8314
- **Precision:** 0.9285
- **Recall:** 0.8883
- **Accuracy:** 0.9908
- *(Best Validation F1: 0.9147 at Epoch 25)*

## 4. Configuration Details
- **Architecture:** Siamese U-Net
- **Decision Threshold:** 0.5
- **Batch Size:** 2
- **Learning Rate:** 0.0001 (Cosine Annealing)
- **Loss:** 0.5 BCE + 0.5 Dice
- **AMP Enabled:** True

## 5. Hardware & Environment
- **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB VRAM)
- **CUDA Version:** 12.4
- **Python:** 3.13.7
- **PyTorch:** 2.6.0+cu124
- **Total Training Duration:** ~8.5 hours
- **Original Git Commit:** c37f340055ee6724a84b88d5a38b95d129b89b9b

---
*Future experiments must use new run directories (`--experiment-name`) and not interfere with these baseline files.*
