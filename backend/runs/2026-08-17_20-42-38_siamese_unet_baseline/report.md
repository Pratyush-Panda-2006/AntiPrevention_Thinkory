# Training Experiment Report

## 1. Run Information
- **Run ID**: 2026-08-17_20-42-38_siamese_unet_baseline
- **Experiment Name**: siamese_unet_baseline
- **Timestamp**: 2026-08-17_20-42-38
- **Status**: completed

## 2. Hardware
- **OS**: Windows 11
- **CPU Cores**: unavailable
- **RAM**: unavailable GB
- **CUDA Available**: True
- **GPU**: NVIDIA GeForce RTX 4050 Laptop GPU
- **GPU VRAM**: 6.0 GB

## 3. Software
- **Python**: 3.13.7
- **PyTorch**: 2.6.0+cu124
- **CUDA Version**: 12.4
- **Git Commit**: c37f340055ee6724a84b88d5a38b95d129b89b9b

## 4. Training Configuration
- **Epochs**: 30
- **Batch Size**: 2
- **Learning Rate**: 0.0001
- **BCE Weight**: 0.5
- **Dice Weight**: 0.5
- **AMP Enabled**: True

## 5. Training Results
| Epoch | Train Loss | Val Loss | Precision | Recall | F1 | IoU | LR | Epoch Time |
| ----: | ---------: | -------: | --------: | -----: | -: | --: | -: | ---------: |
| 24 | 0.0669 | 0.0877 | 0.9225 | 0.9045 | 0.9134 | 0.8407 | 5.00e-05 | 1084.6 |
| 25 | 0.0639 | 0.0971 | 0.9259 | 0.9038 | 0.9147 | 0.8429 | 5.00e-05 | 1061.6 |
| 26 | 0.0640 | 0.0882 | 0.9302 | 0.8905 | 0.9099 | 0.8347 | 5.00e-05 | 1067.3 |
| 27 | 0.0628 | 0.1259 | 0.9223 | 0.8740 | 0.8975 | 0.8141 | 5.00e-05 | 1065.8 |
| 28 | 0.0625 | 0.1270 | 0.9362 | 0.8454 | 0.8885 | 0.7993 | 5.00e-05 | 1067.1 |
| 29 | 0.0608 | 0.0985 | 0.9266 | 0.8910 | 0.9085 | 0.8323 | 2.50e-05 | 1052.7 |
| 30 | 0.0561 | 0.1251 | 0.9124 | 0.8788 | 0.8953 | 0.8104 | 2.50e-05 | 1059.7 |

## 6. Best Model
- **Best Epoch**: 25
- **Best F1**: 0.9147
- **Validation Loss**: 0.0971
