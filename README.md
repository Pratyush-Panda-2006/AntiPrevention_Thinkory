# SAR Change Intelligence - LEVIR-CD

This repository contains the implementation of a Change Detection model (Siamese U-Net) applied to the LEVIR-CD satellite imagery dataset.

## 🚀 Baseline Performance (Model 1)
Our fully trained Model 1 (Baseline V1) achieved the following benchmark scores on the unseen LEVIR-CD Test Set:

* **Test F1 Score:** `0.9080`
* **Test IoU Score:** `0.8314`
* **Precision:** `0.9285`
* **Recall:** `0.8883`
* **Accuracy:** `0.9908`

*Detailed training reports and metric logs can be found in the `results/baseline/` directory.*

## ⚙️ Environment Setup
To reproduce this environment, ensure you have Python 3.13+ installed and a CUDA-capable NVIDIA GPU.

1. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/Scripts/activate  # Windows
   ```

2. **Install dependencies:**
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
   *(Note: Ensure you install the CUDA version of PyTorch compatible with your system).*

## 🧠 Training & Evaluation

**To start a new training run:**
```bash
python scripts/train.py --experiment-name my_experiment_name --batch-size 4 --num-workers 2 --epochs 30
```

**To evaluate the model on the test set:**
```bash
python scripts/test.py --checkpoint runs/<experiment_name>/checkpoints/best.pt
```

**To find the optimal decision threshold:**
```bash
python scripts/sweep_threshold.py --checkpoint runs/<experiment_name>/checkpoints/best.pt
```

## 📁 Repository Structure
* `src/` - Core neural network architectures, data loaders, and training logic.
* `scripts/` - Executable scripts for training, testing, and threshold sweeping.
* `results/` - Publicly available reports and metrics from frozen baselines.
* `runs/` - (Git-ignored) Local directory containing heavy `.pt` model weights and tensorboard logs.
