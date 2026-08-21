import json
from pathlib import Path
from typing import Dict, Any, List

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


class ExperimentReporter:
    """
    Handles end-of-run data serialization, markdown compilation, and plotting.
    """

    def __init__(self, run_dir: str):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self.epoch_metrics_file = self.run_dir / "epoch_metrics.json"
        
        # Initialize an empty list if file doesn't exist (for resumes)
        if not self.epoch_metrics_file.exists():
            with open(self.epoch_metrics_file, "w", encoding="utf-8") as f:
                json.dump([], f)

    def append_epoch_metric(self, metric: Dict[str, Any]):
        """
        Safely flushes one epoch record to the JSON array.
        """
        with open(self.epoch_metrics_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        data.append(metric)
        
        with open(self.epoch_metrics_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def save_json(self, filename: str, data: Dict[str, Any]):
        path = self.run_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def generate_plots(self, history: List[Dict[str, Any]]):
        """
        Generates plots from the training history.
        """
        if plt is None or len(history) == 0:
            return

        plots_dir = self.run_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        epochs = [h["epoch"] for h in history]
        train_loss = [h["train_loss"] for h in history]
        val_loss = [h["val_loss"] for h in history]
        val_f1 = [h["f1"] for h in history]
        val_iou = [h["iou"] for h in history]
        val_prec = [h["precision"] for h in history]
        val_rec = [h["recall"] for h in history]
        lr = [h["learning_rate"] for h in history]

        # 1. Loss
        plt.figure()
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.savefig(plots_dir / "training_vs_validation_loss.png")
        plt.close()

        # 2. F1
        plt.figure()
        plt.plot(epochs, val_f1, label="Val F1", color="green")
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.title("Validation F1")
        plt.legend()
        plt.savefig(plots_dir / "validation_f1.png")
        plt.close()

        # 3. IoU
        plt.figure()
        plt.plot(epochs, val_iou, label="Val IoU", color="purple")
        plt.xlabel("Epoch")
        plt.ylabel("IoU")
        plt.title("Validation IoU")
        plt.legend()
        plt.savefig(plots_dir / "validation_iou.png")
        plt.close()

        # 4. Precision/Recall
        plt.figure()
        plt.plot(epochs, val_prec, label="Precision")
        plt.plot(epochs, val_rec, label="Recall")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("Validation Precision and Recall")
        plt.legend()
        plt.savefig(plots_dir / "precision_recall.png")
        plt.close()

        # 5. Learning Rate
        plt.figure()
        plt.plot(epochs, lr, label="Learning Rate", color="orange")
        plt.xlabel("Epoch")
        plt.ylabel("LR")
        plt.yscale("log")
        plt.title("Learning Rate Schedule")
        plt.legend()
        plt.savefig(plots_dir / "learning_rate.png")
        plt.close()

    def generate_report(self, run_info: Dict[str, Any], status: str, exception_msg: str = None):
        """
        Compiles the complete human-readable report.md and structured report.json.
        """
        # Read the artifacts
        try:
            with open(self.run_dir / "config.json", "r") as f:
                config = json.load(f)
        except Exception:
            config = {}
            
        try:
            with open(self.run_dir / "environment.json", "r") as f:
                env = json.load(f)
        except Exception:
            env = {}
            
        try:
            with open(self.epoch_metrics_file, "r") as f:
                history = json.load(f)
        except Exception:
            history = []

        # Find best epoch based on F1
        best_epoch = None
        if history:
            best_epoch = max(history, key=lambda x: x.get("f1", 0.0))
            
        # Write JSON Report
        report_json = {
            "run_info": run_info,
            "status": status,
            "exception": exception_msg,
            "environment": env,
            "config": config,
            "history": history,
            "best_epoch_metrics": best_epoch,
            "final_epoch_metrics": history[-1] if history else None,
        }
        self.save_json("report.json", report_json)
        
        # Write Summary JSON
        if best_epoch:
            summary_json = {
                "run_id": run_info.get("run_id"),
                "status": status,
                "best_epoch": best_epoch.get("epoch"),
                "best_f1": best_epoch.get("f1"),
                "best_iou": best_epoch.get("iou"),
                "best_precision": best_epoch.get("precision"),
                "best_recall": best_epoch.get("recall"),
                "final_epoch": history[-1].get("epoch") if history else None,
                "final_f1": history[-1].get("f1") if history else None,
                "training_seconds": run_info.get("duration", 0),
                "git_commit": env.get("git_commit", "unknown")
            }
            self.save_json("summary.json", summary_json)

        # Write Markdown Report
        md = [
            "# Training Experiment Report\n",
            "## 1. Run Information",
            f"- **Run ID**: {run_info.get('run_id')}",
            f"- **Experiment Name**: {run_info.get('experiment_name')}",
            f"- **Timestamp**: {run_info.get('timestamp')}",
            f"- **Status**: {status}",
            ""
        ]
        
        if exception_msg:
            md.extend([
                "## ⚠️ Failure Report",
                f"```text\n{exception_msg}\n```\n"
            ])

        md.extend([
            "## 2. Hardware",
            f"- **OS**: {env.get('os')} {env.get('os_release')}",
            f"- **CPU Cores**: {env.get('cpu_count_logical')}",
            f"- **RAM**: {env.get('system_ram_gb')} GB",
            f"- **CUDA Available**: {env.get('cuda_available')}",
            f"- **GPU**: {env.get('gpu_name', 'N/A')}",
            f"- **GPU VRAM**: {env.get('gpu_total_memory_gb', 'N/A')} GB",
            ""
        ])

        md.extend([
            "## 3. Software",
            f"- **Python**: {env.get('python_version')}",
            f"- **PyTorch**: {env.get('pytorch_version')}",
            f"- **CUDA Version**: {env.get('cuda_version', 'N/A')}",
            f"- **Git Commit**: {env.get('git_commit')}",
            ""
        ])

        md.extend([
            "## 4. Training Configuration",
            f"- **Epochs**: {config.get('epochs')}",
            f"- **Batch Size**: {config.get('batch_size')}",
            f"- **Learning Rate**: {config.get('learning_rate')}",
            f"- **BCE Weight**: {config.get('bce_weight')}",
            f"- **Dice Weight**: {config.get('dice_weight')}",
            f"- **AMP Enabled**: {config.get('use_amp')}",
            ""
        ])
        
        if history:
            md.extend([
                "## 5. Training Results",
                "| Epoch | Train Loss | Val Loss | Precision | Recall | F1 | IoU | LR | Epoch Time |",
                "| ----: | ---------: | -------: | --------: | -----: | -: | --: | -: | ---------: |"
            ])
            for h in history:
                md.append(
                    f"| {h['epoch']} | {h['train_loss']:.4f} | {h['val_loss']:.4f} "
                    f"| {h['precision']:.4f} | {h['recall']:.4f} | {h['f1']:.4f} "
                    f"| {h['iou']:.4f} | {h['learning_rate']:.2e} | {h.get('epoch_seconds', 0):.1f} |"
                )
            md.append("")
            
            if best_epoch:
                md.extend([
                    "## 6. Best Model",
                    f"- **Best Epoch**: {best_epoch['epoch']}",
                    f"- **Best F1**: {best_epoch['f1']:.4f}",
                    f"- **Validation Loss**: {best_epoch['val_loss']:.4f}",
                    ""
                ])

        with open(self.run_dir / "report.md", "w", encoding="utf-8") as f:
            f.write("\n".join(md))
            
        # Try to generate plots
        self.generate_plots(history)
