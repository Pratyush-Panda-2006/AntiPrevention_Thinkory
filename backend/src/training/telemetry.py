import time
import platform
import subprocess
from typing import Dict, Any, Optional

import torch


class SystemTelemetry:
    """
    Collects hardware, OS, and software environment metrics.
    Gracefully handles missing dependencies like psutil.
    """

    @staticmethod
    def get_environment_info() -> Dict[str, Any]:
        """
        Snapshot of the current execution environment.
        """
        env = {
            "os": platform.system(),
            "os_release": platform.release(),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "hostname": platform.node(),
            "cuda_available": torch.cuda.is_available(),
        }

        # Try to get git info
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            ).decode("utf-8").strip()
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
            ).decode("utf-8").strip()
            status = subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
            ).decode("utf-8").strip()

            env["git_commit"] = commit
            env["git_branch"] = branch
            env["git_is_dirty"] = bool(status)
        except Exception:
            env["git_commit"] = "unavailable"
            env["git_branch"] = "unavailable"
            env["git_is_dirty"] = False

        # Try to get RAM / CPU via psutil
        try:
            import psutil
            env["cpu_count_physical"] = psutil.cpu_count(logical=False)
            env["cpu_count_logical"] = psutil.cpu_count(logical=True)
            env["system_ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
        except ImportError:
            env["cpu_count_physical"] = "unavailable"
            env["cpu_count_logical"] = "unavailable"
            env["system_ram_gb"] = "unavailable"

        # GPU static info
        if torch.cuda.is_available():
            env["cuda_version"] = torch.version.cuda
            env["cudnn_version"] = torch.backends.cudnn.version()
            env["gpu_count"] = torch.cuda.device_count()
            env["gpu_name"] = torch.cuda.get_device_name(0)
            env["gpu_total_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
            )

        return env

    @staticmethod
    def get_live_metrics() -> Dict[str, Any]:
        """
        Live CPU/RAM metrics.
        """
        metrics = {}
        try:
            import psutil
            process = psutil.Process()
            metrics["process_ram_gb"] = round(process.memory_info().rss / (1024**3), 2)
            metrics["system_ram_percent"] = psutil.virtual_memory().percent
        except ImportError:
            pass
        return metrics


class GPUTelemetry:
    """
    Collects live GPU memory and utilization metrics.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.is_cuda = device.type == "cuda"
        
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)

    def get_metrics(self) -> Dict[str, Any]:
        metrics = {
            "gpu_allocated_gb": 0.0,
            "gpu_reserved_gb": 0.0,
            "gpu_peak_allocated_gb": 0.0,
            "gpu_peak_reserved_gb": 0.0,
        }

        if not self.is_cuda:
            return metrics

        try:
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
            peak_allocated = torch.cuda.max_memory_allocated(self.device)
            peak_reserved = torch.cuda.max_memory_reserved(self.device)

            metrics["gpu_allocated_gb"] = round(allocated / (1024**3), 2)
            metrics["gpu_reserved_gb"] = round(reserved / (1024**3), 2)
            metrics["gpu_peak_allocated_gb"] = round(peak_allocated / (1024**3), 2)
            metrics["gpu_peak_reserved_gb"] = round(peak_reserved / (1024**3), 2)
        except Exception:
            pass

        return metrics


class TimingTelemetry:
    """
    Tracks and smooths timing for ETA calculation.
    Uses an Exponential Moving Average (EMA).
    """

    def __init__(self, smoothing: float = 0.1):
        self.smoothing = smoothing
        self.ema_batch_time: Optional[float] = None
        
        self.epoch_start_time: float = 0.0
        self.batch_start_time: float = 0.0

        self.global_start_time = time.time()
        self.accumulated_training_seconds = 0.0

    def start_epoch(self):
        self.epoch_start_time = time.time()

    def start_batch(self):
        self.batch_start_time = time.time()

    def end_batch(self) -> float:
        duration = time.time() - self.batch_start_time
        
        if self.ema_batch_time is None:
            self.ema_batch_time = duration
        else:
            self.ema_batch_time = (
                self.smoothing * duration + 
                (1.0 - self.smoothing) * self.ema_batch_time
            )
            
        return duration

    def get_epoch_elapsed(self) -> float:
        if self.epoch_start_time == 0.0:
            return 0.0
        return time.time() - self.epoch_start_time

    def get_run_elapsed(self) -> float:
        return (time.time() - self.global_start_time) + self.accumulated_training_seconds

    def estimate_epoch_eta(self, remaining_batches: int) -> float:
        if self.ema_batch_time is None:
            return 0.0
        return self.ema_batch_time * remaining_batches

    def estimate_run_eta(
        self, 
        remaining_batches_current_epoch: int, 
        remaining_epochs: int, 
        batches_per_epoch: int
    ) -> float:
        if self.ema_batch_time is None:
            return 0.0
        
        total_remaining_batches = (
            remaining_batches_current_epoch + 
            (remaining_epochs * batches_per_epoch)
        )
        return self.ema_batch_time * total_remaining_batches

    @staticmethod
    def format_time(seconds: float) -> str:
        """Format seconds to HH:MM:SS or MM:SS."""
        if seconds < 0:
            return "00:00"
        
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"
