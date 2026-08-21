import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducible experiments.

    This controls randomness from:
        - Python
        - NumPy
        - PyTorch CPU
        - PyTorch CUDA

    Note:
        Deterministic execution can reduce performance on some
        GPU operations. We prioritize reproducibility for our
        baseline experiments.
    """

    random.seed(seed)

    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Make CUDA operations as deterministic as possible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """
    Seed a PyTorch DataLoader worker.

    This is used when num_workers > 0.
    """

    worker_seed = (
        torch.initial_seed()
        % 2**32
    )

    np.random.seed(worker_seed)

    random.seed(worker_seed)


def create_generator(seed: int = 42):
    """
    Create a reproducible PyTorch generator.

    Useful for DataLoaders and samplers.
    """

    generator = torch.Generator()

    generator.manual_seed(seed)

    return generator