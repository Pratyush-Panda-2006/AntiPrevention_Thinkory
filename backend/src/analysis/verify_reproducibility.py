import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Add src to Python path
sys.path.append(str(PROJECT_ROOT / "src"))

import random

import numpy as np
import torch

from training.reproducibility import set_seed


def main():

    print("=" * 60)
    print("REPRODUCIBILITY VERIFICATION")
    print("=" * 60)

    seed = 42

    # ---------------------------------------------------------
    # First sequence
    # ---------------------------------------------------------

    set_seed(seed)

    python_a = random.random()

    numpy_a = np.random.rand()

    torch_a = torch.rand(5)

    # ---------------------------------------------------------
    # Second sequence with same seed
    # ---------------------------------------------------------

    set_seed(seed)

    python_b = random.random()

    numpy_b = np.random.rand()

    torch_b = torch.rand(5)

    # ---------------------------------------------------------
    # Compare
    # ---------------------------------------------------------

    print("\nPython:")
    print(f"First:  {python_a}")
    print(f"Second: {python_b}")

    print("\nNumPy:")
    print(f"First:  {numpy_a}")
    print(f"Second: {numpy_b}")

    print("\nPyTorch:")
    print(f"First:  {torch_a}")
    print(f"Second: {torch_b}")

    assert python_a == python_b

    assert numpy_a == numpy_b

    assert torch.equal(
        torch_a,
        torch_b,
    )

    print("\n✓ Python randomness is reproducible")
    print("✓ NumPy randomness is reproducible")
    print("✓ PyTorch randomness is reproducible")

    print("\n" + "=" * 60)
    print("REPRODUCIBILITY TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()