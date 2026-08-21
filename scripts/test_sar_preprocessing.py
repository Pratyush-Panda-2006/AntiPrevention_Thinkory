from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from preprocessing.sar_loader import (
    VV_MIN_DB,
    VV_MAX_DB,
    VH_MIN_DB,
    VH_MAX_DB,
    normalize_sar_tensor,
)


def main():
    print("=" * 70)
    print("SAR PREPROCESSING / NORMALIZATION TEST")
    print("=" * 70)

    # =========================================================
    # TUM-style already-dB input
    # =========================================================

    tum = np.array(
        [
            [
                [-30.0, -22.98, -10.0, 5.63, 20.0],
                [-100.0, -15.0, 0.0, 6.0, np.nan],
            ],
            [
                [-40.0, -32.33, -15.0, -2.53, 5.0],
                [-100.0, -20.0, -5.0, -1.0, np.nan],
            ],
        ],
        dtype=np.float32,
    )

    tum_norm, tum_valid = normalize_sar_tensor(
        tum,
        is_linear=False,
        return_validity=True,
    )

    print("\nTUM / ALREADY-dB TEST")
    print("VV normalized:")
    print(tum_norm[0])

    print("VH normalized:")
    print(tum_norm[1])

    print("Validity:")
    print(tum_valid)

    assert tum_valid.shape == (2, 5)

    assert tum_norm.dtype == np.float32

    assert tum_norm.min() >= -1e-6
    assert tum_norm.max() <= 1.0 + 1e-6

    # Exact lower/upper bounds.
    assert np.isclose(
        tum_norm[0, 0, 0],
        0.0,
    )

    assert np.isclose(
        tum_norm[0, 0, 3],
        1.0,
    )

    assert np.isclose(
        tum_norm[1, 0, 1],
        0.0,
    )

    assert np.isclose(
        tum_norm[1, 0, 3],
        1.0,
    )

    # =========================================================
    # CDSE-style linear input
    # =========================================================

    linear = np.array(
        [
            [
                [1.0, 10.0, 0.1, 0.0],
                [np.nan, 2.0, 3.0, 4.0],
            ],
            [
                [1.0, 0.1, 0.01, 0.0],
                [np.nan, 0.2, 0.3, 0.4],
            ],
        ],
        dtype=np.float32,
    )

    linear_norm, linear_valid = normalize_sar_tensor(
        linear,
        is_linear=True,
        return_validity=True,
    )

    print("\nCDSE / LINEAR-POWER TEST")
    print("Normalized VV:")
    print(linear_norm[0])

    print("Normalized VH:")
    print(linear_norm[1])

    print("Validity:")
    print(linear_valid)

    assert linear_norm.dtype == np.float32

    assert linear_norm.min() >= -1e-6
    assert linear_norm.max() <= 1.0 + 1e-6

    # Zero/NaN linear pixels are invalid.
    # [0, 3] is 0.0, [1, 0] is np.nan
    assert not linear_valid[0, 3]
    assert not linear_valid[1, 0]

    # Invalid pixels must become zero.
    assert np.isclose(
        linear_norm[0, 0, 3],
        0.0,
    )

    assert np.isclose(
        linear_norm[1, 1, 0],
        0.0,
    )

    # =========================================================
    # Constants
    # =========================================================

    print("\nLOCKED CONSTANTS")
    print(
        f"VV: [{VV_MIN_DB}, {VV_MAX_DB}] dB"
    )

    print(
        f"VH: [{VH_MIN_DB}, {VH_MAX_DB}] dB"
    )

    assert VV_MIN_DB == -22.98
    assert VV_MAX_DB == 5.63
    assert VH_MIN_DB == -32.33
    assert VH_MAX_DB == -2.53

    # =========================================================
    # Final
    # =========================================================

    print()
    print("=" * 70)
    print("SAR PREPROCESSING TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()