from pathlib import Path
import sys

import numpy as np
import rasterio


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from preprocessing.sar_loader import (
    normalize_sar_tensor,
    VV_MIN_DB,
    VV_MAX_DB,
    VH_MIN_DB,
    VH_MAX_DB,
)


CITY = "abudhabi"

SAR_ROOT = (
    PROJECT_ROOT
    / "data"
    / "sar"
    / "tum_oscd"
    / "multisensor_fusion_CD"
    / "S1"
    / CITY
)


def find_single_tif(directory):
    files = sorted(
        directory.glob("*.tif")
    )

    if len(files) != 1:
        raise RuntimeError(
            f"Expected exactly 1 TIFF in {directory}, "
            f"found {len(files)}:\n{files}"
        )

    return files[0]


def main():
    print("=" * 70)
    print("REAL TUM SAR PREPROCESSING TEST")
    print("=" * 70)
    print(f"City: {CITY}")

    t1_path = find_single_tif(
        SAR_ROOT
        / "imgs_1"
        / "transformed"
    )

    t2_path = find_single_tif(
        SAR_ROOT
        / "imgs_2"
        / "transformed"
    )

    print()
    print(f"T1: {t1_path.name}")
    print(f"T2: {t2_path.name}")

    # ---------------------------------------------------------
    # Read T1
    # ---------------------------------------------------------

    with rasterio.open(t1_path) as src:
        t1 = src.read().astype(
            np.float32,
            copy=False,
        )

        t1_width = src.width
        t1_height = src.height
        t1_count = src.count
        t1_crs = src.crs
        t1_transform = src.transform

    # ---------------------------------------------------------
    # Read T2
    # ---------------------------------------------------------

    with rasterio.open(t2_path) as src:
        t2 = src.read().astype(
            np.float32,
            copy=False,
        )

        t2_width = src.width
        t2_height = src.height
        t2_count = src.count
        t2_crs = src.crs
        t2_transform = src.transform

    # ---------------------------------------------------------
    # Basic structure
    # ---------------------------------------------------------

    print()
    print("STRUCTURE")
    print(f"T1 shape: {t1.shape}")
    print(f"T2 shape: {t2.shape}")

    assert t1.shape[0] == 2
    assert t2.shape[0] == 2

    assert t1.shape[1:] == t2.shape[1:]

    assert t1_count == 2
    assert t2_count == 2

    # ---------------------------------------------------------
    # Raw statistics
    # ---------------------------------------------------------

    print()
    print("RAW TUM dB STATISTICS")

    for name, arr in [
        ("T1 VV", t1[0]),
        ("T1 VH", t1[1]),
        ("T2 VV", t2[0]),
        ("T2 VH", t2[1]),
    ]:

        finite = np.isfinite(arr)

        print(
            f"{name}: "
            f"min={np.nanmin(arr):.4f}, "
            f"max={np.nanmax(arr):.4f}, "
            f"mean={np.nanmean(arr):.4f}, "
            f"std={np.nanstd(arr):.4f}, "
            f"valid={finite.sum()}/{arr.size}"
        )

        assert finite.all(), (
            f"{name} contains NaN/Inf."
        )

    # ---------------------------------------------------------
    # Normalize
    # ---------------------------------------------------------

    print()
    print("RUNNING LOCKED NORMALIZATION")

    t1_norm, t1_valid = normalize_sar_tensor(
        t1,
        is_linear=False,
        return_validity=True,
    )

    t2_norm, t2_valid = normalize_sar_tensor(
        t2,
        is_linear=False,
        return_validity=True,
    )

    print(
        f"T1 normalized shape: {t1_norm.shape}"
    )
    print(
        f"T2 normalized shape: {t2_norm.shape}"
    )

    # ---------------------------------------------------------
    # Validate output
    # ---------------------------------------------------------

    assert t1_norm.dtype == np.float32
    assert t2_norm.dtype == np.float32

    assert t1_norm.shape == t1.shape
    assert t2_norm.shape == t2.shape

    assert np.isfinite(t1_norm).all()
    assert np.isfinite(t2_norm).all()

    assert (
        t1_norm.min() >= -1e-6
        and t1_norm.max() <= 1.0 + 1e-6
    )

    assert (
        t2_norm.min() >= -1e-6
        and t2_norm.max() <= 1.0 + 1e-6
    )

    # ---------------------------------------------------------
    # Validate validity masks
    # ---------------------------------------------------------

    assert t1_valid.shape == t1.shape[1:]
    assert t2_valid.shape == t2.shape[1:]

    assert t1_valid.dtype == np.bool_
    assert t2_valid.dtype == np.bool_

    print()
    print("VALIDITY")
    print(
        f"T1 valid pixels: "
        f"{int(t1_valid.sum())}/{t1_valid.size}"
    )

    print(
        f"T2 valid pixels: "
        f"{int(t2_valid.sum())}/{t2_valid.size}"
    )

    # ---------------------------------------------------------
    # Check normalized statistics
    # ---------------------------------------------------------

    print()
    print("NORMALIZED STATISTICS")

    for name, arr in [
        ("T1 VV", t1_norm[0]),
        ("T1 VH", t1_norm[1]),
        ("T2 VV", t2_norm[0]),
        ("T2 VH", t2_norm[1]),
    ]:

        print(
            f"{name}: "
            f"min={arr.min():.6f}, "
            f"max={arr.max():.6f}, "
            f"mean={arr.mean():.6f}, "
            f"std={arr.std():.6f}"
        )

    # ---------------------------------------------------------
    # Verify constants
    # ---------------------------------------------------------

    assert VV_MIN_DB == -22.98
    assert VV_MAX_DB == 5.63

    assert VH_MIN_DB == -32.33
    assert VH_MAX_DB == -2.53

    print()
    print("LOCKED NORMALIZATION")
    print(
        f"VV: [{VV_MIN_DB}, {VV_MAX_DB}] dB"
    )
    print(
        f"VH: [{VH_MIN_DB}, {VH_MAX_DB}] dB"
    )

    # ---------------------------------------------------------
    # Final
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("REAL TUM SAR PREPROCESSING TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()