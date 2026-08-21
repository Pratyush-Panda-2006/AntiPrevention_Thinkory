"""
src/preprocessing/sar_loader.py
================================

SAR preprocessing utilities for the Sentinel-1 change-detection pipeline.

Locked SAR representation
-------------------------
Training data:
    Sigma0 ellipsoid
    Orthorectified
    Already in dB

Live CDSE data:
    SIGMA0_ELLIPSOID
    Orthorectified
    VV/VH returned in linear power
    Converted locally to dB

Locked normalization
--------------------
VV:
    [-22.98, 5.63] dB -> [0, 1]

VH:
    [-32.33, -2.53] dB -> [0, 1]

Invalid-data handling
---------------------
A separate validity mask is preserved.

The mask is NOT a third model input channel.

For TUM data:
    finite dB values are valid.

For CDSE data:
    NaN / Inf / non-positive linear-power pixels are treated as invalid.
    Invalid pixels are represented by 0.0 after normalization, while their
    validity is preserved separately.

Public API
----------
decode_geotiff_response(response_bytes, return_validity=False)

normalize_sar_tensor(
    arr,
    is_linear=False,
    return_validity=False,
)

to_torch_tensor(arr)

load_sar_pair_for_inference(
    t1_bytes,
    t2_bytes,
    is_linear=True,
    return_tensors=True,
    return_validity_mask=False,
)
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Locked SAR normalization constants
# ============================================================================

VV_MIN_DB: float = -22.98
VV_MAX_DB: float = 5.63

VH_MIN_DB: float = -32.33
VH_MAX_DB: float = -2.53


# Minimum linear-power threshold used only to avoid log(0).
_LINEAR_EPS: float = 1e-10


# ============================================================================
# Validation helpers
# ============================================================================

def _validate_sar_array(arr: np.ndarray) -> None:
    """
    Validate SAR input shape.

    Expected:
        (2, H, W)

    Band order:
        0 = VV
        1 = VH
    """

    if not isinstance(arr, np.ndarray):
        raise TypeError(
            f"Expected numpy.ndarray, got {type(arr).__name__}."
        )

    if arr.ndim != 3:
        raise ValueError(
            f"Expected 3-D SAR array (C, H, W), got shape {arr.shape}."
        )

    if arr.shape[0] != 2:
        raise ValueError(
            "Expected exactly 2 SAR channels in order "
            "(VV, VH), got shape "
            f"{arr.shape}."
        )


# ============================================================================
# GeoTIFF decoder
# ============================================================================

def decode_geotiff_response(
    response_bytes: bytes,
    return_validity: bool = False,
):
    """
    Decode a 2-band GeoTIFF response.

    Parameters
    ----------
    response_bytes:
        Raw GeoTIFF bytes.

    return_validity:
        If True, also return a boolean validity mask with shape (H, W).

    Returns
    -------
    arr:
        Float32 array with shape (2, H, W).

    validity:
        Boolean array with shape (H, W), returned only when requested.

    Notes
    -----
    The CDSE evalscript returns:
        Band 1 = VV
        Band 2 = VH

    Validity is initially inferred from finite values.

    Non-positive linear-power pixels are handled later by the normalization
    function when is_linear=True.
    """

    try:
        import rasterio
        from rasterio.io import MemoryFile
    except ImportError as exc:
        raise ImportError(
            "rasterio is required for GeoTIFF decoding. "
            "Install it with: python -m pip install rasterio"
        ) from exc

    with MemoryFile(response_bytes) as mem_file:
        with mem_file.open() as dataset:

            if dataset.count != 2:
                raise ValueError(
                    "Expected exactly 2 bands (VV, VH), "
                    f"got {dataset.count}."
                )

            arr = dataset.read().astype(
                np.float32,
                copy=False,
            )

            # Any non-finite value is invalid.
            validity = np.all(
                np.isfinite(arr),
                axis=0,
            )

            # Optional raster mask support.
            try:
                masks = dataset.read_masks()

                raster_validity = np.all(
                    masks > 0,
                    axis=0,
                )

                validity &= raster_validity

            except Exception:
                # Some in-memory GeoTIFFs may not expose useful masks.
                pass

    _validate_sar_array(arr)

    invalid_count = int((~validity).sum())

    logger.debug(
        "Decoded SAR GeoTIFF: shape=%s invalid_pixels=%d",
        arr.shape,
        invalid_count,
    )

    if return_validity:
        return arr, validity

    return arr


# ============================================================================
# SAR normalization
# ============================================================================

def normalize_sar_tensor(
    arr: np.ndarray,
    is_linear: bool = False,
    return_validity: bool = False,
):
    """
    Convert SAR to dB when needed and apply locked per-band normalization.

    Parameters
    ----------
    arr:
        SAR array of shape (2, H, W).

    is_linear:
        True:
            Input is linear power (CDSE).

        False:
            Input is already sigma0 dB (TUM).

    return_validity:
        If True, return a boolean validity mask alongside the normalized array.

    Returns
    -------
    normalized:
        Float32 array in [0, 1], shape (2, H, W).

    validity:
        Boolean array (H, W) when return_validity=True.

    Processing
    ----------
    TUM:
        dB
        ↓
        validate finite pixels
        ↓
        per-band clipping
        ↓
        per-band [0,1]

    CDSE:
        linear power
        ↓
        finite + positive validity check
        ↓
        10*log10
        ↓
        per-band clipping
        ↓
        per-band [0,1]

    Invalid pixels:
        normalized to 0.0
        validity mask = False
    """

    _validate_sar_array(arr)

    arr = np.asarray(
        arr,
        dtype=np.float32,
    )

    # ------------------------------------------------------------------
    # Determine validity
    # ------------------------------------------------------------------

    if is_linear:

        # Linear SAR power should be finite and strictly positive.
        validity = np.all(
            np.isfinite(arr),
            axis=0,
        ) & np.all(
            arr > 0.0,
            axis=0,
        )

    else:

        # TUM data is already in dB.
        validity = np.all(
            np.isfinite(arr),
            axis=0,
        )

    # ------------------------------------------------------------------
    # Convert linear -> dB only for valid positive values
    # ------------------------------------------------------------------

    if is_linear:

        safe = np.where(
            np.isfinite(arr) & (arr > 0.0),
            np.maximum(arr, _LINEAR_EPS),
            _LINEAR_EPS,
        )

        arr_db = (
            10.0
            * np.log10(safe)
        ).astype(
            np.float32,
            copy=False,
        )

    else:

        arr_db = arr.astype(
            np.float32,
            copy=False,
        )

    # ------------------------------------------------------------------
    # Per-band normalization
    # ------------------------------------------------------------------

    normalized = np.zeros_like(
        arr_db,
        dtype=np.float32,
    )

    # VV
    vv = np.clip(
        arr_db[0],
        VV_MIN_DB,
        VV_MAX_DB,
    )

    normalized[0] = (
        (vv - VV_MIN_DB)
        / (VV_MAX_DB - VV_MIN_DB)
    )

    # VH
    vh = np.clip(
        arr_db[1],
        VH_MIN_DB,
        VH_MAX_DB,
    )

    normalized[1] = (
        (vh - VH_MIN_DB)
        / (VH_MAX_DB - VH_MIN_DB)
    )

    # ------------------------------------------------------------------
    # Explicitly zero invalid pixels.
    # ------------------------------------------------------------------

    normalized[:, ~validity] = 0.0

    normalized = normalized.astype(
        np.float32,
        copy=False,
    )

    logger.debug(
        "SAR normalization complete: "
        "shape=%s VV_range=[%.4f, %.4f] "
        "VH_range=[%.4f, %.4f] "
        "valid_pixels=%d invalid_pixels=%d",
        normalized.shape,
        float(normalized[0].min()),
        float(normalized[0].max()),
        float(normalized[1].min()),
        float(normalized[1].max()),
        int(validity.sum()),
        int((~validity).sum()),
    )

    if return_validity:
        return normalized, validity

    return normalized


# ============================================================================
# PyTorch conversion
# ============================================================================

def to_torch_tensor(arr: np.ndarray):
    """
    Convert a (C, H, W) NumPy array to torch.float32.
    """

    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for tensor conversion."
        ) from exc

    _validate_sar_array(arr)

    contiguous = np.ascontiguousarray(
        arr,
        dtype=np.float32,
    )

    return torch.from_numpy(
        contiguous
    )


# ============================================================================
# Full inference pipeline
# ============================================================================

def load_sar_pair_for_inference(
    t1_bytes: bytes,
    t2_bytes: bytes,
    is_linear: bool = True,
    return_tensors: bool = True,
    return_validity_mask: bool = False,
):
    """
    Decode, normalize and convert a Sentinel-1 T1/T2 pair.

    Parameters
    ----------
    t1_bytes:
        Raw T1 GeoTIFF bytes.

    t2_bytes:
        Raw T2 GeoTIFF bytes.

    is_linear:
        True for CDSE linear-power responses.
        False for already-dB TUM-like inputs.

    return_tensors:
        Return torch tensors if True.

    return_validity_mask:
        If True, return T1/T2 validity masks as well.

    Returns
    -------
    If return_validity_mask=False:
        (t1, t2)

    If return_validity_mask=True:
        (t1, t2, valid_t1, valid_t2)

    Tensor/NumPy type depends on return_tensors.
    """

    # ---------------------------------------------------------------
    # Decode
    # ---------------------------------------------------------------

    t1_raw, t1_valid_decode = decode_geotiff_response(
        t1_bytes,
        return_validity=True,
    )

    t2_raw, t2_valid_decode = decode_geotiff_response(
        t2_bytes,
        return_validity=True,
    )

    # ---------------------------------------------------------------
    # Normalize
    # ---------------------------------------------------------------

    t1_norm, t1_valid = normalize_sar_tensor(
        t1_raw,
        is_linear=is_linear,
        return_validity=True,
    )

    t2_norm, t2_valid = normalize_sar_tensor(
        t2_raw,
        is_linear=is_linear,
        return_validity=True,
    )

    # Combine decoder-level and preprocessing-level validity.
    t1_valid &= t1_valid_decode
    t2_valid &= t2_valid_decode

    # ---------------------------------------------------------------
    # Return NumPy if requested
    # ---------------------------------------------------------------

    if not return_tensors:

        if return_validity_mask:
            return (
                t1_norm,
                t2_norm,
                t1_valid,
                t2_valid,
            )

        return t1_norm, t2_norm

    # ---------------------------------------------------------------
    # Convert to torch
    # ---------------------------------------------------------------

    t1_tensor = to_torch_tensor(
        t1_norm
    )

    t2_tensor = to_torch_tensor(
        t2_norm
    )

    if not return_validity_mask:
        return (
            t1_tensor,
            t2_tensor,
        )

    import torch

    t1_valid_tensor = torch.from_numpy(
        np.ascontiguousarray(
            t1_valid,
            dtype=np.bool_,
        )
    )

    t2_valid_tensor = torch.from_numpy(
        np.ascontiguousarray(
            t2_valid,
            dtype=np.bool_,
        )
    )

    return (
        t1_tensor,
        t2_tensor,
        t1_valid_tensor,
        t2_valid_tensor,
    )