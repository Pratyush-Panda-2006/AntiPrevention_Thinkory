"""
src/preprocessing/sar_loader.py
================================
SAR data loading and normalization utilities for the Sentinel-1 ingestion
pipeline.

These functions bridge the raw GeoTIFF bytes returned by the CDSE Processing
API and the (C, H, W) float32 tensor format expected by the Siamese UNet /
SNUNet-CD models.

Public API
----------
    decode_geotiff_response(response_bytes) -> np.ndarray
        Parse raw GeoTIFF bytes → float32 NumPy array (C, H, W).

    normalize_sar_tensor(arr, clip_min_db, clip_max_db) -> np.ndarray
        Log-scale dB normalization + min-max to [0, 1].

    to_torch_tensor(arr) -> torch.Tensor
        Convert (C, H, W) NumPy → FloatTensor (no copy if already contiguous).

    load_sar_pair_for_inference(t1_bytes, t2_bytes, ...) -> tuple[Tensor, Tensor]
        Full decode → normalize → torch pipeline for a T1/T2 pair.

Normalization strategy
----------------------
Sentinel-1 linear power backscatter values are:
  1. Clipped to avoid log(0): values < 1e-10 are set to 1e-10.
  2. Converted to decibels: dB = 10 * log10(linear_power)
  3. Clipped to a physically sensible dB range [clip_min_db, clip_max_db].
     Default: [-30 dB, 0 dB] — covers typical vegetated/urban surfaces.
  4. Min-max scaled to [0.0, 1.0].

This matches common SAR deep-learning preprocessing (e.g., used in
SpaceNet 6, BigEarth-SAR, and similar benchmarks).
"""

from __future__ import annotations

import io
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default dB clipping range for Sentinel-1 GRD GAMMA0 backscatter.
# VV: typically -18 to 0 dB (land); VH: -25 to -5 dB (land).
# Using -30 / 0 is deliberately wide to be safe across both polarisations
# and across diverse land-cover types.
_DEFAULT_CLIP_MIN_DB: float = -30.0
_DEFAULT_CLIP_MAX_DB: float = 0.0

# Minimum linear power threshold to avoid log(0)
_LINEAR_EPS: float = 1e-10


# ── GeoTIFF decoder ────────────────────────────────────────────────────────────

def decode_geotiff_response(response_bytes: bytes) -> np.ndarray:
    """
    Parse raw GeoTIFF bytes from the CDSE Processing API into a NumPy array.

    The Sentinel Hub evalscript in ``sentinel_client.py`` returns a 2-band
    float32 GeoTIFF where:
        Band 1 → VV backscatter (linear power)
        Band 2 → VH backscatter (linear power)

    Parameters
    ----------
    response_bytes : bytes
        Raw GeoTIFF content as returned by ``requests.Response.content``.

    Returns
    -------
    np.ndarray
        Shape ``(C, H, W)`` with ``C=2``, dtype ``float32``.
        Values are **linear power** (not yet in dB or normalised).

    Raises
    ------
    ImportError
        If ``rasterio`` is not installed.
    ValueError
        If the GeoTIFF cannot be read or has unexpected band count.
    """
    try:
        import rasterio
        from rasterio.io import MemoryFile
    except ImportError as exc:
        raise ImportError(
            "rasterio is required for GeoTIFF decoding. "
            "Install it with: pip install rasterio"
        ) from exc

    with MemoryFile(response_bytes) as mem_file:
        with mem_file.open() as dataset:
            n_bands = dataset.count
            if n_bands != 2:
                raise ValueError(
                    f"Expected 2 bands (VV, VH) in GeoTIFF, got {n_bands}. "
                    "Check the evalscript in sentinel_client.py."
                )

            # rasterio returns (bands, rows, cols) — already (C, H, W)
            arr = dataset.read().astype(np.float32)

    n_nan = int(np.sum(np.isnan(arr)))
    if n_nan > 0:
        logger.debug(
            "GeoTIFF decoded: shape=%s  (%d NaN no-data pixels present — "
            "will be zeroed during normalization)",
            arr.shape, n_nan,
        )
    else:
        logger.debug(
            "GeoTIFF decoded: shape=%s  linear_range=[%.4e, %.4e]",
            arr.shape, float(np.nanmin(arr)), float(np.nanmax(arr)),
        )
    return arr  # (2, H, W), float32, linear power


# ── Normalization ──────────────────────────────────────────────────────────────

def normalize_sar_tensor(
    arr: np.ndarray,
    clip_min_db: float = _DEFAULT_CLIP_MIN_DB,
    clip_max_db: float = _DEFAULT_CLIP_MAX_DB,
) -> np.ndarray:
    """
    Normalize a linear-power SAR array to ``[0, 1]`` via dB conversion.

    Steps
    -----
    1. Clip values below ``_LINEAR_EPS`` (avoids ``log(0)``).
    2. Convert to decibels: ``dB = 10 * log10(linear)``.
    3. Clip the dB values to ``[clip_min_db, clip_max_db]``.
    4. Min-max normalize to ``[0.0, 1.0]``.

    Parameters
    ----------
    arr : np.ndarray
        Input array with shape ``(C, H, W)`` in **linear power** units.
    clip_min_db : float
        Lower dB clipping bound. Default: ``-30.0`` dB.
    clip_max_db : float
        Upper dB clipping bound. Default: ``0.0`` dB.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(C, H, W)`` with values in ``[0.0, 1.0]``.

    Notes
    -----
    The clipping bounds apply **independently** to each band so that both
    VV and VH are individually well-normalized despite their different
    typical dynamic ranges.
    """
    if arr.ndim != 3:
        raise ValueError(
            f"Expected 3-D array (C, H, W), got shape {arr.shape}."
        )

    # Step 0: Replace NaN / Inf arising from SAR no-data / swath edges.
    # No-data pixels (NaN) map to 0.0 linear power, which is below _LINEAR_EPS
    # and therefore clips to clip_min_db → normalizes to 0.0 (black background).
    n_nan = int(np.sum(np.isnan(arr)))
    n_inf = int(np.sum(np.isinf(arr)))
    if n_nan > 0 or n_inf > 0:
        logger.debug(
            "SAR array contains %d NaN and %d Inf values "
            "(swath edges / no-data). Replacing with 0.0.",
            n_nan, n_inf,
        )
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # Step 1: Floor at epsilon
    arr_safe = np.clip(arr, _LINEAR_EPS, None)

    # Step 2: Convert to dB
    arr_db = 10.0 * np.log10(arr_safe)

    # Step 3: Clip to [min_db, max_db]
    arr_db = np.clip(arr_db, clip_min_db, clip_max_db)

    # Step 4: Min-max scale per-band to [0, 1]
    db_range = clip_max_db - clip_min_db
    arr_norm = (arr_db - clip_min_db) / db_range

    # Ensure float32 output
    arr_norm = arr_norm.astype(np.float32)

    logger.debug(
        "SAR normalized: shape=%s  dB_range=[%.1f, %.1f]  "
        "output_range=[%.4f, %.4f]",
        arr_norm.shape, clip_min_db, clip_max_db,
        float(arr_norm.min()), float(arr_norm.max()),
    )
    return arr_norm


# ── PyTorch conversion ─────────────────────────────────────────────────────────

def to_torch_tensor(arr: np.ndarray):
    """
    Convert a ``(C, H, W)`` float32 NumPy array to a PyTorch FloatTensor.

    Parameters
    ----------
    arr : np.ndarray
        Shape ``(C, H, W)``, dtype ``float32``.

    Returns
    -------
    torch.Tensor
        Shape ``(C, H, W)``, dtype ``torch.float32``.

    Notes
    -----
    Uses ``torch.from_numpy`` which shares memory when the array is
    C-contiguous — no unnecessary copy.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for tensor conversion. "
            "Install it with: pip install torch"
        ) from exc

    if arr.ndim != 3:
        raise ValueError(
            f"Expected 3-D array (C, H, W), got shape {arr.shape}."
        )

    contiguous = np.ascontiguousarray(arr, dtype=np.float32)
    return torch.from_numpy(contiguous)


# ── Full pipeline ──────────────────────────────────────────────────────────────

def load_sar_pair_for_inference(
    t1_bytes: bytes,
    t2_bytes: bytes,
    clip_min_db: float = _DEFAULT_CLIP_MIN_DB,
    clip_max_db: float = _DEFAULT_CLIP_MAX_DB,
    return_tensors: bool = True,
):
    """
    Full decode → normalize → (optionally) torch pipeline for a T1/T2 pair.

    This is the convenience entry-point for the inference pipeline.  It
    takes the raw GeoTIFF byte-strings returned by ``SentinelHubClient``
    and produces tensors ready to be passed directly into a
    ``SiameseUNet`` or ``SNUNetCD`` model.

    Parameters
    ----------
    t1_bytes : bytes
        Raw GeoTIFF response for the T1 (before) image.
    t2_bytes : bytes
        Raw GeoTIFF response for the T2 (after) image.
    clip_min_db : float
        Lower dB clipping bound for normalization. Default: ``-30.0``.
    clip_max_db : float
        Upper dB clipping bound for normalization. Default: ``0.0``.
    return_tensors : bool
        If ``True`` (default), return ``torch.Tensor`` objects.
        If ``False``, return float32 NumPy arrays.

    Returns
    -------
    tuple
        ``(t1, t2)`` where each element has shape ``(C, H, W)`` with
        ``C=2`` (VV=index 0, VH=index 1).

        Types:
            - ``torch.Tensor`` when ``return_tensors=True``
            - ``np.ndarray`` (float32) when ``return_tensors=False``

    Example
    -------
    >>> t1_tensor, t2_tensor = load_sar_pair_for_inference(t1_bytes, t2_bytes)
    >>> t1_tensor.shape
    torch.Size([2, 512, 512])
    >>> model = SiameseUNet(in_channels=2)
    >>> logits = model(t1_tensor.unsqueeze(0), t2_tensor.unsqueeze(0))
    """
    # Decode
    t1_raw = decode_geotiff_response(t1_bytes)
    t2_raw = decode_geotiff_response(t2_bytes)

    # Normalize
    t1_norm = normalize_sar_tensor(t1_raw, clip_min_db, clip_max_db)
    t2_norm = normalize_sar_tensor(t2_raw, clip_min_db, clip_max_db)

    if not return_tensors:
        return t1_norm, t2_norm

    # Convert to torch
    t1_tensor = to_torch_tensor(t1_norm)
    t2_tensor = to_torch_tensor(t2_norm)

    return t1_tensor, t2_tensor
