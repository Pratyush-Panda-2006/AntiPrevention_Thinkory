"""
src/api/routers/cdse.py
=======================
Endpoints for interacting with Copernicus Data Space Ecosystem (CDSE)
Sentinel-1 SAR ingestion.
"""

from __future__ import annotations

import os
import time
from typing import Dict
from fastapi import APIRouter, HTTPException, status

from src.api.schemas import (
    BandStats,
    SentinelFetchRequest,
    SentinelFetchResponse,
    TileStats,
)
import numpy as np
from src.data_ingestion.sentinel_client import (
    CDSEAuthManager,
    fetch_sentinel1_pair,
    SentinelAPIError,
)
from src.api.services.visualization import array_to_base64_png, sar_dualpol_to_rgb

router = APIRouter(prefix="/cdse", tags=["Copernicus Sentinel-1 Ingestion"])


def _calculate_band_stats(band_data) -> BandStats:
    import numpy as np
    clean = np.nan_to_num(band_data, nan=0.0)
    return BandStats(
        min=round(float(np.min(clean)), 4),
        max=round(float(np.max(clean)), 4),
        mean=round(float(np.mean(clean)), 4),
        std=round(float(np.std(clean)), 4),
    )


@router.get("/auth-status", summary="Check CDSE Authentication Status")
async def get_auth_status() -> Dict[str, object]:
    """Check if CDSE credentials are configured and test OAuth2 token retrieval."""
    client_id = os.environ.get("CDSE_CLIENT_ID", "")
    client_secret = os.environ.get("CDSE_CLIENT_SECRET", "")

    if not client_id or not client_secret:
        return {
            "status": "unconfigured",
            "authenticated": False,
            "message": "CDSE_CLIENT_ID or CDSE_CLIENT_SECRET is missing from environment.",
        }

    try:
        auth_mgr = CDSEAuthManager()
        t0 = time.perf_counter()
        token = auth_mgr.get_token()
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "status": "authenticated",
            "authenticated": True,
            "latency_ms": latency_ms,
            "token_prefix": token[:12] + "...",
            "expires_in_seconds": auth_mgr.expires_in,
        }
    except Exception as e:
        return {
            "status": "error",
            "authenticated": False,
            "error": str(e),
        }


@router.post(
    "/fetch-pair",
    response_model=SentinelFetchResponse,
    summary="Fetch Sentinel-1 SAR Pair from CDSE",
)
async def fetch_sar_pair(req: SentinelFetchRequest) -> SentinelFetchResponse:
    """
    Fetch dual-polarization (VV/VH) Sentinel-1 GRD SAR imagery for T1 and T2 date ranges,
    normalize to [0, 1] dB range, and generate false-color previews with channel statistics.
    """
    try:
        t1_np, t2_np = fetch_sentinel1_pair(
            bbox=req.bbox.to_list(),
            date_t1_range=req.date_range_t1,
            date_t2_range=req.date_range_t2,
            output_resolution=req.resolution,
        )
        # fetch_sentinel1_pair returns normalized float32 np.ndarray (2, H, W)

        # Compute stats for VV (index 0) and VH (index 1)
        t1_stats = TileStats(
            vv=_calculate_band_stats(t1_np[0]),
            vh=_calculate_band_stats(t1_np[1]),
        )
        t2_stats = TileStats(
            vv=_calculate_band_stats(t2_np[0]),
            vh=_calculate_band_stats(t2_np[1]),
        )

        # False-color RGB images
        t1_rgb = sar_dualpol_to_rgb(t1_np)
        t2_rgb = sar_dualpol_to_rgb(t2_np)

        t1_b64 = array_to_base64_png(t1_rgb)
        t2_b64 = array_to_base64_png(t2_rgb)

        return SentinelFetchResponse(
            status="success",
            bbox=req.bbox.to_list(),
            date_range_t1=req.date_range_t1,
            date_range_t2=req.date_range_t2,
            resolution=req.resolution,
            t1_stats=t1_stats,
            t2_stats=t2_stats,
            t1_preview_base64=t1_b64,
            t2_preview_base64=t2_b64,
        )

    except SentinelAPIError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"CDSE Sentinel API error: {e}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch SAR pair: {e}",
        )
