"""
src/api/routers/detect.py
=========================
Change Detection endpoints:
- Automated Sentinel-1 SAR change detection from coordinates
- Custom uploaded image pair change detection
"""

from __future__ import annotations

import io
import time
from typing import Optional
import numpy as np
from PIL import Image
import torch
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from src.api.schemas import (
    ChangeDetectionResponse,
    DetectSentinelRequest,
)
from src.data_ingestion.sentinel_client import (
    fetch_sentinel1_pair,
    SentinelAPIError,
)
from src.api.services.model_service import ModelService
from src.api.services.change_analyzer import extract_changed_regions
from src.api.services.visualization import (
    array_to_base64_png,
    generate_change_mask_image,
    generate_heatmap_image,
    generate_overlay_image,
    sar_dualpol_to_rgb,
)

router = APIRouter(prefix="/detect", tags=["Change Detection"])


@router.post(
    "/sentinel",
    response_model=ChangeDetectionResponse,
    summary="End-to-End Sentinel-1 Change Intelligence",
)
async def detect_sentinel_changes(req: DetectSentinelRequest) -> ChangeDetectionResponse:
    """
    Automated Sentinel-1 Change Detection:
    1. Fetches dual-polarization SAR imagery for T1 and T2 from Copernicus CDSE.
    2. Runs forward inference through Siamese Change Detection neural network.
    3. Identifies changed clusters, geographic coordinates, and severity levels.
    4. Generates false-color SAR previews, binary masks, and confidence heatmaps.
    """
    t0 = time.perf_counter()
    try:
        # Step 1: Ingestion from CDSE
        t1_np, t2_np = fetch_sentinel1_pair(
            bbox=req.bbox.to_list(),
            date_t1_range=req.date_range_t1,
            date_t2_range=req.date_range_t2,
            output_resolution=req.resolution,
        )
        # fetch_sentinel1_pair returns normalized float32 np.ndarray (2, H, W)
        t1_tensor = torch.from_numpy(t1_np)
        t2_tensor = torch.from_numpy(t2_np)

        # Step 2: Model Inference
        model_service = ModelService.get_instance()
        prob_map, binary_mask = model_service.predict_change_sar(
            t1_tensor=t1_tensor,
            t2_tensor=t2_tensor,
            model_name=req.model_name,
            threshold=req.threshold,
        )

        # Step 3: Intelligence & Region Analysis
        regions, total_changed_km2 = extract_changed_regions(
            binary_mask=binary_mask,
            prob_map=prob_map,
            min_region_area_px=req.min_region_area_px,
            bbox=req.bbox.to_list(),
        )

        # Step 4: Visualizations
        t1_rgb = sar_dualpol_to_rgb(t1_np)
        t2_rgb = sar_dualpol_to_rgb(t2_np)
        mask_rgb = generate_change_mask_image(binary_mask)
        heatmap_rgb = generate_heatmap_image(prob_map, colormap_name="turbo")
        overlay_rgb = generate_overlay_image(t2_rgb, binary_mask)

        t1_b64 = array_to_base64_png(t1_rgb)
        t2_b64 = array_to_base64_png(t2_rgb)
        mask_b64 = array_to_base64_png(mask_rgb)
        heatmap_b64 = array_to_base64_png(heatmap_rgb)
        overlay_b64 = array_to_base64_png(overlay_rgb)

        total_px = binary_mask.size
        changed_px = int(np.sum(binary_mask > 0))
        change_pct = round((changed_px / total_px) * 100.0, 3)
        elapsed = round(time.perf_counter() - t0, 3)

        return ChangeDetectionResponse(
            status="success",
            model_used=req.model_name,
            threshold=req.threshold,
            total_pixels=total_px,
            changed_pixels=changed_px,
            change_percentage=change_pct,
            total_changed_area_sq_km=total_changed_km2,
            num_change_clusters=len(regions),
            regions=regions,
            t1_preview_base64=t1_b64,
            t2_preview_base64=t2_b64,
            change_mask_base64=mask_b64,
            confidence_heatmap_base64=heatmap_b64,
            overlay_base64=overlay_b64,
            execution_time_sec=elapsed,
        )

    except SentinelAPIError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Sentinel API error: {e}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Change detection failed: {e}",
        )


@router.post(
    "/upload",
    response_model=ChangeDetectionResponse,
    summary="Change Detection on Uploaded Image Pair",
)
async def detect_uploaded_images(
    image_t1: UploadFile = File(..., description="Reference (T1) image file (PNG/JPEG)"),
    image_t2: UploadFile = File(..., description="Target (T2) image file (PNG/JPEG)"),
    model_name: str = Form("siamese_unet", description="Model architecture: 'siamese_unet' or 'snunet_cd'"),
    threshold: float = Form(0.5, ge=0.0, le=1.0, description="Decision threshold"),
    min_region_area_px: int = Form(10, ge=1, description="Minimum cluster pixel area"),
) -> ChangeDetectionResponse:
    """
    Change Detection on custom uploaded image pair (RGB or Grayscale).
    """
    t0 = time.perf_counter()
    try:
        t1_bytes = await image_t1.read()
        t2_bytes = await image_t2.read()

        pil_t1 = Image.open(io.BytesIO(t1_bytes)).convert("RGB")
        pil_t2 = Image.open(io.BytesIO(t2_bytes)).convert("RGB")

        # Standardize size (ensure same dimensions)
        if pil_t1.size != pil_t2.size:
            pil_t2 = pil_t2.resize(pil_t1.size, Image.Resampling.BILINEAR)

        # Convert to tensor [3, H, W] in [0, 1]
        t1_np = np.array(pil_t1, dtype=np.float32) / 255.0
        t2_np = np.array(pil_t2, dtype=np.float32) / 255.0

        t1_tensor = torch.from_numpy(t1_np).permute(2, 0, 1)
        t2_tensor = torch.from_numpy(t2_np).permute(2, 0, 1)

        model_service = ModelService.get_instance()
        prob_map, binary_mask = model_service.predict_change_rgb(
            t1_tensor=t1_tensor,
            t2_tensor=t2_tensor,
            model_name=model_name,
            threshold=threshold,
        )

        regions, _ = extract_changed_regions(
            binary_mask=binary_mask,
            prob_map=prob_map,
            min_region_area_px=min_region_area_px,
        )

        mask_rgb = generate_change_mask_image(binary_mask)
        heatmap_rgb = generate_heatmap_image(prob_map, colormap_name="turbo")
        overlay_rgb = generate_overlay_image((t2_np * 255).astype(np.uint8), binary_mask)

        t1_b64 = array_to_base64_png((t1_np * 255).astype(np.uint8))
        t2_b64 = array_to_base64_png((t2_np * 255).astype(np.uint8))
        mask_b64 = array_to_base64_png(mask_rgb)
        heatmap_b64 = array_to_base64_png(heatmap_rgb)
        overlay_b64 = array_to_base64_png(overlay_rgb)

        total_px = binary_mask.size
        changed_px = int(np.sum(binary_mask > 0))
        change_pct = round((changed_px / total_px) * 100.0, 3)
        elapsed = round(time.perf_counter() - t0, 3)

        return ChangeDetectionResponse(
            status="success",
            model_used=model_name,
            threshold=threshold,
            total_pixels=total_px,
            changed_pixels=changed_px,
            change_percentage=change_pct,
            total_changed_area_sq_km=None,
            num_change_clusters=len(regions),
            regions=regions,
            t1_preview_base64=t1_b64,
            t2_preview_base64=t2_b64,
            change_mask_base64=mask_b64,
            confidence_heatmap_base64=heatmap_b64,
            overlay_base64=overlay_b64,
            execution_time_sec=elapsed,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference on uploaded images failed: {e}",
        )
