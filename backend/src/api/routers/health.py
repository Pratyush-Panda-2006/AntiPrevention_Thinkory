"""
src/api/routers/health.py
=========================
System health, GPU/VRAM statistics, and environment status endpoints.
"""

from __future__ import annotations

import torch
from fastapi import APIRouter

from src.api.schemas import HealthResponse
from src.api.services.model_service import ModelService
from src import __version__ as pkg_version

router = APIRouter(tags=["Health & Diagnostics"])


@router.get("/health", response_model=HealthResponse, summary="API Health Check")
@router.get("/status", response_model=HealthResponse, summary="System Status")
async def get_health_status() -> HealthResponse:
    """
    Returns current API health, PyTorch version, CUDA status, and loaded models.
    """
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"

    vram_total = None
    vram_used = None
    if cuda_avail:
        vram_total = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        vram_used = round(torch.cuda.memory_allocated(0) / (1024**3), 2)

    model_service = ModelService.get_instance()
    loaded_models = list(model_service._models.keys())

    return HealthResponse(
        status="healthy",
        app_version=str(pkg_version),
        torch_version=torch.__version__,
        cuda_available=cuda_avail,
        device_name=device_name,
        vram_total_gb=vram_total,
        vram_used_gb=vram_used,
        loaded_models=loaded_models,
    )
