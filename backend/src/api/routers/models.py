"""
src/api/routers/models.py
=========================
Model catalog and architecture discovery endpoints.
"""

from __future__ import annotations

from typing import List
from fastapi import APIRouter

from src.api.schemas import ModelInfo
from src.api.services.model_service import ModelService

router = APIRouter(prefix="/models", tags=["Models & Architecture"])


@router.get("", response_model=List[ModelInfo], summary="List Available Models")
async def list_models() -> List[ModelInfo]:
    """Returns the catalog of registered and pre-loaded change detection models."""
    service = ModelService.get_instance()
    return service.get_available_models()
