"""
src/api/main.py
===============
FastAPI Application entry point for the SAR Change Intelligence System.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

# Ensure repo root is loaded and .env is present
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(REPO_ROOT / ".env")

from src.api.routers.cdse import router as cdse_router
from src.api.routers.detect import router as detect_router
from src.api.routers.health import router as health_router
from src.api.routers.models import router as models_router
from src.api.services.model_service import ModelService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up model services and check CDSE credentials upon startup."""
    logger.info("Initializing SAR Change Intelligence Server...")
    
    # Warm up models
    ModelService.get_instance()
    
    cdse_id = os.environ.get("CDSE_CLIENT_ID")
    if cdse_id:
        logger.info(f"CDSE Credentials detected for client: {cdse_id[:8]}...")
    else:
        logger.warning("CDSE_CLIENT_ID not found in environment!")
        
    logger.info("API Server ready to accept requests.")
    yield
    logger.info("Shutting down SAR Change Intelligence Server...")


app = FastAPI(
    title="SAR Change Intelligence API",
    description=(
        "Production REST API for automated Synthetic Aperture Radar (SAR) and optical "
        "satellite Change Detection. Provides Sentinel-1 GRD ingestion from Copernicus CDSE, "
        "Siamese Neural Network inference (Siamese U-Net / SNUNet-CD), cluster extraction, "
        "and calibrated intelligence scoring."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Enable CORS for frontend clients (React, Vite, Next.js, Streamlit, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount sub-routers under /api/v1
API_V1_PREFIX = "/api/v1"
app.include_router(health_router, prefix=API_V1_PREFIX)
app.include_router(cdse_router, prefix=API_V1_PREFIX)
app.include_router(detect_router, prefix=API_V1_PREFIX)
app.include_router(models_router, prefix=API_V1_PREFIX)

# Also expose /health at root level for load balancers
app.include_router(health_router)


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root path to interactive Swagger documentation."""
    return RedirectResponse(url="/docs")
