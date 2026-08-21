# src/data_ingestion/__init__.py
"""
Data ingestion package for the SIH-2026 Backend.

Provides clients for fetching Sentinel-1 SAR imagery from the
Copernicus Data Space Ecosystem (CDSE) Processing API.
"""

from data_ingestion.sentinel_client import (
    CDSEAuthManager,
    SentinelHubClient,
    SentinelAPIError,
    SentinelSceneNotFoundError,
    fetch_sentinel1_pair,
)

__all__ = [
    "CDSEAuthManager",
    "SentinelHubClient",
    "SentinelAPIError",
    "SentinelSceneNotFoundError",
    "fetch_sentinel1_pair",
]
