from __future__ import annotations

from typing import List, Optional, Tuple
from pydantic import BaseModel, Field


class BBox(BaseModel):
    """Geographic Bounding Box in WGS84 [min_lon, min_lat, max_lon, max_lat]."""
    min_lon: float = Field(..., description="Westernmost longitude", ge=-180.0, le=180.0)
    min_lat: float = Field(..., description="Southernmost latitude", ge=-90.0, le=90.0)
    max_lon: float = Field(..., description="Easternmost longitude", ge=-180.0, le=180.0)
    max_lat: float = Field(..., description="Northernmost latitude", ge=-90.0, le=90.0)

    def to_list(self) -> List[float]:
        return [self.min_lon, self.min_lat, self.max_lon, self.max_lat]


class BandStats(BaseModel):
    min: float
    max: float
    mean: float
    std: float


class TileStats(BaseModel):
    vv: BandStats
    vh: BandStats


class SentinelFetchRequest(BaseModel):
    """Request to fetch and preview a Sentinel-1 SAR pair from CDSE."""
    bbox: BBox = Field(..., description="Geographic bounding box [min_lon, min_lat, max_lon, max_lat]")
    date_range_t1: Tuple[str, str] = Field(
        ("2024-01-01", "2024-01-20"),
        description="Reference (T1) date acquisition window (YYYY-MM-DD, YYYY-MM-DD)"
    )
    date_range_t2: Tuple[str, str] = Field(
        ("2024-06-01", "2024-06-20"),
        description="Target (T2) date acquisition window (YYYY-MM-DD, YYYY-MM-DD)"
    )
    resolution: Tuple[int, int] = Field(
        (256, 256),
        description="Tile resolution in pixels (height, width)"
    )


class SentinelFetchResponse(BaseModel):
    """Response containing fetched SAR tile metadata and preview images."""
    status: str = "success"
    bbox: List[float]
    date_range_t1: Tuple[str, str]
    date_range_t2: Tuple[str, str]
    resolution: Tuple[int, int]
    t1_stats: TileStats
    t2_stats: TileStats
    t1_preview_base64: str
    t2_preview_base64: str


class ChangedRegion(BaseModel):
    """Individual connected component cluster identified in the change mask."""
    region_id: int
    area_px: int
    approx_area_sq_km: Optional[float] = None
    centroid_xy: Tuple[float, float]
    bbox_xy: Tuple[int, int, int, int]  # (min_row, min_col, max_row, max_col)
    geo_bbox: Optional[Tuple[float, float, float, float]] = None # (min_lon, min_lat, max_lon, max_lat)
    mean_change_prob: float
    severity: str = Field("Medium", description="Severity level: Low, Medium, High, Critical, Uncertain")
    label: str = Field("Unclassified Change", description="Interpreted change classification label")


class DetectSentinelRequest(BaseModel):
    """Request to fetch Sentinel-1 data live from CDSE and run Change Detection."""
    bbox: BBox = Field(..., description="Geographic bounding box [min_lon, min_lat, max_lon, max_lat]")
    date_range_t1: Tuple[str, str] = Field(
        ("2024-01-01", "2024-01-20"),
        description="Reference (T1) date window"
    )
    date_range_t2: Tuple[str, str] = Field(
        ("2024-06-01", "2024-06-20"),
        description="Target (T2) date window"
    )
    resolution: Tuple[int, int] = Field(
        (256, 256),
        description="Tile resolution in pixels (height, width)"
    )
    model_name: str = Field(
        "siamese_unet",
        description="Model architecture to use: 'siamese_unet' or 'snunet_cd'"
    )
    threshold: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Decision threshold for binary classification"
    )
    min_region_area_px: int = Field(
        10,
        ge=1,
        description="Minimum area in pixels for a change cluster to be retained"
    )


class ChangeDetectionResponse(BaseModel):
    """Full Change Intelligence analysis output."""
    status: str = "success"
    model_used: str
    threshold: float
    total_pixels: int
    changed_pixels: int
    change_percentage: float
    total_changed_area_sq_km: Optional[float] = None
    num_change_clusters: int
    regions: List[ChangedRegion]
    t1_preview_base64: Optional[str] = None
    t2_preview_base64: Optional[str] = None
    change_mask_base64: str
    confidence_heatmap_base64: str
    overlay_base64: Optional[str] = None
    execution_time_sec: float


class HealthResponse(BaseModel):
    status: str = "healthy"
    app_version: str
    torch_version: str
    cuda_available: bool
    device_name: Optional[str] = None
    vram_total_gb: Optional[float] = None
    vram_used_gb: Optional[float] = None
    loaded_models: List[str]


class ModelInfo(BaseModel):
    name: str
    display_name: str
    input_channels: int
    parameters: int
    description: str
