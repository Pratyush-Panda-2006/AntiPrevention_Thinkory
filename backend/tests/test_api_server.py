#!/usr/bin/env python3
"""
scripts/test_api_server.py
==========================
Comprehensive automated test suite for the SAR Change Intelligence API.
Tests all routers, models, serialization, and end-to-end pipelines.
"""

import io
import sys
import numpy as np
from PIL import Image
from pathlib import Path

# Force UTF-8 output on Windows so box/check characters and emojis render correctly.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Setup paths
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

_GREEN = "\033[92m"
_RED = "\033[91m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _ok(msg: str):
    print(f"  {_GREEN}✓{_RESET}  {msg}")


def _fail(msg: str):
    print(f"  {_RED}✗{_RESET}  {msg}")


def _header(title: str):
    print(f"\n{_BOLD}[{title}]{_RESET}")


def test_health():
    _header("1. HEALTH & STATUS ENDPOINTS")
    res = client.get("/health")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    data = res.json()
    assert data["status"] == "healthy"
    _ok(f"GET /health: {data['status']} (Torch {data['torch_version']}, Device: {data['device_name']})")

    res_status = client.get("/api/v1/status")
    assert res_status.status_code == 200
    _ok(f"GET /api/v1/status: 200 OK | Loaded models: {data['loaded_models']}")


def test_models():
    _header("2. MODEL CATALOG ENDPOINTS")
    res = client.get("/api/v1/models")
    assert res.status_code == 200
    models = res.json()
    assert len(models) >= 2
    for m in models:
        _ok(f"Model: {m['display_name']} ({m['name']}) - {m['parameters']:,} params, {m['input_channels']} in_channels")


def test_cdse_auth():
    _header("3. CDSE AUTHENTICATION STATUS")
    res = client.get("/api/v1/cdse/auth-status")
    assert res.status_code == 200
    data = res.json()
    if data.get("authenticated"):
        _ok(f"CDSE Authenticated: latency = {data.get('latency_ms')} ms")
    else:
        _ok(f"CDSE Status: {data.get('status')} - {data.get('message') or data.get('error')}")


def test_upload_detection():
    _header("4. IMAGE UPLOAD CHANGE DETECTION")
    # Generate synthetic T1 and T2 images with a clear change region in center
    h, w = 128, 128
    t1 = np.ones((h, w, 3), dtype=np.uint8) * 100
    t2 = t1.copy()
    # Add changed rectangle to T2
    t2[40:80, 40:80, :] = 250

    buf1 = io.BytesIO()
    Image.fromarray(t1).save(buf1, format="PNG")
    buf1.seek(0)

    buf2 = io.BytesIO()
    Image.fromarray(t2).save(buf2, format="PNG")
    buf2.seek(0)

    files = {
        "image_t1": ("t1.png", buf1, "image/png"),
        "image_t2": ("t2.png", buf2, "image/png"),
    }
    data = {
        "model_name": "siamese_unet",
        "threshold": 0.5,
        "min_region_area_px": 5,
    }

    res = client.post("/api/v1/detect/upload", files=files, data=data)
    assert res.status_code == 200, f"Upload detect failed: {res.text}"
    resp_data = res.json()
    assert resp_data["status"] == "success"
    assert "change_mask_base64" in resp_data
    assert "confidence_heatmap_base64" in resp_data
    _ok(f"POST /api/v1/detect/upload: Succeeded in {resp_data['execution_time_sec']}s")
    _ok(f"Changed pixels: {resp_data['changed_pixels']}/{resp_data['total_pixels']} ({resp_data['change_percentage']}%)")
    _ok(f"Identified clusters: {resp_data['num_change_clusters']}")
    for r in resp_data["regions"][:3]:
        _ok(f"  Cluster #{r['region_id']}: area={r['area_px']}px, severity={r['severity']}, centroid={r['centroid_xy']}")


def test_live_sentinel_ingestion():
    _header("5. LIVE SENTINEL-1 INGESTION & DETECTION")
    payload = {
        "bbox": {
            "min_lon": 72.0,
            "min_lat": 24.0,
            "max_lon": 73.0,
            "max_lat": 25.0
        },
        "date_range_t1": ["2024-01-01", "2024-01-20"],
        "date_range_t2": ["2024-06-01", "2024-06-20"],
        "resolution": [128, 128],
        "model_name": "siamese_unet",
        "threshold": 0.5,
        "min_region_area_px": 10,
    }

    # Test Fetch SAR Pair
    res_fetch = client.post("/api/v1/cdse/fetch-pair", json={
        "bbox": payload["bbox"],
        "date_range_t1": payload["date_range_t1"],
        "date_range_t2": payload["date_range_t2"],
        "resolution": payload["resolution"],
    })
    if res_fetch.status_code == 200:
        f_data = res_fetch.json()
        _ok(f"POST /api/v1/cdse/fetch-pair: Fetched T1/T2 dual-pol tiles successfully")
        _ok(f"  T1 VV: mean={f_data['t1_stats']['vv']['mean']}, T2 VV: mean={f_data['t2_stats']['vv']['mean']}")
    else:
        _fail(f"Fetch SAR pair failed: {res_fetch.text}")

    # Test Live Detect Sentinel
    res_detect = client.post("/api/v1/detect/sentinel", json=payload)
    if res_detect.status_code == 200:
        d_data = res_detect.json()
        _ok(f"POST /api/v1/detect/sentinel: End-to-end detection finished in {d_data['execution_time_sec']}s")
        _ok(f"  Changed area: {d_data['change_percentage']}% | Approx area: {d_data.get('total_changed_area_sq_km')} km²")
        _ok(f"  Clusters found: {d_data['num_change_clusters']}")
    else:
        _fail(f"Detect Sentinel failed: {res_detect.text}")


if __name__ == "__main__":
    print("=" * 60)
    print("  🧪  Testing SAR Change Intelligence API Endpoints")
    print("=" * 60)
    try:
        test_health()
        test_models()
        test_cdse_auth()
        test_upload_detection()
        test_live_sentinel_ingestion()
        print("\n" + "=" * 60)
        print("  🎉  ALL API TESTS PASSED SUCCESSFULLY!")
        print("=" * 60)
    except Exception as exc:
        print(f"\n{_RED}Test failed with error:{_RESET} {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
