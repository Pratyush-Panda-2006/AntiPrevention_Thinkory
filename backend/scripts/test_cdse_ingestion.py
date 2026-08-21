#!/usr/bin/env python3
"""
scripts/test_cdse_ingestion.py
===============================
Standalone verification script for the Sentinel-1 SAR ingestion pipeline.

What this script tests
-----------------------
1. [AUTH]   CDSE OAuth2 token retrieval from environment credentials.
2. [FETCH]  Sentinel-1 GRD tile fetch for T1 and T2 over a test AOI in India.
3. [DECODE] GeoTIFF → NumPy array decoding (shape, dtype, band count).
4. [NORM]   Normalization to [0, 1] (dB pipeline).
5. [TENSOR] Conversion to PyTorch FloatTensor.
6. [MODEL]  Compatibility check: forward pass through SiameseUNet(in_channels=2).
7. [PAIR]   High-level fetch_sentinel1_pair() convenience function.

Test AOI
---------
  Rajasthan / Gujarat border, western India.
  BBox (EPSG:4326): [72.0°E, 24.0°N, 73.0°E, 25.0°N]

  This region was chosen because:
    - High revisit frequency for Sentinel-1 IW mode
    - Open, non-sensitive geography
    - Low persistent cloud cover (dry region — good for SAR sanity check)

Usage
-----
    # From repository root:
    python scripts/test_cdse_ingestion.py

Requirements
------------
    - .env file with CDSE_CLIENT_ID and CDSE_CLIENT_SECRET
    - pip install -r requirements.txt

Exit codes
----------
    0  All checks passed
    1  One or more checks failed
"""

import io
import logging
import os
import sys
import time
import traceback
from pathlib import Path

# Force UTF-8 output on Windows so box/check characters render correctly.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Path setup ─────────────────────────────────────────────────────────────────
# Ensure src/ is importable when running from repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cdse_test")

# ── Test configuration ─────────────────────────────────────────────────────────
# Rajasthan / Gujarat border — open test region in India
TEST_BBOX = [72.0, 24.0, 73.0, 25.0]        # [W, S, E, N] EPSG:4326
TEST_DATE_T1 = ("2024-01-01", "2024-01-20")  # Dry season — good acquisition
TEST_DATE_T2 = ("2024-06-01", "2024-06-20")  # Pre-monsoon
TEST_RESOLUTION = (256, 256)                 # Smaller tile for quick smoke-test


# ── ANSI colour helpers ────────────────────────────────────────────────────────
_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _ok(msg: str) -> None:
    print(f"  {_GREEN}✓{_RESET}  {msg}")


def _fail(msg: str) -> None:
    print(f"  {_RED}✗{_RESET}  {msg}")


def _info(msg: str) -> None:
    print(f"  {_YELLOW}·{_RESET}  {msg}")


def _header(title: str) -> None:
    print(f"\n{_BOLD}[{title}]{_RESET}")


# ── Individual check functions ─────────────────────────────────────────────────

def check_environment() -> bool:
    """Verify that CDSE credentials are present in the environment."""
    _header("ENV")

    # Load .env if present
    try:
        from dotenv import load_dotenv
        env_path = REPO_ROOT / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            _ok(f".env loaded from {env_path}")
        else:
            _info(
                ".env not found — relying on shell environment variables. "
                f"Create it at {env_path} (see .env.example)"
            )
    except ImportError:
        _fail("python-dotenv not installed. Run: pip install python-dotenv")
        return False

    client_id = os.environ.get("CDSE_CLIENT_ID", "")
    client_secret = os.environ.get("CDSE_CLIENT_SECRET", "")

    if not client_id:
        _fail("CDSE_CLIENT_ID is not set.")
        return False

    if not client_secret:
        _fail("CDSE_CLIENT_SECRET is not set.")
        return False

    _ok(f"CDSE_CLIENT_ID   = {client_id[:8]}…{client_id[-4:]}")
    _ok(f"CDSE_CLIENT_SECRET set (length={len(client_secret)})")
    return True


def check_auth() -> "CDSEAuthManager | None":
    """Test OAuth2 token retrieval and return the auth manager on success."""
    _header("AUTH")

    try:
        from data_ingestion.sentinel_client import CDSEAuthManager, SentinelAPIError
    except ImportError as exc:
        _fail(f"Import failed: {exc}")
        return None

    try:
        t0 = time.perf_counter()
        auth = CDSEAuthManager()
        token = auth.get_token()
        elapsed = time.perf_counter() - t0
    except EnvironmentError as exc:
        _fail(str(exc))
        return None
    except Exception as exc:
        _fail(f"Unexpected error during authentication: {exc}")
        traceback.print_exc()
        return None

    if not token or len(token) < 20:
        _fail(f"Token looks invalid (length={len(token)})")
        return None

    _ok(f"Token acquired in {elapsed:.2f} s")
    _ok(f"Token prefix: {token[:12]}…  length={len(token)}")
    _ok(f"Expires in ~{auth.expires_in:.0f} s")

    # Test auto-cache (second call should be instant)
    t0 = time.perf_counter()
    token2 = auth.get_token()
    cache_elapsed = time.perf_counter() - t0

    assert token2 == token, "Cached token mismatch!"
    _ok(f"Token cache hit: second call in {cache_elapsed*1000:.2f} ms")

    return auth


def check_tile_fetch(auth) -> "tuple[bytes, bytes] | None":
    """Fetch T1 and T2 tiles and return raw bytes."""
    _header("FETCH")

    try:
        from data_ingestion.sentinel_client import (
            SentinelHubClient,
            SentinelSceneNotFoundError,
            SentinelAPIError,
        )
    except ImportError as exc:
        _fail(f"Import failed: {exc}")
        return None

    client = SentinelHubClient(auth)

    # ── T1 ────────────────────────────────────────────────────────────────────
    try:
        t0 = time.perf_counter()
        t1_bytes = client.fetch_tile(TEST_BBOX, TEST_DATE_T1, TEST_RESOLUTION)
        elapsed = time.perf_counter() - t0
    except SentinelSceneNotFoundError as exc:
        _fail(f"T1 scene not found: {exc}")
        _info("Try widening TEST_DATE_T1 or adjusting TEST_BBOX.")
        return None
    except SentinelAPIError as exc:
        _fail(f"T1 API error: {exc}")
        return None
    except Exception as exc:
        _fail(f"T1 unexpected error: {exc}")
        traceback.print_exc()
        return None

    _ok(f"T1 tile fetched: {len(t1_bytes)/1024:.1f} KB  ({elapsed:.2f} s)")

    # ── T2 ────────────────────────────────────────────────────────────────────
    try:
        t0 = time.perf_counter()
        t2_bytes = client.fetch_tile(TEST_BBOX, TEST_DATE_T2, TEST_RESOLUTION)
        elapsed = time.perf_counter() - t0
    except SentinelSceneNotFoundError as exc:
        _fail(f"T2 scene not found: {exc}")
        _info("Try widening TEST_DATE_T2 or adjusting TEST_BBOX.")
        return None
    except SentinelAPIError as exc:
        _fail(f"T2 API error: {exc}")
        return None
    except Exception as exc:
        _fail(f"T2 unexpected error: {exc}")
        traceback.print_exc()
        return None

    _ok(f"T2 tile fetched: {len(t2_bytes)/1024:.1f} KB  ({elapsed:.2f} s)")
    return t1_bytes, t2_bytes


def check_decode(t1_bytes: bytes, t2_bytes: bytes) -> "tuple | None":
    """Decode GeoTIFF bytes and validate array shape + dtype."""
    _header("DECODE")

    try:
        from preprocessing.sar_loader import decode_geotiff_response
    except ImportError as exc:
        _fail(f"Import failed: {exc}")
        return None

    for label, raw_bytes in [("T1", t1_bytes), ("T2", t2_bytes)]:
        try:
            arr = decode_geotiff_response(raw_bytes)
        except Exception as exc:
            _fail(f"{label} decode failed: {exc}")
            traceback.print_exc()
            return None

        # Shape check
        if arr.ndim != 3:
            _fail(f"{label}: expected 3D array, got {arr.shape}")
            return None
        if arr.shape[0] != 2:
            _fail(f"{label}: expected 2 bands (VV, VH), got {arr.shape[0]}")
            return None
        if arr.shape[1] != TEST_RESOLUTION[1] or arr.shape[2] != TEST_RESOLUTION[0]:
            _fail(
                f"{label}: expected spatial size {TEST_RESOLUTION}, "
                f"got {arr.shape[1:]}"
            )
            return None
        if str(arr.dtype) != "float32":
            _fail(f"{label}: expected float32, got {arr.dtype}")
            return None

        _ok(
            f"{label}: shape={arr.shape}  dtype={arr.dtype}  "
            f"linear_range=[{arr.min():.4e}, {arr.max():.4e}]"
        )

    t1_arr = decode_geotiff_response(t1_bytes)
    t2_arr = decode_geotiff_response(t2_bytes)
    return t1_arr, t2_arr


def check_normalization(t1_arr, t2_arr) -> "tuple | None":
    """Test dB normalization and verify output range."""
    _header("NORM")

    try:
        from preprocessing.sar_loader import normalize_sar_tensor
    except ImportError as exc:
        _fail(f"Import failed: {exc}")
        return None

    import numpy as np

    for label, arr in [("T1", t1_arr), ("T2", t2_arr)]:
        try:
            norm = normalize_sar_tensor(arr, is_linear=True)
        except Exception as exc:
            _fail(f"{label} normalization failed: {exc}")
            traceback.print_exc()
            return None

        min_val = float(norm.min())
        max_val = float(norm.max())

        if min_val > 0 or max_val > 50 or min_val < -100:
            _fail(
                f"{label}: normalized values out of expected dB range: "
                f"min={min_val:.6f}  max={max_val:.6f}"
            )
            return None

        # Check no NaN / Inf
        if np.any(np.isnan(norm)) or np.any(np.isinf(norm)):
            _fail(f"{label}: NaN or Inf values detected after normalization.")
            return None

        _ok(
            f"{label}: min={min_val:.4f}  max={max_val:.4f}  "
            f"mean={float(norm.mean()):.4f}  std={float(norm.std()):.4f}"
        )

    t1_norm = normalize_sar_tensor(t1_arr, is_linear=True)
    t2_norm = normalize_sar_tensor(t2_arr, is_linear=True)
    return t1_norm, t2_norm


def check_tensor_conversion(t1_norm, t2_norm) -> "tuple | None":
    """Test NumPy → PyTorch tensor conversion."""
    _header("TENSOR")

    try:
        from preprocessing.sar_loader import to_torch_tensor
        import torch
    except ImportError as exc:
        _fail(f"Import failed: {exc}")
        return None

    for label, arr in [("T1", t1_norm), ("T2", t2_norm)]:
        try:
            tensor = to_torch_tensor(arr)
        except Exception as exc:
            _fail(f"{label} tensor conversion failed: {exc}")
            traceback.print_exc()
            return None

        if tensor.shape != torch.Size([2, TEST_RESOLUTION[1], TEST_RESOLUTION[0]]):
            _fail(f"{label}: unexpected tensor shape {tensor.shape}")
            return None
        if tensor.dtype != torch.float32:
            _fail(f"{label}: expected float32, got {tensor.dtype}")
            return None

        _ok(f"{label}: shape={list(tensor.shape)}  dtype={tensor.dtype}")

    t1_tensor = to_torch_tensor(t1_norm)
    t2_tensor = to_torch_tensor(t2_norm)
    return t1_tensor, t2_tensor


def check_model_forward(t1_tensor, t2_tensor) -> bool:
    """
    Run a forward pass through SiameseUNet(in_channels=2) to confirm
    the full ingestion → inference pipeline is end-to-end compatible.
    """
    _header("MODEL")

    try:
        import torch
        from detection.siamese_unet import SiameseUNet
    except ImportError as exc:
        _fail(f"Import failed: {exc}")
        return False

    try:
        model = SiameseUNet(in_channels=2, num_classes=1)
        model.eval()

        with torch.no_grad():
            t1_batch = t1_tensor.unsqueeze(0)   # (1, 2, H, W)
            t2_batch = t2_tensor.unsqueeze(0)   # (1, 2, H, W)
            logits = model(t1_batch, t2_batch)  # (1, 1, H, W)

    except Exception as exc:
        _fail(f"Forward pass failed: {exc}")
        traceback.print_exc()
        return False

    expected_shape = torch.Size([1, 1, TEST_RESOLUTION[1], TEST_RESOLUTION[0]])
    if logits.shape != expected_shape:
        _fail(f"Unexpected output shape: {logits.shape}  (expected {expected_shape})")
        return False

    _ok(f"SiameseUNet(in_channels=2) forward pass OK")
    _ok(f"Output logits: shape={list(logits.shape)}  dtype={logits.dtype}")
    _ok(
        f"Logit stats: min={float(logits.min()):.4f}  "
        f"max={float(logits.max()):.4f}  "
        f"mean={float(logits.mean()):.4f}"
    )
    return True


def check_high_level_api(auth) -> bool:
    """Verify the top-level fetch_sentinel1_pair() convenience function."""
    _header("HIGH-LEVEL API")
    _info("Calling fetch_sentinel1_pair() …")

    try:
        from data_ingestion.sentinel_client import fetch_sentinel1_pair
        import numpy as np
    except ImportError as exc:
        _fail(f"Import failed: {exc}")
        return False

    try:
        t1, t2 = fetch_sentinel1_pair(
            bbox=TEST_BBOX,
            date_t1_range=TEST_DATE_T1,
            date_t2_range=TEST_DATE_T2,
            output_resolution=TEST_RESOLUTION,
            auth=auth,
        )
    except Exception as exc:
        _fail(f"fetch_sentinel1_pair() raised: {exc}")
        traceback.print_exc()
        return False

    for label, arr in [("T1", t1), ("T2", t2)]:
        if arr.shape != (2, TEST_RESOLUTION[1], TEST_RESOLUTION[0]):
            _fail(f"{label}: unexpected shape {arr.shape}")
            return False
        if arr.dtype != np.float32:
            _fail(f"{label}: expected float32, got {arr.dtype}")
            return False
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            _fail(f"{label}: NaN or Inf values detected in output — check normalize_sar_tensor.")
            return False
        arr_min = float(np.nanmin(arr))
        arr_max = float(np.nanmax(arr))
        if arr_min > 0 or arr_max > 50 or arr_min < -100:
            _fail(f"{label}: values outside expected dB range: min={arr_min:.6f}  max={arr_max:.6f}")
            return False
        _ok(
            f"{label}: shape={arr.shape}  "
            f"range=[{arr_min:.4f}, {arr_max:.4f}]"
        )

    return True


def check_request_payload() -> bool:
    """Verify that the API request asks for SIGMA0_ELLIPSOID, orthorectify=True, and VV/VH."""
    _header("PAYLOAD & LOADER LOGIC")
    try:
        from data_ingestion.sentinel_client import _build_request_body
        from preprocessing.sar_loader import normalize_sar_tensor
        import numpy as np
    except ImportError as exc:
        _fail(f"Import failed: {exc}")
        return False

    # Check Request Body
    body = _build_request_body(TEST_BBOX, TEST_DATE_T1, TEST_RESOLUTION)
    proc = body["input"]["data"][0]["processing"]
    
    if proc.get("backCoeff") != "SIGMA0_ELLIPSOID":
        _fail(f"Expected backCoeff='SIGMA0_ELLIPSOID', got {proc.get('backCoeff')}")
        return False
    _ok("backCoeff = SIGMA0_ELLIPSOID")

    if proc.get("orthorectify") is not True:
        _fail("Expected orthorectify=True")
        return False
    _ok("orthorectify = True")

    evalscript = body.get("evalscript", "")
    if "VV" not in evalscript or "VH" not in evalscript:
        _fail("VV or VH missing from evalscript")
        return False
    _ok("VV and VH requested in evalscript")

    # Check loader logic
    # Mock TUM data (already in dB, e.g. -10 dB)
    tum_mock = np.array([[[-10.0]]], dtype=np.float32)
    tum_out = normalize_sar_tensor(tum_mock, is_linear=False)
    if not np.isclose(tum_out[0, 0, 0], -10.0):
        _fail(f"TUM data was modified! Expected -10.0, got {tum_out[0, 0, 0]}")
        return False
    _ok("TUM (is_linear=False) passes through unaltered")

    # Mock CDSE data (linear power, e.g. 0.1 linear -> -10 dB)
    cdse_mock = np.array([[[0.1]]], dtype=np.float32)
    cdse_out = normalize_sar_tensor(cdse_mock, is_linear=True)
    if not np.isclose(cdse_out[0, 0, 0], -10.0):
        _fail(f"CDSE data not correctly converted! Expected -10.0, got {cdse_out[0, 0, 0]}")
        return False
    _ok("CDSE (is_linear=True) correctly converted to dB")

    return True


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    print(
        f"\n{_BOLD}=================================================={_RESET}\n"
        f"{_BOLD}   CDSE Sentinel-1 Ingestion Pipeline -- Test{_RESET}\n"
        f"{_BOLD}=================================================={_RESET}\n"
        f"  AOI  : {TEST_BBOX}\n"
        f"  T1   : {TEST_DATE_T1[0]} -> {TEST_DATE_T1[1]}\n"
        f"  T2   : {TEST_DATE_T2[0]} -> {TEST_DATE_T2[1]}\n"
        f"  Size : {TEST_RESOLUTION[0]}x{TEST_RESOLUTION[1]} px\n"
    )

    failures: list[str] = []

    # 1. Environment
    if not check_environment():
        failures.append("ENV: credentials not configured")

    # 1.5 Payload & Loader Logic
    if not check_request_payload():
        failures.append("PAYLOAD/LOADER: assertion failed")

    # 2. Authentication
    auth = check_auth()
    if auth is None:
        failures.append("AUTH: token acquisition failed")

    # 3. Tile fetch (requires auth)
    raw_pair = None
    if auth is not None:
        raw_pair = check_tile_fetch(auth)
        if raw_pair is None:
            failures.append("FETCH: tile retrieval failed")

    # 4-6. Decode → Normalize → Tensor (requires raw bytes)
    tensors = None
    if raw_pair is not None:
        t1_bytes, t2_bytes = raw_pair

        arr_pair = check_decode(t1_bytes, t2_bytes)
        if arr_pair is None:
            failures.append("DECODE: GeoTIFF parsing failed")
        else:
            norm_pair = check_normalization(*arr_pair)
            if norm_pair is None:
                failures.append("NORM: normalization failed or out of range")
            else:
                tensors = check_tensor_conversion(*norm_pair)
                if tensors is None:
                    failures.append("TENSOR: NumPy→torch conversion failed")

    # 7. Model forward pass (requires tensors)
    if tensors is not None:
        if not check_model_forward(*tensors):
            failures.append("MODEL: forward pass failed")

    # 8. High-level API (requires auth, uses a second fetch)
    if auth is not None:
        _info("Re-using auth manager for high-level API check …")
        if not check_high_level_api(auth):
            failures.append("HIGH-LEVEL API: fetch_sentinel1_pair() failed")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{_BOLD}{'─'*52}{_RESET}")
    if not failures:
        print(f"\n{_GREEN}{_BOLD}All checks passed ✓{_RESET}\n")
        print(
            "  Your Sentinel-1 ingestion pipeline is ready.\n"
            "  Next steps:\n"
            "    • Use fetch_sentinel1_pair() in your inference script.\n"
            "    • Pass (t1_tensor, t2_tensor) directly to model(image_a, image_b).\n"
            "    • Set in_channels=2 when instantiating SiameseUNet or SNUNetCD.\n"
        )
        return 0
    else:
        print(f"\n{_RED}{_BOLD}Some checks FAILED:{_RESET}")
        for f in failures:
            print(f"  {_RED}✗{_RESET}  {f}")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
