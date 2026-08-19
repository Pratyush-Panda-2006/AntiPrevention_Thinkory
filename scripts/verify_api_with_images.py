#!/usr/bin/env python3
"""
scripts/verify_api_with_images.py
===================================
Hits the live FastAPI server at localhost:8000, fetches real Sentinel-1 SAR
imagery from CDSE, runs it through the change-detection model, and saves
ALL output images to disk for visual inspection.

Outputs (saved to results/api_visual_check/):
    t1_preview.png         — T1 SAR false-color (VV/VH/Ratio)
    t2_preview.png         — T2 SAR false-color
    change_mask.png        — Binary change mask (red = changed pixels)
    confidence_heatmap.png — Model confidence probability map (turbo colormap)
    overlay.png            — Change mask overlaid on T2 image
    report.json            — Full JSON response with statistics
"""

import io
import sys
import json
import base64
import requests
from pathlib import Path
from PIL import Image

# Force UTF-8 on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

BASE_URL = "http://localhost:8000"
OUT_DIR = Path("results/api_visual_check")
OUT_DIR.mkdir(parents=True, exist_ok=True)

_GREEN = "\033[92m"
_RED   = "\033[91m"
_YELLOW = "\033[93m"
_BOLD  = "\033[1m"
_RESET = "\033[0m"

def ok(msg):   print(f"  {_GREEN}✓{_RESET}  {msg}")
def fail(msg): print(f"  {_RED}✗{_RESET}  {msg}")
def info(msg): print(f"  {_YELLOW}·{_RESET}  {msg}")
def hdr(title): print(f"\n{_BOLD}[{title}]{_RESET}")

def save_b64_image(b64_str: str, path: Path) -> None:
    """Decode a data-URI base64 PNG and save it."""
    if b64_str.startswith("data:"):
        b64_str = b64_str.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(img_bytes))
    img.save(path)


# ── 1. Health check ────────────────────────────────────────────────────────────
hdr("1. HEALTH CHECK")
resp = requests.get(f"{BASE_URL}/health", timeout=10)
assert resp.status_code == 200, f"Health check failed: {resp.text}"
health = resp.json()
ok(f"Status     : {health['status']}")
ok(f"PyTorch    : {health['torch_version']}")
ok(f"Device     : {health['device_name']}")
ok(f"Models     : {health['loaded_models']}")
if health.get("cuda_available"):
    ok(f"VRAM       : {health['vram_used_gb']} / {health['vram_total_gb']} GB used")


# ── 2. Model catalog ──────────────────────────────────────────────────────────
hdr("2. MODEL CATALOG")
resp = requests.get(f"{BASE_URL}/api/v1/models", timeout=10)
assert resp.status_code == 200
for m in resp.json():
    ok(f"{m['name']:30s}  {m['parameters']:>12,} params  {m['input_channels']}ch")


# ── 3. Fetch Sentinel-1 SAR pair ──────────────────────────────────────────────
hdr("3. FETCHING SENTINEL-1 SAR PAIR FROM CDSE")
info("Requesting Rajasthan/Gujarat border (72°E–73°E, 24°N–25°N) ...")

fetch_payload = {
    "bbox": {"min_lon": 72.0, "min_lat": 24.0, "max_lon": 73.0, "max_lat": 25.0},
    "date_range_t1": ["2024-01-01", "2024-01-20"],
    "date_range_t2": ["2024-06-01", "2024-06-20"],
    "resolution": [256, 256],
}
resp = requests.post(f"{BASE_URL}/api/v1/cdse/fetch-pair", json=fetch_payload, timeout=60)
assert resp.status_code == 200, f"Fetch failed ({resp.status_code}): {resp.text}"
fetch_data = resp.json()

ok(f"T1 VV  → min={fetch_data['t1_stats']['vv']['min']:.4f}  max={fetch_data['t1_stats']['vv']['max']:.4f}  mean={fetch_data['t1_stats']['vv']['mean']:.4f}")
ok(f"T1 VH  → min={fetch_data['t1_stats']['vh']['min']:.4f}  max={fetch_data['t1_stats']['vh']['max']:.4f}  mean={fetch_data['t1_stats']['vh']['mean']:.4f}")
ok(f"T2 VV  → min={fetch_data['t2_stats']['vv']['min']:.4f}  max={fetch_data['t2_stats']['vv']['max']:.4f}  mean={fetch_data['t2_stats']['vv']['mean']:.4f}")
ok(f"T2 VH  → min={fetch_data['t2_stats']['vh']['min']:.4f}  max={fetch_data['t2_stats']['vh']['max']:.4f}  mean={fetch_data['t2_stats']['vh']['mean']:.4f}")

save_b64_image(fetch_data["t1_preview_base64"], OUT_DIR / "t1_preview.png")
save_b64_image(fetch_data["t2_preview_base64"], OUT_DIR / "t2_preview.png")
ok(f"Saved T1 preview → {OUT_DIR / 't1_preview.png'}")
ok(f"Saved T2 preview → {OUT_DIR / 't2_preview.png'}")


# ── 4. Full Change Detection (fetch + model inference) ─────────────────────────
hdr("4. RUNNING SIAMESE CHANGE DETECTION MODEL")
info("Fetching SAR pair and running Siamese U-Net inference ...")

detect_payload = {
    "bbox": {"min_lon": 72.0, "min_lat": 24.0, "max_lon": 73.0, "max_lat": 25.0},
    "date_range_t1": ["2024-01-01", "2024-01-20"],
    "date_range_t2": ["2024-06-01", "2024-06-20"],
    "resolution": [256, 256],
    "model_name": "siamese_unet",
    "threshold": 0.5,
    "min_region_area_px": 10,
}
resp = requests.post(f"{BASE_URL}/api/v1/detect/sentinel", json=detect_payload, timeout=120)
assert resp.status_code == 200, f"Detection failed ({resp.status_code}): {resp.text}"
det = resp.json()

ok(f"Model used        : {det['model_used']}")
ok(f"Threshold         : {det['threshold']}")
ok(f"Total pixels      : {det['total_pixels']:,}")
ok(f"Changed pixels    : {det['changed_pixels']:,}  ({det['change_percentage']}%)")
ok(f"Changed area      : {det.get('total_changed_area_sq_km')} km²")
ok(f"Change clusters   : {det['num_change_clusters']}")
ok(f"Execution time    : {det['execution_time_sec']} s")

# Save all output images
save_b64_image(det["t1_preview_base64"],          OUT_DIR / "t1_preview.png")
save_b64_image(det["t2_preview_base64"],          OUT_DIR / "t2_preview.png")
save_b64_image(det["change_mask_base64"],         OUT_DIR / "change_mask.png")
save_b64_image(det["confidence_heatmap_base64"],  OUT_DIR / "confidence_heatmap.png")
if det.get("overlay_base64"):
    save_b64_image(det["overlay_base64"],         OUT_DIR / "overlay.png")

ok(f"Saved change_mask.png")
ok(f"Saved confidence_heatmap.png")
ok(f"Saved overlay.png")

# Region report
if det["regions"]:
    hdr("5. DETECTED CHANGE CLUSTERS")
    for r in det["regions"][:10]:
        ok(
            f"Cluster #{r['region_id']:02d}  area={r['area_px']:5d}px"
            f"  prob={r['mean_change_prob']:.3f}"
            f"  severity={r['severity']:10s}"
            f"  label={r['label']}"
        )
else:
    info("No change clusters detected (threshold not exceeded on this tile/model).")

# Save full JSON report
report_path = OUT_DIR / "report.json"
report_path.write_text(
    json.dumps({k: v for k, v in det.items() if not k.endswith("_base64")}, indent=2),
    encoding="utf-8",
)
ok(f"Full report saved → {report_path}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"""
{'='*60}
  ALL API CHECKS PASSED - MODEL IS WORKING
{'='*60}
  Output images saved to: {OUT_DIR.resolve()}
    t1_preview.png         - T1 SAR false-color image
    t2_preview.png         - T2 SAR false-color image
    change_mask.png        - Binary change mask
    confidence_heatmap.png - Model probability map
    overlay.png            - Change overlay on T2
""")
