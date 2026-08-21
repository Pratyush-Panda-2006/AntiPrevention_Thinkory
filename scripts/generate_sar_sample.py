import os
import sys
from pathlib import Path
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_ingestion.sentinel_client import fetch_sentinel1_pair
from preprocessing.sar_loader import normalize_sar_tensor, decode_geotiff_response

def report_raster_stats(path_str):
    with rasterio.open(path_str) as ds:
        print(f"--- Raster: {path_str} ---")
        print(f"CRS: {ds.crs}")
        print(f"Width: {ds.width}, Height: {ds.height}")
        print(f"Bands: {ds.count}")
        print(f"Dtype: {ds.dtypes[0]}")
        print(f"Bounds: {ds.bounds}")
        print(f"Transform: {ds.transform}")
        arr = ds.read()
        print(f"Stats: min={np.nanmin(arr):.6f}, max={np.nanmax(arr):.6f}, mean={np.nanmean(arr):.6f}, std={np.nanstd(arr):.6f}")
        return {
            "crs": ds.crs,
            "width": ds.width,
            "height": ds.height,
            "bounds": ds.bounds,
            "transform": ds.transform
        }

def save_img(arr, path, title=""):
    # arr is [0, 1] float32. Convert to uint8 for saving.
    arr_uint8 = (arr * 255).clip(0, 255).astype(np.uint8)
    if arr_uint8.ndim == 3 and arr_uint8.shape[0] == 3: # (3, H, W)
        arr_uint8 = arr_uint8.transpose(1, 2, 0)
    elif arr_uint8.ndim == 2:
        pass
    else:
        raise ValueError(f"Unexpected shape for saving: {arr_uint8.shape}")
    
    img = Image.fromarray(arr_uint8)
    img.save(path)
    print(f"Saved {path}")

def main():
    bbox = [72.0, 24.0, 73.0, 25.0]
    t1_dates = ("2024-01-01", "2024-01-20")
    t2_dates = ("2024-06-01", "2024-06-20")
    res = (256, 256)
    
    save_dir = Path("data/cdse_raw/sample")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. DOWNLOAD
    print("Downloading pair...")
    fetch_sentinel1_pair(bbox, t1_dates, t2_dates, output_resolution=res, save_dir=save_dir)
    
    # 2. VERIFY RAW DATA
    t1_path = save_dir / "t1.tif"
    t2_path = save_dir / "t2.tif"
    s1 = report_raster_stats(t1_path)
    s2 = report_raster_stats(t2_path)
    
    print("\n--- Alignment Verification ---")
    print(f"Identical Width: {s1['width'] == s2['width']}")
    print(f"Identical Height: {s1['height'] == s2['height']}")
    print(f"Identical CRS: {s1['crs'] == s2['crs']}")
    print(f"Identical Spatial Extent (Bounds): {s1['bounds'] == s2['bounds']}")
    print(f"Identical Pixel Grid (Transform): {s1['transform'] == s2['transform']}")
    
    # Read raw bytes for preprocessing
    t1_bytes = t1_path.read_bytes()
    t2_bytes = t2_path.read_bytes()
    
    t1_raw = decode_geotiff_response(t1_bytes)
    t2_raw = decode_geotiff_response(t2_bytes)
    
    # 4. ACTUAL PREPROCESSING RESULT
    print("\n--- Preprocessing Stats (0-1 Normalized) ---")
    t1_norm = normalize_sar_tensor(t1_raw)
    t2_norm = normalize_sar_tensor(t2_raw)
    
    print(f"T1 VV: min={t1_norm[0].min():.4f}, max={t1_norm[0].max():.4f}, mean={t1_norm[0].mean():.4f}, std={t1_norm[0].std():.4f}")
    print(f"T1 VH: min={t1_norm[1].min():.4f}, max={t1_norm[1].max():.4f}, mean={t1_norm[1].mean():.4f}, std={t1_norm[1].std():.4f}")
    print(f"T2 VV: min={t2_norm[0].min():.4f}, max={t2_norm[0].max():.4f}, mean={t2_norm[0].mean():.4f}, std={t2_norm[0].std():.4f}")
    print(f"T2 VH: min={t2_norm[1].min():.4f}, max={t2_norm[1].max():.4f}, mean={t2_norm[1].mean():.4f}, std={t2_norm[1].std():.4f}")
    
    # 3. CREATE HUMAN-VIEWABLE PNGs
    preview_dir = save_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    
    # Visual contrast stretch helper (apply a bit of gamma/stretching purely for visualization if needed, 
    # but the instructions say "DO NOT invent a new normalization scheme for the actual model data. For visualization only, apply a sensible contrast stretch if necessary")
    # Actually, the 0-1 normalization is usually decent. Let's just apply gamma=0.6 to brighten it.
    def vis_transform(x):
        return np.power(x, 0.6)
    
    t1_vv_vis = vis_transform(t1_norm[0])
    t1_vh_vis = vis_transform(t1_norm[1])
    t2_vv_vis = vis_transform(t2_norm[0])
    t2_vh_vis = vis_transform(t2_norm[1])
    
    save_img(t1_vv_vis, preview_dir / "t1_vv.png")
    save_img(t1_vh_vis, preview_dir / "t1_vh.png")
    save_img(t2_vv_vis, preview_dir / "t2_vv.png")
    save_img(t2_vh_vis, preview_dir / "t2_vh.png")
    
    # RGB Composite: R=VV, G=VH, B=VV
    t1_rgb = np.stack([t1_vv_vis, t1_vh_vis, t1_vv_vis], axis=0)
    t2_rgb = np.stack([t2_vv_vis, t2_vh_vis, t2_vv_vis], axis=0)
    
    save_img(t1_rgb, preview_dir / "t1_rgb_composite.png")
    save_img(t2_rgb, preview_dir / "t2_rgb_composite.png")
    
    # Difference
    diff_vv = np.abs(t2_norm[0] - t1_norm[0])
    diff_vh = np.abs(t2_norm[1] - t1_norm[1])
    # stretch diff for visibility
    diff_vv_vis = (diff_vv / (diff_vv.max() + 1e-6))
    diff_vh_vis = (diff_vh / (diff_vh.max() + 1e-6))
    
    save_img(diff_vv_vis, preview_dir / "vv_difference.png")
    save_img(diff_vh_vis, preview_dir / "vh_difference.png")
    
    # 5. CREATE CONTACT SHEET
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Row 1
    axes[0, 0].imshow(t1_vv_vis, cmap='gray'); axes[0, 0].set_title('T1 VV')
    axes[0, 1].imshow(t1_vh_vis, cmap='gray'); axes[0, 1].set_title('T1 VH')
    axes[0, 2].imshow(t1_rgb.transpose(1, 2, 0)); axes[0, 2].set_title('T1 Composite (R=VV,G=VH,B=VV)')
    axes[0, 3].imshow(diff_vv_vis, cmap='hot'); axes[0, 3].set_title('VV Difference')
    
    # Row 2
    axes[1, 0].imshow(t2_vv_vis, cmap='gray'); axes[1, 0].set_title('T2 VV')
    axes[1, 1].imshow(t2_vh_vis, cmap='gray'); axes[1, 1].set_title('T2 VH')
    axes[1, 2].imshow(t2_rgb.transpose(1, 2, 0)); axes[1, 2].set_title('T2 Composite (R=VV,G=VH,B=VV)')
    axes[1, 3].imshow(diff_vh_vis, cmap='hot'); axes[1, 3].set_title('VH Difference')
    
    for ax in axes.flatten():
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(preview_dir / "sample_contact_sheet.png", dpi=150)
    print(f"Saved {preview_dir / 'sample_contact_sheet.png'}")

if __name__ == "__main__":
    main()
