import os
import json
import rasterio
import numpy as np
from pathlib import Path

manifest_path = Path("data/sar/tum_oscd/sar_scene_manifest.json")
base_dir = Path("data/sar/tum_oscd")

with open(manifest_path, "r") as f:
    data = json.load(f)

train_scenes = [s for s in data["scenes"] if s["split"] == "train"]

vv_all = []
vh_all = []

scene_stats = []

has_nans = False
has_infs = False
nodata_values = set()

def compute_stats(arr):
    arr_valid = arr[np.isfinite(arr)]
    return {
        "min": float(np.min(arr_valid)),
        "max": float(np.max(arr_valid)),
        "mean": float(np.mean(arr_valid)),
        "std": float(np.std(arr_valid)),
        "p1": float(np.percentile(arr_valid, 1)),
        "p5": float(np.percentile(arr_valid, 5)),
        "p50": float(np.percentile(arr_valid, 50)),
        "p95": float(np.percentile(arr_valid, 95)),
        "p99": float(np.percentile(arr_valid, 99)),
        "p99_5": float(np.percentile(arr_valid, 99.5)),
        "nans": int(np.isnan(arr).sum()),
        "infs": int(np.isinf(arr).sum())
    }

for scene in train_scenes:
    city = scene["city"]
    
    t1_path = base_dir / scene["t1_image"]
    t2_path = base_dir / scene["t2_image"]
    
    with rasterio.open(t1_path) as src:
        t1_data = src.read()
        t1_nodata = src.nodata
        if t1_nodata is not None:
            nodata_values.add(t1_nodata)
            
    with rasterio.open(t2_path) as src:
        t2_data = src.read()
        t2_nodata = src.nodata
        if t2_nodata is not None:
            nodata_values.add(t2_nodata)
            
    # Assuming band 1 is VV, band 2 is VH
    # Combine T1 and T2 for per-scene stats
    vv_scene = np.concatenate([t1_data[0].flatten(), t2_data[0].flatten()])
    vh_scene = np.concatenate([t1_data[1].flatten(), t2_data[1].flatten()])
    
    s_vv = compute_stats(vv_scene)
    s_vh = compute_stats(vh_scene)
    
    if s_vv["nans"] > 0 or s_vh["nans"] > 0: has_nans = True
    if s_vv["infs"] > 0 or s_vh["infs"] > 0: has_infs = True
    
    scene_stats.append({
        "city": city,
        "vv": s_vv,
        "vh": s_vh
    })
    
    # Store for global computation (might consume some RAM but it's 14 small cities, should be fine)
    # Average size is ~500x500 pixels. 500*500*2 = 500,000 floats. Very small.
    vv_all.append(vv_scene)
    vh_all.append(vh_scene)

vv_global = np.concatenate(vv_all)
vh_global = np.concatenate(vh_all)

s_vv_global = compute_stats(vv_global)
s_vh_global = compute_stats(vh_global)

output = {
    "global_vv": s_vv_global,
    "global_vh": s_vh_global,
    "nodata_values": list(nodata_values),
    "per_scene": scene_stats
}

with open("sar_stats_output.json", "w") as f:
    json.dump(output, f, indent=2)
