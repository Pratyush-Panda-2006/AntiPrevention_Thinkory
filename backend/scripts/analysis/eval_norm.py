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

for scene in train_scenes:
    with rasterio.open(base_dir / scene["t1_image"]) as src:
        t1_data = src.read()
    with rasterio.open(base_dir / scene["t2_image"]) as src:
        t2_data = src.read()
        
    vv_all.append(t1_data[0].flatten())
    vv_all.append(t2_data[0].flatten())
    vh_all.append(t1_data[1].flatten())
    vh_all.append(t2_data[1].flatten())

vv_global = np.concatenate(vv_all)
vh_global = np.concatenate(vh_all)

# Remove invalid if any
vv_valid = vv_global[np.isfinite(vv_global)]
vh_valid = vh_global[np.isfinite(vh_global)]

total_vv = len(vv_valid)
total_vh = len(vh_valid)

print(f"Total VV valid pixels: {total_vv}")
print(f"Total VH valid pixels: {total_vh}")

def evaluate_strategy(name, vv_min, vv_max, vh_min, vh_max):
    vv_clip_low_cnt = np.sum(vv_valid < vv_min)
    vv_clip_high_cnt = np.sum(vv_valid > vv_max)
    vh_clip_low_cnt = np.sum(vh_valid < vh_min)
    vh_clip_high_cnt = np.sum(vh_valid > vh_max)
    
    vv_clip_low = vv_clip_low_cnt / total_vv * 100
    vv_clip_high = vv_clip_high_cnt / total_vv * 100
    vh_clip_low = vh_clip_low_cnt / total_vh * 100
    vh_clip_high = vh_clip_high_cnt / total_vh * 100
    
    print(f"Strategy {name}:")
    print(f"  VV range: [{vv_min:.2f}, {vv_max:.2f}]")
    print(f"    Clipped Low:  {vv_clip_low_cnt:9d} ({vv_clip_low:.2f}%)")
    print(f"    Clipped High: {vv_clip_high_cnt:9d} ({vv_clip_high:.2f}%)")
    print(f"  VH range: [{vh_min:.2f}, {vh_max:.2f}]")
    print(f"    Clipped Low:  {vh_clip_low_cnt:9d} ({vh_clip_low:.2f}%)")
    print(f"    Clipped High: {vh_clip_high_cnt:9d} ({vh_clip_high:.2f}%)")
    
    # Simulate normalization
    vv_norm = np.clip(vv_valid, vv_min, vv_max)
    vv_norm = (vv_norm - vv_min) / (vv_max - vv_min)
    vh_norm = np.clip(vh_valid, vh_min, vh_max)
    vh_norm = (vh_norm - vh_min) / (vh_max - vh_min)
    
    print(f"  VV norm mean: {np.mean(vv_norm):.4f}, std: {np.std(vv_norm):.4f}")
    print(f"  VH norm mean: {np.mean(vh_norm):.4f}, std: {np.std(vh_norm):.4f}")
    print()

# A: [-30, 0] shared
evaluate_strategy("A ([-30, 0] shared)", -30.0, 0.0, -30.0, 0.0)

# B: [P1, P99] shared (combined over all pixels)
combined_p1 = min(np.percentile(vv_valid, 1), np.percentile(vh_valid, 1))
combined_p99 = max(np.percentile(vv_valid, 99), np.percentile(vh_valid, 99))
evaluate_strategy("B (Shared P1-P99)", combined_p1, combined_p99, combined_p1, combined_p99)

# C: [P0.5, P99.5] independent
vv_p05, vv_p995 = np.percentile(vv_valid, 0.5), np.percentile(vv_valid, 99.5)
vh_p05, vh_p995 = np.percentile(vh_valid, 0.5), np.percentile(vh_valid, 99.5)
evaluate_strategy("C (Independent P0.5-P99.5)", vv_p05, vv_p995, vh_p05, vh_p995)

# D: [P1, P99] independent
vv_p1, vv_p99 = np.percentile(vv_valid, 1), np.percentile(vv_valid, 99)
vh_p1, vh_p99 = np.percentile(vh_valid, 1), np.percentile(vh_valid, 99)
evaluate_strategy("D (Independent P1-P99)", vv_p1, vv_p99, vh_p1, vh_p99)

# E: Independent rounded based on stats
evaluate_strategy("E (Independent rounded -25/5 and -35/-5)", -25.0, 5.0, -35.0, -5.0)
