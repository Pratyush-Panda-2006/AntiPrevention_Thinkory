import os
import numpy as np
import rasterio

train_cities = [
    'abudhabi', 'aguasclaras', 'beihai', 'beirut', 'bercy', 'bordeaux',
    'chongqing', 'dubai', 'hongkong', 'milano', 'nantes', 'paris',
    'pisa', 'rennes'
]
sar_root = r"d:\Projects\border surv\data\sar\tum_oscd\multisensor_fusion_CD\S1"
label_root = r"d:\Projects\border surv\data\sar\tum_oscd\oscd_labels\train"

results = {}

for city in train_cities:
    label_path = os.path.join(label_root, f"{city}.tif")
    if not os.path.exists(label_path):
        label_path = os.path.join(label_root.replace("train", "test"), f"{city}.tif")
    if not os.path.exists(label_path):
        print(f"Missing {city}")
        continue
    
    with rasterio.open(label_path) as src:
        label = src.read(1)
        # OSCD format is usually 1 (no change) and 2 (change)
        change_pixels = (label == 2).sum()
        valid_pixels = (label > 0).sum()
        ratio = change_pixels / valid_pixels if valid_pixels > 0 else 0
        h, w = label.shape
        size_pixels = h * w
    
    s1_t1_dir = os.path.join(sar_root, city, "imgs_1", "transformed")
    s1_t1_files = [f for f in os.listdir(s1_t1_dir) if f.endswith(".tif")] if os.path.exists(s1_t1_dir) else []
    vv_mean = vv_std = vh_mean = vh_std = 0
    if s1_t1_files:
        with rasterio.open(os.path.join(s1_t1_dir, s1_t1_files[0])) as src:
            t1 = src.read()
            vv_mean = np.nanmean(t1[0])
            vv_std = np.nanstd(t1[0])
            vh_mean = np.nanmean(t1[1])
            vh_std = np.nanstd(t1[1])
            
    results[city] = {
        "size": f"{h}x{w}",
        "total_pix": size_pixels,
        "change_ratio": ratio,
        "vv_mean": vv_mean,
        "vv_std": vv_std,
        "vh_mean": vh_mean,
        "vh_std": vh_std,
    }

print(f"{'City':<15} | {'Size':<10} | {'Total Pix':<10} | {'Change %':<10} | {'VV Mean':<10} | {'VH Mean':<10}")
print("-" * 75)
for city in train_cities:
    if city not in results:
        continue
    res = results[city]
    print(f"{city:<15} | {res['size']:<10} | {res['total_pix']:<10} | {res['change_ratio']*100:>8.2f}% | {res['vv_mean']:>10.2f} | {res['vh_mean']:>10.2f}")
