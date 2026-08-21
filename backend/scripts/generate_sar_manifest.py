import os
import json
import rasterio
from pathlib import Path

base_s1 = Path("data/sar/tum_oscd/multisensor_fusion_CD/S1")
base_labels = Path("data/sar/tum_oscd/oscd_labels")
manifest_path = Path("data/sar/tum_oscd/sar_scene_manifest.json")

train_cities = ['abudhabi', 'aguasclaras', 'beihai', 'beirut', 'bercy', 'bordeaux', 'cupertino', 'hongkong', 'mumbai', 'nantes', 'paris', 'pisa', 'rennes', 'saclay_e']
test_cities = ['brasilia', 'chongqing', 'dubai', 'lasvegas', 'milano', 'montpellier', 'norcia', 'rio', 'saclay_w', 'valencia']
all_cities = sorted(train_cities + test_cities)

manifest_data = []

def get_acquisition_hour(filename):
    # S1A_IW_GRDH_1SDV_YYYYMMDDTHHMMSS_YYYYMMDDTHHMMSS_...
    parts = filename.split('_')
    for part in parts:
        if 'T' in part and len(part) == 15: # YYYYMMDDTHHMMSS
            # Return HH
            return part[9:11]
    return None

for city in all_cities:
    split = "train" if city in train_cities else "test"
    
    t1_dir = base_s1 / city / "imgs_1" / "transformed"
    t2_dir = base_s1 / city / "imgs_2" / "transformed"
    
    t1_files = list(t1_dir.glob("*.tif"))
    if len(t1_files) != 1:
        raise ValueError(f"Expected exactly 1 T1 file for {city}, found {len(t1_files)}")
    
    t1_file = t1_files[0]
    t1_hour = get_acquisition_hour(t1_file.name)
    
    t2_candidates = list(t2_dir.glob("*.tif"))
    if len(t2_candidates) == 0:
        raise ValueError(f"No T2 files found for {city}")
    elif len(t2_candidates) == 1:
        t2_file = t2_candidates[0]
    else:
        # Match by hour
        matched = []
        for cand in t2_candidates:
            if get_acquisition_hour(cand.name) == t1_hour:
                matched.append(cand)
        if len(matched) != 1:
            raise ValueError(f"Could not uniquely match T2 for {city} by hour. Found {len(matched)} matches out of {len(t2_candidates)} candidates.")
        t2_file = matched[0]
        
    mask_files = list(base_labels.rglob(f"{city}*.tif"))
    # Filter to ensure we get exactly the right mask, some masks might be named differently
    # Let's check exactly {city}.tif or {city}-cm.tif
    valid_masks = [m for m in mask_files if m.name == f"{city}.tif" or m.name == f"{city}-cm.tif"]
    
    if len(valid_masks) != 1:
        raise ValueError(f"Expected exactly 1 mask file for {city}, found {len(valid_masks)}: {valid_masks}")
    
    mask_file = valid_masks[0]
    
    with rasterio.open(t1_file) as src:
        t1_w, t1_h = src.width, src.height
        t1_b = src.count
    with rasterio.open(t2_file) as src:
        t2_w, t2_h = src.width, src.height
        t2_b = src.count
    with rasterio.open(mask_file) as src:
        m_w, m_h = src.width, src.height
    
    if (t1_w, t1_h) != (t2_w, t2_h) or (t1_w, t1_h) != (m_w, m_h):
        raise ValueError(f"Dimension mismatch for {city}: T1({t1_w}x{t1_h}), T2({t2_w}x{t2_h}), Mask({m_w}x{m_h})")
    
    if t1_b != t2_b:
        raise ValueError(f"Band count mismatch for {city}: T1={t1_b}, T2={t2_b}")
    
    manifest_data.append({
        "city": city,
        "split": split,
        "t1_image": str(t1_file.relative_to(base_s1.parent.parent)), # Relative to tum_oscd
        "t2_image": str(t2_file.relative_to(base_s1.parent.parent)),
        "mask": str(mask_file.relative_to(base_labels.parent)),      # Relative to tum_oscd
        "image_height": t1_h,
        "image_width": t1_w,
        "band_count": t1_b,
        "mask_height": m_h,
        "mask_width": m_w
    })

manifest_path.parent.mkdir(parents=True, exist_ok=True)
with open(manifest_path, 'w') as f:
    json.dump({"scenes": manifest_data}, f, indent=4)

print("Mapping:")
for entry in manifest_data:
    print(f"City: {entry['city']} | Split: {entry['split']} | T1: {Path(entry['t1_image']).name} | T2: {Path(entry['t2_image']).name}")

print("\nSummary:")
print(f"{len(manifest_data)}/24 valid scenes")
print("0 missing")
print("0 ambiguous")
print("0 dimension mismatches")
