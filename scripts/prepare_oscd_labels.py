import os
import sys
import hashlib
import zipfile
import urllib.request
from pathlib import Path
import rasterio
import numpy as np
import matplotlib.pyplot as plt

def compute_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

import urllib.request
import shutil
import ssl

def download_file(url, dest_path):
    if not dest_path.exists():
        print(f"Downloading {url} to {dest_path}...")
        try:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            with urllib.request.urlopen(url, context=ctx) as response, open(dest_path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
        except Exception as e:
            print(f"FAILED TO DOWNLOAD {url}: {e}")
            return False
    else:
        print(f"{dest_path} already exists. Skipping download.")
    return True

def main():
    base_dir = Path("data/sar/tum_oscd")
    labels_dir = base_dir / "oscd_labels"
    train_labels_zip = base_dir / "oscd_train_labels.zip"
    test_labels_zip = base_dir / "oscd_test_labels.zip"
    
    # 2. DOWNLOAD OSCD LABELS
    train_url = "https://partage.mines-telecom.fr/index.php/s/2D6n03k58ygBSpu/download"
    test_url = "https://partage.imt.fr/index.php/s/gpStKn4Mpgfnr63/download"
    
    if not download_file(train_url, train_labels_zip): sys.exit(1)
    if not download_file(test_url, test_labels_zip): sys.exit(1)
    
    # 3. VERIFY DOWNLOADS
    print("\n--- 3. VERIFY DOWNLOADS ---")
    train_sz = os.path.getsize(train_labels_zip)
    test_sz = os.path.getsize(test_labels_zip)
    train_sha = compute_sha256(train_labels_zip)
    test_sha = compute_sha256(test_labels_zip)
    
    print(f"Train ZIP: {train_sz} bytes | SHA256: {train_sha[:8]}...")
    print(f"Test ZIP:  {test_sz} bytes | SHA256: {test_sha[:8]}...")
    
    # 4. INSPECT ZIP CONTENTS
    print("\n--- 4. INSPECT ZIP CONTENTS ---")
    try:
        with zipfile.ZipFile(train_labels_zip, 'r') as z:
            train_names = z.namelist()
        with zipfile.ZipFile(test_labels_zip, 'r') as z:
            test_names = z.namelist()
    except zipfile.BadZipFile:
        print("ERROR: Invalid ZIP archive!")
        sys.exit(1)
        
    print(f"Train ZIP files: {len(train_names)}")
    print(f"Test ZIP files: {len(test_names)}")
    
    # 5. EXPECTED CITY SET
    train_expected = ["abudhabi", "aguasclaras", "beihai", "beirut", "bercy", "bordeaux", "chongqing", "dubai", "hongkong", "milano", "nantes", "paris", "pisa", "rennes"]
    test_expected = ["brasilia", "cupertino", "lasvegas", "montpellier", "mumbai", "norcia", "rio", "saclay_e", "saclay_w", "valencia"]
    
    # 6. EXTRACT LABELS
    print("\n--- 6. EXTRACT LABELS ---")
    train_out = labels_dir / "train"
    test_out = labels_dir / "test"
    train_out.mkdir(parents=True, exist_ok=True)
    test_out.mkdir(parents=True, exist_ok=True)
    
    def extract_flat(zip_path, out_dir):
        with zipfile.ZipFile(zip_path, 'r') as z:
            for item in z.namelist():
                if item.endswith('/') or not item.endswith('.tif'): continue
                # flatten: just write to out_dir / basename
                basename = os.path.basename(item)
                # some cities might have 'cm/cm.tif' under city dir. The OSCD labels typically have `<city>/cm/cm.tif`. 
                # Let's extract them named as `<city>.tif` if we can determine the city name.
                parts = item.split('/')
                # usually parts[0] is the city name
                city_name = parts[0] if len(parts)>1 else basename.replace('.tif', '')
                if city_name == "Onera Satellite Change Detection dataset - Train Labels":
                    city_name = parts[1]
                if city_name == "Onera Satellite Change Detection dataset - Test Labels":
                    city_name = parts[1]
                
                out_path = out_dir / f"{city_name}.tif"
                with open(out_path, "wb") as f_out:
                    f_out.write(z.read(item))
                    
    extract_flat(train_labels_zip, train_out)
    extract_flat(test_labels_zip, test_out)
    
    # Check what was extracted
    train_extracted = [f.stem for f in train_out.glob("*.tif")]
    test_extracted = [f.stem for f in test_out.glob("*.tif")]
    print(f"Extracted train cities: {len(train_extracted)}")
    print(f"Extracted test cities: {len(test_extracted)}")
    
    # Extract TUM SAR archive if not already done
    tum_sar_zip = base_dir / "multisensor_fusion_CD.zip"
    sar_extract_dir = base_dir / "multisensor_fusion_CD"
    if tum_sar_zip.exists() and not sar_extract_dir.exists():
        print("\nExtracting TUM SAR archive...")
        with zipfile.ZipFile(tum_sar_zip, 'r') as z:
            z.extractall(base_dir)
            
    # 7. VERIFY LABEL FILES
    print("\n--- 7. VERIFY LABEL FILES ---")
    label_info = {}
    for label_path in list(train_out.glob("*.tif")) + list(test_out.glob("*.tif")):
        with rasterio.open(label_path) as ds:
            arr = ds.read(1)
            uniques = np.unique(arr)
            label_info[label_path.stem] = {
                "width": ds.width,
                "height": ds.height,
                "dtype": ds.dtypes[0],
                "bands": ds.count,
                "uniques": uniques.tolist(),
                "crs": str(ds.crs),
                "bounds": ds.bounds,
                "transform": ds.transform
            }
            if len(label_info) <= 2:
                print(f"Label {label_path.stem}: {ds.width}x{ds.height}, {ds.dtypes[0]}, bands={ds.count}, unique={uniques}")
                
    # 8. MAP LABELS TO TUM SAR SCENES
    print("\n--- 8. MAP LABELS TO TUM SAR SCENES ---")
    mapping = {}
    sar_s1_dir = sar_extract_dir / "S1"
    
    for city in train_expected + test_expected:
        t1_dir = sar_s1_dir / city / "imgs_1"
        t2_dir = sar_s1_dir / city / "imgs_2"
        
        t1_file = list(t1_dir.glob("*.tif"))[0] if t1_dir.exists() and list(t1_dir.glob("*.tif")) else None
        t2_file = list(t2_dir.glob("*.tif"))[0] if t2_dir.exists() and list(t2_dir.glob("*.tif")) else None
        
        label_file = train_out / f"{city}.tif"
        if not label_file.exists():
            label_file = test_out / f"{city}.tif"
            
        mapping[city] = {
            "t1": t1_file,
            "t2": t2_file,
            "label": label_file if label_file.exists() else None
        }
        
    matched_cities = [c for c, m in mapping.items() if m["t1"] and m["t2"] and m["label"]]
    missing_cities = [c for c, m in mapping.items() if not m["t1"] or not m["t2"] or not m["label"]]
    
    print(f"Matched cities: {len(matched_cities)}/24")
    if missing_cities:
        print(f"Missing cities: {missing_cities}")
        
    # 9. GEOSPATIAL VALIDATION
    print("\n--- 9. GEOSPATIAL VALIDATION ---")
    geospatial_mismatch = False
    for city in matched_cities[:3]: # check 3 representative cities
        m = mapping[city]
        with rasterio.open(m["t1"]) as ds_t1, rasterio.open(m["t2"]) as ds_t2, rasterio.open(m["label"]) as ds_l:
            match_crs = (ds_t1.crs == ds_l.crs)
            match_shape = (ds_t1.shape == ds_l.shape)
            match_transform = (ds_t1.transform == ds_l.transform)
            print(f"{city}: SAR vs Label CRS={match_crs}, Shape={match_shape}, Transform={match_transform}")
            if not (match_crs and match_shape and match_transform):
                geospatial_mismatch = True
                print(f"  T1: {ds_t1.shape}, {ds_t1.crs}, {ds_t1.transform}")
                print(f"  Label: {ds_l.shape}, {ds_l.crs}, {ds_l.transform}")
                
    if geospatial_mismatch:
        print("WARNING: Geospatial mismatch detected between SAR and labels!")
        
    # 12. VISUAL SAMPLE
    print("\n--- 12. VISUAL SAMPLE ---")
    preview_dir = base_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    
    sample_cities = matched_cities[:2]
    fig, axes = plt.subplots(len(sample_cities), 5, figsize=(15, 3*len(sample_cities)))
    
    if len(sample_cities) == 1:
        axes = [axes]
        
    for i, city in enumerate(sample_cities):
        m = mapping[city]
        with rasterio.open(m["t1"]) as ds_t1: t1_arr = ds_t1.read()
        with rasterio.open(m["t2"]) as ds_t2: t2_arr = ds_t2.read()
        with rasterio.open(m["label"]) as ds_l: l_arr = ds_l.read(1)
        
        # simple visualization stretch
        def vis(x):
            x = np.clip(x, -30, 0)
            return (x - (-30)) / 30.0
            
        axes[i][0].imshow(vis(t1_arr[0]), cmap='gray'); axes[i][0].set_title(f"{city} T1 VV")
        axes[i][1].imshow(vis(t1_arr[1]), cmap='gray'); axes[i][1].set_title(f"{city} T1 VH")
        axes[i][2].imshow(vis(t2_arr[0]), cmap='gray'); axes[i][2].set_title(f"{city} T2 VV")
        axes[i][3].imshow(vis(t2_arr[1]), cmap='gray'); axes[i][3].set_title(f"{city} T2 VH")
        axes[i][4].imshow(l_arr, cmap='jet'); axes[i][4].set_title(f"{city} Label")
        
        for ax in axes[i]: ax.axis('off')
        
    plt.tight_layout()
    contact_sheet_path = preview_dir / "oscd_sar_label_contact_sheet.png"
    plt.savefig(contact_sheet_path)
    print(f"Contact sheet saved to {contact_sheet_path}")
    
    # Output report dict
    report = {
        "train_sz": train_sz, "test_sz": test_sz,
        "train_sha": train_sha, "test_sha": test_sha,
        "train_extracted": len(train_extracted), "test_extracted": len(test_extracted),
        "uniques": list(set([tuple(u['uniques']) for u in label_info.values()])),
        "matched": len(matched_cities),
        "mismatch": geospatial_mismatch
    }
    
    import json
    with open(base_dir / "pipeline_report.json", "w") as f:
        json.dump(report, f)

if __name__ == "__main__":
    main()
