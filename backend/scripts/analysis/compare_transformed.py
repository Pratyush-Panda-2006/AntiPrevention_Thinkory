import rasterio
from pathlib import Path
import sys

def report_stats(path, name):
    if not path.exists():
        print(f"NOT FOUND: {name} at {path}")
        return
    with rasterio.open(path) as ds:
        print(f"\n--- {name} ---")
        print(f"Path: {path}")
        print(f"Width: {ds.width}, Height: {ds.height}")
        print(f"Bands: {ds.count}")
        print(f"Dtype: {ds.dtypes[0]}")
        print(f"CRS: {ds.crs}")
        print(f"Transform: {ds.transform}")
        print(f"Bounds: {ds.bounds}")

def main():
    base_sar = Path("data/sar/tum_oscd/multisensor_fusion_CD/S1/abudhabi")
    label_path = Path("data/sar/tum_oscd/oscd_labels/train/abudhabi.tif")
    
    t1_raw = list((base_sar / "imgs_1").glob("S1*.tif"))[0]
    t1_trans = list((base_sar / "imgs_1" / "transformed").glob("S1*.tif"))[0]
    
    t2_raw = list((base_sar / "imgs_2").glob("S1*.tif"))[0]
    t2_trans = list((base_sar / "imgs_2" / "transformed").glob("S1*.tif"))[0]
    
    report_stats(t1_raw, "T1 RAW")
    report_stats(t1_trans, "T1 TRANSFORMED")
    report_stats(t2_raw, "T2 RAW")
    report_stats(t2_trans, "T2 TRANSFORMED")
    report_stats(label_path, "OSCD LABEL")
    
if __name__ == "__main__":
    main()
