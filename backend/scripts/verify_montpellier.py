import rasterio

def print_info(path, label):
    with rasterio.open(path) as src:
        print(f"--- {label} ---")
        print(f"File: {path.name}")
        print(f"Width: {src.width}, Height: {src.height}")
        print(f"CRS: {src.crs}")
        print(f"Transform: {src.transform}")
        print(f"Pixel size: ({src.transform[0]}, {src.transform[4]})")
        print(f"Dtype: {src.dtypes[0]}")
        print(f"Bounds: {src.bounds}")

import pathlib
base_s1 = pathlib.Path("data/sar/tum_oscd/multisensor_fusion_CD/S1/montpellier")
t1_trans = list((base_s1 / "imgs_1" / "transformed").glob("*.tif"))[0]
t2_trans_cands = list((base_s1 / "imgs_2" / "transformed").glob("*.tif"))

base_mask = pathlib.Path("data/sar/tum_oscd/oscd_labels")
mask_path = list(base_mask.rglob("montpellier*.tif"))[0]

print_info(t1_trans, "T1 Transformed")
for i, cand in enumerate(t2_trans_cands):
    print_info(cand, f"T2 Candidate {i+1}")

print_info(mask_path, "OSCD Mask")
