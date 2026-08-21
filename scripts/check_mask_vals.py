import rasterio
import numpy as np
import pathlib

base_mask = pathlib.Path("data/sar/tum_oscd/oscd_labels")
mask_path = list(base_mask.rglob("montpellier*.tif"))[0]

with rasterio.open(mask_path) as src:
    mask = src.read(1)
    unique_vals = np.unique(mask)
    print(f"Mask values: {unique_vals}")

# The reference code says: `label = self.read_img(mask_path, [1])[0] - 1`
# Let's see what values it actually has.
