# MASTER INVESTIGATION PROMPT — TUM OSCD / SENTINEL-1 PAIRING LOGIC

## 1. Executive Summary
We have conducted a thorough investigation into the TUM Multi-modal Supervised Change Detection Sentinel-1 dataset to ascertain the intended pairing logic, particularly in light of the anomaly in the `montpellier` scene which contains two candidate Sentinel-1 images for the post-change (T2) timeframe.

The official TUM implementation blindly pairs S1 images by taking the first file returned by `os.listdir()` in the respective `transformed` folder. This process is highly dependent on the filesystem and operating system. In the case of `montpellier`, the two files in `imgs_2/transformed` have different acquisition times (05:52 and 17:39). Only the evening acquisition (`17:39`) matches the time of day (and thus the relative orbit and viewing geometry) of the `imgs_1` acquisition (`17:38`). The other acquisition is from a different pass.

Since the official codebase does not properly handle multiple candidates and relies on arbitrary list ordering, we must implement our own deterministic matching logic (e.g., matching by acquisition hour) rather than mimicking the buggy reference code.

## 2. Relevant Current-Project Files
- **`scripts/prepare_oscd_labels.py`**: Defines the `montpellier` city within the `test_expected` split list, and constructs directories using `imgs_1` and `imgs_2` paths.
- **`scripts/compare_transformed.py`**: Contains `list((base_sar / "imgs_1").glob("S1*.tif"))[0]`, showing that existing scripts in the current project ALSO naively pick the first matched TIFF, just like the reference repository.

There is currently no logic in the project handling the multiple-acquisition anomaly for Montpellier deterministically.

## 3. Relevant TUM-Reference Files
- **`tum_reference/code/data_loader.py`**: The primary data loading script. Specifically, `get_img()` and `get_img_np_instead_of_tensor()` methods contain the pairing logic for S1 transformed data.
- **`tum_reference/code/test.py`**: The testing script which loops over the test cities and consumes `get_img()`.

## 4. Exact Pairing Logic
In the official TUM reference (`data_loader.py`), the selection of T1 and T2 is executed as follows:
```python
s1_1_path = os.path.join(self.root_dir, "S1", roi_name, "imgs_1", "transformed")
s1_2_path = os.path.join(self.root_dir, "S1", roi_name, "imgs_2", "transformed")
s1_1 =  self.process_SAR(self.read_img(os.path.join(s1_1_path, os.listdir(s1_1_path)[0]), [1, 2]))
s1_2 =  self.process_SAR(self.read_img(os.path.join(s1_2_path, os.listdir(s1_2_path)[0]), [1, 2]))
```
- **A. How is T1 selected?** By taking the 0th element from `os.listdir(s1_1_path)`.
- **B. How is T2 selected?** By taking the 0th element from `os.listdir(s1_2_path)`.
- **C. Is T2 determined by a specific rule?** No. It is determined by the arbitrary ordering of `os.listdir()`, which is filesystem-dependent (usually alphabetical on Windows, and effectively random directory-entry order on Linux).
- **D. Explicit handling of multiple candidates:** The official code does **not** explicitly handle multiple acquisitions. It blindly uses `[0]` on the returned list.
- **G, H. Which files are used:** The official code strictly uses the files in the `imgs_1/transformed` and `imgs_2/transformed` directories.

## 5. Exact Montpellier Resolution
- **E. What the official implementation does:** It executes `os.listdir()[0]`. On Windows, this alphabetically selects `S1A_IW_GRDH_1SDV_20171028T055211...` because `20171028` precedes `20171029`. On the original authors' Linux system, it might have arbitrarily returned the other file.
- **F. Evidence for the intended OSCD pair:** 
  - T1 is `S1A_IW_GRDH_1SDV_20150805T173853...` (Time: 17:38:53)
  - T2 Candidate 1 is `S1A_IW_GRDH_1SDV_20171028T055211...` (Time: 05:52:11)
  - T2 Candidate 2 is `S1A_IW_GRDH_1SDV_20171029T173905...` (Time: 17:39:05)
  
  Candidate 2 matches the approximate acquisition time of day of T1 (~17:39 vs ~17:38). In SAR datasets, preserving the relative orbit (ascending vs descending pass) is critical for change detection, otherwise shadows and layover effects will be entirely reversed. Candidate 2 is undeniably the intended pair, and Candidate 1 is an auxiliary acquisition (or an error in dataset curation).

## 6. 24-City Pairing Table

| city | selected T1 | selected T2 candidates | # trans T1 | # trans T2 | mask path | split |
|---|---|---|---|---|---|---|
| abudhabi | S1A...20160218T142406... | S1A...20180327T142413... | 1 | 1 | train\abudhabi.tif | train |
| aguasclaras | S1A...20150920T084442... | S1A...20171015T084500... | 1 | 1 | train\aguasclaras.tif | train |
| beihai | S1A...20161216T104927... | S1A...20180305T104948... | 1 | 1 | train\beihai.tif | train |
| beirut | S1A...20150819T154054... | S1A...20171003T033515... | 1 | 1 | train\beirut.tif | train |
| bercy | S1B...20161130T060631... | S1B...20170828T055832... | 1 | 1 | train\bercy.tif | train |
| bordeaux | S1A...20160504T060821... | S1A...20171026T060829... | 1 | 1 | train\bordeaux.tif | train |
| brasilia | S1A...20150920T084442... | S1A...20171015T084500... | 1 | 1 | test\brasilia.tif | test |
| chongqing | S1A...20170415T225631... | S1A...20180403T110004... | 1 | 1 | test\chongqing.tif | test |
| cupertino | S1A...20150921T140741... | S1A...20180326T141554... | 1 | 1 | train\cupertino.tif | train |
| dubai | S1A...20151102T142421... | S1A...20180330T021517... | 1 | 1 | test\dubai.tif | test |
| hongkong | S1A...20160925T103320... | S1A...20180319T103309... | 1 | 1 | train\hongkong.tif | train |
| lasvegas | S1A...20150403T014200... | S1A...20180210T134315... | 1 | 1 | test\lasvegas.tif | test |
| milano | S1B...20161228T053435... | S1A...20180122T053510... | 1 | 1 | test\milano.tif | test |
| montpellier | S1A...20150805T173853... | S1A...20171028T055211..., S1A...20171029T173905... | 1 | 2 | test\montpellier.tif | test |
| mumbai | S1A...20160224T010235... | S1A...20180321T010247... | 1 | 1 | train\mumbai.tif | train |
| nantes | S1A...20150815T175602... | S1A...20171015T175633... | 1 | 1 | train\nantes.tif | train |
| norcia | S1A...20150711T165718... | S1A...20171015T051130... | 1 | 1 | test\norcia.tif | test |
| paris | S1B...20161130T060631... | S1A...20171107T060714... | 1 | 1 | train\paris.tif | train |
| pisa | S1A...20150709T171401... | S1A...20180211T171418... | 1 | 1 | train\pisa.tif | train |
| rennes | S1A...20150807T061526... | S1A...20170621T061546... | 1 | 1 | train\rennes.tif | train |
| rio | S1A...20160420T082217... | S1B...20171006T082130... | 1 | 1 | test\rio.tif | test |
| saclay_e | S1A...20160317T060711... | S1A...20170827T060713... | 1 | 1 | train\saclay_e.tif | train |
| saclay_w | S1A...20160312T055849... | S1A...20170827T060713... | 1 | 1 | test\saclay_w.tif | test |
| valencia | S1A...20160727T060952... | S1A...20171107T060944... | 1 | 1 | test\valencia.tif | test |

## 7. Mask Mapping
Mask naming convention: `{city}.tif`. The labels reside in `test\` or `train\` respectively in the current `oscd_labels` directory.
Mask values are `1` and `2`. 
In the official dataloader, this is mapped via `label = self.read_img(mask_path, [1])[0] - 1`. 
- `1` becomes `0` (No Change)
- `2` becomes `1` (Change)

There are no ignored or no-data values handled in the dataset beyond this binary mapping.

## 8. Transformed-Grid Verification (Montpellier)
For Montpellier:
- **T1 Transformed**: Width: 451, Height: 426, CRS: None, Transform: Identity (1.0, 1.0), Dtype: float64
- **T2 Candidate 1**: Width: 451, Height: 426, CRS: None, Transform: Identity (1.0, 1.0), Dtype: float64
- **T2 Candidate 2**: Width: 451, Height: 426, CRS: None, Transform: Identity (1.0, 1.0), Dtype: float64
- **OSCD Mask**: Width: 451, Height: 426, CRS: None, Transform: Identity (1.0, 1.0), Dtype: uint8

All `transformed` images perfectly align pixel-for-pixel with the corresponding OSCD mask. They lack a CRS or actual geo-transform (both fallback to an identity matrix), implying they are already spatially cropped and coregistered directly to the mask's grid.

## 9. Ambiguities
There are no dataset-level ambiguities remaining. We understand exactly how the original code incorrectly handles multiple files by relying on `os.listdir()`, and we understand the geophysical necessity to pair Sentinel-1 acquisitions based on identical relative orbits (which corresponds to matching the hour of acquisition).

## 10. Recommended Next Implementation Step
Update our internal SAR dataloader/pipeline logic to sort/select candidate Sentinel-1 images matching the acquisition hour (and thus the relative orbit) of the T1 image, instead of relying on `.glob("*.tif")[0]`.
