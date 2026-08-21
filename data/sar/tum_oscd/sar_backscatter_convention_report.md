# SAR Backscatter Convention Report: TUM OSCD Dataset

## 1. Executive Summary
The TUM OSCD SAR dataset contains Sentinel-1 GRD imagery that has been processed to **geometrically orthorectified Sigma Naught ($\sigma^0$) in decibels (dB)**. This was determined by analyzing the dataset's value ranges (which are centered around -11 dB with negative values, proving they are pre-log-scaled) and the authors' stated methodology of downloading directly from Google Earth Engine (GEE). 

The current local CDSE pipeline requests Radiometric Terrain Corrected Gamma Naught ($\gamma^0_{T}$) in linear power. This is a severe semantic mismatch. To align the live client with the training data, the CDSE client must be updated to request `SIGMA0_ELLIPSOID` with geometric orthorectification.

## 2. TUM Representation
- **Quantity:** Calibrated Sigma Naught ($\sigma^0$)
- **Scale:** Decibels (dB)
- **Terrain Correction:** Geometric only (Orthorectification)
- **Radiometric Terrain Flattening:** No
- **Source Engine:** Google Earth Engine default `COPERNICUS/S1_GRD` pipeline.

## 3. Evidence
1. **Paper Statement:** Ebel, Saha & Zhu (2021) state: *"The (ascending orbit) Sentinel-1 SAR observations are downloaded via Google Earth Engine... and coordinate-transformed via GDAL."* They do not mention applying custom radiometric terrain flattening (RTC), which is a complex custom procedure in GEE.
2. **GEE Default Pipeline:** The default GEE `COPERNICUS/S1_GRD` collection applies thermal noise removal, radiometric calibration to **Sigma Naught**, and orthorectification using SRTM. It then log-scales the output to decibels.
3. **Data Inspection:** Inspecting the raw TUM GeoTIFFs (e.g., Montpellier) reveals a range of approx `[-36.1, +24.7]` with a mean of `-11.0`. These are undeniably decibel values. If they were linear power, they would be strictly positive and mostly clustered near 0.1.

## 4. Current CDSE Client Implementation
Inspection of `src/data_ingestion/sentinel_client.py` reveals:
- **backCoeff:** `GAMMA0_TERRAIN`
- **orthorectify:** `True`
- **DEM:** `COPERNICUS_30`
- **Output Units:** `LINEAR_POWER`
- **Pipeline:** It downloads linear Gamma Naught, which `src/preprocessing/sar_loader.py` then converts to decibels and clips. 
- **Mismatch:** The client is generating Radiometric Terrain Corrected $\gamma^0$, while the model will be trained on Geometric-only $\sigma^0$. 

## 5. CDSE / Sentinel Hub Available Representations
Sentinel Hub supports the following relevant coefficients:
- **`SIGMA0_ELLIPSOID`**: Calibrated backscatter referenced to the flat ellipsoid. Can be geometrically orthorectified (`orthorectify: True`). This preserves slope-induced brightness (slopes facing the radar are brighter).
- **`GAMMA0_ELLIPSOID`**: Backscatter normalized by the incidence angle relative to the ellipsoid.
- **`GAMMA0_TERRAIN`**: Radiometric Terrain Corrected (RTC) backscatter. Normalizes by the local illuminated terrain area using a DEM. Slope-induced brightness variations are flattened out.
*(Note: As expected physically, there is no `SIGMA0_TERRAIN` option, as terrain-normalizing sigma yields gamma).*

## 6. Sentinel Hub vs openEO Comparison
Both Sentinel Hub Process API and openEO support identical semantic generation of these products:
- Sentinel Hub Process API: `processing.backCoeff = "SIGMA0_ELLIPSOID"` + `processing.orthorectify = True`
- openEO: `sar_backscatter(coefficient="sigma0-ellipsoid", ...)` followed by orthorectification processes.

## 7. TUM vs CDSE Semantic Match Decision Table

| TUM Representation | Closest CDSE Representation | Exactly Reproducible? | Reason |
| :--- | :--- | :--- | :--- |
| $\sigma^0$ ellipsoid (dB) + Orthorectification | `SIGMA0_ELLIPSOID` (linear) + `orthorectify: True` → locally converted to dB | **Almost Exact** | Semantically identical. Minor numerical differences will arise from DEM choices (GEE SRTM vs CDSE Copernicus), interpolation kernels (bilinear vs nearest), and SNAP processing version drifts. |

## 8. Final Recommendation

**TRAINING REPRESENTATION:**
Sigma0 Ellipsoid, Orthorectified, Decibels (GEE Default)

**LIVE CDSE REPRESENTATION:**
`backCoeff = SIGMA0_ELLIPSOID`, `orthorectify = True`, converted to dB locally.

**ACCESS ROUTE:**
Sentinel Hub Process API

**EXACT MATCH:**
No

**IF NO — DOCUMENTED MISMATCH:**
Differences in underlying DEMs (GEE's SRTM 30m vs CDSE's Copernicus 30m) and backend orthorectification interpolation algorithms will result in slight sub-pixel spatial shifts and minor radiometric variations. The fundamental semantic quantity (Sigma Naught without radiometric flattening) matches.

**RATIONALE:**
Since the TUM dataset is fixed to GEE's default Sigma0 output, the live client must match this semantic representation to prevent domain shift. Training on Sigma0 (where hillsides are bright) and inferencing on Gamma0_Terrain (where hillsides are flattened) would severely degrade performance in topographically varied AOIs.

## 9. Limitations & Note on Testing
- No test cities were used to inform this analysis.
- The `sar_loader.py` must be carefully modified to accept the fact that TUM data is *already* in decibels, while the CDSE data requires local linear-to-dB conversion.
