# TUM OSCD SAR Normalization Report

## 1. Methodology
This analysis is strictly based on the 14 official training scenes defined in `sar_scene_manifest.json`. The 10 test scenes were strictly excluded from statistical computations to guarantee no data leakage. 
For each scene, both T1 and T2 transformed SAR GeoTIFFs were parsed. The VV and VH bands were flattened and concatenated to compute global statistics across the entire training split.

## 2. Full Statistics (14 Training Scenes)
Global statistics were computed over all valid pixels for each polarization independently.

### VV Polarization
- **Min:** -42.94 dB
- **Max:** 40.95 dB
- **Mean:** -9.59 dB
- **Std:** 5.98 dB
- **Percentiles:**
  - P1: -22.98 dB
  - P5: -19.75 dB
  - P50 (Median): -9.51 dB
  - P95: 0.12 dB
  - P99: 5.63 dB
  - P99.5: 7.97 dB

### VH Polarization
- **Min:** -52.54 dB
- **Max:** 38.09 dB
- **Mean:** -16.49 dB
- **Std:** 6.08 dB
- **Percentiles:**
  - P1: -32.33 dB
  - P5: -26.64 dB
  - P50 (Median): -15.99 dB
  - P95: -7.25 dB
  - P99: -2.53 dB
  - P99.5: -0.42 dB

## 3. Nodata / Invalid Pixels Analysis
- **NaN/Inf Count:** `0` across all training arrays.
- **Explicit Nodata:** No explicit `nodata` metadata flag was present in the TUM GeoTIFFs.
- **Minimum Value:** The absolute minimum value in the entire TUM training set is `-52.54 dB`. 
- **CDSE Compatibility:** In the live CDSE pipeline, `NaN` or `0.0` linear values are floored to a safety epsilon (`1e-10`), which results in `-100.0 dB`. 
- **Conclusion:** There are **no artificial -100 dB** values in the TUM dataset. -100 dB is strictly an artificial artifact of the CDSE preprocessing. Because -100 dB falls significantly below the physical SAR range (e.g., -52 dB), it can safely be handled by standard lower-bound clipping (mapping to 0.0) without requiring an explicit nodata mask.

## 4. Candidate Normalization Comparison

We evaluated four normalization strategies mapping physical bounds to `[0, 1]`:

| Strategy | Bounds | VV Clipped (Low / High) | VH Clipped (Low / High) | Norm Mean (VV / VH) |
| :--- | :--- | :--- | :--- | :--- |
| **A (Shared [-30, 0])** | `[-30.0, 0.0]` | 0.01% / 5.19% | 1.80% / 0.44% | 0.67 / 0.45 |
| **B (Shared P1-P99)** | `[-32.3, 5.6]` | 0.00% / 1.00% | 1.00% / 0.08% | 0.60 / 0.42 |
| **C (Indep. P0.5-P99.5)** | VV: `[-24.0, 8.0]`, VH: `[-35.8, -0.4]` | 0.50% / 0.50% | 0.50% / 0.50% | 0.45 / 0.54 |
| **D (Indep. P1-P99)** | VV: `[-23.0, 5.6]`, VH: `[-32.3, -2.5]` | 1.00% / 1.00% | 1.00% / 1.00% | 0.46 / 0.53 |
| **E (Indep. Rounded)** | VV: `[-25.0, 5.0]`, VH: `[-35.0, -5.0]` | 0.25% / 1.20% | 0.58% / 2.33% | 0.51 / 0.61 |

**Analysis:**
- Strategy A is too narrow for VV, heavily clipping high-intensity urban returns (5.19%).
- Strategy B protects outliers but leaves VH significantly darker than VV, which may cause channel imbalance in the network.
- Independent scaling (Strategies C, D, E) appropriately normalizes each band's dynamic range.
- Strategy E provides clean, rounded integers highly representative of the P1-P99 distribution while maintaining well-centered normalized means (~0.5).

## 5. CDSE Compatibility Guarantee
The CDSE live pipeline (tested previously) returned a physical dB range spanning up to `8.7 dB`, matching the upper physical bounds of the TUM distribution (`P99.5 = 7.97 dB`). The lower end is floored at `-100.0 dB` due to nodata zero-padding. 
Because the lower normalization bounds (e.g., `-25` or `-35`) sit well above `-100.0`, the artificial CDSE nodata values will cleanly clip to the lower bound and normalize to exactly `0.0`. The distributions are semantically aligned and compatible.

## 6. Final Recommendation

**Recommended Strategy:** Independent Per-Band Normalization (Derived from Strategy E).

**Exact Constants:**
- **VV Band:** `[-25.0, 5.0]` dB
- **VH Band:** `[-35.0, -5.0]` dB

**Implementation Rules:**
- Normalization must be performed **independently** per band.
- Values below the lower bound (including CDSE's `-100 dB` nodata) clip to `0.0` after min-max scaling.
- Values above the upper bound clip to `1.0` after min-max scaling.
- No explicit NaN/nodata mask is required for the neural network, as invalid pixels cleanly default to `0.0`.

These bounds are strictly derived from the 14 training scenes and are now FROZEN for all downstream operations (training, validation, test, and live inference).
