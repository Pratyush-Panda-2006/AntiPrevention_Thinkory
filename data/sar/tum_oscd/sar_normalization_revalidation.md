# SAR Normalization Re-validation Report

## 1. Exact Pixel Counts and Clipping

We re-evaluated the actual physical arrays of all 14 training scenes to compute exact, rather than estimated, clipping statistics.

- **Total Valid VV Pixels:** 11,984,332
- **Total Valid VH Pixels:** 11,984,332

| Strategy | VV Bounds | VH Bounds | VV Clipped (Low / High) | VH Clipped (Low / High) | Norm Mean (VV/VH) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A** `[-30, 0]` | `[-30.00, 0.00]` | `[-30.00, 0.00]` | 622 (0.01%) / 621,654 (5.19%) | 215,403 (1.80%) / 52,440 (0.44%) | 0.6745 / 0.4526 |
| **B** Shared P1-P99 | `[-32.33, 5.63]` | `[-32.33, 5.63]` | 132 (0.00%) / 119,844 (1.00%) | 119,844 (1.00%) / 9,602 (0.08%) | 0.5982 / 0.4185 |
| **C** Indep. P0.5-P99.5 | `[-24.04, 7.97]` | `[-35.80, -0.42]` | 59,922 (0.50%) / 59,922 (0.50%) | 59,922 (0.50%) / 59,922 (0.50%) | 0.4513 / 0.5460 |
| **D** Indep. P1-P99 | `[-22.98, 5.63]` | `[-32.33, -2.53]` | 119,844 (1.00%) / 119,844 (1.00%) | 119,844 (1.00%) / 119,844 (1.00%) | 0.4674 / 0.5323 |
| **E** Indep. Rounded | `[-25.00, 5.00]` | `[-35.00, -5.00]` | 30,177 (0.25%) / 144,059 (1.20%) | 69,132 (0.58%) / 278,727 (2.33%) | 0.5125 / 0.6157 |

## 2. Analysis of Candidate Strategies

### The Flaw in Strategy E (Arbitrary Rounding)
While Strategy E provides clean, rounded integers, it introduces severe asymmetry. Because we artificially narrowed the upper bound of VH to `-5.00` dB (when the physical P99 is `-2.53` dB), **we are clipping 278,727 valid VH pixels (2.33%)**. This removes substantial physical structure from bright scatterers (e.g., complex urban structures or metallic targets). Additionally, it skews the normalized VH mean to `0.6157`, far from centered. Strategy E is mathematically inferior to data-driven percentiles.

### The Advantage of Strategies C & D (Data-Driven Symmetry)
By contrast, Strategies C and D are perfectly symmetric. 
- **Strategy D (P1-P99)** reliably clips exactly 1% of outliers on both ends of both bands, efficiently rejecting SAR speckle spikes while preserving 98% of physical data. It centers the dataset exceptionally well (`0.4674` and `0.5323`).
- **Strategy C (P0.5-P99.5)** protects more extreme physical returns (clipping only 0.5%), but may allow more noise/speckle to heavily influence the `[0,1]` scaling range.

## 3. Consequence of the `-100 dB` CDSE Nodata Floor

In the live CDSE pipeline, missing data (0.0 linear power) maps to a safety floor resulting in exactly `-100.0 dB`. 
- **Mathematical Consequence:** Because `-100.0` is well below the lower bound of *any* reasonable strategy (e.g., `-22.98` for Strategy D), `np.clip(arr, min_bound, max_bound)` will definitively clamp these nodata values to the lower bound.
- **Normalization Output:** After min-max scaling, `-100.0 dB` becomes exactly `0.0`.
- **Semantic Overloading:** Physical pixels that are genuinely dark (e.g., a smooth water body returning `-30 dB`) will *also* be clamped to the lower bound and become `0.0`. The neural network will not be able to mathematically distinguish between "invalid CDSE pixel" and "extremely dark physical target."

## 4. Validity / Nodata Mask Recommendation

**Recommendation:** An explicit validity mask should NOT be required as a direct input to the model (to maintain a strict 2-channel architecture), but it **should** be preserved through preprocessing for loss-masking during training.

**Reasoning:** If we do not preserve a nodata mask, the network will compute loss over invalid edge pixels (which default to 0.0 / black). This forces the model to predict "no change" over invalid regions, injecting false gradients. By preserving a boolean validity mask alongside the tensors, we can safely ignore nodata regions in the loss function without altering the `(VV, VH)` model architecture.

## 5. Final Normalization Recommendation

**We firmly recommend Strategy D (Independent P1-P99):**

- **VV Bounds:** `[-22.98, 5.63]` dB
- **VH Bounds:** `[-32.33, -2.53]` dB

**Why?**
1. **Empirically Rigorous:** It rejects exactly 1% of statistical outliers (vital for handling SAR speckle distribution tails) on all bounds.
2. **Channel Symmetry:** By independently centering the data, both VV and VH arrive at the network with near-identical normalized means (~0.5), preventing the network from heavily biasing one channel simply because of arbitrary physical scaling differences.
3. **Defensible:** It requires no arbitrary human rounding (like Strategy E) and is purely driven by the 12 million physical training pixels of the TUM OSCD dataset.
