# OSCD Validation Split Analysis

## 1. Scene Analysis (All 14 Training Cities)

The OSCD dataset provides 14 official training cities and 10 official test cities. To train and validate our models effectively without touching the official test cities, we must extract a validation split from the 14 training cities. 

The following table details the metadata for all 14 official training scenes:

| City | Size (H x W) | Total Pixels | Change Ratio | VV Mean (dB) | VH Mean (dB) |
|---|---|---|---|---|---|
| **abudhabi** | 799 x 785 | 627,215 | 3.76% | -13.48 | -21.58 |
| **aguasclaras** | 471 x 525 | 247,275 | 1.64% | -8.17 | -14.45 |
| **beihai** | 902 x 772 | 696,344 | 2.49% | -11.47 | -17.63 |
| **beirut** | 1180 x 1070 | 1,262,600 | 2.69% | -8.93 | -15.69 |
| **bercy** | 395 x 360 | 142,200 | 0.74% | -3.88 | -8.70 |
| **bordeaux** | 517 x 461 | 238,337 | 1.00% | -6.35 | -12.90 |
| **cupertino** | 1015 x 788 | 799,820 | 2.37% | -7.97 | -16.13 |
| **hongkong** | 695 x 540 | 375,300 | 3.56% | -11.06 | -17.11 |
| **mumbai** | 858 x 557 | 477,906 | 2.56% | -7.63 | -14.62 |
| **nantes** | 522 x 582 | 303,804 | 1.14% | -6.72 | -13.85 |
| **paris** | 408 x 390 | 159,120 | 0.29% | -3.97 | -11.01 |
| **pisa** | 776 x 718 | 557,168 | 1.64% | -9.19 | -16.19 |
| **rennes** | 339 x 563 | 190,857 | 2.58% | -6.46 | -14.25 |
| **saclay_e** | 639 x 688 | 439,632 | 0.99% | -8.20 | -14.82 |

> **Note**: High VV/VH means highly dense/bright urban structures. Low VV/VH usually indicates water, desert, or flat land.

---

## 2. Recommended Validation Split

We recommend the following 3 cities for the validation set:
1. **Hong Kong**
2. **Mumbai**
3. **Paris**

By choosing these three, we extract **~1.01M pixels (~15.5%)** for validation, leaving **~5.49M pixels (~84.5%)** for training. This closely adheres to an ideal 85/15 volumetric split for machine learning datasets.

---

## 3. Resulting Training Set

The remaining 11 cities will stay in the training set:
`abudhabi`, `aguasclaras`, `beihai`, `beirut`, `bercy`, `bordeaux`, `cupertino`, `nantes`, `pisa`, `rennes`, `saclay_e`.

---

## 4. Reason for Each Validation Choice

The goal is to select 3 validation cities that provide reasonable diversity rather than simply taking the first, last, or random three. 

* **Hong Kong (High Change, Lower Backscatter)**: At 3.56% change, it tests the model's ability to detect high volumes of change in a coastal, hilly topology where the overall SAR backscatter is lower (-11.06 dB VV).
* **Mumbai (Mid Change, Mid Backscatter)**: At 2.56% change and -7.63 dB VV mean, this is the "average" scene. It provides a balanced, highly representative urban environment to gauge baseline model health.
* **Paris (Extremely Low Change, High Backscatter)**: At just 0.29% change and an intensely bright -3.97 dB VV mean, this scene tests a critical edge case. It evaluates the model's ability to resist false positives in incredibly dense, bright urban cores where almost nothing has actually changed.

### Why not other cities?
* **Beirut**: It contains 1.26M pixels (nearly 20% of the entire dataset alone). Removing it from training would severely starve the model of training data.
* **Abu Dhabi**: It is the only major desert environment in the training set. If we move it to validation, the model never learns desert signatures and will likely fail on official test cities like Dubai and Las Vegas.

---

## 5. Limitations Caused by Having Only 14 Training Scenes

* **Geographical Starvation**: With only 14 scenes, taking *any* 3 scenes away removes ~21% of the geographical diversity. For example, removing Hong Kong takes away a unique high-rise coastal topology. The model simply has fewer unique typologies to learn from.
* **Class Imbalance Fragility**: The dataset has extreme variance in change ratios (0.29% in Paris vs 3.76% in Abu Dhabi). With so few scenes, the exact combination of validation cities causes wild swings in the overall class imbalance of the remaining training pool, which can make the training loss unstable if not weighted properly.
