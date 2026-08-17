# SAR Change Intelligence System — Final Hackathon Plan (v3)

> **Team context:** Full developer team, ample time, good GPU access.
> **Guiding principle:** benchmark first, select second, deploy third — with a
> working demo at every checkpoint, never just a roadmap.

---

## 1. Product Definition

> Given SAR imagery of an area at T1 and T2 (and ideally T3...Tn), detect what
> materially changed, localize it, classify it, filter out natural/environmental
> change, add temporal context, score it transparently with calibrated
> confidence, explain the evidence, and present it to a human analyst for
> review — including honestly flagging when the system is unsure.

**Primary pitch framing:**
> "Optical satellites can't see through clouds or at night. Ours can. We built
> a system that watches land change over time using radar, explains *why* it
> flagged something, tells a human how urgent it is — and says 'I don't know'
> when it isn't sure, instead of faking confidence."

**Use-case order (lead with these):** illegal construction/deforestation monitoring → disaster response (flood/landslide extent) → general infrastructure monitoring. Defense/border applications are mentioned only if directly asked, as one possible downstream use — never the headline.

---

## 2. Team Allocation

| Track | Owns | Deliverable |
|---|---|---|
| **ML — Detection** | Preprocessing, co-registration, Siamese U-Net, SNUNet-CD, training, evaluation, confidence calibration | Trained change-detection model + benchmark table + calibrated confidence outputs |
| **ML — Classification & Scoring** | Region extraction, feature engineering, Random Forest + XGBoost, rule-based priority engine, uncertainty logic | Region classifier + explainable, tunable priority score + "Uncertain" state handling |
| **Backend** | FastAPI, PostGIS, event schema, inference service, endpoints | Working API serving real model output |
| **Frontend** | Map (MapLibre/OpenLayers), before/after slider, timeline, event review panel, uncertainty UI, expansion animation (stretch) | Full interactive dashboard |
| **Data/Infra** | Dataset acquisition, Sentinel-1 pulls (Copernicus + Google Earth Engine as faster alt), dataset splits, config management, GPU scheduling | Clean, versioned datasets + reproducible configs |
| **Pitch/Presentation** | Demo script, honesty slide, team-narrative framing, rehearsal | Locked 90-second demo script + backup plan |

Run all tracks in parallel from day one. Backend/frontend build against a mocked API response shaped like the real one so they never wait on the model team.

State this explicitly in the pitch: *"This was built by a team that split ML, backend, and frontend from day one and integrated continuously — not bolted together the night before."*

---

## 3. Dataset Plan (staged, not mixed)

```
LEVIR-CD                    → pipeline validation (fast, clean, optical)
        ↓
Sentinel-1 + OSCD           → primary SAR training/evaluation
        ↓
Custom Sentinel-1 pairs     → demo-day real-world proof (2-3 pairs, pre-cached)
```

- SYSU-CD / SECOND / DSIFN kept as optional stretch datasets for extra robustness claims, not required for the core story.
- For real Sentinel-1 data, evaluate **Google Earth Engine's pre-calibrated, terrain-corrected backscatter** as a faster alternative to running the full SNAP pipeline from raw GRD — SNAP is reliable but slow (batch runs on hundreds of scenes have taken multiple days in reported workflows), so don't default to raw-SNAP-for-everything if a faster path exists for prototyping.
- **Never preprocess SAR pairs live in front of judges.** All demo-day pairs are processed and cached in advance.

---

## 4. Model Track

**Core models (trained and evaluated properly, not just run once):**
1. Siamese U-Net — baseline
2. SNUNet-CD — main candidate

**Framing for judges:** present SNUNet-CD as *"a strong, well-validated 2021-era benchmark architecture we adapted for SAR"* — not as state-of-the-art. This is more defensible under technical questioning than implying it's cutting-edge, since newer architectures now outperform it on public benchmarks.

**Stretch (only if core checkpoints finish early):** ChangeFormer or TinyCD as a third comparison point.

**Ablations worth running (genuinely interesting findings, not filler):**
- VV-only vs VV+VH input channels
- No speckle filter vs light speckle filter
- Optical (LEVIR-CD) F1 vs real SAR F1 — expect a real gap

**Confidence calibration — new in v3:**
Raw model confidence (softmax output) is not the same as true probability of correctness, and judges with ML background will notice if it's presented as if it were. Add a calibration step (temperature scaling or isotonic regression, fit on validation data) before any confidence number is shown to an analyst or a judge. This is roughly an hour of work and is a strong rigor signal.

**The SAR-realism honesty slide — new in v3:**
Prepare one slide, ready to show proactively or on question:
> "Our optical F1 (LEVIR-CD) is in the high-80s. Our real Sentinel-1 SAR F1 is meaningfully lower — this gap is expected and documented in SAR change-detection literature, and it's exactly why naive threshold-based systems fail on radar. Our pipeline is built around that reality, not around a benchmark number."

Own this gap before a judge finds it. It converts a weakness into a credibility signal.

---

## 5. Region Classification & Priority Scoring

```
Binary change mask
    ↓
Connected components / region extraction
    ↓
Region features (area, shape, VV/VH deltas, texture, persistence, expansion rate)
    ↓
Random Forest (baseline) + XGBoost (main candidate) → change category
```

Categories (first version, small and defensible):
1. Built-environment / construction
2. Road / linear infrastructure
3. Vegetation / land-cover
4. Water / environmental
5. Other / uncertain

**Priority engine — rule-based, transparent, tunable (not a strict if/else):**

```
priority_score =
    w1 * change_confidence
  + w2 * persistence
  + w3 * expansion_rate
  + w4 * region_significance
  + w5 * classification_confidence
  - w6 * natural_change_probability
```

Weights are exposed in the dashboard/config — visibly not a black box.

Levels: 0–25 LOW · 26–50 MEDIUM · 51–75 HIGH · 76–100 CRITICAL (prototype defaults, stated explicitly as such, not validated operational standards).

**Long-term model path (staged honestly):**
- **Phase A (hackathon):** rule-based, multi-factor, weighted, tunable — as above.
- **Phase B (post-hackathon / stretch):** every analyst decision (Confirm/Uncertain/Reject) becomes a labeled example. Once enough reviewed events exist, train a learned re-ranker on the same tabular features XGBoost already uses. Pitch this explicitly: *"our priority engine is rule-based and transparent today, and is designed to become a learned model as soon as we have analyst-labeled data — which our review workflow already collects."*

**Uncertainty as a first-class state — new in v3:**
"Uncertain" is not just a review-panel button — it's a distinct visual and logical state throughout the system:
- If natural-change probability and human-built probability are close, or persistence data is insufficient, the system outputs **"Uncertain — insufficient evidence"** instead of forcing a LOW/MEDIUM/HIGH call.
- The dashboard renders this with its own distinct color/badge (not a muted version of LOW).
- State this in the pitch directly: *"the system is designed to say 'I don't know' rather than force a confident-sounding wrong answer."* This directly preempts the "AI overclaims" concern any thoughtful judge will be listening for.

---

## 6. Full Dashboard (Frontend)

Screens:
1. Overview / map with event markers
2. Interactive map (MapLibre/OpenLayers)
3. Before/after slider
4. Change-mask overlay
5. Timeline view (per-event confidence over observations)
6. Model confidence panel (calibrated values, clearly labeled as such)
7. Priority card with visible evidence + weights
8. Uncertainty state (distinct styling, as above)
9. Analyst review panel: Confirm / Uncertain / Reject
10. Historical event page

**Stretch goal (frontend-heavy, mostly cheap since it's your strong track):**
If Checkpoint 4 finishes early, add a simple **expansion-over-time view**: the same site shown across 3–4 timestamps with the change region visibly growing, using data already produced by the temporal-persistence pipeline. This demonstrates the temporal USP visually to non-technical judges without new ML work.

---

## 7. Backend / Infrastructure

```
FastAPI
   ↓
Model inference service
   ↓
PostgreSQL + PostGIS
   ↓
Object/raster storage
   ↓
Web dashboard
```

Endpoints: `GET /scenes`, `GET /events`, `GET /events/{id}`, `GET /events/{id}/timeline`, `POST /inference`, `POST /review`, `GET /map/events`, `GET /health`.

Event schema includes: `change_confidence` (calibrated), `classification`, `classification_confidence`, `natural_change_probability`, `priority_score`, `priority_level` (including `UNCERTAIN`), `created_at`.

---

## 8. Four Demo-Ready Checkpoints

Each is a complete, working thing you could demo immediately after finishing it.

### Checkpoint 1 — "It detects change"
- LEVIR-CD pipeline end-to-end (loader → Siamese U-Net → mask)
- F1/IoU/Precision/Recall logged, reproducible (config + seed recorded)

### Checkpoint 2 — "It works on real SAR"
- Sentinel-1+OSCD trained and evaluated (VV/VH channels)
- SNUNet-CD benchmarked against Siamese U-Net on the same split
- 2–3 custom Sentinel-1 pairs pre-processed and cached
- Confidence calibration applied to outputs

### Checkpoint 3 — "It explains itself"
- Region extraction → Random Forest + XGBoost classification
- Rule-based priority score with visible weighted evidence
- Uncertainty state implemented and visually distinct

### Checkpoint 4 — "It's a product"
- FastAPI + PostGIS backend serving real events
- Full dashboard live, including uncertainty UI
- Stretch: expansion-over-time view if time allows

---

## 9. The 90-Second Demo Script (locked, word-for-word)

**0:00–0:15 — Hook**
> "Optical satellites can't see through clouds or at night. Ours can."
*(Show a cloud-obscured optical image next to a clear SAR image of the same site.)*

**0:15–0:40 — Live pipeline**
*(On the dashboard: select a before/after pair → change mask appears → region highlighted → classification label appears.)*
> "Here's a real Sentinel-1 pair. The system detects the changed region, classifies it, and estimates how likely it is to be natural versus built change."

**0:40–1:05 — The priority card (most airtime — this is the differentiator)**
*(Show the full event card: confidence, area, persistence, expansion, priority level, evidence list.)*
> "This is the part that matters: not just 'something changed,' but why we think so, how confident we are — calibrated, not raw model output — and how urgent it is. And if the evidence is thin, it says so."
*(Show one "Uncertain" example.)*
> "It doesn't force a confident-sounding wrong answer when it isn't sure."

**1:05–1:25 — One honest technical slide**
*(Benchmark table: Siamese U-Net vs SNUNet-CD, optical F1 vs real SAR F1.)*
> "Our optical F1 is strong. Our real SAR F1 is lower — that's expected, it's a harder, noisier modality, and our pipeline is built around that reality, not around a cherry-picked benchmark number."

**1:25–1:30 — Close**
> "Today it's transparent, rule-based prioritization. Every analyst review we collect becomes training data for a learned version of this — that's the roadmap."

---

## 10. Risks

| Risk | Mitigation |
|---|---|
| SAR preprocessing takes too long | Pre-process demo pairs in advance; evaluate Google Earth Engine as a faster alternative to raw SNAP; never process live |
| Model training doesn't converge in time | Checkpoint 1 (LEVIR-CD/optical) is the fallback demo if SAR training stalls |
| Team blocked waiting on model output | Backend/frontend build against a mocked API contract from day one |
| Overclaiming in the pitch | Only state benchmark numbers actually produced; label anything untested as "future work"; lead with the SAR-realism slide rather than waiting to be asked |
| Raw confidence mistaken for calibrated probability | Calibration step is mandatory before any number reaches the UI |
| System looks like a black box | Priority weights and evidence are always visibly rendered; "Uncertain" is a first-class state, not hidden |

---

## 11. One-Sentence Pitch

> **An explainable, SAR-based change-detection platform that tells analysts not just "something changed," but what changed, how confident the system honestly is, how persistent it is, and how urgently it deserves human review — including when it doesn't know — built and benchmarked end-to-end, not just proposed.**
