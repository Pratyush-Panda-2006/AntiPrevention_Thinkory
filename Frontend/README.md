# National Remote Sensing Centre (NRSC) - SAR Change Intelligence Suite

A unified multi-temporal Synthetic Aperture Radar (SAR) intelligence platform designed for orbital change detection, tactical defense monitoring, 3D terrain analysis, and satellite telemetry.

---

## Interconnected Suite Modules

All views are connected with universal global navigation, quick action launchpads, and live status routing:

| Page | Module | Key Features |
| :--- | :--- | :--- |
| [`index.html`](file:///c:/Users/prema.PRATYUSH/Downloads/stitch_sar_change_intelligence_explorer/index.html) / [`explorer.html`](file:///c:/Users/prema.PRATYUSH/Downloads/stitch_sar_change_intelligence_explorer/explorer.html) | **SAR Intelligence Explorer** | Multi-temporal acquisition controls (T1/T2 baseline dates), draggable ROI bounding box with live area calculation, live radar sweep, coordinate HUD reticle, target lock terminal logs, saved vector presets. |
| [`overview.html`](file:///c:/Users/prema.PRATYUSH/Downloads/stitch_sar_change_intelligence_explorer/overview.html) | **Defense Command Suite** | Strategic command dashboard, mission readiness, target tracker, threat clusters (Alpha, Beta, Gamma) with routing to analytics, executive metrics, audio feedback, radar sweep animations. |
| [`studio.html`](file:///c:/Users/prema.PRATYUSH/Downloads/stitch_sar_change_intelligence_explorer/studio.html) | **3D Terrain Studio** | Interactive before/after split-view slider (T1 baseline vs. T2 AI change mask), colormap toggles (Turbo, Viridis, Binary Red), layer opacity/brightness/contrast controls, radiometric speckle filter alerts. |
| [`analytics.html`](file:///c:/Users/prema.PRATYUSH/Downloads/stitch_sar_change_intelligence_explorer/analytics.html) | **Cluster Audit & Analytics** | Interactive anomaly cluster table with GSAP-powered fly-to map navigation, filter buttons, zero-detection simulation toggle, export actions (PDF, GeoJSON, CSV), and live coordinate readouts. |
| [`telemetry.html`](file:///c:/Users/prema.PRATYUSH/Downloads/stitch_sar_change_intelligence_explorer/telemetry.html) | **RISAT-2B Telemetry Hub** | Real-time IST clock, dynamic downlink latency jitter readout, GPU VRAM allocation monitor, CUDA RTX compute specs, and historical 7-day system uptime telemetry. |

---

## Cleaned Directory Structure

```
stitch_sar_change_intelligence_explorer/
├── index.html          # Main entry point (SAR Intelligence Explorer)
├── explorer.html       # Direct alias to SAR Explorer
├── overview.html       # Defense Command Overview Dashboard
├── studio.html         # 3D Terrain Intelligence Split-View Studio
├── analytics.html      # SAR Change Analytics & Cluster Audit Table
├── telemetry.html      # RISAT-2B Real-time Telemetry & Health Hub
├── README.md           # Documentation & Navigation Map
└── assets/
    └── screenshots/    # High-resolution screenshots of each suite view
        ├── analytics.png
        ├── explorer.png
        ├── overview.png
        ├── studio.png
        └── telemetry.png
```
