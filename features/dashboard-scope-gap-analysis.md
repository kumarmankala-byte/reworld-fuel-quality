# Dashboard Scope Gap Analysis & Implementation Requirements
**Reworld Haverhill — Waste Intelligence Dashboard**
*Bright AI · Workstream 2 · Version 1.0 · 2026-05-05*
*Prepared for: Customer Signoff*

---

## Status Legend

| Symbol | Meaning |
|---|---|
| ✅ Live | Implemented and serving real data |
| ⚠️ Demo Preview | UI exists, data is synthetic or partially connected |
| ❌ Not Built | Required by scope, not yet started |
| 🔧 Bug | Built but broken — needs fix |

---

## User Roles

This document is organized against three primary users identified for this dashboard:

| Role | Primary Need |
|---|---|
| **Crane Operator** | Know where to pick and deposit in the pit to homogenize BTU; be alerted to hotspots or hazards |
| **Control Room Operator** | See what fuel quality is coming into the furnace 15–20 min ahead; respond to thermal/contaminant alerts |
| **Shift Manager** | Estimate total waste volume, composition mix, and homogeneity of what arrived during the shift |

---

## Scope Coverage Matrix

| Scope Item | Status | Crane Op | Control Room | Shift Mgr |
|---|---|---|---|---|
| Moisture content (IR, chute) | ✅ Live | — | Primary | Supporting |
| Chute fill level | ✅ Live | — | Primary | Supporting |
| Tipping floor thermal monitoring | ✅ Live | Supporting | Primary | Supporting |
| West pit thermal monitoring | ✅ Live | Primary | Supporting | — |
| Load uniformity (tipping floor thermal std dev) | ✅ Live | — | Supporting | Primary |
| Predicted BTU / HHV | ⚠️ Phase 4 | — | Primary | Primary |
| Waste classification (11 categories) | ❌ Missing | Supporting | Primary | Primary |
| Plastic fraction estimate | ⚠️ Phase 3 | — | Supporting | Supporting |
| Organic fraction estimate | ⚠️ Phase 3 | — | Supporting | Supporting |
| Contaminant detection — tipping floor | ⚠️ Phase 2 | — | Primary | — |
| Contaminant detection — pit cameras | ❌ Missing | Supporting | Primary | — |
| Homogenous load detection | ⚠️ Phase 1 (building) | — | Primary | Primary |
| Pit zone BTU/composition map | ⚠️ Phase 2 | Primary | — | Supporting |
| Crane homogenisation guidance | ⚠️ Phase 2 | Primary | — | — |
| Shift-level waste summary | ❌ Missing | — | — | Primary |
| Pixel frequency heatmap (west pit) | 🔧 Bug | Supporting | Supporting | — |

---

## Gap 1 — Waste Classification (11 Categories)

**Scope requirement:** Classify waste as Yard Waste, Food, Wood, Paper, Cardboard, Plastics, Textiles, Rubber, Leather, Misc. Organics, Dirt/Ashes using IR & RGB cameras going into the Hopper.

**Current status:** ❌ Not built. No classification output exists anywhere in the dashboard. No training pipeline, no model, no UI panel.

**Primary users:** Control Room Operator (to anticipate BTU mix), Shift Manager (to understand composition of received waste).

### What needs to be built

#### 1a. Data labeling pipeline
- Set up a labeling tool (e.g., Label Studio or CVAT) pointed at the existing RGB frames from `achute` and `chuteb`
- Define a labeling schema with the 11 scope categories
- Target: **200 labeled frames per category minimum** (2,200 total labels) before model training can begin
- Recommend: start with the 4 highest-BTU-impact categories first — Plastics, Paper/Cardboard, Food, Yard Waste
- Assign: Reworld floor staff or Bright AI annotators — **requires Reworld signoff on who labels**

#### 1b. Classification model
- Input: RGB JPEG frames from `achute` or `chuteb` (1280×720)
- Model: lightweight CNN or ViT variant (e.g., MobileNetV3 or EfficientNet-B0) running on JupyterHub GPU
- Output per frame: top-3 waste categories with confidence scores + dominant category
- Training data: labeled frames from 1a
- Validation: held-out labeled set + Reworld operator review of edge cases
- Inference cadence: once per camera frame arrival (triggered by S3 upload, ~1 frame/min)

#### 1c. Dashboard panel — Waste Composition View (new panel, Furnace Feed tab)
Add a "Waste Composition" section to the **Furnace Feed** tab showing:

```
Chute A — Current Composition Estimate         Chute B — Current Composition Estimate
┌─────────────────────────────────────────┐    ┌─────────────────────────────────────────┐
│  Dominant type:  PLASTICS  (38%)         │    │  Dominant type:  PAPER/CARD  (54%)      │
│  ░░░░░░░░░░░░░░  Plastics    38%         │    │  ░░░░░░░░░░░░░░  Paper/Card   54%        │
│  ░░░░░░░░░░      Paper/Card  29%         │    │  ░░░░          Food          22%         │
│  ░░░░            Yard Waste  16%         │    │  ░░            Plastics      14%         │
│  ░░              Other       17%         │    │  ░░            Other         10%         │
└─────────────────────────────────────────┘    └─────────────────────────────────────────┘
  Last updated: 2026-05-05 09:28               Last updated: 2026-05-05 09:31
  Model confidence: 84%                        Model confidence: 71%
```

- Time series chart below: rolling 2-hour stacked bar chart of composition % (15-min buckets)
- Phase: **Q3 2026** (dependent on labeling completing by end of May 2026)

#### 1d. Operator actions integration
When dominant category is Yard Waste or Food (high moisture, low BTU), auto-generate an action:
> "Chute A: High organic fraction detected — expect moisture spike in 15 min. Consider reducing feed rate."

---

## Gap 2 — Predicted BTU / HHV (Live Signal)

**Scope requirement:** Predict approximate BTU (HHV) from IR & RGB data at the chute.

**Current status:** ⚠️ Demo Preview only (Phase 4). Chart renders but uses synthetic data. Blocked on historian-camera data overlap (gap closes ~summer 2026 per dashboard footnote).

**Primary users:** Control Room Operator (combustion adjustment), Shift Manager (shift fuel quality summary).

### What needs to be built

#### 2a. Historian data integration
- Obtain plant historian export from Reworld Haverhill (DCS/SCADA): steam output (kg/h), grate temperature, secondary air, O2 reading — aligned to timestamps
- The gap is that historian timestamps do not yet overlap with camera timestamps; once they do, correlation is possible
- **Action required from Reworld:** confirm historian export format (CSV, OPC-UA, PI System) and access method
- Target integration: **summer 2026**

#### 2b. BTU regression model
- Inputs: moisture_index (live), plastic_fraction (Phase 3), organic_fraction (Phase 3), fill_level_pct, chute temp signals
- Target: HHV in MJ/kg (from historian boiler efficiency back-calculation)
- Model: ridge regression or gradient boosting on rolling 30-day training window
- Validation: compare predicted HHV to historian-derived HHV on held-out periods

#### 2c. Dashboard panel update — Predicted HHV (live, Furnace Feed tab)
- Replace synthetic demo chart with live predictions
- Add green band overlay showing target combustion zone (11–12 MJ/kg as per current demo)
- Add action thresholds: `> 13 MJ/kg → pre-cool airflow`, `< 9 MJ/kg → increase feed rate` (as shown in demo)
- Add confidence interval band on the chart

---

## Gap 3 — Contaminant Detection (Pit Cameras)

**Scope requirement:** Detect contaminants from **Pit & Tipping floor cameras** — prior to going into hopper: bulky items (water heaters, propane tanks), compressed gas cylinders, large textiles (carpets, mattresses), large quantities of homogenous fuel.

**Current status:**
- Tipping floor: ⚠️ Demo Preview (Phase 2, Q3 2026)
- **Pit cameras (west pit): ❌ Not planned or built** — the scope explicitly names "Pit & Tipping floor cameras" but only tipping floor appears in any phase plan

**Primary users:** Control Room Operator (safety alert), Crane Operator (directed removal before items enter hopper).

### What needs to be built

#### 3a. Tipping floor contaminant model (Phase 2 — scope already captured)
- Object detection model (YOLOv8 or similar) on RGB frames from tipping1/2/3
- Classes: propane cylinder, compressed gas cylinder, mattress, carpet, water heater
- Requires: ~50–100 labeled examples per class (noted in current demo)
- Output: bounding-box alert with camera, zone, timestamp, confidence, recommended action
- **Reworld action required:** confirm which tipping floor camera angles have best coverage of each contaminant type

#### 3b. Pit (west pit) contaminant detection — NEW, not yet planned
- Same model architecture as 3a, retrained or fine-tuned on west-pit RGB frames
- West-pit RGB frames are currently fetched on-demand from S3 — need to add to sync pipeline
- Note: west-pit has a known camera clock delay of up to 48 hours; model output timestamp must use S3 upload time, not filename timestamp
- Dashboard change: add contaminant alert card to **Safety Monitor** tab under the west pit section, mirroring the tipping floor contaminant panel structure

#### 3c. Alert escalation for safety-critical contaminants
- Propane/gas cylinders: red CRITICAL alert, sound (browser `Audio API`), persist until operator dismisses
- Water heaters/large metal: orange WARNING, appear in Operator Actions
- Large textiles: yellow CAUTION (grate jam risk), appear in Operator Actions
- **Gap:** current dashboard has no persistent alert acknowledgement mechanism — alerts clear on next refresh

---

## Gap 4 — Homogenous Load Detection (Tipping Floor)

**Scope requirement:** Detect large quantities of homogenous fuel (e.g., truckload of paper) before it enters the pit, to allow pre-emptive combustion adjustment.

**Current status:** ⚠️ Phase 1 — "Building Now". Demo UI exists with synthetic examples (paper/cardboard, plastic film). Load uniformity chart (thermal spatial std dev) is live.

**Primary users:** Control Room Operator (advance combustion adjustment), Shift Manager (load pattern tracking).

### What needs to be built

#### 4a. Homogenous load classifier (unsupervised, no labeling required per demo note)
- Method: thermal spatial std dev < threshold → flag as single-material load (this signal is already live as the uniformity chart)
- Add: when std dev dips below threshold, trigger RGB classification to name the material type (reuse Gap 1 model)
- BTU impact estimator: lookup table of material type → expected BTU range → furnace lead time calculation
- Output: "Paper/Cardboard load on Tipping 2 — LOW BTU slug in ~75 min. Prepare grate adjustment."

#### 4b. Dashboard panel — make live (Tipping Floor tab)
- Remove "DEMO PREVIEW" badge and synthetic data once 4a is implemented
- Add the material name and BTU direction to the alert card
- Add history log: last 5 homogenous load detections with timestamps and BTU outcomes (once historian data is available)

---

## Gap 5 — Shift-Level Summary (Shift Manager View)

**Scope requirement:** Support shift managers in estimating how much waste arrived and how homogenous it is.

**Current status:** ❌ Not built. No shift summary view, no tonnage estimates, no homogeneity KPI, no reporting.

**Primary users:** Shift Manager exclusively.

### What needs to be built

#### 5a. New dashboard tab — "Shift Summary"
Add a fourth tab (or sub-tab under Operator View) with the following sections:

**Section 1 — Shift Snapshot (header cards)**
```
Shift: Day / 06:00–18:00           As of: 2026-05-05 14:30
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Loads        │  │ Homogenous   │  │ Avg Moisture │  │ Avg BTU Est  │
│ Detected     │  │ Load Flags   │  │ (Chute B)    │  │ (Chute B)    │
│    47        │  │    3  (6%)   │  │   0.58       │  │   11.4 MJ/kg │
│ tipping floor│  │ ⚠ 1 critical │  │  MODERATE    │  │  IN TARGET   │
└──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
```

**Section 2 — Composition Mix (rolling shift)**
- Stacked bar chart: waste composition % over the shift (from Gap 1 classification model)
- Y-axis: fraction of frames per category
- X-axis: time in shift

**Section 3 — Waste Volume Proxy**
- Note: no weigh-bridge integration exists — volume is estimated from tipping floor camera coverage
- Show: load count (thermal uniform dips = single truck deposit events) as a proxy
- Show: chute fill level average and variance as an indicator of throughput consistency
- **Reworld action required:** confirm if weigh-bridge data is available via historian for actual tonnage

**Section 4 — Thermal Incidents**
- Table of all alerts in the shift: timestamp, camera, type, peak temp, status (active/cleared)
- Export button: download shift report as CSV

#### 5b. Shift boundary configuration
- Add a settings endpoint `GET /api/config` and `POST /api/config` for shift start/end times
- Default: Day 06:00–18:00, Night 18:00–06:00
- Shift summary aggregates data within the active shift window

---

## Gap 6 — Pixel Frequency Heatmap Blank (West Pit)

**Current status:** 🔧 Bug. The "West Pit — % of Frames Where Pixel Exceeded 50°C" chart renders as a fully blank/dark panel in the Safety Monitor tab (slides 7 right panel). The heatmap data (`west-pit_hot50.npy`) exists in `data/eda_cache/` based on README, but the chart appears empty.

**Affected users:** Control Room Operator, Crane Operator.

### What needs to be fixed

- Check that `data/eda_cache/west-pit_hot50.npy` is non-zero: `np.load('data/eda_cache/west-pit_hot50.npy').max()` — if it returns 0.0, no pixels have ever exceeded 50°C in the west pit, which is consistent with the live data (current west pit max is 51°C, borderline)
- If the array is all-zeros, the Plotly heatmap renders as a uniform dark panel with no visible color variation — this is a data situation, not a code bug
- **Fix option A:** lower the threshold in `pit_tipping_eda.ipynb` to show % of frames exceeding 35°C instead of 50°C (caution threshold), which will produce visible data
- **Fix option B:** replace the static heatmap with a "rolling 7-day pixel count above caution threshold" that auto-scales, with a note: "No pixels have exceeded 50°C" when truly empty
- Also verify the `_clean()` function in `dashboard_api.py` is not converting near-zero floats to null before they reach the chart

---

## Gap 7 — Alert Acknowledgement & Persistence

**Current status:** ❌ Not built. All alerts in the Operator Actions panel and banner reset on each refresh. There is no way for an operator to acknowledge, dismiss, or log a response to an alert.

**Affected users:** Crane Operator, Control Room Operator.

### What needs to be built

- Add a `POST /api/alert/ack` endpoint accepting `{alert_id, operator, note}`
- Store acknowledgements in a local SQLite file (`data/alerts.db`) with timestamp, operator id, and optional note
- Show acknowledged alerts with a ✓ badge in the Operator Actions panel
- Unacknowledged critical alerts (CRITICAL level) should persist across refreshes until explicitly dismissed
- Scope: MVP is local file-based; no external notification system (email/SMS) required in Workstream 2

---

## Gap 8 — West Pit RGB Sync Not in Dashboard Sync

**Current status:** ⚠️ Operational gap. The dashboard's "Sync from S3" button only syncs chute A and chute B CSV data. West pit and tipping floor data requires manually re-running `pit_tipping_eda.ipynb` in a separate notebook.

**Affected users:** All.

### What needs to be built

- Add west-pit and tipping floor CSV sync + EDA cache regeneration to the `POST /api/sync` endpoint
- Move the EDA notebook logic (signal extraction for west-pit and tipping floor) into a standalone Python script `scripts/pit_tipping_signals.py` (mirroring `chute_signals.py`)
- This script is then called by the sync job after the chute sync completes
- Estimated effort: 1–2 days (mostly code extraction from notebook)

---

## Gap 9 — Contaminant Scope: Pit Cameras Not Included in Phase Plan

**Current status:** ❌ The scope slide explicitly names "Pit & Tipping floor cameras" for contaminant detection. The current phase plan (Phase 2, Q3 2026) covers tipping floor only. Pit (west pit) contaminant detection has no assigned phase.

**Recommendation:** Add **Phase 2b** to the roadmap for pit contaminant detection, with Q4 2026 target. Key difference from tipping floor: west pit RGB has up to 48-hour clock lag, so model output must use S3 upload timestamps throughout. Also: the crane arm and grapple will appear in west pit frames and must be masked from contaminant detection (to avoid false positives).

---

## Implementation Phases Summary

| Phase | Feature | Target | Labeling Needed | Reworld Action Required |
|---|---|---|---|---|
| **Live** | Moisture index, fill level, tipping thermal, pit thermal, load uniformity | Now | — | — |
| **Phase 1** | Homogenous load detector (live, remove demo badge) | Q3 2026 | None (unsupervised) | Confirm BTU lookup table values |
| **Phase 2** | Tipping floor contaminant detection | Q3 2026 | 50–100 labels/class | Confirm labeling responsibility |
| **Phase 2b** | Pit (west pit) contaminant detection | Q4 2026 | 50–100 labels/class (transfer from Phase 2) | Confirm west pit RGB coverage suitability |
| **Phase 3** | Waste classification (11 categories) | Q3–Q4 2026 | 200 labels/category | Assign labeler; start immediately |
| **Phase 3** | Plastic fraction + organic fraction (live) | Q4 2026 | Derived from Phase 3 labels | — |
| **Phase 4** | Predicted HHV (live) | Summer 2026+ | — | **Historian export: format + access method** |
| **Ongoing** | Pixel frequency heatmap fix | Sprint 1 | — | — |
| **Ongoing** | West pit sync integrated into Sync button | Sprint 1 | — | — |
| **Ongoing** | Alert acknowledgement + persistence | Sprint 2 | — | — |
| **New** | Shift Manager Summary tab | Q3 2026 | — | Confirm if weigh-bridge data available |

---

## Open Questions Requiring Reworld Signoff

| # | Question | Owner | Impact if unresolved |
|---|---|---|---|
| Q1 | Who labels the waste classification training data — Reworld floor staff or Bright AI? | Reworld | Phase 3 cannot start without labelers |
| Q2 | What format is the historian data in (CSV export, OPC-UA, PI System, other)? | Reworld | Phase 4 (BTU prediction) blocked |
| Q3 | Is weigh-bridge data available for actual tonnage per shift? | Reworld | Shift Summary shows load count proxy only, not real tonnage |
| Q4 | Are the scope BTU thresholds confirmed: `< 9 MJ/kg = increase feed rate`, `> 13 MJ/kg = pre-cool airflow`? | Reworld | Operator actions for BTU alerts can't be written without these |
| Q5 | Which tipping floor camera angles have best coverage of propane cylinders and bulky items? | Reworld | Affects labeling effort and model accuracy for Phase 2 |
| Q6 | Should contaminant alerts trigger an audible alarm in the control room or just a visual? | Reworld | Determines if browser Audio API or external paging system is needed |
| Q7 | Is the Pit Zone Composition Map guidance (crane pick/deposit instructions) safe to automate, or must it remain advisory only? | Reworld (Safety / Operations) | Legal/safety scoping for Phase 2 crane guidance feature |
| Q8 | Shift definition: confirm Day = 06:00–18:00, Night = 18:00–06:00, and whether there is a third shift | Reworld | Shift Summary tab configuration |
