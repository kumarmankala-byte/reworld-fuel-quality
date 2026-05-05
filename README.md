# Reworld Haverhill — Waste Intelligence Dashboard

FastAPI + Plotly dashboard for thermal-camera and IR-sensor data from the Reworld Haverhill waste-to-energy facility. Served through JupyterHub's proxy — no port-forwarding or networking changes required.

---

## Quick start

1. Open `dashboard_launch.ipynb`
2. Run **Cell 1** — kills any existing server on port 8050, starts a fresh one, prints the shareable URL
3. Share the URL with anyone who has a JupyterHub login

To restart cleanly after editing `dashboard_api.py`, just re-run Cell 1 — it kills the old process first.

---

## Directory layout

```
fastapi-jupyter-dashboard/
├── dashboard_api.py          # FastAPI backend — the entire app lives here
├── dashboard_launch.ipynb    # Launch / stop / log notebook
├── README.md                 # This file
│
├── data/                     # Processed signals and caches
│   ├── achute_signals/
│   │   └── signals.csv       # Chute A: fill %, moisture index, temps per frame
│   ├── chuteb_signals/
│   │   └── signals.csv       # Chute B: same schema
│   ├── eda_cache/            # Output of pit_tipping_eda.ipynb (west-pit + tipping)
│   │   ├── west-pit_signals.csv
│   │   ├── west-pit_max.npy          # 384×512 peak-temperature heatmap
│   │   ├── west-pit_hot50.npy        # 384×512 fraction-above-50°C map
│   │   ├── tipping{1,2,3}_signals.csv
│   │   └── tipping{1,2,3}_{max,mean,std,...}.npy
│   ├── s3_upload_times/      # Per-camera JSON: {stem → S3 LastModified epoch}
│   │   └── {achute,chuteb,west-pit,tipping1,2,3}.json
│   ├── camera_images/        # On-demand S3 fetches (west-pit + fallback for others)
│   └── west_pit_images/      # Legacy cache dir (still used by /api/west-pit/latest-image alias)
│
├── camera_data/              # Raw camera files synced from S3
│   ├── reworld-haverhill-achute/
│   │   ├── data_type=csv/    # IR thermal CSVs
│   │   └── data_type=rgb/    # 1280×720 JPEG frames
│   ├── reworld-haverhill-chuteb/   (same structure)
│   ├── reworld-haverhill-tipping{1,2,3}/
│   └── reworld-west-pit/     # CSV only — RGB images fetched from S3 on demand
│
└── scripts/                  # Processing scripts (moved from /home/shared/reworld/scripts)
    ├── chute_signals.py      # Computes achute / chuteb signals from raw CSV dirs
    ├── venv/                 # Python env used by chute_signals.py (numpy+pandas)
    └── ...
```

---

## Cameras and sensors

| Camera key  | Location            | Data type    | RGB resolution | Lead time to furnace |
|-------------|---------------------|--------------|---------------|----------------------|
| `achute`    | Chute A             | IR + RGB     | 1280×720      | 15–20 min            |
| `chuteb`    | Chute B             | IR + RGB     | 1280×720      | 15–20 min            |
| `west-pit`  | West waste pit      | IR + RGB     | 1280×720      | 45–80 min            |
| `tipping1`  | Tipping floor cam 1 | IR + RGB     | 1280×720      | 60–120 min           |
| `tipping2`  | Tipping floor cam 2 | IR + RGB     | 1280×720      | 60–120 min           |
| `tipping3`  | Tipping floor cam 3 | IR + RGB     | 1280×720      | 60–120 min           |

IR sensor resolution: **512 × 384 pixels** per frame (stored as CSV matrices).
RGB cameras capture 1280×720 JPEG frames alongside each CSV.

### S3 bucket layout

Bucket: `s3://bai-stl128tetra-uw2-data-field-data`

```
site=haverhill/facility=a/device_id=reworld-haverhill-achute/data_type={csv,rgb}/
site=haverhill/facility=chute/device_id=reworld-haverhill-chuteb/data_type={csv,rgb}/
site=haverhill/facility=tipping/device_id=reworld-haverhill-tipping1/data_type={csv,rgb}/
site=haverhill/facility=Tipping/device_id=reworld-haverhill-tipping2/data_type={csv,rgb}/
site=haverhill/facility=tipping/device_id=reworld-haverhill-tipping3/data_type={csv,rgb}/  (also facility=Tipping)
site=reworld/facility=west-pit/device_id=reworld-west-pit/data_type={csv,rgb}/
```

Note: tipping2 uses `facility=Tipping` (capital T). tipping3 appears under both casings.
Note: west-pit is under `site=reworld`, not `site=haverhill`.

RGB path structure within a prefix:
```
year={yyyy}/month={mm}/day={dd}/hour={hh}/minute={mm}/{YYYYMMDDHHmmSS}.jpg
```

---

## API endpoints

### `GET /`
Returns the full dashboard HTML shell (~27 KB). Static — no data. JavaScript fetches `/api/data` after load.

### `GET /api/data`
Returns all chart specs and summary stats as JSON. Called by the dashboard JS on every refresh.

Response shape:
```json
{
  "summary": {
    "pit_max": 42.3, "pit_level": "CAUTION", "pit_color": "#eab308", "pit_icon": "🟡",
    "pit_alert_frames": 120, "pit_total_frames": 220, "pit_hotspot_pct": 54,
    "tip_max": 38.1, "tip_level": "CAUTION", "tip_color": "...", "tip_icon": "...",
    "chute_fill": 78.5, "chute_moisture": 0.412, "moist_label": "MODERATE",
    "achute_fill": 75.1, "achute_moisture": 0.632,
    "chute_status": "NORMAL", "chute_color": "#22c55e",
    "actions": [["green", "Pit temperature normal..."], ...],
    "data_through": "2026-05-01 05:54", "generated": "2026-05-01 16:54:00"
  },
  "charts": {
    "pit_temp": { /* Plotly figure dict */ },
    "pit_maxmap": { ... },
    "pit_h50": { ... },
    "tipping_temp": { ... },
    "tipping_uniform": { ... },
    "achute_fill": { ... }, "achute_moisture": { ... }, "achute_combined": { ... },
    "chute_fill": { ... }, "chute_moisture": { ... }, "chute_combined": { ... },
    "plastic_frac": { ... }, "organic_frac": { ... }, "btu_pred": { ... }
  }
}
```

### `GET /api/image/{camera}`
Returns the latest RGB JPEG for a camera. `camera` must be one of: `achute`, `chuteb`, `west-pit`, `tipping1`, `tipping2`, `tipping3`.

Strategy:
1. Look up the latest stem (timestamp filename) from the pre-cached S3 upload-time index (`data/s3_upload_times/{cam}.json`)
2. Search the local `camera_data/{cam}/data_type=rgb/` tree for `{stem}.jpg` (fast path, avoids S3 hit for synced cameras)
3. If not found locally, fetch from S3 using the RGB prefix and date-based path structure, cache in `data/camera_images/{cam}/`
4. On failure, return a 1280×720 SVG placeholder

Response headers: `Cache-Control: no-store`, `X-Uploaded-At: {datetime} UTC`

west-pit has no local RGB files — it always fetches from S3 (step 3).

### `GET /api/west-pit/latest-image`
Legacy alias for `/api/image/west-pit`. Kept for bookmark compatibility.

### `POST /api/sync`
Starts a background S3 sync + signal recompute job. Non-blocking — returns immediately.

Steps performed:
1. `aws s3 sync` CSV files for achute from S3 into `camera_data/reworld-haverhill-achute/data_type=csv/`
2. `aws s3 sync` CSV files for chuteb from S3 into `camera_data/reworld-haverhill-chuteb/data_type=csv/`
3. Run `scripts/chute_signals.py` for achute → outputs `data/achute_signals/signals.csv`
4. Run `scripts/chute_signals.py` for chuteb → outputs `data/chuteb_signals/signals.csv`
5. Refresh S3 upload-time index for all 6 cameras (`data/s3_upload_times/*.json`)

Note: west-pit and tipping floor CSVs are NOT synced by this step — they require rerunning `pit_tipping_eda.ipynb` to regenerate the eda_cache. That notebook reads from `camera_data/reworld-west-pit/` and `camera_data/reworld-haverhill-tipping{1,2,3}/`.

### `GET /api/sync/status`
Polls sync progress. Returns `{running, log, started, finished, ok}`.

### `GET /healthz`
Returns `{"status": "ok"}`.

---

## Dashboard tabs

### Operator View (default tab)
Designed for control-room operators. Three sections:

**Instrument clusters** — large-format numbers for at-a-glance status:
- West Pit: max temperature, % of readings above 50°C
- Chute B: fill level %, moisture index, feed status chip
- Tipping Floor: thermal status chip, max temp recorded

**Operator Actions** — plain-English prioritised action list, auto-generated from live data. Color-coded: red = urgent, orange = warning, yellow = caution, green = nominal, blue = informational.

**Camera feeds** (6 cameras, native 16:9 aspect ratio):
- Row 1: Chute A | Chute B (2-column)
- Row 2: West Pit (full width)
- Row 3: Tipping 1 | Tipping 2 | Tipping 3 (3-column)

Images are served via `/api/image/{camera}` with cache-busting on every refresh. `object-fit: contain` preserves native aspect ratio with no cropping.

### Safety Monitor
For control-room and floor supervisor. Shows:
- West Pit live status card (peak temp, alert level)
- Tipping Floor thermal card
- West Pit temperature time-series chart
- West Pit peak-temperature heatmap (512×384, native IR aspect ratio via `scaleanchor`)
- West Pit % of frames exceeding 50°C heatmap
- Contaminant detection placeholder (Phase 2)

### Furnace Feed
For combustion engineers and control room. Shows both Chute A and Chute B:
- 4 stat cards: Chute A fill, Chute A moisture, Chute B fill, Chute B moisture
- 3 rows of side-by-side charts: combined (fill + moisture + max temp), fill only, moisture only

### Tipping Floor
For floor supervisors and dispatch. Shows:
- Tipping floor thermal uniformity chart (low std = single-material load)
- Homogenous Load Detector placeholder (Phase 1 — building)
- Pit Zone Composition Map placeholder (Phase 2 — Q3 2026)

---

## Data ordering: S3 upload time vs. camera timestamp

Camera filenames encode the **camera's internal clock** (e.g. `20260428164207.csv`). These can be significantly delayed relative to actual upload time due to buffering, connectivity drops, or camera restarts. Observed worst-case delays: west-pit ~48 hours, tipping2 ~23.5 hours.

The dashboard sorts all signals DataFrames by **S3 `LastModified` timestamp** (when the file actually reached S3), not by the filename timestamp. This prevents stale data from appearing as "current" and ensures the displayed time series matches operational reality.

Upload times are pre-fetched via `aws s3 ls --recursive` for each camera prefix and cached in `data/s3_upload_times/{cam}.json` as `{stem: epoch}`. This cache is refreshed at the end of each Sync operation.

---

## Key implementation notes

### Launching the server correctly

The launch notebook starts uvicorn with:
```python
subprocess.Popen(
    [PYTHON, '-m', 'uvicorn', 'dashboard_api:app',
     '--host', '0.0.0.0', '--port', str(PORT), '--log-level', 'warning',
     '--app-dir', '/home/shared/kumar/library/fastapi-jupyter-dashboard'],
    cwd='/home/shared/kumar/library/fastapi-jupyter-dashboard',
    ...
)
```

`--app-dir` is essential — without it uvicorn uses Python's module search path which may not include the project directory. `cwd` is also set for consistency. **Do not use the `scripts/` subdirectory as cwd** (there used to be a copy of `dashboard_api.py` there that caused confusion after the directory consolidation in May 2026).

### Heatmap aspect ratio (IR sensor)

IR sensor produces 512-wide × 384-tall data. To render at native aspect ratio in Plotly:
```python
yaxis=dict(autorange='reversed', scaleanchor='x', scaleratio=1)
```
Pass `h=None` to `_fig()` so Plotly auto-sizes the height instead of constraining it to a fixed pixel value.

### RGB camera aspect ratio

All RGB cameras output 1280×720 (16:9). In the HTML:
```css
.cam-img { aspect-ratio: 16/9; object-fit: contain; background: #0a0d14; }
```
`object-fit: contain` letterboxes if the actual image differs from 16:9; `cover` would crop. No distortion.

### moisture_masked column

`moisture_index` is unreliable when the chute is nearly empty (fill < 20%) — the sensor reads the bare chute wall. `moisture_masked` is computed in `load_data()` by masking those readings to NaN:
```python
df['moisture_masked'] = df['moisture_index'].where(df['fill_level_pct'] >= 20)
```
Applied to both achute and chuteb. Chart functions use `moisture_masked` so masked periods appear as gaps in the line.

### NaN safety

`json.dumps` crashes on `float('nan')`. All chart dicts pass through `_clean()` which recursively replaces NaN/Inf with `None` (renders as line gaps in Plotly.js).

### Dark theme

All charts share `DARK` layout dict. Pass through `_fig(fig, title, h=280, **extra_layout)`. Override any key with `extra_layout` kwargs.

### Chart label reuse

`chart_chute_fill`, `chart_chute_moisture`, and `chart_chute_combined` accept an optional `label` parameter (default `'Chute B'`). Pass `label='Chute A'` to reuse them for achute without duplication.

### S3 upload-time cache

`data/s3_upload_times/{cam}.json` maps `"YYYYMMDDHHmmSS" → epoch_float`. Loaded once per process into `_s3_upload_cache` (in-memory dict). Cache is cleared and rebuilt at the end of each Sync. Call `_s3_upload_times(cam)` anywhere to get the dict.

### Per-camera image caching

For cameras without local RGB files (currently only west-pit), images are fetched from S3 and cached in `data/camera_images/{cam}/`. Only the single latest frame is kept — older files are deleted after a successful fetch.

---

## How to update west-pit / tipping floor data

These cameras are not synced by the dashboard's Sync button. To update:

1. Open `pit_tipping_eda.ipynb` (in `/home/shared/kumar/analysis/`)
2. Update the S3 sync cells if needed
3. Re-run the notebook — it regenerates `data/eda_cache/{west-pit,tipping*}_*.{csv,npy}`
4. Hit Refresh in the dashboard

West-pit RGB images are fetched on-demand from S3 via `/api/image/west-pit` — no local sync needed for the latest image.

---

## Dependencies

All available in the JupyterHub conda environment (`/opt/conda`):

```
fastapi
uvicorn[standard]
plotly >= 6.0
pandas
numpy
psutil
```

`chute_signals.py` runs in `scripts/venv/` (numpy + pandas only). The dashboard server always uses `sys.executable` (the conda kernel) to launch uvicorn.

### Tab initialisation — always call `showTab()` before `refresh()`

Tab visibility is controlled by the CSS rule `.tab{display:none}.tab.active{display:block}`. On page load, all tabs also get `visibility:hidden` so nothing flashes before data arrives.

The critical rule: **the init block must call `showTab('operator')` before `refresh()`**:

```javascript
document.querySelectorAll('.tab').forEach(el => el.style.visibility = 'hidden');
showTab('operator');   // ← adds 'active' class to the operator tab div
refresh();             // ← fetches data, populates DOM, sets visibility:visible
```

Without the `showTab()` call, the operator tab div has no `active` class on first load so it stays `display:none`. `applySummary()` still runs and writes values to the hidden DOM, but nothing is visible. Switching to another tab and back triggers `showTab('operator')` which adds the class and everything appears — making the bug look like a render race when it is actually a missing class.

If the default tab is ever changed, update both the `nb active` attribute on the nav button **and** the `showTab()` argument in the init block.

### Plotly 6+ note

`colorbar.titlefont` was removed in Plotly 6. Use:
```python
colorbar=dict(title=dict(text='Label', font=dict(color='#cbd5e1', size=10)))
```

---

## Access URL

```
https://jupyterhub.uw2.prod.core.bai-infra.net/user/kumar.mankala@bright.ai/kumar-reworld/proxy/8050/
```

Pattern: `{JUPYTERHUB_PUBLIC_URL}/proxy/{PORT}/`

The env var `JUPYTERHUB_PUBLIC_URL` is set automatically inside JupyterHub containers. Relative URLs in the HTML (e.g. `fetch('./api/data')`, `src="./api/image/achute"`) resolve correctly through the proxy.
