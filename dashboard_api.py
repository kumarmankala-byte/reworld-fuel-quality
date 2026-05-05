"""
Reworld Haverhill — Waste Intelligence Dashboard
FastAPI backend. Start from dashboard_launch.ipynb.
Access via jupyter-server-proxy at .../proxy/8050/

Endpoints
---------
GET  /                          HTML shell (static; JS fetches /api/data on load)
GET  /api/data                  All Plotly chart specs + summary stats as JSON
GET  /api/image/{camera}        Latest RGB JPEG for camera (achute|chuteb|west-pit|tipping1|2|3)
                                  - Checks local camera_data/ first, fetches from S3 if absent
                                  - Returns SVG placeholder on failure
GET  /api/west-pit/latest-image Legacy alias for /api/image/west-pit
POST /api/sync                  Background job: S3 sync (achute+chuteb) → chute_signals.py →
                                  S3 upload-time refresh for all cameras
GET  /api/sync/status           Poll sync job state: {running, log, started, finished, ok}
GET  /healthz                   {"status": "ok"}

Dashboard tabs
--------------
Operator View   Large-format instrument clusters (pit/chute/tipping), plain-English action list,
                and 6 live RGB camera feeds (Chute A+B, West Pit, Tipping 1/2/3) at native 16:9.
Safety Monitor  West pit temperature charts and IR heatmaps. Alert banner on WARNING/CRITICAL.
Furnace Feed    Chute A and Chute B side-by-side: fill level, moisture index, combined chart.
Tipping Floor   Tipping floor uniformity chart + future-feature placeholders.

Data ordering
-------------
All signals DataFrames are sorted by S3 upload time (LastModified), NOT by the filename
timestamp embedded by the camera. Camera clocks can be delayed by up to 48 h relative to
actual upload. Upload times are cached in data/s3_upload_times/{cam}.json.

See README.md for full documentation.
"""

import io
import json
import os
import re
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from PIL import Image as _PILImage

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR          = Path('/home/shared/kumar/library/fastapi-jupyter-dashboard')
DATA_DIR          = BASE_DIR / 'data'
CAM_DIR           = BASE_DIR / 'camera_data'
SCRIPTS_DIR       = BASE_DIR / 'scripts'
CHUTEB_CSV        = DATA_DIR / 'chuteb_signals/signals.csv'
ACHUTE_CSV        = DATA_DIR / 'achute_signals/signals.csv'
CACHE_DIR         = DATA_DIR / 'eda_cache'
UPLOAD_TIMES_DIR  = DATA_DIR / 's3_upload_times'   # per-camera JSON: stem → S3 LastModified epoch
IMAGES_DIR        = DATA_DIR / 'camera_images'     # cached latest RGB frames per camera

# ── S3 prefixes (used to fetch upload timestamps, not to sync data) ───────────
_S3B = 's3://bai-stl128tetra-uw2-data-field-data'
S3_PREFIXES: dict[str, list[str]] = {
    'achute':   [f'{_S3B}/site=haverhill/facility=a/device_id=reworld-haverhill-achute/data_type=csv/'],
    'chuteb':   [f'{_S3B}/site=haverhill/facility=chute/device_id=reworld-haverhill-chuteb/data_type=csv/'],
    'west-pit': [f'{_S3B}/site=reworld/facility=west-pit/device_id=reworld-west-pit/data_type=csv/'],
    'tipping1': [f'{_S3B}/site=haverhill/facility=tipping/device_id=reworld-haverhill-tipping1/data_type=csv/'],
    'tipping2': [f'{_S3B}/site=haverhill/facility=Tipping/device_id=reworld-haverhill-tipping2/data_type=csv/'],
    'tipping3': [
        f'{_S3B}/site=haverhill/facility=tipping/device_id=reworld-haverhill-tipping3/data_type=csv/',
        f'{_S3B}/site=haverhill/facility=Tipping/device_id=reworld-haverhill-tipping3/data_type=csv/',
    ],
}

# local RGB directories (fast path; west-pit has no local jpgs so it's absent)
CAM_RGB_DIRS = {
    'achute':   CAM_DIR / 'reworld-haverhill-achute/data_type=rgb',
    'chuteb':   CAM_DIR / 'reworld-haverhill-chuteb/data_type=rgb',
    'tipping1': CAM_DIR / 'reworld-haverhill-tipping1/data_type=rgb',
    'tipping2': CAM_DIR / 'reworld-haverhill-tipping2/data_type=rgb',
    'tipping3': CAM_DIR / 'reworld-haverhill-tipping3/data_type=rgb',
}

# S3 RGB prefixes for on-demand fetch when local file is absent
S3_RGB_PREFIXES = {
    'achute':   f'{_S3B}/site=haverhill/facility=a/device_id=reworld-haverhill-achute/data_type=rgb/',
    'chuteb':   f'{_S3B}/site=haverhill/facility=chute/device_id=reworld-haverhill-chuteb/data_type=rgb/',
    'tipping1': f'{_S3B}/site=haverhill/facility=tipping/device_id=reworld-haverhill-tipping1/data_type=rgb/',
    'tipping2': f'{_S3B}/site=haverhill/facility=Tipping/device_id=reworld-haverhill-tipping2/data_type=rgb/',
    'tipping3': f'{_S3B}/site=haverhill/facility=tipping/device_id=reworld-haverhill-tipping3/data_type=rgb/',
    'west-pit': f'{_S3B}/site=reworld/facility=west-pit/device_id=reworld-west-pit/data_type=rgb/',
}

# ── IR overlay ────────────────────────────────────────────────────────────────
# Cameras that have local IR BMP data synced (fast path)
CAM_IR_DIRS = {
    'achute':   CAM_DIR / 'reworld-haverhill-achute/data_type=ir',
    'tipping1': CAM_DIR / 'reworld-haverhill-tipping1/data_type=ir',
}

# S3 IR prefixes — stem and path structure match the CSV/RGB files exactly
S3_IR_PREFIXES = {
    'achute':   f'{_S3B}/site=haverhill/facility=a/device_id=reworld-haverhill-achute/data_type=ir/',
    'chuteb':   f'{_S3B}/site=haverhill/facility=chute/device_id=reworld-haverhill-chuteb/data_type=ir/',
    'west-pit': f'{_S3B}/site=reworld/facility=west-pit/device_id=reworld-west-pit/data_type=ir/',
    'tipping1': f'{_S3B}/site=haverhill/facility=tipping/device_id=reworld-haverhill-tipping1/data_type=ir/',
    'tipping2': f'{_S3B}/site=haverhill/facility=Tipping/device_id=reworld-haverhill-tipping2/data_type=ir/',
    'tipping3': f'{_S3B}/site=haverhill/facility=tipping/device_id=reworld-haverhill-tipping3/data_type=ir/',
}

CAM_DEVICE_ID = {
    'achute':   'reworld-haverhill-achute',
    'chuteb':   'reworld-haverhill-chuteb',
    'west-pit': 'reworld-west-pit',
    'tipping1': 'reworld-haverhill-tipping1',
    'tipping2': 'reworld-haverhill-tipping2',
    'tipping3': 'reworld-haverhill-tipping3',
}

_IR_HOMOGRAPHIES: dict = json.loads((SCRIPTS_DIR / 'ir_homographies.json').read_text())

# Build inferno colormap LUT at startup (256×3 uint8)
def _build_inferno_lut() -> np.ndarray:
    import matplotlib.cm
    lut = matplotlib.cm.get_cmap('inferno', 256)(np.arange(256))[:, :3]
    return (lut * 255).astype(np.uint8)

_INFERNO_LUT = _build_inferno_lut()


_SVG_NO_FEED = (
    b'<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720">'
    b'<rect width="1280" height="720" fill="#1a1f2e"/>'
    b'<text x="640" y="360" text-anchor="middle" dominant-baseline="middle" '
    b'fill="#334155" font-size="20" font-family="sans-serif">No feed available</text>'
    b'</svg>'
)

# ── S3 upload-time helpers ────────────────────────────────────────────────────
_s3_upload_cache: dict[str, dict[str, float]] = {}   # cam → {stem: epoch}


def _fetch_s3_upload_times() -> None:
    """List all CSV objects in S3 for every camera and persist upload timestamps."""
    UPLOAD_TIMES_DIR.mkdir(parents=True, exist_ok=True)
    for cam, prefixes in S3_PREFIXES.items():
        idx: dict[str, float] = {}
        for prefix in prefixes:
            try:
                r = subprocess.run(
                    ['aws', 's3', 'ls', '--recursive', prefix],
                    capture_output=True, text=True, timeout=120,
                )
                for line in r.stdout.splitlines():
                    # "2026-04-30 17:16:55   1234 path/to/20260428164207.csv"
                    m = re.match(
                        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+\d+\s+.*?(\d{14})\.csv$',
                        line.strip(),
                    )
                    if m:
                        epoch = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S').timestamp()
                        stem  = m.group(2)
                        # keep earliest upload time when a stem appears in multiple prefixes
                        if stem not in idx or epoch < idx[stem]:
                            idx[stem] = epoch
            except Exception:
                pass
        (UPLOAD_TIMES_DIR / f'{cam}.json').write_text(json.dumps(idx))
    _s3_upload_cache.clear()   # force reload on next /api/data call


def _s3_upload_times(cam: str) -> dict[str, float]:
    if cam not in _s3_upload_cache:
        p = UPLOAD_TIMES_DIR / f'{cam}.json'
        _s3_upload_cache[cam] = json.loads(p.read_text()) if p.exists() else {}
    return _s3_upload_cache[cam]


def _sort_by_s3_upload(df: pd.DataFrame, cam: str) -> pd.DataFrame:
    """Sort a signals DataFrame by S3 upload time; fall back to filename timestamp."""
    idx = _s3_upload_times(cam)
    if not idx:
        return df.sort_values('timestamp')
    stems = df['timestamp'].dt.strftime('%Y%m%d%H%M%S')
    ut = [idx.get(s, float('nan')) for s in stems]
    return (df.assign(_ut=ut)
              .sort_values('_ut', na_position='last')
              .drop(columns=['_ut'])
              .reset_index(drop=True))


app = FastAPI(title="Reworld Haverhill — Waste Intelligence")

# ── sync state ────────────────────────────────────────────────────────────────
_sync_state = {"running": False, "log": [], "started": None, "finished": None, "ok": None}

def _recompute_eda_incremental(cam: str, device_id: str, log: list) -> int:
    """Bulk-sync new CSV frames from S3 and append computed stats to eda_cache.

    Uses aws s3 sync (parallelised) to download all frames newer than cutoff,
    then processes them locally.  Returns the count of new rows added.
    """
    import csv as _csv
    from scipy.stats import entropy as _scipy_entropy

    out_path  = CACHE_DIR / f'{cam}_signals.csv'
    local_dir = CAM_DIR / device_id / 'data_type=csv'
    local_dir.mkdir(parents=True, exist_ok=True)

    # Determine the last processed timestamp
    cutoff = None
    if out_path.exists():
        try:
            cutoff = (pd.to_datetime(
                          pd.read_csv(out_path, usecols=['timestamp'])['timestamp'])
                        .max().to_pydatetime().replace(tzinfo=None))
        except Exception:
            pass

    # Identify which stems are new (in S3 index, newer than cutoff)
    idx = _s3_upload_times(cam)
    new_stems: dict[str, datetime] = {}
    for stem in idx:
        try:
            ts = datetime.strptime(stem, '%Y%m%d%H%M%S')
            if cutoff is None or ts > cutoff:
                new_stems[stem] = ts
        except ValueError:
            pass

    if not new_stems:
        log.append(f'  {cam}: already up to date')
        return 0

    log.append(f'  {cam}: {len(new_stems)} new frames — syncing from S3…')

    # Bulk download via aws s3 sync (much faster than individual cp calls)
    for pfx in S3_PREFIXES.get(cam, []):
        r = subprocess.run(
            ['aws', 's3', 'sync', pfx, str(local_dir), '--only-show-errors'],
            capture_output=True, text=True, timeout=600,
        )
        if r.stderr.strip():
            log.append(f'    sync warning: {r.stderr.strip()[:200]}')

    # Process every new stem that is now available locally
    rows: list[dict] = []
    for stem, ts in sorted(new_stems.items(), key=lambda x: x[1]):
        local_files = list(local_dir.rglob(f'{stem}.csv'))
        if not local_files:
            continue
        try:
            data_rows: list[list[float]] = []
            with open(local_files[0]) as f:
                for line in _csv.reader(f):
                    try:
                        data_rows.append([float(x) for x in line if x.strip()])
                    except ValueError:
                        pass
            if not data_rows:
                continue
            arr = np.array(data_rows, dtype=np.float32).flatten()
            if arr.size < 100:
                continue

            hot_35 = int((arr > 35).sum())
            hot_50 = int((arr > 50).sum())
            n      = len(arr)
            hist, _ = np.histogram(arr, bins=50)
            rows.append({
                'temp_min':    round(float(arr.min()), 2),
                'temp_max':    round(float(arr.max()), 2),
                'temp_mean':   round(float(arr.mean()), 2),
                'temp_median': round(float(np.median(arr)), 2),
                'temp_std':    round(float(arr.std()), 2),
                'temp_p10':    round(float(np.percentile(arr, 10)), 2),
                'temp_p90':    round(float(np.percentile(arr, 90)), 2),
                'temp_iqr':    round(float(np.percentile(arr, 75) - np.percentile(arr, 25)), 2),
                'hot_35':      hot_35,
                'hot_50':      hot_50,
                'hot_frac_35': round(hot_35 / n, 4),
                'hot_frac_50': round(hot_50 / n, 4),
                'entropy':     round(float(_scipy_entropy(hist + 1e-10)), 4),
                'timestamp':   ts.strftime('%Y-%m-%d %H:%M:%S'),
                'camera':      cam,
            })
        except Exception:
            continue

    if rows:
        new_df = pd.DataFrame(rows)
        if out_path.exists():
            pd.concat([pd.read_csv(out_path), new_df], ignore_index=True).to_csv(out_path, index=False)
        else:
            new_df.to_csv(out_path, index=False)

    log.append(f'  {cam}: added {len(rows)} rows')
    return len(rows)


def _run_sync():
    _sync_state.update({"running": True, "log": [], "started": datetime.now().isoformat(),
                         "finished": None, "ok": None})
    log = _sync_state["log"]
    venv = SCRIPTS_DIR / "venv/bin/python"
    py   = str(venv) if venv.exists() else "python3"
    ok   = True

    # ── Step 1: refresh S3 upload-time index so we know what's new ────────────
    log.append("▶ Fetching S3 upload timestamps…")
    try:
        _fetch_s3_upload_times()
        log.append("  ✓ done")
    except Exception as e:
        log.append(f"  ✗ {e}")

    # ── Step 2: full sync + recompute for chute cameras ───────────────────────
    chute_steps = [
        ("S3 sync — achute",
         ["aws", "s3", "sync",
          f"{_S3B}/site=haverhill/facility=a/device_id=reworld-haverhill-achute/data_type=csv/",
          str(CAM_DIR / "reworld-haverhill-achute/data_type=csv/"),
          "--only-show-errors"]),
        ("S3 sync — chuteb",
         ["aws", "s3", "sync",
          f"{_S3B}/site=haverhill/facility=chute/device_id=reworld-haverhill-chuteb/data_type=csv/",
          str(CAM_DIR / "reworld-haverhill-chuteb/data_type=csv/"),
          "--only-show-errors"]),
        ("Recompute — achute",
         [py, str(SCRIPTS_DIR / "chute_signals.py"),
          str(CAM_DIR / "reworld-haverhill-achute"),
          "--out", str(DATA_DIR / "achute_signals"),
          "--no-viz", "--cal-empty", "10.8", "--cal-full", "90.0"]),
        ("Recompute — chuteb",
         [py, str(SCRIPTS_DIR / "chute_signals.py"),
          str(CAM_DIR / "reworld-haverhill-chuteb"),
          "--out", str(DATA_DIR / "chuteb_signals"),
          "--no-viz", "--cal-empty", "24.0", "--cal-full", "90.4"]),
    ]
    for label, cmd in chute_steps:
        log.append(f"▶ {label}…")
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if r.returncode == 0:
                log.append("  ✓ done")
            else:
                log.append(f"  ✗ {r.stderr.strip()[:200]}")
                ok = False
        except Exception as e:
            log.append(f"  ✗ {e}")
            ok = False

    # ── Step 3: incremental update for pit/tipping cameras ────────────────────
    log.append("▶ Updating pit + tipping signals…")
    eda_cams = [
        ('west-pit', 'reworld-west-pit'),
        ('tipping1', 'reworld-haverhill-tipping1'),
        ('tipping2', 'reworld-haverhill-tipping2'),
        ('tipping3', 'reworld-haverhill-tipping3'),
    ]
    for cam, device_id in eda_cams:
        try:
            _recompute_eda_incremental(cam, device_id, log)
        except Exception as e:
            log.append(f"  ✗ {cam}: {e}")
            ok = False

    # ── Step 4: final timestamp refresh to catch late-arriving uploads ─────────
    log.append("▶ Refreshing upload timestamps (final)…")
    try:
        _fetch_s3_upload_times()
        log.append("  ✓ done")
    except Exception as e:
        log.append(f"  ✗ {e}")

    _sync_state.update({"running": False, "finished": datetime.now().isoformat(), "ok": ok})


# ── chart theme ───────────────────────────────────────────────────────────────
DARK = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(15,17,23,0.5)',
    font=dict(color='#cbd5e1', size=11, family='Inter, sans-serif'),
    margin=dict(l=48, r=16, t=38, b=36),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#94a3b8', size=10)),
    xaxis=dict(gridcolor='rgba(255,255,255,0.06)', zeroline=False,
               tickfont=dict(color='#64748b', size=10)),
    yaxis=dict(gridcolor='rgba(255,255,255,0.06)', zeroline=False,
               tickfont=dict(color='#64748b', size=10)),
)

def _clean(obj):
    """Recursively replace NaN/Inf floats with None so JSON serializes cleanly."""
    if isinstance(obj, float):
        return None if (obj != obj or obj == float('inf') or obj == float('-inf')) else obj
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj

def _fig(fig, title, h=280, **extra_layout):
    layout = {**DARK, "title": dict(text=title, font=dict(size=13, color='#94a3b8'))}
    if h is not None:
        layout["height"] = h
    layout.update(extra_layout)
    fig.update_layout(**layout)
    return _clean(fig.to_dict())

def _alert(t):
    if t >= 60: return "CRITICAL", "#ef4444", "🔴"
    if t >= 50: return "WARNING",  "#f97316", "🟠"
    if t >= 35: return "CAUTION",  "#eab308", "🟡"
    return "NORMAL", "#22c55e", "🟢"

# ── data loading ──────────────────────────────────────────────────────────────
def load_data():
    chuteb = _sort_by_s3_upload(pd.read_csv(CHUTEB_CSV, parse_dates=['timestamp']), 'chuteb')
    achute = _sort_by_s3_upload(pd.read_csv(ACHUTE_CSV, parse_dates=['timestamp']), 'achute')
    pit    = _sort_by_s3_upload(pd.read_csv(CACHE_DIR / 'west-pit_signals.csv',  parse_dates=['timestamp']), 'west-pit')
    t1     = _sort_by_s3_upload(pd.read_csv(CACHE_DIR / 'tipping1_signals.csv',  parse_dates=['timestamp']), 'tipping1')
    t2     = _sort_by_s3_upload(pd.read_csv(CACHE_DIR / 'tipping2_signals.csv',  parse_dates=['timestamp']), 'tipping2')
    t3     = _sort_by_s3_upload(pd.read_csv(CACHE_DIR / 'tipping3_signals.csv',  parse_dates=['timestamp']), 'tipping3')
    chuteb['moisture_masked'] = chuteb['moisture_index'].where(chuteb['fill_level_pct'] >= 20)
    achute['moisture_masked'] = achute['moisture_index'].where(achute['fill_level_pct'] >= 20)

    def _ds(p):   # downsample 4× (384×512 → 96×128) to keep JSON size small
        return np.load(p).astype(np.float32)[::4, ::4].round(2).tolist()

    return dict(chuteb=chuteb, achute=achute, pit=pit, t1=t1, t2=t2, t3=t3,
                pit_max_map=_ds(CACHE_DIR / 'west-pit_max.npy'),
                pit_h50_map=_ds(CACHE_DIR / 'west-pit_hot50.npy'))

# ── chart builders ────────────────────────────────────────────────────────────
def chart_pit_temp(pit):
    ts = pit['timestamp'].dt.strftime('%Y-%m-%d %H:%M').tolist()
    fig = go.Figure([
        go.Scatter(x=ts, y=pit['temp_max'].round(1).tolist(), name='Max °C',
                   line=dict(color='#ef4444', width=2),
                   fill='tozeroy', fillcolor='rgba(239,68,68,0.07)'),
        go.Scatter(x=ts, y=pit['temp_mean'].round(1).tolist(), name='Mean °C',
                   line=dict(color='#fb923c', width=1, dash='dot')),
    ])
    for y, label, color in [(60,'60°C — Smoldering','#ef4444'),
                             (50,'50°C — Warning','#f97316'),
                             (35,'35°C — Caution','#eab308')]:
        fig.add_hline(y=y, line_color=color, line_dash='dash', line_width=1,
                      annotation_text=label, annotation_font_color=color,
                      annotation_font_size=10)
    return _fig(fig, 'West Pit — Temperature Over Time', h=300,
                yaxis=dict(**DARK['yaxis'], title='°C'))

def chart_pit_maxmap(pit_max_map):
    fig = go.Figure(go.Heatmap(z=pit_max_map, colorscale='Inferno', zmin=20, zmax=65,
                               colorbar=dict(
                                   title=dict(text='°C', font=dict(color='#cbd5e1', size=10)),
                                   tickfont=dict(color='#cbd5e1', size=10))))
    return _fig(fig, 'West Pit — Peak Temperature Map', h=None,
                xaxis=dict(showticklabels=False, showgrid=False),
                yaxis=dict(showticklabels=False, showgrid=False, autorange='reversed',
                           scaleanchor='x', scaleratio=1))

def chart_pit_h50(pit_h50_map):
    cs = [[0,'#0f1117'],[0.3,'#7c3aed'],[0.7,'#f97316'],[1,'#ef4444']]
    fig = go.Figure(go.Heatmap(z=pit_h50_map, colorscale=cs, zmin=0, zmax=1,
                               colorbar=dict(
                                   title=dict(text='Fraction of frames',
                                              font=dict(color='#cbd5e1', size=10)),
                                   tickformat='.0%',
                                   tickfont=dict(color='#cbd5e1', size=10))))
    return _fig(fig, 'West Pit — % of Frames Where Pixel Exceeded 50°C', h=None,
                xaxis=dict(showticklabels=False, showgrid=False),
                yaxis=dict(showticklabels=False, showgrid=False, autorange='reversed',
                           scaleanchor='x', scaleratio=1))

def chart_tipping_temp(t1, t2, t3):
    fig = go.Figure()
    for df, name, color in [(t1,'Tipping 1','#3b82f6'),
                             (t2,'Tipping 2','#60a5fa'),
                             (t3,'Tipping 3','#93c5fd')]:
        ts = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M').tolist()
        fig.add_trace(go.Scatter(x=ts, y=df['temp_max'].round(1).tolist(),
                                 name=name, line=dict(color=color, width=1.5)))
    fig.add_hline(y=35, line_color='#eab308', line_dash='dot', line_width=1,
                  annotation_text='35°C anomaly threshold', annotation_font_color='#eab308',
                  annotation_font_size=10)
    return _fig(fig, 'Tipping Floor — Max Temperature per Camera', h=270,
                yaxis=dict(**DARK['yaxis'], title='°C'))

def chart_tipping_uniformity(t1, t2, t3):
    fig = go.Figure()
    for df, name, color in [(t1,'Tipping 1','#3b82f6'),
                             (t2,'Tipping 2','#60a5fa'),
                             (t3,'Tipping 3','#93c5fd')]:
        ts = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M').tolist()
        fig.add_trace(go.Scatter(x=ts, y=df['temp_std'].round(3).tolist(),
                                 name=name, line=dict(color=color, width=1.5)))
    fig.add_hline(y=0.6, line_color='#a855f7', line_dash='dash', line_width=1,
                  annotation_text='Below → uniform / single-material load',
                  annotation_font_color='#a855f7', annotation_font_size=10)
    return _fig(fig, 'Tipping Floor — Load Uniformity  (low = single-material load)', h=260,
                yaxis=dict(**DARK['yaxis'], title='Spatial Std Dev (°C)'))

def chart_chute_fill(chuteb, label='Chute B'):
    ts = chuteb['timestamp'].dt.strftime('%Y-%m-%d %H:%M').tolist()
    fig = go.Figure([
        go.Scatter(x=ts, y=chuteb['fill_level_pct'].tolist(), name='Fill %',
                   line=dict(color='#38bdf8', width=2),
                   fill='tozeroy', fillcolor='rgba(56,189,248,0.09)')
    ])
    drain = chuteb[(chuteb['timestamp'] >= '2026-04-30 18:00') &
                   (chuteb['timestamp'] <= '2026-04-30 22:00')]
    if len(drain):
        fig.add_vrect(
            x0=drain['timestamp'].min().strftime('%Y-%m-%d %H:%M'),
            x1=drain['timestamp'].max().strftime('%Y-%m-%d %H:%M'),
            fillcolor='rgba(99,102,241,0.13)', line_width=0,
            annotation_text='Drain event', annotation_position='top left',
            annotation_font_color='#818cf8', annotation_font_size=10)
    return _fig(fig, f'{label} — Fill Level', h=250,
                yaxis=dict(**DARK['yaxis'], title='Fill %', range=[0, 105]))

def chart_chute_moisture(chuteb, label='Chute B'):
    ts = chuteb['timestamp'].dt.strftime('%Y-%m-%d %H:%M').tolist()
    fig = go.Figure([
        go.Scatter(x=ts, y=chuteb['moisture_masked'].tolist(), name='Moisture Index',
                   line=dict(color='#34d399', width=2))
    ])
    hi = chuteb[(chuteb['timestamp'] >= '2026-04-30 04:00') &
                (chuteb['timestamp'] <= '2026-04-30 06:30')]
    if len(hi):
        fig.add_vrect(
            x0=hi['timestamp'].min().strftime('%Y-%m-%d %H:%M'),
            x1=hi['timestamp'].max().strftime('%Y-%m-%d %H:%M'),
            fillcolor='rgba(251,191,36,0.11)', line_width=0,
            annotation_text='High-BTU: dry + full', annotation_position='top left',
            annotation_font_color='#fbbf24', annotation_font_size=10)
    return _fig(fig, f'{label} — Moisture Index  (masked when fill < 20%)', h=250,
                yaxis=dict(**DARK['yaxis'], title='0 = dry  ·  1 = wet', range=[-0.05, 1.05]))

def chart_chute_combined(chuteb, label='Chute B'):
    ts = chuteb['timestamp'].dt.strftime('%Y-%m-%d %H:%M').tolist()
    fig = go.Figure([
        go.Scatter(x=ts, y=chuteb['fill_level_pct'].tolist(), name='Fill %',
                   line=dict(color='#38bdf8', width=1.5), yaxis='y'),
        go.Scatter(x=ts, y=chuteb['moisture_masked'].tolist(), name='Moisture',
                   line=dict(color='#34d399', width=1.5), yaxis='y2'),
        go.Scatter(x=ts, y=chuteb['max_temp_c'].tolist(), name='Max °C',
                   line=dict(color='#f87171', width=1, dash='dot'), yaxis='y3'),
    ])
    layout_extra = dict(
        yaxis =dict(title='Fill %',    range=[0,110],  gridcolor='rgba(255,255,255,0.06)',
                    tickfont=dict(color='#38bdf8', size=10)),
        yaxis2=dict(title='Moisture',  range=[0,1.1],  overlaying='y', side='right',
                    tickfont=dict(color='#34d399', size=10), showgrid=False),
        yaxis3=dict(title='Max °C',    range=[5,55],   overlaying='y', side='right',
                    position=0.94, tickfont=dict(color='#f87171', size=10), showgrid=False),
    )
    return _fig(fig, f'{label} — Fill / Moisture / Max Temp', h=300, **layout_extra)

# ── synthetic placeholders ────────────────────────────────────────────────────
def _synth_plastic(chuteb):
    rng = np.random.default_rng(42)
    base  = (1 - chuteb['moisture_masked'].fillna(0.5)).values
    vals  = pd.Series(np.clip(base * 0.65 + rng.normal(0, 0.08, len(base)) + 0.10, 0.05, 0.82) * 100
                      ).rolling(5, min_periods=1, center=True).mean()
    ts = chuteb['timestamp'].dt.strftime('%Y-%m-%d %H:%M').tolist()
    fig = go.Figure([
        go.Scatter(x=ts, y=vals.round(1).tolist(), name='Plastic %',
                   line=dict(color='#a78bfa', width=2),
                   fill='tozeroy', fillcolor='rgba(167,139,250,0.08)')
    ])
    fig.add_hline(y=40, line_color='#f97316', line_dash='dash', line_width=1,
                  annotation_text='> 40% → high-BTU alert', annotation_font_color='#f97316',
                  annotation_font_size=10)
    return _fig(fig, 'Plastic Fraction %  ·  DEMO PREVIEW (Phase 3)', h=240,
                yaxis=dict(**DARK['yaxis'], title='Plastic %', range=[0,90]))

def _synth_organic(chuteb):
    rng = np.random.default_rng(7)
    base = chuteb['moisture_masked'].fillna(0.5).values
    vals = pd.Series(np.clip(base * 0.55 + rng.normal(0, 0.07, len(base)) + 0.05, 0.02, 0.65) * 100
                     ).rolling(5, min_periods=1, center=True).mean()
    ts = chuteb['timestamp'].dt.strftime('%Y-%m-%d %H:%M').tolist()
    fig = go.Figure([
        go.Scatter(x=ts, y=vals.round(1).tolist(), name='Organic %',
                   line=dict(color='#4ade80', width=2),
                   fill='tozeroy', fillcolor='rgba(74,222,128,0.07)')
    ])
    return _fig(fig, 'Organic Fraction %  ·  DEMO PREVIEW (Phase 3)', h=240,
                yaxis=dict(**DARK['yaxis'], title='Organic %', range=[0,70]))

def _synth_btu(chuteb):
    rng  = np.random.default_rng(13)
    pl   = (1 - chuteb['moisture_masked'].fillna(0.5)).values * 0.65 + 0.10
    og   = chuteb['moisture_masked'].fillna(0.5).values * 0.55 + 0.05
    vals = pd.Series(np.clip(11.0 + 4.5*pl - 3.0*og + rng.normal(0, 0.3, len(pl)), 6.0, 18.0)
                     ).rolling(8, min_periods=1, center=True).mean()
    ts = chuteb['timestamp'].dt.strftime('%Y-%m-%d %H:%M').tolist()
    fig = go.Figure([
        go.Scatter(x=ts, y=vals.round(2).tolist(), name='HHV (MJ/kg)',
                   line=dict(color='#fbbf24', width=2))
    ])
    fig.add_hrect(y0=9, y1=13, fillcolor='rgba(34,197,94,0.05)', line_width=0)
    for y, label, color in [(13,'> 13 → pre-cool airflow','#f97316'),
                             (9,'< 9 → increase feed rate','#60a5fa')]:
        fig.add_hline(y=y, line_color=color, line_dash='dash', line_width=1,
                      annotation_text=label, annotation_font_color=color,
                      annotation_font_size=10)
    return _fig(fig, 'Predicted HHV (MJ/kg)  ·  DEMO PREVIEW (Phase 4)', h=240,
                yaxis=dict(**DARK['yaxis'], title='MJ/kg'))

# ── summary stats ─────────────────────────────────────────────────────────────
def build_summary(d):
    pit   = d['pit']
    cb    = d['chuteb']
    t1, t2, t3 = d['t1'], d['t2'], d['t3']

    latest_pit  = pit.iloc[-1]
    pit_lvl, pit_col, pit_icon = _alert(float(latest_pit['temp_max']))
    tip_max = max(t1['temp_max'].max(), t2['temp_max'].max(), t3['temp_max'].max())
    tip_lvl, tip_col, tip_icon = _alert(float(tip_max))
    latest_cb = cb.iloc[-1]
    moist = (float(latest_cb['moisture_masked'])
             if pd.notna(latest_cb.get('moisture_masked', float('nan')))
             else float(latest_cb['moisture_index']))

    pit_alert_frames = int((pit['temp_max'] > 50).sum())
    pit_total_frames = len(pit)
    pit_hotspot_pct  = int(round(pit_alert_frames / pit_total_frames * 100)) if pit_total_frames else 0

    fill  = round(float(latest_cb['fill_level_pct']), 1)
    moist_label = 'DRY' if moist < 0.25 else 'WET' if moist > 0.65 else 'MODERATE'
    if fill < 20:
        chute_status, chute_color = 'CHUTE EMPTY', '#818cf8'
    elif moist < 0.25 and fill > 70:
        chute_status, chute_color = 'HIGH BTU', '#f97316'
    else:
        chute_status, chute_color = 'NORMAL', '#22c55e'

    actions = []
    if pit_lvl == 'CRITICAL':
        actions.append(['red',    f'URGENT — Pit at {round(float(latest_pit["temp_max"]),1)}°C. Direct crane to break up hotspot immediately.'])
    elif pit_lvl == 'WARNING':
        actions.append(['orange', f'Pit at {round(float(latest_pit["temp_max"]),1)}°C — direct crane to mix hotspot zone.'])
    elif pit_lvl == 'CAUTION':
        actions.append(['yellow', f'Pit at {round(float(latest_pit["temp_max"]),1)}°C elevated — monitor, crane on standby.'])
    else:
        actions.append(['green',  'Pit temperature normal — no action required.'])
    if fill < 20:
        actions.append(['blue',   'Chute draining — monitor fill level, expect low feed period.'])
    elif moist < 0.25 and fill > 70:
        actions.append(['orange', 'HIGH BTU load in chute — notify control room. PA flow adjustment needed in 15–20 min.'])
    else:
        actions.append(['green',  'Chute feed normal — no combustion adjustment needed.'])
    tip_max_val = round(float(tip_max), 1)
    if tip_max_val > 40:
        actions.append(['orange', f'Tipping floor: {tip_max_val}°C anomaly — check cameras for battery or smoldering material.'])
    elif tip_max_val > 35:
        actions.append(['yellow', f'Tipping floor: {tip_max_val}°C warm spot detected — check cameras.'])
    else:
        actions.append(['green',  'Tipping floor clear — no anomalies detected.'])

    latest_ac = d['achute'].iloc[-1]
    ac_moist = (float(latest_ac['moisture_masked'])
                if pd.notna(latest_ac.get('moisture_masked', float('nan')))
                else float(latest_ac['moisture_index']))
    ac_moist_label = 'DRY' if ac_moist < 0.25 else 'WET' if ac_moist > 0.65 else 'MODERATE'

    tip1_max = round(float(t1['temp_max'].max()), 1)
    tip2_max = round(float(t2['temp_max'].max()), 1)
    tip3_max = round(float(t3['temp_max'].max()), 1)

    return dict(
        pit_max=round(float(latest_pit['temp_max']), 1),
        pit_level=pit_lvl, pit_color=pit_col, pit_icon=pit_icon,
        pit_alert_frames=pit_alert_frames, pit_total_frames=pit_total_frames,
        pit_hotspot_pct=pit_hotspot_pct,
        tip_max=tip_max_val,
        tip_level=tip_lvl, tip_color=tip_col, tip_icon=tip_icon,
        tip1_max=tip1_max, tip2_max=tip2_max, tip3_max=tip3_max,
        chute_fill=fill,
        chute_moisture=round(moist, 3),
        moist_label=moist_label,
        achute_fill=round(float(latest_ac['fill_level_pct']), 1),
        achute_moisture=round(ac_moist, 3),
        achute_moist_label=ac_moist_label,
        chute_status=chute_status, chute_color=chute_color,
        actions=actions,
        data_through=pit['timestamp'].max().strftime('%Y-%m-%d %H:%M'),
        generated=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    )

# ── API routes ────────────────────────────────────────────────────────────────
@app.get('/api/data')
def api_data():
    d = load_data()
    charts = dict(
        pit_temp         = chart_pit_temp(d['pit']),
        pit_maxmap       = chart_pit_maxmap(d['pit_max_map']),
        pit_h50          = chart_pit_h50(d['pit_h50_map']),
        tipping_temp     = chart_tipping_temp(d['t1'], d['t2'], d['t3']),
        tipping_uniform  = chart_tipping_uniformity(d['t1'], d['t2'], d['t3']),
        chute_fill       = chart_chute_fill(d['chuteb']),
        chute_moisture   = chart_chute_moisture(d['chuteb']),
        chute_combined   = chart_chute_combined(d['chuteb']),
        achute_fill      = chart_chute_fill(d['achute'], label='Chute A'),
        achute_moisture  = chart_chute_moisture(d['achute'], label='Chute A'),
        achute_combined  = chart_chute_combined(d['achute'], label='Chute A'),
        plastic_frac     = _synth_plastic(d['chuteb']),
        organic_frac     = _synth_organic(d['chuteb']),
        btu_pred         = _synth_btu(d['chuteb']),
    )
    summary = build_summary(d)
    wp_idx = _s3_upload_times('west-pit')
    if wp_idx:
        latest_stem = max(wp_idx, key=lambda s: wp_idx[s])
        summary['pit_image_ts']       = datetime.strptime(latest_stem, '%Y%m%d%H%M%S').strftime('%Y-%m-%d %H:%M')
        summary['pit_image_uploaded'] = datetime.fromtimestamp(wp_idx[latest_stem]).strftime('%Y-%m-%d %H:%M')
    else:
        summary['pit_image_ts'] = summary['pit_image_uploaded'] = None
    return JSONResponse(content={"summary": summary, "charts": charts},
                        headers={"Cache-Control": "no-store"})

@app.post('/api/sync')
def api_sync():
    if _sync_state["running"]:
        return {"status": "already_running", "log": _sync_state["log"]}
    t = threading.Thread(target=_run_sync, daemon=True)
    t.start()
    return {"status": "started"}

@app.get('/api/sync/status')
def api_sync_status():
    return {k: v for k, v in _sync_state.items()}

@app.get('/healthz')
def health():
    return {"status": "ok"}


def _ir_colorize(gray: np.ndarray) -> np.ndarray:
    """Apply inferno colormap to a (H,W) uint8 grayscale array → (H,W,3) uint8 RGB."""
    idx = gray.astype(np.uint8)
    return _INFERNO_LUT[idx]


def _find_nearest_ir_bmp(cam: str, rgb_stem: str):
    """Return the locally-available BMP path nearest to rgb_stem."""
    ir_dir = CAM_IR_DIRS.get(cam)
    if not ir_dir or not ir_dir.exists():
        return None
    bmps = sorted(ir_dir.rglob('*.bmp'))
    if not bmps:
        return None
    try:
        target = datetime.strptime(rgb_stem, '%Y%m%d%H%M%S')
    except ValueError:
        return bmps[-1]

    def dist(p):
        try:
            return abs((datetime.strptime(p.stem, '%Y%m%d%H%M%S') - target).total_seconds())
        except ValueError:
            return float('inf')

    return min(bmps, key=dist)


def _get_ir_bmp(cam: str, stem: str):
    """Return a BMP path for cam/stem: local dir first, then S3 fetch with caching."""
    # Fast path: local sync'd data
    local = _find_nearest_ir_bmp(cam, stem)
    if local is not None:
        return local

    # S3 fetch using the exact stem (IR and CSV share timestamps)
    prefix = S3_IR_PREFIXES.get(cam)
    if not prefix:
        return None

    try:
        ts = datetime.strptime(stem, '%Y%m%d%H%M%S')
    except ValueError:
        return None

    ir_cache = IMAGES_DIR / cam / 'ir'
    ir_cache.mkdir(parents=True, exist_ok=True)
    cached = ir_cache / f'{stem}.bmp'

    if not cached.exists():
        for old in ir_cache.glob('*.bmp'):
            old.unlink(missing_ok=True)
        s3_path = (f'{prefix}year={ts.year}/month={ts.month:02d}/day={ts.day:02d}/'
                   f'hour={ts.hour:02d}/minute={ts.minute:02d}/{stem}.bmp')
        r = subprocess.run(['aws', 's3', 'cp', s3_path, str(cached)],
                           capture_output=True, timeout=30)
        if r.returncode != 0 or not cached.exists():
            return None

    return cached


def _resolve_rgb_path(cam: str):
    """Return (local_path, latest_stem, upload_epoch) or raise HTTPException."""
    idx = _s3_upload_times(cam)
    if not idx:
        return None, None, None

    latest_stem = max(idx, key=lambda s: idx[s])
    ts = datetime.strptime(latest_stem, '%Y%m%d%H%M%S')

    local_path = None
    if cam in CAM_RGB_DIRS:
        matches = list(CAM_RGB_DIRS[cam].rglob(f'{latest_stem}.jpg'))
        if matches:
            local_path = matches[0]

    if local_path is None:
        cache_dir = IMAGES_DIR / cam
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached = cache_dir / f'{latest_stem}.jpg'
        if not cached.exists():
            prefix = S3_RGB_PREFIXES.get(cam, '')
            s3_path = (f'{prefix}year={ts.year}/month={ts.month:02d}/day={ts.day:02d}/'
                       f'hour={ts.hour:02d}/minute={ts.minute:02d}/{latest_stem}.jpg')
            r = subprocess.run(['aws', 's3', 'cp', s3_path, str(cached)],
                               capture_output=True, timeout=30)
            if r.returncode != 0 or not cached.exists():
                return None, None, None
            for old in cache_dir.glob('*.jpg'):
                if old.name != cached.name:
                    old.unlink(missing_ok=True)
        local_path = cached

    return local_path, latest_stem, idx[latest_stem]


def _cam_overlay_response(cam: str, alpha: float = 0.45):
    """Return an inferno-IR-over-RGB blended JPEG. Falls back to plain RGB if no IR."""
    local_path, latest_stem, upload_epoch = _resolve_rgb_path(cam)
    if local_path is None:
        return Response(content=_SVG_NO_FEED, media_type='image/svg+xml',
                        headers={'Cache-Control': 'no-store'})

    uploaded_at = datetime.fromtimestamp(upload_epoch).strftime('%Y-%m-%d %H:%M UTC')
    bmp_path = _get_ir_bmp(cam, latest_stem)
    device_id = CAM_DEVICE_ID.get(cam, '')
    H = _IR_HOMOGRAPHIES.get(device_id)

    if bmp_path is None or H is None:
        return FileResponse(str(local_path), media_type='image/jpeg',
                            headers={'Cache-Control': 'no-store', 'X-Uploaded-At': uploaded_at})

    rgb_img = _PILImage.open(str(local_path)).convert('RGB')
    w, h = rgb_img.size

    ir_gray = np.array(_PILImage.open(str(bmp_path)).convert('L'))
    ir_rgb = _PILImage.fromarray(_ir_colorize(ir_gray), 'RGB')

    sx, tx = H[0][0], H[0][2]
    sy, ty = H[1][1], H[1][2]
    inv = (1/sx, 0.0, -tx/sx, 0.0, 1/sy, -ty/sy)

    ir_warped = ir_rgb.transform((w, h), _PILImage.AFFINE, inv, _PILImage.BILINEAR)

    # Build alpha mask: 1.0 inside the warped IR region, 0 outside
    mask_src = _PILImage.new('L', ir_rgb.size, 255)
    mask_warped = mask_src.transform((w, h), _PILImage.AFFINE, inv, _PILImage.NEAREST)
    alpha_channel = _PILImage.fromarray(
        (np.array(mask_warped) * alpha).astype(np.uint8), 'L')

    ir_rgba = ir_warped.convert('RGBA')
    ir_rgba.putalpha(alpha_channel)
    result = _PILImage.alpha_composite(rgb_img.convert('RGBA'), ir_rgba).convert('RGB')

    buf = io.BytesIO()
    result.save(buf, format='JPEG', quality=85)
    return Response(content=buf.getvalue(), media_type='image/jpeg',
                    headers={'Cache-Control': 'no-store', 'X-Uploaded-At': uploaded_at})


CAM_LABELS = {
    'achute':   'Chute A',
    'chuteb':   'Chute B',
    'west-pit': 'West Pit',
    'tipping1': 'Tipping Floor 1',
    'tipping2': 'Tipping Floor 2',
    'tipping3': 'Tipping Floor 3',
}

SIGNALS_CSVS = {
    'achute': ACHUTE_CSV,
    'chuteb': CHUTEB_CSV,
}

def _temp_range(cam: str, stem: str):
    """Return (min_c, max_c, mean_c) from the nearest available temperature data."""
    # 1. Signals summary CSV (achute / chuteb)
    sig_path = SIGNALS_CSVS.get(cam)
    if sig_path and sig_path.exists():
        try:
            df = pd.read_csv(sig_path, usecols=['timestamp', 'min_temp_c', 'max_temp_c', 'waste_mean_temp_c'])
            target = datetime.strptime(stem, '%Y%m%d%H%M%S')
            df['_d'] = pd.to_datetime(df['timestamp']).apply(lambda t: abs((t.to_pydatetime().replace(tzinfo=None) - target).total_seconds()))
            row = df.loc[df['_d'].idxmin()]
            return round(float(row['min_temp_c']), 1), round(float(row['max_temp_c']), 1), round(float(row['waste_mean_temp_c']), 1)
        except Exception:
            pass

    # 2. Raw per-frame IR CSV (tipping1 has these locally)
    device_id = CAM_DEVICE_ID.get(cam, '')
    csv_dir = CAM_DIR / device_id / 'data_type=csv'
    if csv_dir.exists():
        try:
            target = datetime.strptime(stem, '%Y%m%d%H%M%S')
            all_csvs = sorted(csv_dir.rglob('*.csv'))
            if all_csvs:
                def dist(p):
                    try:
                        return abs((datetime.strptime(p.stem, '%Y%m%d%H%M%S') - target).total_seconds())
                    except ValueError:
                        return float('inf')
                nearest = min(all_csvs, key=dist)
                arr = np.loadtxt(str(nearest), delimiter=',', dtype=np.float32, max_rows=384)
                return round(float(arr.min()), 1), round(float(arr.max()), 1), round(float(arr.mean()), 1)
        except Exception:
            pass

    return None, None, None


def _cam_image_response(cam: str):
    """Return the latest RGB frame for a camera, fetching from S3 if needed."""
    local_path, latest_stem, upload_epoch = _resolve_rgb_path(cam)
    if local_path is None:
        return Response(content=_SVG_NO_FEED, media_type='image/svg+xml',
                        headers={'Cache-Control': 'no-store'})
    uploaded_at = datetime.fromtimestamp(upload_epoch).strftime('%Y-%m-%d %H:%M UTC')
    return FileResponse(str(local_path), media_type='image/jpeg',
                        headers={'Cache-Control': 'no-store', 'X-Uploaded-At': uploaded_at})


@app.get('/api/image/{camera}')
def api_image(camera: str, overlay: int = 0):
    if camera not in {*CAM_RGB_DIRS, *S3_RGB_PREFIXES}:
        raise HTTPException(status_code=404, detail='Unknown camera')
    if overlay:
        return _cam_overlay_response(camera)
    return _cam_image_response(camera)


@app.get('/api/image/{camera}/meta')
def api_image_meta(camera: str):
    if camera not in {*CAM_RGB_DIRS, *S3_RGB_PREFIXES}:
        raise HTTPException(status_code=404, detail='Unknown camera')

    idx = _s3_upload_times(camera)
    if not idx:
        raise HTTPException(status_code=404, detail='No data available')

    stem = max(idx, key=lambda s: idx[s])
    ts = datetime.strptime(stem, '%Y%m%d%H%M%S')
    upload_epoch = idx[stem]

    def _s3_url(prefix, ext):
        if not prefix:
            return None
        return (f'{prefix}year={ts.year}/month={ts.month:02d}/day={ts.day:02d}/'
                f'hour={ts.hour:02d}/minute={ts.minute:02d}/{stem}.{ext}')

    temp_min, temp_max, temp_mean = _temp_range(camera, stem)
    ir_available = (camera in S3_IR_PREFIXES) and (CAM_DEVICE_ID.get(camera, '') in _IR_HOMOGRAPHIES)

    return JSONResponse({
        'camera':      camera,
        'label':       CAM_LABELS.get(camera, camera),
        'stem':        stem,
        'captured_at': ts.strftime('%Y-%m-%d %H:%M:%S UTC'),
        'uploaded_at': datetime.fromtimestamp(upload_epoch).strftime('%Y-%m-%d %H:%M:%S UTC'),
        'rgb_s3_path': _s3_url(S3_RGB_PREFIXES.get(camera), 'jpg'),
        'ir_s3_path':  _s3_url(S3_IR_PREFIXES.get(camera), 'bmp') if ir_available else None,
        'ir_available': ir_available,
        'temp_min_c':  temp_min,
        'temp_max_c':  temp_max,
        'temp_mean_c': temp_mean,
    })


# Legacy alias kept so any bookmarked URLs still work
@app.get('/api/west-pit/latest-image')
def west_pit_latest_image():
    return _cam_image_response('west-pit')

# ── HTML shell ────────────────────────────────────────────────────────────────
HTML_SHELL = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Reworld Haverhill — Waste Intelligence</title>
<script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet"/>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0f1117;--card:#1a1f2e;--card2:#1d2235;--border:rgba(255,255,255,0.07);
  --text:#e2e8f0;--muted:#64748b;--subtle:#1e2840;
  --red:#ef4444;--orange:#f97316;--yellow:#eab308;
  --green:#22c55e;--blue:#3b82f6;--purple:#a855f7;
  --sky:#38bdf8;
}
body{background:var(--bg);color:var(--text);font-family:'Inter',sans-serif;font-size:14px;line-height:1.5}

/* ── header ── */
.hdr{background:var(--card);border-bottom:1px solid var(--border);
     padding:12px 24px;display:flex;align-items:center;gap:14px;position:sticky;top:0;z-index:50}
.hdr-dot{width:9px;height:9px;border-radius:50%;background:var(--green);
          box-shadow:0 0 7px var(--green);flex-shrink:0}
.hdr h1{font-size:16px;font-weight:700;letter-spacing:-.3px}
.hdr .sub{font-size:11px;color:var(--muted)}
.spacer{flex:1}
.hdr-meta{font-size:11px;color:var(--muted);text-align:right}

/* ── alert banner ── */
#banner{padding:9px 24px;font-size:13px;font-weight:500;display:none}

/* ── nav ── */
.nav{display:flex;border-bottom:1px solid var(--border);
     padding:0 24px;background:var(--card);gap:2px;flex-wrap:wrap}
.nb{padding:11px 16px;font-size:13px;font-weight:500;color:var(--muted);
    cursor:pointer;border:none;border-bottom:2px solid transparent;
    background:none;transition:color .15s,border-color .15s;white-space:nowrap}
.nb:hover{color:var(--text)}
.nb.active{color:var(--text);border-bottom-color:var(--blue)}
.nb .dot{display:inline-block;width:7px;height:7px;border-radius:50%;
          margin-right:5px;vertical-align:middle}

/* ── action bar ── */
.actions{display:flex;gap:10px;padding:12px 24px;background:rgba(26,31,46,.6);
          border-bottom:1px solid var(--border);align-items:center;flex-wrap:wrap}
.btn{padding:6px 14px;border-radius:7px;font-size:12px;font-weight:600;
     cursor:pointer;border:1px solid;transition:opacity .15s}
.btn:disabled{opacity:.45;cursor:not-allowed}
.btn-refresh{background:rgba(59,130,246,.15);color:#93c5fd;border-color:rgba(59,130,246,.3)}
.btn-sync{background:rgba(34,197,94,.12);color:#86efac;border-color:rgba(34,197,94,.25)}
.btn-auto{background:rgba(99,102,241,.12);color:#a5b4fc;border-color:rgba(99,102,241,.25)}
.btn-ir{background:rgba(99,102,241,.12);color:#a5b4fc;border-color:rgba(99,102,241,.25)}

/* ── camera lightbox ── */
#cam-modal{display:none;position:fixed;inset:0;z-index:9999;background:rgba(0,0,0,.88);
           align-items:center;justify-content:center}
#cam-modal.open{display:flex}
.modal-inner{background:#1a1f2e;border-radius:14px;overflow:hidden;
             display:flex;flex-direction:column;max-width:96vw;max-height:96vh}
.modal-hdr{display:flex;align-items:center;justify-content:space-between;
           padding:12px 16px;border-bottom:1px solid var(--border);flex-shrink:0}
.modal-title{font-size:14px;font-weight:700;color:var(--text)}
.modal-body{display:flex;overflow:hidden;flex:1;min-height:0}
.modal-img-wrap{flex:1;display:flex;align-items:center;justify-content:center;
                background:#0a0d14;overflow:hidden}
#modal-img{max-width:100%;max-height:calc(96vh - 56px);object-fit:contain;display:block}
.modal-sidebar{width:270px;min-width:270px;overflow-y:auto;padding:14px 16px;
               border-left:1px solid var(--border);font-size:12px}
.modal-close{background:none;border:none;color:var(--muted);font-size:24px;
             cursor:pointer;line-height:1;padding:2px 6px;border-radius:5px}
.modal-close:hover{color:var(--text);background:rgba(255,255,255,.06)}
.mlbl{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;
      color:var(--muted);margin:12px 0 5px}
.mlbl:first-child{margin-top:0}
.mrow{margin:3px 0;line-height:1.5;color:var(--text)}
.mkey{color:var(--muted)}
.mpath{color:#93c5fd;font-size:10px;word-break:break-all;margin-top:2px}
#modal-colorbar-section{margin-bottom:4px}
.colorbar-row{display:flex;gap:8px;align-items:stretch;margin-top:6px}
#modal-colorbar{border-radius:3px;flex-shrink:0}
#modal-temp-labels{font-size:10px;color:#94a3b8;display:flex;flex-direction:column;
                   justify-content:space-between;padding:2px 0}
.cam-img{cursor:pointer}
.status-text{font-size:11px;color:var(--muted);margin-left:4px}

/* ── page body ── */
.page{padding:18px 24px;max-width:1380px}
.tab{display:none}.tab.active{display:block}

/* ── section label ── */
.sec{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.08em;
     color:var(--muted);margin:18px 0 14px;display:flex;align-items:center;gap:8px}
.sec::after{content:'';flex:1;height:1px;background:var(--border)}
.ptag{font-size:10px;font-weight:600;padding:2px 8px;border-radius:99px;
      text-transform:uppercase;letter-spacing:.05em}

/* ── stat cards ── */
.cards{display:grid;gap:12px;margin-bottom:16px}
.c3{grid-template-columns:repeat(3,1fr)}
.c4{grid-template-columns:repeat(4,1fr)}
.card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:14px 16px}
.card .lbl{font-size:11px;color:var(--muted);text-transform:uppercase;
            letter-spacing:.06em;margin-bottom:5px}
.card .val{font-size:24px;font-weight:700;line-height:1.1}
.card .unit{font-size:12px;color:var(--muted);margin-left:2px}
.card .csub{font-size:11px;color:var(--muted);margin-top:3px}
.badge{display:inline-flex;align-items:center;gap:5px;font-size:11px;font-weight:600;
       padding:3px 10px;border-radius:99px;margin-top:5px}

/* ── chart cards ── */
.cc{background:var(--card);border:1px solid var(--border);border-radius:10px;
    padding:14px;margin-bottom:14px}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px}
.row3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-bottom:14px}

/* ── loading overlay ── */
.loader{text-align:center;padding:60px 24px;color:var(--muted)}
.spinner{display:inline-block;width:20px;height:20px;border:2px solid var(--muted);
          border-top-color:var(--blue);border-radius:50%;
          animation:spin .8s linear infinite;margin-right:8px;vertical-align:middle}
@keyframes spin{to{transform:rotate(360deg)}}

/* ── placeholder cards ── */
.ph{background:var(--card2);border:1px dashed rgba(168,85,247,.3);
    border-radius:10px;padding:14px;margin-bottom:14px;position:relative}
.ph .ph-badge{position:absolute;top:11px;right:13px;font-size:10px;font-weight:700;
               padding:2px 8px;border-radius:99px;background:rgba(168,85,247,.15);
               color:var(--purple);border:1px solid rgba(168,85,247,.25)}
.ph h4{font-size:13px;font-weight:600;color:#c084fc;margin-bottom:5px}
.ph .ph-desc{font-size:12px;color:var(--muted);margin-bottom:10px}
.ph .demo-chip{display:inline-block;font-size:10px;font-weight:700;
                background:rgba(168,85,247,.13);color:#c084fc;
                padding:2px 8px;border-radius:4px;margin-bottom:8px;letter-spacing:.05em}

/* ── detection demo ── */
.det-box{background:rgba(0,0,0,.3);border-radius:8px;padding:12px;
          border:1px solid rgba(255,255,255,.06)}
.det-row{display:flex;gap:10px;align-items:flex-start;margin-bottom:6px}
.det-row:last-child{margin-bottom:0}
.det-ico{font-size:20px;flex-shrink:0;line-height:1.3}
.det-title{font-size:13px;font-weight:700}
.det-meta{font-size:11px;color:var(--muted);margin-top:1px}
.det-action{font-size:11px;font-weight:600;color:#fbbf24;margin-top:3px}
.conf{display:inline-block;font-size:11px;font-weight:700;margin-top:5px;
      padding:2px 8px;border-radius:4px}

/* ── zone map ── */
.zg{display:grid;grid-template-columns:1fr 1fr 1fr;gap:5px;max-width:420px}
.zc{padding:9px;border-radius:7px;text-align:center}
.zc .zl{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.05em}
.zc .zb{font-size:11px;font-weight:600;margin-top:1px}
.zhigh{background:rgba(239,68,68,.13);border:1px solid rgba(239,68,68,.22)}
.zmid {background:rgba(251,191,36,.10);border:1px solid rgba(251,191,36,.18)}
.zlow {background:rgba(56,189,248,.10);border:1px solid rgba(56,189,248,.18)}
.crane-rec{background:rgba(34,197,94,.08);border:1px solid rgba(34,197,94,.18);
            border-radius:7px;padding:11px;margin-top:8px;font-size:12px}

/* ── sync log ── */
#sync-log{font-size:11px;color:var(--muted);margin-left:10px;max-width:500px;
           white-space:nowrap;overflow:hidden;text-overflow:ellipsis}

/* ── camera feeds ── */
.cam-section{margin:16px 0 8px;font-size:11px;font-weight:700;text-transform:uppercase;
             letter-spacing:.07em;color:var(--muted)}
.cam-grid-2{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px}
.cam-grid-3{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:14px}
.cam-grid-1{margin-bottom:14px}
.cam-panel{background:var(--card);border:1px solid var(--border);border-radius:10px;overflow:hidden}
.cam-label{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;
           color:var(--muted);padding:9px 13px 5px}
.cam-img{width:100%;display:block;aspect-ratio:16/9;object-fit:contain;background:#0a0d14}
.cam-fname{font-size:10px;color:#334155;padding:4px 13px 8px;white-space:nowrap;
           overflow:hidden;text-overflow:ellipsis}

/* ── operator view ── */
.op-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:16px}
.op-cluster{background:var(--card);border:1px solid var(--border);border-radius:12px;overflow:hidden}
.op-cluster-hdr{padding:12px 18px 10px;border-bottom:1px solid var(--border)}
.op-cluster-title{font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:var(--muted)}
.op-cluster-sub{font-size:10px;color:#334155;margin-top:2px}
.op-instruments{display:grid;grid-template-columns:1fr 1fr;gap:0}
.op-instr{padding:20px 18px;border-right:1px solid var(--border);border-bottom:1px solid var(--border)}
.op-instr:nth-child(even){border-right:none}
.op-instr:last-child:nth-child(odd){grid-column:span 2;border-right:none}
.op-instr-lbl{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:8px}
.op-num{font-size:58px;font-weight:700;line-height:1;letter-spacing:-2px}
.op-unit{font-size:18px;font-weight:500;color:var(--muted);margin-left:4px;vertical-align:bottom;line-height:1}
.op-sub{font-size:12px;color:var(--muted);margin-top:6px}
.op-chip{display:inline-block;font-size:15px;font-weight:700;padding:6px 16px;border-radius:99px;margin-top:8px;letter-spacing:.04em}
.op-chip-lg{display:inline-block;font-size:22px;font-weight:700;padding:10px 22px;border-radius:99px;letter-spacing:.03em}
.op-sbar{padding:14px 24px;display:flex;align-items:center;gap:16px;margin-bottom:16px;border-radius:12px}
.op-sbar-icon{font-size:32px;flex-shrink:0}
.op-sbar-text{font-size:20px;font-weight:700}
.op-sbar-sub{font-size:13px;margin-top:2px;opacity:.8}
.op-actions-card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:18px 20px}
.op-actions-title{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:14px}
.op-action{display:flex;align-items:flex-start;gap:12px;padding:10px 0;border-bottom:1px solid var(--border)}
.op-action:last-child{border-bottom:none;padding-bottom:0}
.op-action-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0;margin-top:4px}
.op-action-text{font-size:15px;font-weight:500;line-height:1.4}

/* ── responsive ── */
@media(max-width:960px){
  .c3,.c4,.row2,.row3,.zg,.op-grid,.cam-grid-2,.cam-grid-3{grid-template-columns:1fr}
}
</style>
</head>
<body>

<div class="hdr">
  <div class="hdr-dot" id="status-dot"></div>
  <div>
    <h1>Reworld Haverhill — Waste Intelligence</h1>
    <div class="sub">Camera-based fuel quality &amp; safety monitoring</div>
  </div>
  <div class="spacer"></div>
  <div class="hdr-meta">
    <div id="hdr-data-through">—</div>
    <div id="hdr-generated" style="color:#334155">—</div>
  </div>
</div>

<div id="banner"></div>

<div class="nav">
  <button class="nb active" onclick="showTab('operator')" id="tab-operator">
    <span class="dot" id="op-nav-dot" style="background:var(--green)"></span>Operator View
  </button>
  <button class="nb" onclick="showTab('safety')" id="tab-safety">
    <span class="dot" id="pit-dot"></span>Safety Monitor
  </button>
  <button class="nb" onclick="showTab('furnace')" id="tab-furnace">
    <span class="dot" style="background:var(--sky)"></span>Furnace Feed
  </button>
  <button class="nb" onclick="showTab('tipping')" id="tab-tipping">
    <span class="dot" style="background:var(--blue)"></span>Tipping Floor
  </button>
</div>

<div class="actions">
  <button class="btn btn-refresh" id="btn-refresh" onclick="refresh()">⟳ Refresh</button>
  <button class="btn btn-auto"    id="btn-auto"    onclick="toggleAuto()">⏱ Auto-refresh: OFF</button>
  <button class="btn btn-sync"    id="btn-sync"    onclick="syncS3()">☁ Sync from S3</button>
  <button class="btn btn-ir"      id="btn-ir"      onclick="toggleIR()">🌡 IR: OFF</button>
  <span class="status-text" id="status-text">Loading…</span>
  <span id="sync-log"></span>
</div>

<div class="page">

<!-- LOADING -->
<div id="loading" class="loader">
  <span class="spinner"></span> Loading dashboard data…
</div>

<!-- ═══════════════ OPERATOR VIEW ═══════════════ -->
<div class="tab" id="operator">

  <div class="op-sbar" id="op-sbar">
    <div class="op-sbar-icon" id="op-sbar-icon">—</div>
    <div>
      <div class="op-sbar-text" id="op-sbar-text">Loading…</div>
      <div class="op-sbar-sub" id="op-sbar-sub"></div>
    </div>
  </div>

  <div class="op-grid">
    <!-- PIT -->
    <div class="op-cluster">
      <div class="op-cluster-hdr">
        <div class="op-cluster-title">West Pit</div>
        <div class="op-cluster-sub">45–80 min lead time to furnace</div>
      </div>
      <div class="op-instruments">
        <div class="op-instr">
          <div class="op-instr-lbl">Max Temp</div>
          <div><span class="op-num" id="op-pit-max">—</span><span class="op-unit">°C</span></div>
          <div id="op-pit-chip" class="op-chip" style="margin-top:10px">—</div>
        </div>
        <div class="op-instr">
          <div class="op-instr-lbl">Readings Above 50°C</div>
          <div><span class="op-num" id="op-pit-pct" style="font-size:52px">—</span><span class="op-unit">%</span></div>
          <div class="op-sub" id="op-pit-frames-sub">— of — frames</div>
        </div>
      </div>
    </div>

    <!-- CHUTE B -->
    <div class="op-cluster">
      <div class="op-cluster-hdr">
        <div class="op-cluster-title">Furnace Feed — Chute A &amp; B</div>
        <div class="op-cluster-sub">15–20 min lead time to furnace</div>
      </div>
      <div class="op-instruments">
        <div class="op-instr">
          <div class="op-instr-lbl">Chute A — Fill</div>
          <div><span class="op-num" id="op-a-fill" style="color:var(--sky)">—</span><span class="op-unit">%</span></div>
        </div>
        <div class="op-instr">
          <div class="op-instr-lbl">Chute A — Moisture</div>
          <div><span class="op-num" id="op-a-moist" style="color:#34d399;font-size:48px">—</span></div>
          <div class="op-sub" id="op-a-moist-label">—</div>
        </div>
        <div class="op-instr">
          <div class="op-instr-lbl">Chute B — Fill</div>
          <div><span class="op-num" id="op-fill" style="color:var(--sky)">—</span><span class="op-unit">%</span></div>
        </div>
        <div class="op-instr">
          <div class="op-instr-lbl">Chute B — Moisture</div>
          <div><span class="op-num" id="op-moist" style="color:#34d399;font-size:48px">—</span></div>
          <div class="op-sub" id="op-moist-label">—</div>
        </div>
        <div class="op-instr" style="grid-column:span 2;border-bottom:none">
          <div class="op-instr-lbl">Feed Status (Chute B)</div>
          <div id="op-chute-chip" class="op-chip-lg">—</div>
        </div>
      </div>
    </div>

    <!-- TIPPING FLOOR -->
    <div class="op-cluster">
      <div class="op-cluster-hdr">
        <div class="op-cluster-title">Tipping Floor</div>
        <div class="op-cluster-sub">60–120 min lead time to furnace</div>
      </div>
      <div class="op-instruments">
        <div class="op-instr" style="grid-column:span 2">
          <div class="op-instr-lbl">Thermal Status</div>
          <div><span class="op-num" id="op-tip-max" style="font-size:52px">—</span><span class="op-unit">°C</span></div>
          <div id="op-tip-chip" class="op-chip-lg" style="margin-top:4px">—</div>
        </div>
        <div class="op-instr">
          <div class="op-instr-lbl">Tipping 1 — Max Temp</div>
          <div><span class="op-num" id="op-tip1-max" style="font-size:42px">—</span><span class="op-unit">°C</span></div>
        </div>
        <div class="op-instr">
          <div class="op-instr-lbl">Tipping 2 — Max Temp</div>
          <div><span class="op-num" id="op-tip2-max" style="font-size:42px">—</span><span class="op-unit">°C</span></div>
        </div>
        <div class="op-instr" style="border-bottom:none">
          <div class="op-instr-lbl">Tipping 3 — Max Temp</div>
          <div><span class="op-num" id="op-tip3-max" style="font-size:42px">—</span><span class="op-unit">°C</span></div>
        </div>
        <div class="op-instr" style="border-bottom:none">
          <div class="op-instr-lbl">Contaminant Detection</div>
          <div class="op-chip" style="background:rgba(168,85,247,.15);color:#c084fc;
               border:1px solid rgba(168,85,247,.3);margin-top:8px;font-size:12px">
            Phase 2 — Not yet active
          </div>
        </div>
      </div>
    </div>
  </div><!-- /op-grid -->

  <div class="op-actions-card">
    <div class="op-actions-title">Operator Actions</div>
    <div id="op-action-list">
      <div class="op-action"><div class="op-action-dot" style="background:var(--muted)"></div>
        <div class="op-action-text" style="color:var(--muted)">Loading…</div></div>
    </div>
  </div>

  <!-- Camera feeds -->
  <div class="cam-section">Chute Cameras</div>
  <div class="cam-grid-2">
    <div class="cam-panel">
      <div class="cam-label">Chute A</div>
      <img id="cam-achute" class="cam-img" alt="Chute A" src="./api/image/achute" onclick="openModal('achute')"
           onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%221280%22 height=%22720%22><rect fill=%22%231a1f2e%22 width=%221280%22 height=%22720%22/><text x=%22640%22 y=%22360%22 text-anchor=%22middle%22 dominant-baseline=%22middle%22 fill=%22%23334155%22 font-size=%2220%22 font-family=%22sans-serif%22>No feed available</text></svg>'"/>
      <div class="cam-fname" id="cam-achute-name">—</div>
    </div>
    <div class="cam-panel">
      <div class="cam-label">Chute B</div>
      <img id="cam-chuteb" class="cam-img" alt="Chute B" src="./api/image/chuteb" onclick="openModal('chuteb')"
           onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%221280%22 height=%22720%22><rect fill=%22%231a1f2e%22 width=%221280%22 height=%22720%22/><text x=%22640%22 y=%22360%22 text-anchor=%22middle%22 dominant-baseline=%22middle%22 fill=%22%23334155%22 font-size=%2220%22 font-family=%22sans-serif%22>No feed available</text></svg>'"/>
      <div class="cam-fname" id="cam-chuteb-name">—</div>
    </div>
  </div>

  <div class="cam-section">West Pit</div>
  <div class="cam-grid-1">
    <div class="cam-panel">
      <div class="cam-label">West Pit</div>
      <img id="cam-westpit" class="cam-img" alt="West Pit" src="./api/image/west-pit" onclick="openModal('west-pit')"
           onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%221280%22 height=%22720%22><rect fill=%22%231a1f2e%22 width=%221280%22 height=%22720%22/><text x=%22640%22 y=%22360%22 text-anchor=%22middle%22 dominant-baseline=%22middle%22 fill=%22%23334155%22 font-size=%2220%22 font-family=%22sans-serif%22>No feed available</text></svg>'"/>
      <div class="cam-fname" id="cam-westpit-name">—</div>
    </div>
  </div>

  <div class="cam-section">Tipping Floor Cameras</div>
  <div class="cam-grid-3">
    <div class="cam-panel">
      <div class="cam-label">Tipping Floor 1</div>
      <img id="cam-tipping1" class="cam-img" alt="Tipping 1" src="./api/image/tipping1" onclick="openModal('tipping1')"
           onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%221280%22 height=%22720%22><rect fill=%22%231a1f2e%22 width=%221280%22 height=%22720%22/><text x=%22640%22 y=%22360%22 text-anchor=%22middle%22 dominant-baseline=%22middle%22 fill=%22%23334155%22 font-size=%2220%22 font-family=%22sans-serif%22>No feed available</text></svg>'"/>
      <div class="cam-fname" id="cam-tipping1-name">—</div>
    </div>
    <div class="cam-panel">
      <div class="cam-label">Tipping Floor 2</div>
      <img id="cam-tipping2" class="cam-img" alt="Tipping 2" src="./api/image/tipping2" onclick="openModal('tipping2')"
           onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%221280%22 height=%22720%22><rect fill=%22%231a1f2e%22 width=%221280%22 height=%22720%22/><text x=%22640%22 y=%22360%22 text-anchor=%22middle%22 dominant-baseline=%22middle%22 fill=%22%23334155%22 font-size=%2220%22 font-family=%22sans-serif%22>No feed available</text></svg>'"/>
      <div class="cam-fname" id="cam-tipping2-name">—</div>
    </div>
    <div class="cam-panel">
      <div class="cam-label">Tipping Floor 3</div>
      <img id="cam-tipping3" class="cam-img" alt="Tipping 3" src="./api/image/tipping3" onclick="openModal('tipping3')"
           onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%221280%22 height=%22720%22><rect fill=%22%231a1f2e%22 width=%221280%22 height=%22720%22/><text x=%22640%22 y=%22360%22 text-anchor=%22middle%22 dominant-baseline=%22middle%22 fill=%22%23334155%22 font-size=%2220%22 font-family=%22sans-serif%22>No feed available</text></svg>'"/>
      <div class="cam-fname" id="cam-tipping3-name">—</div>
    </div>
  </div>

</div><!-- /operator -->

<!-- ═══════════════ SAFETY ═══════════════ -->
<div class="tab" id="safety">
  <div class="sec">
    Live Monitoring &nbsp;
    <span class="ptag" style="background:rgba(239,68,68,.13);color:#fca5a5;border:1px solid rgba(239,68,68,.22)">Control Room</span>
    <span class="ptag" style="background:rgba(59,130,246,.13);color:#93c5fd;border:1px solid rgba(59,130,246,.22);margin-left:3px">Floor Supervisor</span>
  </div>

  <div class="cards c3" id="safety-cards">
    <div class="card" id="card-pit">
      <div class="lbl">West Pit — Current Status</div>
      <div class="val" id="pit-val">—<span class="unit">°C</span></div>
      <div class="csub">Peak temperature, latest frame</div>
      <div class="badge" id="pit-badge">—</div>
    </div>
    <div class="card">
      <div class="lbl">Pit — Frames Above 50°C</div>
      <div class="val" style="color:var(--orange)" id="pit-alert-frames">—</div>
      <div class="csub" id="pit-total-frames">out of — frames</div>
    </div>
    <div class="card" id="card-tip">
      <div class="lbl">Tipping Floor — Max Recorded</div>
      <div class="val" id="tip-val">—<span class="unit">°C</span></div>
      <div class="csub">Across all 3 cameras</div>
      <div class="badge" id="tip-badge">—</div>
    </div>
  </div>

  <!-- Live camera image -->
  <div class="row2" style="margin-bottom:14px">
    <div class="cc" style="padding:12px">
      <div class="lbl" style="margin-bottom:8px">West Pit — Latest RGB Frame</div>
      <img id="wp-img"
           src="./api/west-pit/latest-image"
           style="width:100%;display:block;border-radius:6px;aspect-ratio:16/9;object-fit:cover;background:var(--card2)"
           onerror="document.getElementById('wp-img-err').style.display='block';this.style.display='none'"/>
      <div id="wp-img-err" style="display:none;padding:20px;text-align:center;color:var(--muted);font-size:11px">Image unavailable — run Sync to fetch</div>
      <div id="wp-img-meta" style="font-size:10px;color:var(--muted);margin-top:6px;display:flex;justify-content:space-between">
        <span id="wp-img-ts"></span><span id="wp-img-uploaded"></span>
      </div>
    </div>
    <div class="cc"><div id="c-pit_temp"></div></div>
  </div>
  <div class="row2">
    <div class="cc"><div id="c-pit_maxmap"></div></div>
    <div class="cc"><div id="c-pit_h50"></div></div>
  </div>
  <div class="cc"><div id="c-tipping_temp"></div></div>

  <!-- Contaminant placeholder -->
  <div class="ph">
    <div class="ph-badge">Phase 2 — Q3 2026</div>
    <div class="demo-chip">DEMO PREVIEW</div>
    <h4>Contaminant Detection — Tipping Floor</h4>
    <p class="ph-desc">Detects hazardous objects (propane cylinders, compressed gas, mattresses, water heaters) from tipping floor RGB images. Example alerts shown below.</p>
    <div class="det-box">
      <div class="det-row">
        <div class="det-ico">🚨</div>
        <div>
          <div class="det-title" style="color:var(--orange)">Propane Cylinder Detected</div>
          <div class="det-meta">Camera: Tipping 1 · Zone: Left-centre · 2026-04-30 09:15</div>
          <div class="det-action">⚡ Remove from floor before pit entry</div>
          <span class="conf" style="background:rgba(249,115,22,.13);color:var(--orange)">Confidence: 89%</span>
        </div>
      </div>
      <div class="det-row">
        <div class="det-ico">⚠️</div>
        <div>
          <div class="det-title" style="color:var(--yellow)">Mattress Detected</div>
          <div class="det-meta">Camera: Tipping 2 · Zone: Right · 2026-04-30 11:42</div>
          <div class="det-action">⚡ Remove — grate jam risk</div>
          <span class="conf" style="background:rgba(234,179,8,.12);color:var(--yellow)">Confidence: 94%</span>
        </div>
      </div>
    </div>
    <p style="font-size:11px;color:var(--muted);margin-top:8px">Requires ~50–100 labeled examples per object class.</p>
  </div>
</div><!-- /safety -->

<!-- ═══════════════ FURNACE FEED ═══════════════ -->
<div class="tab" id="furnace">
  <div class="sec">
    Live Signals — Chute A &amp; Chute B (15–20 min furnace lead time) &nbsp;
    <span class="ptag" style="background:rgba(56,189,248,.12);color:#7dd3fc;border:1px solid rgba(56,189,248,.22)">Combustion Engineer</span>
    <span class="ptag" style="background:rgba(59,130,246,.12);color:#93c5fd;border:1px solid rgba(59,130,246,.22);margin-left:3px">Control Room</span>
  </div>

  <div class="cards c4">
    <div class="card">
      <div class="lbl">Chute A — Fill Level</div>
      <div class="val" style="color:var(--sky)" id="ca-fill">—<span class="unit">%</span></div>
      <div class="csub">Latest frame</div>
    </div>
    <div class="card">
      <div class="lbl">Chute A — Moisture</div>
      <div class="val" style="color:#34d399" id="ca-moist">—</div>
      <div class="csub">0 = dry (high BTU) · 1 = wet</div>
    </div>
    <div class="card">
      <div class="lbl">Chute B — Fill Level</div>
      <div class="val" style="color:var(--sky)" id="cb-fill">—<span class="unit">%</span></div>
      <div class="csub">Latest frame</div>
    </div>
    <div class="card">
      <div class="lbl">Chute B — Moisture</div>
      <div class="val" style="color:#34d399" id="cb-moist">—</div>
      <div class="csub">0 = dry (high BTU) · 1 = wet</div>
    </div>
  </div>

  <div class="row2">
    <div class="cc"><div id="c-achute_combined"></div></div>
    <div class="cc"><div id="c-chute_combined"></div></div>
  </div>
  <div class="row2">
    <div class="cc"><div id="c-achute_fill"></div></div>
    <div class="cc"><div id="c-chute_fill"></div></div>
  </div>
  <div class="row2">
    <div class="cc"><div id="c-achute_moisture"></div></div>
    <div class="cc"><div id="c-chute_moisture"></div></div>
  </div>

  <!-- Plastic fraction -->
  <div class="ph">
    <div class="ph-badge">Phase 3 — Q4 2026</div>
    <div class="demo-chip">DEMO PREVIEW</div>
    <h4>Plastic Fraction Estimator — Chute B</h4>
    <p class="ph-desc">% of visible waste that is plastic. Plastic has ~3× the energy content of mixed MSW — a reading above 40% signals a high-BTU slug 15–20 min ahead. Synthetic data shown.</p>
    <div class="cc" style="border:none;padding:0;background:none"><div id="c-plastic_frac"></div></div>
    <p style="font-size:11px;color:var(--muted);margin-top:4px">Requires ~200 expert-labeled frames.</p>
  </div>

  <div class="row2">
    <div class="ph">
      <div class="ph-badge">Phase 3 — Q4 2026</div>
      <div class="demo-chip">DEMO PREVIEW</div>
      <h4>Organic Fraction</h4>
      <p class="ph-desc">High organic = low BTU, high moisture.</p>
      <div id="c-organic_frac"></div>
    </div>
    <div class="ph">
      <div class="ph-badge">Phase 4 — historian-dependent</div>
      <div class="demo-chip">DEMO PREVIEW</div>
      <h4>Predicted HHV (MJ/kg)</h4>
      <p class="ph-desc">Combines all signals into one fuel quality number. Green band = target combustion zone. Blocked on historian-camera data overlap (gap closes ~summer 2026).</p>
      <div id="c-btu_pred"></div>
    </div>
  </div>
</div><!-- /furnace -->

<!-- ═══════════════ TIPPING FLOOR ═══════════════ -->
<div class="tab" id="tipping">
  <div class="sec">
    Live Signals — Tipping Floor (60–120 min furnace lead time) &nbsp;
    <span class="ptag" style="background:rgba(99,102,241,.12);color:#a5b4fc;border:1px solid rgba(99,102,241,.22)">Floor Supervisor</span>
    <span class="ptag" style="background:rgba(59,130,246,.12);color:#93c5fd;border:1px solid rgba(59,130,246,.22);margin-left:3px">Dispatch</span>
  </div>

  <div class="cc">
    <div id="c-tipping_uniform"></div>
    <p style="font-size:11px;color:var(--muted);margin-top:7px;padding-left:2px">
      Low spatial std = thermally uniform load = likely single-material truck deposit.
      Dips below the purple line warrant a pre-emptive combustion adjustment 60–120 min in advance.
    </p>
  </div>

  <!-- Homogenous load placeholder -->
  <div class="ph" style="border-color:rgba(96,165,250,.28)">
    <div class="ph-badge" style="background:rgba(96,165,250,.13);color:#93c5fd;border-color:rgba(96,165,250,.28)">Phase 1 — Building Now</div>
    <div class="demo-chip" style="background:rgba(96,165,250,.1);color:#93c5fd">DEMO PREVIEW</div>
    <h4 style="color:#93c5fd">Homogenous Load Detector</h4>
    <p class="ph-desc">When a truck deposits a single-material load, the system names it and estimates the BTU impact ~75 min before it hits the furnace. No labeling required — unsupervised.</p>
    <div class="det-box">
      <div class="det-row">
        <div class="det-ico">📦</div>
        <div>
          <div class="det-title" style="color:#93c5fd">Homogenous Load — Paper / Cardboard</div>
          <div class="det-meta">Camera: Tipping 2 · 2026-04-30 07:22</div>
          <div class="det-action" style="color:#60a5fa">⚡ LOW BTU slug in ~75 min — prepare grate adjustment</div>
        </div>
      </div>
      <div class="det-row">
        <div class="det-ico">🛍️</div>
        <div>
          <div class="det-title" style="color:var(--orange)">Homogenous Load — Plastic Film</div>
          <div class="det-meta">Camera: Tipping 1 · 2026-04-30 14:10</div>
          <div class="det-action" style="color:#fbbf24">⚡ HIGH BTU spike in ~90 min — pre-cool airflow</div>
        </div>
      </div>
    </div>
  </div>

  <!-- Pit zone map placeholder -->
  <div class="ph">
    <div class="ph-badge">Phase 2 — Q3 2026</div>
    <div class="demo-chip">DEMO PREVIEW</div>
    <h4>Pit Zone Composition Map — Crane Homogenisation Guidance</h4>
    <p class="ph-desc">Updated every 5 min. Shows BTU distribution across the pit and directs the crane operator where to pick and deposit to blend toward target HHV.</p>
    <div>
      <div class="zg">
        <div class="zc zhigh"><div class="zl" style="color:#fca5a5">HIGH BTU</div><div class="zb" style="color:#fca5a5">Plastic-rich</div></div>
        <div class="zc zmid" ><div class="zl" style="color:#fde68a">MID</div><div class="zb" style="color:#fde68a">Mixed</div></div>
        <div class="zc zhigh"><div class="zl" style="color:#fca5a5">HIGH BTU</div><div class="zb" style="color:#fca5a5">Plastic-rich</div></div>
        <div class="zc zlow" ><div class="zl" style="color:#7dd3fc">LOW BTU</div><div class="zb" style="color:#7dd3fc">Organic-dom.</div></div>
        <div class="zc zhigh"><div class="zl" style="color:#fca5a5">HIGH BTU</div><div class="zb" style="color:#fca5a5">Plastic-rich</div></div>
        <div class="zc zmid" ><div class="zl" style="color:#fde68a">MID</div><div class="zb" style="color:#fde68a">Paper / card</div></div>
      </div>
      <div class="crane-rec">
        <strong style="color:var(--green)">Crane recommendation:</strong><br/>
        PICK from bottom-left (Low BTU — organic-dominant)<br/>
        DEPOSIT near top-left or top-centre (blend into High BTU zones)<br/>
        Estimated blend HHV: <strong style="color:#4ade80">11.8 MJ/kg</strong> (target 11–12)
      </div>
    </div>
    <p style="font-size:11px;color:var(--muted);margin-top:8px">Requires ~150 labeled frames with per-zone composition estimates.</p>
  </div>
</div><!-- /tipping -->

</div><!-- /page -->

<script>
// ── chart map: id → div element ────────────────────────────────────────────
const CHART_IDS = [
  'c-pit_temp','c-pit_maxmap','c-pit_h50',
  'c-tipping_temp','c-tipping_uniform',
  'c-achute_fill','c-achute_moisture','c-achute_combined',
  'c-chute_fill','c-chute_moisture','c-chute_combined',
  'c-plastic_frac','c-organic_frac','c-btu_pred',
];
const KEY_MAP = {
  'c-pit_temp':'pit_temp','c-pit_maxmap':'pit_maxmap','c-pit_h50':'pit_h50',
  'c-tipping_temp':'tipping_temp','c-tipping_uniform':'tipping_uniform',
  'c-achute_fill':'achute_fill','c-achute_moisture':'achute_moisture',
  'c-achute_combined':'achute_combined',
  'c-chute_fill':'chute_fill','c-chute_moisture':'chute_moisture',
  'c-chute_combined':'chute_combined',
  'c-plastic_frac':'plastic_frac','c-organic_frac':'organic_frac',
  'c-btu_pred':'btu_pred',
};
const CFG = {responsive:true, displayModeBar:false};

let autoTimer = null;
let autoOn = false;
let firstLoad = true;
let irOverlay = false;

// ── alert colour helpers ──────────────────────────────────────────────────
function alertStyle(level){
  const m = {
    CRITICAL:{color:'#ef4444',bg:'rgba(239,68,68,.14)'},
    WARNING: {color:'#f97316',bg:'rgba(249,115,22,.14)'},
    CAUTION: {color:'#eab308',bg:'rgba(234,179,8,.12)'},
    NORMAL:  {color:'#22c55e',bg:'rgba(34,197,94,.12)'},
  };
  return m[level] || m.NORMAL;
}

const ACTION_COLORS = {
  red:    '#ef4444', orange: '#f97316', yellow: '#eab308',
  green:  '#22c55e', blue:   '#818cf8',
};

// ── apply summary stats to DOM ────────────────────────────────────────────
function applySummary(s){
  document.getElementById('hdr-data-through').textContent = 'Data through: ' + s.data_through;
  document.getElementById('hdr-generated').textContent    = 'Refreshed: '    + s.generated;

  // ── OPERATOR VIEW ────────────────────────────────────────────────────────
  const worstColor = s.pit_color;
  const overallIcon = s.pit_level === 'NORMAL' && s.tip_level === 'NORMAL' ? '✅' : s.pit_icon;
  const overallMsg  = s.pit_level !== 'NORMAL' ? `Pit ${s.pit_level}: ${s.pit_max}°C`
                    : s.tip_level !== 'NORMAL'  ? `Tipping floor ${s.tip_level}: ${s.tip_max}°C`
                    : s.chute_status !== 'NORMAL' ? `Chute: ${s.chute_status}`
                    : 'All systems nominal';
  const sbar = document.getElementById('op-sbar');
  sbar.style.background = s.pit_level === 'NORMAL' && s.tip_level === 'NORMAL' && s.chute_status === 'NORMAL'
    ? 'rgba(34,197,94,.09)' : `${worstColor}1e`;
  sbar.style.border = `1px solid ${worstColor}44`;
  document.getElementById('op-sbar-icon').textContent = overallIcon;
  document.getElementById('op-sbar-text').textContent = overallMsg;
  document.getElementById('op-sbar-text').style.color = worstColor;
  document.getElementById('op-sbar-sub').textContent  =
    `Data through ${s.data_through}  ·  Refreshed ${s.generated}`;
  document.getElementById('op-nav-dot').style.cssText =
    `background:${worstColor};box-shadow:0 0 6px ${worstColor};` +
    `display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:5px;vertical-align:middle`;
  document.getElementById('op-pit-max').textContent = s.pit_max;
  document.getElementById('op-pit-max').style.color  = s.pit_color;
  const pitChip = document.getElementById('op-pit-chip');
  pitChip.textContent = s.pit_icon + ' ' + s.pit_level;
  pitChip.style.cssText = `background:${s.pit_color}22;color:${s.pit_color};` +
    `border:1px solid ${s.pit_color}55;display:inline-block;font-size:15px;` +
    `font-weight:700;padding:6px 16px;border-radius:99px;margin-top:10px;letter-spacing:.04em`;
  document.getElementById('op-pit-pct').textContent        = s.pit_hotspot_pct;
  document.getElementById('op-pit-pct').style.color        = s.pit_color;
  document.getElementById('op-pit-frames-sub').textContent =
    s.pit_alert_frames + ' of ' + s.pit_total_frames + ' readings';
  document.getElementById('op-a-fill').textContent        = s.achute_fill;
  document.getElementById('op-a-moist').textContent       = s.achute_moisture;
  document.getElementById('op-a-moist-label').textContent = s.achute_moist_label;
  document.getElementById('op-fill').textContent          = s.chute_fill;
  document.getElementById('op-moist').textContent         = s.chute_moisture;
  document.getElementById('op-moist-label').textContent   = s.moist_label;

  document.getElementById('op-tip1-max').textContent = s.tip1_max;
  document.getElementById('op-tip1-max').style.color  = s.tip_color;
  document.getElementById('op-tip2-max').textContent = s.tip2_max;
  document.getElementById('op-tip2-max').style.color  = s.tip_color;
  document.getElementById('op-tip3-max').textContent = s.tip3_max;
  document.getElementById('op-tip3-max').style.color  = s.tip_color;
  const cc = document.getElementById('op-chute-chip');
  cc.textContent = s.chute_status;
  cc.style.cssText = `background:${s.chute_color}22;color:${s.chute_color};` +
    `border:1px solid ${s.chute_color}55;display:inline-block;font-size:22px;` +
    `font-weight:700;padding:10px 22px;border-radius:99px;letter-spacing:.03em`;
  const tc = document.getElementById('op-tip-chip');
  tc.textContent = s.tip_icon + ' ' + s.tip_level;
  tc.style.cssText = `background:${s.tip_color}22;color:${s.tip_color};` +
    `border:1px solid ${s.tip_color}55;display:inline-block;font-size:22px;` +
    `font-weight:700;padding:10px 22px;border-radius:99px;letter-spacing:.03em`;
  document.getElementById('op-tip-max').textContent = s.tip_max;
  document.getElementById('op-tip-max').style.color  = s.tip_color;
  document.getElementById('op-action-list').innerHTML = s.actions.map(([col, txt]) =>
    `<div class="op-action">
       <div class="op-action-dot" style="background:${ACTION_COLORS[col]};box-shadow:0 0 5px ${ACTION_COLORS[col]}88;margin-top:5px"></div>
       <div class="op-action-text" style="color:${ACTION_COLORS[col]}">${txt}</div>
     </div>`
  ).join('');

  const ps = alertStyle(s.pit_level);
  document.getElementById('pit-val').innerHTML   = s.pit_max + '<span class="unit">°C</span>';
  document.getElementById('pit-badge').textContent = s.pit_icon + ' ' + s.pit_level;
  document.getElementById('pit-badge').style.cssText =
    `background:${ps.bg};color:${ps.color};border:1px solid ${ps.color}44`;
  document.getElementById('card-pit').style.borderColor = ps.color + '44';
  document.getElementById('pit-dot').style.cssText =
    `background:${ps.color};box-shadow:0 0 6px ${ps.color}`;

  document.getElementById('pit-alert-frames').textContent = s.pit_alert_frames;
  document.getElementById('pit-total-frames').textContent =
    'out of ' + s.pit_total_frames + ' frames (last 4 days)';

  const ts = alertStyle(s.tip_level);
  document.getElementById('tip-val').innerHTML   = s.tip_max + '<span class="unit">°C</span>';
  document.getElementById('tip-val').style.color = ts.color;
  document.getElementById('tip-badge').textContent = s.tip_icon + ' ' + s.tip_level;
  document.getElementById('tip-badge').style.cssText =
    `background:${ts.bg};color:${ts.color};border:1px solid ${ts.color}44`;

  document.getElementById('ca-fill').innerHTML   = s.achute_fill + '<span class="unit">%</span>';
  document.getElementById('ca-moist').textContent = s.achute_moisture;
  document.getElementById('cb-fill').innerHTML   = s.chute_fill + '<span class="unit">%</span>';
  document.getElementById('cb-moist').textContent = s.chute_moisture;

  // alert banner
  const banner = document.getElementById('banner');
  if(s.pit_level === 'CRITICAL' || s.pit_level === 'WARNING'){
    const bg = s.pit_level === 'CRITICAL' ? '#7f1d1d' : '#7c2d12';
    banner.style.background = bg;
    banner.style.display    = 'block';
    banner.textContent      = s.pit_icon + ' ' + s.pit_level + ': West Pit at ' +
      s.pit_max + '°C — ' + s.pit_alert_frames + ' of ' +
      s.pit_total_frames + ' frames exceed 50°C. Crane intervention may be required.';
  } else {
    banner.style.display = 'none';
  }
}

// ── render / update charts ────────────────────────────────────────────────
function renderCharts(charts){
  for(const divId of CHART_IDS){
    const el  = document.getElementById(divId);
    if(!el) continue;
    const key = KEY_MAP[divId];
    const spec = charts[key];
    if(!spec) continue;
    if(firstLoad){
      Plotly.newPlot(el, spec.data, spec.layout, CFG);
    } else {
      Plotly.react(el, spec.data, spec.layout, CFG);
    }
  }
  firstLoad = false;
}

// ── main refresh ──────────────────────────────────────────────────────────
async function refresh(){
  const btn = document.getElementById('btn-refresh');
  const st  = document.getElementById('status-text');
  btn.disabled = true;
  st.textContent = 'Loading…';
  document.getElementById('status-dot').style.background = '#eab308';

  try{
    const resp  = await fetch('./api/data', {cache:'no-store'});
    const payload = await resp.json();
    applySummary(payload.summary);
    renderCharts(payload.charts);
    // Reload all camera images (respects IR overlay state)
    _reloadCams(Date.now());
    document.getElementById('loading').style.display = 'none';
    document.querySelectorAll('.tab').forEach(el => el.style.visibility = 'visible');
    st.textContent = 'Last refreshed ' + new Date().toLocaleTimeString();
    document.getElementById('status-dot').style.cssText =
      'width:9px;height:9px;border-radius:50%;background:var(--green);box-shadow:0 0 7px var(--green);flex-shrink:0';
  } catch(e){
    st.textContent = 'Error loading data — ' + e.message;
    document.getElementById('status-dot').style.background = '#ef4444';
  } finally {
    btn.disabled = false;
  }
}

// ── auto-refresh ──────────────────────────────────────────────────────────
function toggleAuto(){
  autoOn = !autoOn;
  const btn = document.getElementById('btn-auto');
  if(autoOn){
    btn.textContent = '⏱ Auto-refresh: ON (5 min)';
    btn.style.background = 'rgba(34,197,94,.15)';
    btn.style.color = '#86efac';
    autoTimer = setInterval(refresh, 5 * 60 * 1000);
  } else {
    btn.textContent = '⏱ Auto-refresh: OFF';
    btn.style.background = 'rgba(99,102,241,.12)';
    btn.style.color = '#a5b4fc';
    clearInterval(autoTimer);
  }
}

// ── IR overlay toggle ─────────────────────────────────────────────────────
const CAM_ROUTES = [
  ['cam-achute',   'achute'],
  ['cam-chuteb',   'chuteb'],
  ['cam-westpit',  'west-pit'],
  ['cam-tipping1', 'tipping1'],
  ['cam-tipping2', 'tipping2'],
  ['cam-tipping3', 'tipping3'],
];

function _reloadCams(ts){
  const ov = irOverlay ? '&overlay=1' : '';
  CAM_ROUTES.forEach(([id, cam]) => {
    const el = document.getElementById(id);
    if(el) el.src = `./api/image/${cam}?ts=${ts}${ov}`;
  });
}

function toggleIR(){
  irOverlay = !irOverlay;
  const btn = document.getElementById('btn-ir');
  if(irOverlay){
    btn.textContent = '🌡 IR: ON';
    btn.style.background = 'rgba(249,115,22,.15)';
    btn.style.color = '#fdba74';
    btn.style.borderColor = 'rgba(249,115,22,.3)';
  } else {
    btn.textContent = '🌡 IR: OFF';
    btn.style.background = 'rgba(99,102,241,.12)';
    btn.style.color = '#a5b4fc';
    btn.style.borderColor = 'rgba(99,102,241,.25)';
  }
  _reloadCams(Date.now());
}

// ── S3 sync ───────────────────────────────────────────────────────────────
let syncPoll = null;
async function syncS3(){
  const btn = document.getElementById('btn-sync');
  const log = document.getElementById('sync-log');
  btn.disabled = true;
  log.textContent = 'Starting sync…';
  try{
    await fetch('./api/sync', {method:'POST', cache:'no-store'});
    syncPoll = setInterval(async ()=>{
      const r = await fetch('./api/sync/status', {cache:'no-store'});
      const s = await r.json();
      log.textContent = s.log.length ? s.log[s.log.length-1] : '…';
      if(!s.running){
        clearInterval(syncPoll);
        log.textContent = s.ok ? '✓ Sync complete — refreshing…' : '✗ Sync error (check logs)';
        btn.disabled = false;
        if(s.ok) setTimeout(refresh, 800);
      }
    }, 2000);
  } catch(e){
    log.textContent = 'Sync failed: ' + e.message;
    btn.disabled = false;
  }
}

// ── tab switching ─────────────────────────────────────────────────────────
function showTab(id){
  document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.nb').forEach(el => el.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  document.getElementById('tab-' + id).classList.add('active');
  setTimeout(()=> window.dispatchEvent(new Event('resize')), 60);
}

// ── init ──────────────────────────────────────────────────────────────────
document.querySelectorAll('.tab').forEach(el => el.style.visibility = 'hidden');
showTab('operator');
refresh();

// ── camera lightbox ──────────────────────────────────────────────────────
let modalCam = null;
let modalIR  = false;

const CAM_DISPLAY = {
  'achute':'Chute A','chuteb':'Chute B','west-pit':'West Pit',
  'tipping1':'Tipping Floor 1','tipping2':'Tipping Floor 2','tipping3':'Tipping Floor 3'
};
const INFERNO_STOPS = [
  '#000004','#1b0c41','#4a0c6b','#781c6d',
  '#a52c60','#cf4446','#ed6925','#fb9b06','#f7d13d','#fcffa4'
];

function _drawColorbar(minC, maxC){
  const canvas = document.getElementById('modal-colorbar');
  const ctx = canvas.getContext('2d');
  const h = canvas.height;
  const grad = ctx.createLinearGradient(0, h, 0, 0);
  INFERNO_STOPS.forEach((c, i) => grad.addColorStop(i / (INFERNO_STOPS.length - 1), c));
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, canvas.width, h);
  const wrap = document.getElementById('modal-temp-labels');
  wrap.innerHTML = '';
  for(let i = 5; i >= 0; i--){
    const span = document.createElement('span');
    span.textContent = (minC !== null && maxC !== null)
      ? (minC + (maxC - minC) * (i / 5)).toFixed(1) + '°C'
      : (i * 20) + '%';
    wrap.appendChild(span);
  }
}

function _setModalIRState(on){
  modalIR = on;
  const btn = document.getElementById('modal-btn-ir');
  if(on){
    btn.textContent = '🌡 IR: ON';
    btn.style.cssText = 'background:rgba(249,115,22,.15);color:#fdba74;border-color:rgba(249,115,22,.3)';
  } else {
    btn.textContent = '🌡 IR: OFF';
    btn.style.cssText = 'background:rgba(99,102,241,.12);color:#a5b4fc;border-color:rgba(99,102,241,.25)';
  }
  document.getElementById('modal-colorbar-section').style.display = on ? 'block' : 'none';
  if(modalCam){
    const ov = on ? '&overlay=1' : '';
    document.getElementById('modal-img').src = `./api/image/${modalCam}?ts=${Date.now()}${ov}`;
  }
}

function toggleModalIR(){ _setModalIRState(!modalIR); }

async function openModal(cam){
  modalCam = cam;
  document.getElementById('modal-cam-label').textContent = CAM_DISPLAY[cam] || cam;
  _setModalIRState(irOverlay);
  document.getElementById('cam-modal').classList.add('open');
  document.body.style.overflow = 'hidden';
  ['meta-captured','meta-uploaded','meta-temp-min','meta-temp-max',
   'meta-temp-mean','meta-rgb-path','meta-ir-path'].forEach(id => {
    const el = document.getElementById(id);
    if(el) el.textContent = '…';
  });
  try{
    const r = await fetch(`./api/image/${cam}/meta`, {cache:'no-store'});
    const m = await r.json();
    document.getElementById('meta-captured').textContent = m.captured_at || '—';
    document.getElementById('meta-uploaded').textContent = m.uploaded_at || '—';
    document.getElementById('meta-rgb-path').textContent = m.rgb_s3_path || '—';
    const irRow = document.getElementById('meta-ir-path-row');
    if(m.ir_s3_path){
      document.getElementById('meta-ir-path').textContent = m.ir_s3_path;
      irRow.style.display = '';
    } else {
      irRow.style.display = 'none';
    }
    const hasTemp = m.temp_min_c !== null && m.temp_max_c !== null;
    document.getElementById('meta-temp-section').style.display = hasTemp ? '' : 'none';
    if(hasTemp){
      document.getElementById('meta-temp-min').textContent  = m.temp_min_c + ' °C';
      document.getElementById('meta-temp-max').textContent  = m.temp_max_c + ' °C';
      document.getElementById('meta-temp-mean').textContent = m.temp_mean_c != null ? m.temp_mean_c + ' °C' : '—';
    }
    _drawColorbar(hasTemp ? m.temp_min_c : null, hasTemp ? m.temp_max_c : null);
    document.getElementById('modal-colorbar-section').style.display = modalIR ? 'block' : 'none';
  } catch(e){ console.warn('meta fetch failed:', e); }
}

function closeModal(){
  document.getElementById('cam-modal').classList.remove('open');
  document.body.style.overflow = '';
  modalCam = null;
}

document.addEventListener('keydown', e => { if(e.key === 'Escape') closeModal(); });
document.getElementById('cam-modal').addEventListener('click', e => {
  if(e.target === document.getElementById('cam-modal')) closeModal();
});
</script>

<!-- ── camera lightbox modal ── -->
<div id="cam-modal">
  <div class="modal-inner">
    <div class="modal-hdr">
      <span id="modal-cam-label" class="modal-title">Camera</span>
      <div style="display:flex;align-items:center;gap:10px">
        <button id="modal-btn-ir" class="btn btn-ir" onclick="toggleModalIR()">🌡 IR: OFF</button>
        <button class="modal-close" onclick="closeModal()" title="Close (Esc)">×</button>
      </div>
    </div>
    <div class="modal-body">
      <div class="modal-img-wrap">
        <img id="modal-img" alt="Camera feed"/>
      </div>
      <div class="modal-sidebar">
        <!-- Colorbar (visible when IR is on) -->
        <div id="modal-colorbar-section" style="display:none">
          <div class="mlbl">Temperature Scale</div>
          <div class="colorbar-row">
            <canvas id="modal-colorbar" width="24" height="180"></canvas>
            <div id="modal-temp-labels"></div>
          </div>
        </div>

        <div class="mlbl">Frame Info</div>
        <div class="mrow"><span class="mkey">Captured: </span><span id="meta-captured">…</span></div>
        <div class="mrow"><span class="mkey">Uploaded: </span><span id="meta-uploaded">…</span></div>

        <div class="mlbl">Temperature</div>
        <div id="meta-temp-section">
          <div class="mrow"><span class="mkey">Min: </span><span id="meta-temp-min">…</span></div>
          <div class="mrow"><span class="mkey">Max: </span><span id="meta-temp-max">…</span></div>
          <div class="mrow"><span class="mkey">Mean: </span><span id="meta-temp-mean">…</span></div>
        </div>

        <div class="mlbl">S3 Location</div>
        <div style="margin-bottom:8px">
          <div class="mkey" style="font-size:10px">RGB</div>
          <div id="meta-rgb-path" class="mpath">…</div>
        </div>
        <div id="meta-ir-path-row">
          <div class="mkey" style="font-size:10px">IR</div>
          <div id="meta-ir-path" class="mpath">…</div>
        </div>
      </div>
    </div>
  </div>
</div>
</body>
</html>"""

@app.get('/', response_class=HTMLResponse)
def root():
    return HTML_SHELL
