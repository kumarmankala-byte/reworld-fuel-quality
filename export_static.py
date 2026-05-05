"""
Generate a self-contained static HTML snapshot of the Reworld Haverhill dashboard.

Produces a single .html file that works offline (no JupyterHub, no S3, no server).
Camera images are embedded as base64 data URIs; Plotly charts are serialised as JSON
and rendered client-side via the Plotly CDN.

Usage (from JupyterHub terminal or a notebook cell):
    python /home/shared/kumar/library/fastapi-jupyter-dashboard/export_static.py

Output:
    /home/shared/kumar/library/fastapi-jupyter-dashboard/
        reworld_haverhill_<YYYYMMDD_HHMM>.html
"""

import base64
import json
import sys
from datetime import datetime
from pathlib import Path

# Make dashboard_api importable without starting uvicorn
DASH_DIR = Path('/home/shared/kumar/library/fastapi-jupyter-dashboard')
sys.path.insert(0, str(DASH_DIR))

import dashboard_api as api  # noqa: E402  (import after sys.path patch)

# ── 1. Load data and build all charts ─────────────────────────────────────────
print('Loading data…')
d = api.load_data()

print('Building charts…')
charts = dict(
    pit_temp        = api.chart_pit_temp(d['pit']),
    pit_maxmap      = api.chart_pit_maxmap(d['pit_max_map']),
    pit_h50         = api.chart_pit_h50(d['pit_h50_map']),
    tipping_temp    = api.chart_tipping_temp(d['t1'], d['t2'], d['t3']),
    tipping_uniform = api.chart_tipping_uniformity(d['t1'], d['t2'], d['t3']),
    chute_fill      = api.chart_chute_fill(d['chuteb']),
    chute_moisture  = api.chart_chute_moisture(d['chuteb']),
    chute_combined  = api.chart_chute_combined(d['chuteb']),
    achute_fill     = api.chart_chute_fill(d['achute'], label='Chute A'),
    achute_moisture = api.chart_chute_moisture(d['achute'], label='Chute A'),
    achute_combined = api.chart_chute_combined(d['achute'], label='Chute A'),
    plastic_frac    = api._synth_plastic(d['chuteb']),
    organic_frac    = api._synth_organic(d['chuteb']),
    btu_pred        = api._synth_btu(d['chuteb']),
)

print('Building summary…')
summary = api.build_summary(d)

# Enrich summary with west-pit image timestamps (same logic as /api/data)
wp_idx = api._s3_upload_times('west-pit')
if wp_idx:
    latest_stem = max(wp_idx, key=lambda s: wp_idx[s])
    summary['pit_image_ts']       = datetime.strptime(latest_stem, '%Y%m%d%H%M%S').strftime('%Y-%m-%d %H:%M')
    summary['pit_image_uploaded'] = datetime.fromtimestamp(wp_idx[latest_stem]).strftime('%Y-%m-%d %H:%M')
else:
    summary['pit_image_ts'] = summary['pit_image_uploaded'] = None


# ── 2. Resolve camera images → base64 data URIs ───────────────────────────────
SVG_NO_FEED = (
    'data:image/svg+xml,'
    '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720">'
    '<rect width="1280" height="720" fill="%231a1f2e"/>'
    '<text x="640" y="360" text-anchor="middle" dominant-baseline="middle" '
    'fill="%23334155" font-size="20" font-family="sans-serif">No feed available</text>'
    '</svg>'
)

def _cam_src(cam: str) -> tuple[str, str]:
    """Return (data_uri, timestamp_label) for a camera's latest local JPEG."""
    idx = api._s3_upload_times(cam)
    if not idx:
        return SVG_NO_FEED, ''
    latest_stem = max(idx, key=lambda s: idx[s])
    uploaded_at = datetime.fromtimestamp(idx[latest_stem]).strftime('%Y-%m-%d %H:%M UTC')

    # Try local camera_data dir first (matches api._cam_image_response fast path)
    if cam in api.CAM_RGB_DIRS:
        matches = list(api.CAM_RGB_DIRS[cam].rglob(f'{latest_stem}.jpg'))
        if matches:
            data = matches[0].read_bytes()
            b64  = base64.b64encode(data).decode()
            return f'data:image/jpeg;base64,{b64}', uploaded_at

    # Fall back to per-camera image cache dir (west-pit, or after a failed local lookup)
    cache_dir = api.IMAGES_DIR / cam
    cached    = cache_dir / f'{latest_stem}.jpg'
    if cached.exists():
        data = cached.read_bytes()
        b64  = base64.b64encode(data).decode()
        return f'data:image/jpeg;base64,{b64}', uploaded_at

    return SVG_NO_FEED, ''


print('Embedding camera images…')
cam_srcs: dict[str, tuple[str, str]] = {}
for cam in ['achute', 'chuteb', 'west-pit', 'tipping1', 'tipping2', 'tipping3']:
    src, ts = _cam_src(cam)
    cam_srcs[cam] = (src, ts)
    label = '(embedded)' if src.startswith('data:image/jpeg') else '(placeholder)'
    print(f'  {cam}: {label}')


# ── 3. Helper to emit a chart block ──────────────────────────────────────────
def _chart_block(div_id: str, chart_dict: dict) -> str:
    data_json   = json.dumps(chart_dict.get('data', []))
    layout_json = json.dumps(chart_dict.get('layout', {}))
    return (
        f'<div id="{div_id}"></div>'
        f'<script>Plotly.newPlot("{div_id}",{data_json},{layout_json},'
        f'{{responsive:true,displayModeBar:false}});</script>'
    )


# ── 4. Build inline JS that populates all the summary stat DOM nodes ──────────
def _js_summary(s: dict) -> str:
    action_color_map = {
        'red':    '#ef4444', 'orange': '#f97316',
        'yellow': '#eab308', 'green':  '#22c55e', 'blue': '#3b82f6',
    }
    actions_html = ''.join(
        '<div class="op-action">'
        '<div class="op-action-dot" style="background:' + action_color_map.get(c, '#64748b') + '"></div>'
        '<div class="op-action-text">' + txt + '</div>'
        '</div>'
        for c, txt in s.get('actions', [])
    )

    pit_icon     = s.get('pit_icon', '')
    pit_level    = s.get('pit_level', '')
    pit_color    = s.get('pit_color', '#22c55e')
    pit_max      = str(s.get('pit_max', '—'))
    pit_pct      = str(s.get('pit_hotspot_pct', '—'))
    pit_alert    = str(s.get('pit_alert_frames', '—'))
    pit_total    = str(s.get('pit_total_frames', '—'))
    tip_icon     = s.get('tip_icon', '')
    tip_level    = s.get('tip_level', '')
    tip_color    = s.get('tip_color', '#22c55e')
    tip_max      = str(s.get('tip_max', '—'))
    tip1_max     = str(s.get('tip1_max', '—'))
    tip2_max     = str(s.get('tip2_max', '—'))
    tip3_max     = str(s.get('tip3_max', '—'))
    chute_status = s.get('chute_status', '')
    chute_color  = s.get('chute_color', '#22c55e')
    chute_fill   = str(s.get('chute_fill', '—'))
    chute_moist  = str(s.get('chute_moisture', '—'))
    moist_label  = s.get('moist_label', '—')
    achute_fill  = str(s.get('achute_fill', '—'))
    achute_moist = str(s.get('achute_moisture', '—'))
    achute_mlbl  = s.get('achute_moist_label', '—')
    data_through = s.get('data_through', '')
    generated    = s.get('generated', '')
    pit_img_ts   = s.get('pit_image_ts') or 'unknown'
    pit_img_up   = s.get('pit_image_uploaded') or 'unknown'

    def _set(eid, val):
        return f"document.getElementById('{eid}').textContent={json.dumps(str(val))};"
    def _style(eid, css):
        return f"document.getElementById('{eid}').style.cssText={json.dumps(css)};"
    def _html(eid, val):
        return f"document.getElementById('{eid}').innerHTML={json.dumps(str(val))};"

    lines = [
        _set('hdr-data-through', 'Data through: ' + data_through),
        _set('hdr-generated', 'Generated: ' + generated),
        "document.getElementById('status-dot').style.background='#22c55e';",
        _set('op-sbar-icon', pit_icon),
        _set('op-sbar-text', 'West Pit: ' + pit_level + ' — ' + pit_max + '°C'),
        "document.getElementById('op-sbar').style.background='rgba(26,31,46,0.6)';",
        _set('op-pit-max', pit_max),
        f"document.getElementById('op-pit-max').style.color='{pit_color}';",
        _set('op-pit-chip', pit_icon + ' ' + pit_level),
        _style('op-pit-chip', f'background:rgba(0,0,0,.25);color:{pit_color};border:1px solid {pit_color}44;margin-top:10px;'),
        _set('op-pit-pct', pit_pct),
        f"document.getElementById('op-pit-pct').style.color='{pit_color}';",
        _set('op-pit-frames-sub', pit_alert + ' of ' + pit_total + ' frames'),
        _set('op-a-fill', achute_fill),
        _set('op-a-moist', achute_moist),
        _set('op-a-moist-label', achute_mlbl),
        _set('op-fill', chute_fill),
        _set('op-moist', chute_moist),
        _set('op-moist-label', moist_label),
        _set('op-chute-chip', chute_status),
        _style('op-chute-chip', f'background:rgba(0,0,0,.25);color:{chute_color};border:1px solid {chute_color}44;'),
        _set('op-tip-chip', tip_icon + ' ' + tip_level),
        _style('op-tip-chip', f'background:rgba(0,0,0,.25);color:{tip_color};border:1px solid {tip_color}44;'),
        _set('op-tip1-max', tip1_max),
        _set('op-tip2-max', tip2_max),
        _set('op-tip3-max', tip3_max),
        f"document.getElementById('op-action-list').innerHTML={json.dumps(actions_html)};",
        _html('pit-val', pit_max + '<span class="unit">°C</span>'),
        f"document.getElementById('pit-val').style.color='{pit_color}';",
        _set('pit-badge', pit_icon + ' ' + pit_level),
        _style('pit-badge', f'background:rgba(0,0,0,.25);color:{pit_color};border:1px solid {pit_color}44;'),
        _set('pit-alert-frames', pit_alert + ' frames'),
        _set('pit-total-frames', 'out of ' + pit_total + ' frames'),
        _html('tip-val', tip_max + '<span class="unit">°C</span>'),
        f"document.getElementById('tip-val').style.color='{tip_color}';",
        _set('tip-badge', tip_icon + ' ' + tip_level),
        _style('tip-badge', f'background:rgba(0,0,0,.25);color:{tip_color};border:1px solid {tip_color}44;'),
        _html('ca-fill', achute_fill + '<span class="unit">%</span>'),
        _set('ca-moist', achute_moist),
        _html('cb-fill', chute_fill + '<span class="unit">%</span>'),
        _set('cb-moist', chute_moist),
        f"document.getElementById('pit-dot').style.background='{pit_color}';",
        f"document.getElementById('op-nav-dot').style.background='{pit_color}';",
        _set('wp-img-ts', 'Camera: ' + pit_img_ts),
        _set('wp-img-uploaded', 'Uploaded: ' + pit_img_up),
        "document.getElementById('static-note').style.display='flex';",
    ]
    return '\n'.join(lines)


# ── 5. Assemble the HTML ──────────────────────────────────────────────────────
print('Assembling HTML…')

# Camera image helpers
def _cam_img(cam_key: str, alt: str) -> str:
    src, ts = cam_srcs[cam_key]
    ts_html = f'<div class="cam-fname">{ts}</div>' if ts else ''
    return f'<img class="cam-img" alt="{alt}" src="{src}"/>{ts_html}'


generated = datetime.now().strftime('%Y%m%d_%H%M')

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Reworld Haverhill — Waste Intelligence (Snapshot {generated})</title>
<script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet"/>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg:#0f1117;--card:#1a1f2e;--card2:#1d2235;--border:rgba(255,255,255,0.07);
  --text:#e2e8f0;--muted:#64748b;--subtle:#1e2840;
  --red:#ef4444;--orange:#f97316;--yellow:#eab308;
  --green:#22c55e;--blue:#3b82f6;--purple:#a855f7;
  --sky:#38bdf8;
}}
body{{background:var(--bg);color:var(--text);font-family:'Inter',sans-serif;font-size:14px;line-height:1.5}}
.hdr{{background:var(--card);border-bottom:1px solid var(--border);
     padding:12px 24px;display:flex;align-items:center;gap:14px;position:sticky;top:0;z-index:50}}
.hdr-dot{{width:9px;height:9px;border-radius:50%;background:var(--green);
          box-shadow:0 0 7px var(--green);flex-shrink:0}}
.hdr h1{{font-size:16px;font-weight:700;letter-spacing:-.3px}}
.hdr .sub{{font-size:11px;color:var(--muted)}}
.spacer{{flex:1}}
.hdr-meta{{font-size:11px;color:var(--muted);text-align:right}}
#banner{{padding:9px 24px;font-size:13px;font-weight:500;display:none}}
.nav{{display:flex;border-bottom:1px solid var(--border);
     padding:0 24px;background:var(--card);gap:2px;flex-wrap:wrap}}
.nb{{padding:11px 16px;font-size:13px;font-weight:500;color:var(--muted);
    cursor:pointer;border:none;border-bottom:2px solid transparent;
    background:none;transition:color .15s,border-color .15s;white-space:nowrap}}
.nb:hover{{color:var(--text)}}
.nb.active{{color:var(--text);border-bottom-color:var(--blue)}}
.nb .dot{{display:inline-block;width:7px;height:7px;border-radius:50%;
          margin-right:5px;vertical-align:middle}}
.static-note{{display:none;align-items:center;gap:10px;padding:9px 24px;
              background:rgba(99,102,241,.10);border-bottom:1px solid rgba(99,102,241,.2);
              font-size:12px;color:#a5b4fc}}
.page{{padding:18px 24px;max-width:1380px}}
.tab{{display:none}}.tab.active{{display:block}}
.sec{{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.08em;
     color:var(--muted);margin:18px 0 14px;display:flex;align-items:center;gap:8px}}
.sec::after{{content:'';flex:1;height:1px;background:var(--border)}}
.ptag{{font-size:10px;font-weight:600;padding:2px 8px;border-radius:99px;
      text-transform:uppercase;letter-spacing:.05em}}
.cards{{display:grid;gap:12px;margin-bottom:16px}}
.c3{{grid-template-columns:repeat(3,1fr)}}
.c4{{grid-template-columns:repeat(4,1fr)}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:14px 16px}}
.card .lbl{{font-size:11px;color:var(--muted);text-transform:uppercase;
            letter-spacing:.06em;margin-bottom:5px}}
.card .val{{font-size:24px;font-weight:700;line-height:1.1}}
.card .unit{{font-size:12px;color:var(--muted);margin-left:2px}}
.card .csub{{font-size:11px;color:var(--muted);margin-top:3px}}
.badge{{display:inline-flex;align-items:center;gap:5px;font-size:11px;font-weight:600;
       padding:3px 10px;border-radius:99px;margin-top:5px}}
.cc{{background:var(--card);border:1px solid var(--border);border-radius:10px;
    padding:14px;margin-bottom:14px}}
.row2{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px}}
.row3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-bottom:14px}}
.ph{{background:var(--card2);border:1px dashed rgba(168,85,247,.3);
    border-radius:10px;padding:14px;margin-bottom:14px;position:relative}}
.ph .ph-badge{{position:absolute;top:11px;right:13px;font-size:10px;font-weight:700;
               padding:2px 8px;border-radius:99px;background:rgba(168,85,247,.15);
               color:var(--purple);border:1px solid rgba(168,85,247,.25)}}
.ph h4{{font-size:13px;font-weight:600;color:#c084fc;margin-bottom:5px}}
.ph .ph-desc{{font-size:12px;color:var(--muted);margin-bottom:10px}}
.ph .demo-chip{{display:inline-block;font-size:10px;font-weight:700;
                background:rgba(168,85,247,.13);color:#c084fc;
                padding:2px 8px;border-radius:4px;margin-bottom:8px;letter-spacing:.05em}}
.det-box{{background:rgba(0,0,0,.3);border-radius:8px;padding:12px;
          border:1px solid rgba(255,255,255,.06)}}
.det-row{{display:flex;gap:10px;align-items:flex-start;margin-bottom:6px}}
.det-row:last-child{{margin-bottom:0}}
.det-ico{{font-size:20px;flex-shrink:0;line-height:1.3}}
.det-title{{font-size:13px;font-weight:700}}
.det-meta{{font-size:11px;color:var(--muted);margin-top:1px}}
.det-action{{font-size:11px;font-weight:600;color:#fbbf24;margin-top:3px}}
.conf{{display:inline-block;font-size:11px;font-weight:700;margin-top:5px;
      padding:2px 8px;border-radius:4px}}
.zg{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:5px;max-width:420px}}
.zc{{padding:9px;border-radius:7px;text-align:center}}
.zc .zl{{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.05em}}
.zc .zb{{font-size:11px;font-weight:600;margin-top:1px}}
.zhigh{{background:rgba(239,68,68,.13);border:1px solid rgba(239,68,68,.22)}}
.zmid {{background:rgba(251,191,36,.10);border:1px solid rgba(251,191,36,.18)}}
.zlow {{background:rgba(56,189,248,.10);border:1px solid rgba(56,189,248,.18)}}
.crane-rec{{background:rgba(34,197,94,.08);border:1px solid rgba(34,197,94,.18);
            border-radius:7px;padding:11px;margin-top:8px;font-size:12px}}
.cam-section{{margin:16px 0 8px;font-size:11px;font-weight:700;text-transform:uppercase;
             letter-spacing:.07em;color:var(--muted)}}
.cam-grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px}}
.cam-grid-3{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:14px}}
.cam-grid-1{{margin-bottom:14px}}
.cam-panel{{background:var(--card);border:1px solid var(--border);border-radius:10px;overflow:hidden}}
.cam-label{{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;
           color:var(--muted);padding:9px 13px 5px}}
.cam-img{{width:100%;display:block;aspect-ratio:16/9;object-fit:contain;background:#0a0d14}}
.cam-fname{{font-size:10px;color:#334155;padding:4px 13px 8px;white-space:nowrap;
           overflow:hidden;text-overflow:ellipsis}}
.op-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:16px}}
.op-cluster{{background:var(--card);border:1px solid var(--border);border-radius:12px;overflow:hidden}}
.op-cluster-hdr{{padding:12px 18px 10px;border-bottom:1px solid var(--border)}}
.op-cluster-title{{font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:var(--muted)}}
.op-cluster-sub{{font-size:10px;color:#334155;margin-top:2px}}
.op-instruments{{display:grid;grid-template-columns:1fr 1fr;gap:0}}
.op-instr{{padding:20px 18px;border-right:1px solid var(--border);border-bottom:1px solid var(--border)}}
.op-instr:nth-child(even){{border-right:none}}
.op-instr:last-child:nth-child(odd){{grid-column:span 2;border-right:none}}
.op-instr-lbl{{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:8px}}
.op-num{{font-size:58px;font-weight:700;line-height:1;letter-spacing:-2px}}
.op-unit{{font-size:18px;font-weight:500;color:var(--muted);margin-left:4px;vertical-align:bottom;line-height:1}}
.op-sub{{font-size:12px;color:var(--muted);margin-top:6px}}
.op-chip{{display:inline-block;font-size:15px;font-weight:700;padding:6px 16px;border-radius:99px;margin-top:8px;letter-spacing:.04em}}
.op-chip-lg{{display:inline-block;font-size:22px;font-weight:700;padding:10px 22px;border-radius:99px;letter-spacing:.03em}}
.op-sbar{{padding:14px 24px;display:flex;align-items:center;gap:16px;margin-bottom:16px;border-radius:12px}}
.op-sbar-icon{{font-size:32px;flex-shrink:0}}
.op-sbar-text{{font-size:20px;font-weight:700}}
.op-sbar-sub{{font-size:13px;margin-top:2px;opacity:.8}}
.op-actions-card{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:18px 20px}}
.op-actions-title{{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:14px}}
.op-action{{display:flex;align-items:flex-start;gap:12px;padding:10px 0;border-bottom:1px solid var(--border)}}
.op-action:last-child{{border-bottom:none;padding-bottom:0}}
.op-action-dot{{width:10px;height:10px;border-radius:50%;flex-shrink:0;margin-top:4px}}
.op-action-text{{font-size:15px;font-weight:500;line-height:1.4}}
@media(max-width:960px){{
  .c3,.c4,.row2,.row3,.zg,.op-grid,.cam-grid-2,.cam-grid-3{{grid-template-columns:1fr}}
}}
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

<div id="static-note" class="static-note">
  <span>📄</span>
  <span><strong>Static snapshot</strong> — generated {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC.
  Data and images are embedded; charts are fully interactive.
  To get a fresh snapshot, re-run <code>export_static.py</code> in JupyterHub.</span>
</div>

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

<div class="page">

<!-- ═══════════════ OPERATOR VIEW ═══════════════ -->
<div class="tab active" id="operator">

  <div class="op-sbar" id="op-sbar">
    <div class="op-sbar-icon" id="op-sbar-icon">—</div>
    <div>
      <div class="op-sbar-text" id="op-sbar-text">Loading…</div>
      <div class="op-sbar-sub" id="op-sbar-sub"></div>
    </div>
  </div>

  <div class="op-grid">
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

    <div class="op-cluster">
      <div class="op-cluster-hdr">
        <div class="op-cluster-title">Tipping Floor</div>
        <div class="op-cluster-sub">60–120 min lead time to furnace</div>
      </div>
      <div class="op-instruments">
        <div class="op-instr" style="grid-column:span 2">
          <div class="op-instr-lbl">Thermal Status</div>
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
  </div>

  <div class="op-actions-card">
    <div class="op-actions-title">Operator Actions</div>
    <div id="op-action-list"><div class="op-action"><div class="op-action-dot"></div><div class="op-action-text" style="color:var(--muted)">Loading…</div></div></div>
  </div>

  <div class="cam-section">Chute Cameras</div>
  <div class="cam-grid-2">
    <div class="cam-panel"><div class="cam-label">Chute A</div>{_cam_img('achute','Chute A')}</div>
    <div class="cam-panel"><div class="cam-label">Chute B</div>{_cam_img('chuteb','Chute B')}</div>
  </div>

  <div class="cam-section">West Pit</div>
  <div class="cam-grid-1">
    <div class="cam-panel"><div class="cam-label">West Pit</div>{_cam_img('west-pit','West Pit')}</div>
  </div>

  <div class="cam-section">Tipping Floor Cameras</div>
  <div class="cam-grid-3">
    <div class="cam-panel"><div class="cam-label">Tipping Floor 1</div>{_cam_img('tipping1','Tipping 1')}</div>
    <div class="cam-panel"><div class="cam-label">Tipping Floor 2</div>{_cam_img('tipping2','Tipping 2')}</div>
    <div class="cam-panel"><div class="cam-label">Tipping Floor 3</div>{_cam_img('tipping3','Tipping 3')}</div>
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

  <div class="row2" style="margin-bottom:14px">
    <div class="cc" style="padding:12px">
      <div class="lbl" style="margin-bottom:8px">West Pit — Latest RGB Frame</div>
      {_cam_img('west-pit','West Pit')}
      <div id="wp-img-meta" style="font-size:10px;color:var(--muted);margin-top:6px;display:flex;justify-content:space-between">
        <span id="wp-img-ts"></span><span id="wp-img-uploaded"></span>
      </div>
    </div>
    <div class="cc">{_chart_block('c-pit_temp', charts['pit_temp'])}</div>
  </div>
  <div class="row2">
    <div class="cc">{_chart_block('c-pit_maxmap', charts['pit_maxmap'])}</div>
    <div class="cc">{_chart_block('c-pit_h50', charts['pit_h50'])}</div>
  </div>
  <div class="cc">{_chart_block('c-tipping_temp', charts['tipping_temp'])}</div>

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
    <div class="cc">{_chart_block('c-achute_combined', charts['achute_combined'])}</div>
    <div class="cc">{_chart_block('c-chute_combined', charts['chute_combined'])}</div>
  </div>
  <div class="row2">
    <div class="cc">{_chart_block('c-achute_fill', charts['achute_fill'])}</div>
    <div class="cc">{_chart_block('c-chute_fill', charts['chute_fill'])}</div>
  </div>
  <div class="row2">
    <div class="cc">{_chart_block('c-achute_moisture', charts['achute_moisture'])}</div>
    <div class="cc">{_chart_block('c-chute_moisture', charts['chute_moisture'])}</div>
  </div>

  <div class="ph">
    <div class="ph-badge">Phase 3 — Q4 2026</div>
    <div class="demo-chip">DEMO PREVIEW</div>
    <h4>Waste Composition Estimation</h4>
    <p class="ph-desc">Estimates plastic, organic, and inert fractions from chute camera imagery. Inputs predicted HHV to combustion control. Charts below are synthetic previews.</p>
    <div class="row3">
      <div class="cc">{_chart_block('c-plastic_frac', charts['plastic_frac'])}</div>
      <div class="cc">{_chart_block('c-organic_frac', charts['organic_frac'])}</div>
      <div class="cc">{_chart_block('c-btu_pred', charts['btu_pred'])}</div>
    </div>
  </div>
</div><!-- /furnace -->

<!-- ═══════════════ TIPPING FLOOR ═══════════════ -->
<div class="tab" id="tipping">
  <div class="sec">
    Tipping Floor Analysis (60–120 min furnace lead time) &nbsp;
    <span class="ptag" style="background:rgba(59,130,246,.13);color:#93c5fd;border:1px solid rgba(59,130,246,.22)">Floor Supervisor</span>
    <span class="ptag" style="background:rgba(99,102,241,.13);color:#a5b4fc;border:1px solid rgba(99,102,241,.22);margin-left:3px">Dispatch</span>
  </div>

  <div class="cc">{_chart_block('c-tipping_uniform', charts['tipping_uniform'])}</div>

  <div class="ph">
    <div class="ph-badge">Phase 1 — Building</div>
    <h4>Homogenous Load Detector</h4>
    <p class="ph-desc">Identifies single-material loads (tyres, pallets, carpet) from tipping floor cameras to flag elevated BTU variance risk. Low uniformity std + high fill → alert.</p>
    <div class="det-box">
      <div class="det-row">
        <div class="det-ico">🔵</div>
        <div>
          <div class="det-title" style="color:var(--blue)">Homogenous Load Detected — Tipping 3</div>
          <div class="det-meta">Inferred type: Carpet/textiles · Zone: Centre · 2026-04-30 14:22</div>
          <div class="det-action">⚡ Disperse load before crane transfer — low BTU material</div>
          <span class="conf" style="background:rgba(59,130,246,.13);color:var(--blue)">Confidence: 81%</span>
        </div>
      </div>
    </div>
  </div>

  <div class="ph">
    <div class="ph-badge">Phase 2 — Q3 2026</div>
    <h4>Pit Zone Composition Map</h4>
    <p class="ph-desc">9-zone grid showing estimated BTU density across the west pit — inferred from tipping-floor load composition and crane drop history. Demo layout below.</p>
    <div class="zg">
      <div class="zc zhigh"><div class="zl">NW</div><div class="zb">High BTU</div></div>
      <div class="zc zmid" ><div class="zl">N</div> <div class="zb">Mixed</div></div>
      <div class="zc zlow" ><div class="zl">NE</div><div class="zb">Low BTU</div></div>
      <div class="zc zmid" ><div class="zl">W</div> <div class="zb">Mixed</div></div>
      <div class="zc zhigh"><div class="zl">C</div>  <div class="zb">High BTU</div></div>
      <div class="zc zmid" ><div class="zl">E</div>  <div class="zb">Mixed</div></div>
      <div class="zc zlow" ><div class="zl">SW</div><div class="zb">Low BTU</div></div>
      <div class="zc zmid" ><div class="zl">S</div>  <div class="zb">Mixed</div></div>
      <div class="zc zhigh"><div class="zl">SE</div><div class="zb">High BTU</div></div>
    </div>
    <div class="crane-rec">
      <strong style="color:#86efac">🏗 Crane Recommendation:</strong>
      Move material from NW → NE to balance feed BTU.
      Estimated furnace impact in 50–70 min.
    </div>
  </div>
</div><!-- /tipping -->

</div><!-- /page -->

<script>
// Tab switching
function showTab(id){{
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.nb').forEach(b=>b.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  var btn=document.getElementById('tab-'+id);
  if(btn) btn.classList.add('active');
}}

// Populate summary stats from embedded data
{_js_summary(summary)}
</script>

</body>
</html>"""

# ── 6. Write output file ───────────────────────────────────────────────────────
out_path = DASH_DIR / f'reworld_haverhill_{generated}.html'
out_path.write_text(html, encoding='utf-8')
size_mb = out_path.stat().st_size / 1_048_576
print(f'\nDone! → {out_path}  ({size_mb:.1f} MB)')
print('Upload that file to Google Drive and share it with the customer.')
print('They download it and open in any browser — no server or login needed.')
