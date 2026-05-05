"""
Prepare files for manual Google Drive upload.

Generates into .mcp_upload_tmp/:
  - Camera — <name>.html   (one per camera, JPEG embedded as data URI)
  - moisture_trends.html   (interactive Plotly charts for Chute A & B)

Run from JupyterHub terminal:
  python /home/shared/kumar/library/fastapi-jupyter-dashboard/prepare_drive_upload.py

Then upload the files in .mcp_upload_tmp/ to Google Drive manually.
"""

import base64
import io
import json
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image

DASH_DIR = Path('/home/shared/kumar/library/fastapi-jupyter-dashboard')
OUT_DIR  = DASH_DIR / 'google_drive_upload_tmp'
OUT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(DASH_DIR))
import dashboard_api as api

CAMERAS = {
    'achute':   'Camera — Chute A',
    'chuteb':   'Camera — Chute B',
    'west-pit': 'Camera — West Pit',
    'tipping1': 'Camera — Tipping 1',
    'tipping2': 'Camera — Tipping 2',
    'tipping3': 'Camera — Tipping 3',
}

THUMB_W, THUMB_H, THUMB_Q = 320, 180, 50

CAMERA_HTML = """\
<!DOCTYPE html><html><head><meta charset="UTF-8">
<style>body{{margin:0;background:#0f1117;display:flex;flex-direction:column;\
align-items:center;justify-content:center;min-height:100vh;\
font-family:sans-serif;color:#94a3b8}}\
h2{{font-size:14px;margin-bottom:8px;text-align:center}}\
p{{font-size:11px;color:#475569;margin-top:6px}}</style></head>
<body><h2>{label}</h2>\
<img src="data:image/jpeg;base64,{b64}" style="max-width:640px;border-radius:8px">\
<p>{ts}</p></body></html>"""


def _latest_jpeg(cam: str) -> tuple[bytes | None, str]:
    """Return (jpeg_bytes, timestamp_label) for a camera's latest frame."""
    idx = api._s3_upload_times(cam)
    if not idx:
        return None, ''
    stem = max(idx, key=lambda s: idx[s])
    ts   = datetime.strptime(stem, '%Y%m%d%H%M%S').strftime('%Y-%m-%d %H:%M UTC')

    if cam in api.CAM_RGB_DIRS:
        matches = list(api.CAM_RGB_DIRS[cam].rglob(f'{stem}.jpg'))
        if matches:
            return matches[0].read_bytes(), ts

    cached = api.IMAGES_DIR / cam / f'{stem}.jpg'
    if cached.exists():
        return cached.read_bytes(), ts

    return None, ''


def _thumbnail(data: bytes) -> str:
    """Resize JPEG to THUMB_W×THUMB_H and return base64 string."""
    img = Image.open(io.BytesIO(data)).convert('RGB')
    img.thumbnail((THUMB_W, THUMB_H), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=THUMB_Q, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def build_camera_html():
    print('Building camera HTML files…')
    for cam, label in CAMERAS.items():
        data, ts = _latest_jpeg(cam)
        if data is None:
            print(f'  {cam}: no image found, skipping')
            continue
        b64  = _thumbnail(data)
        html = CAMERA_HTML.format(label=label, b64=b64, ts=ts)
        path = OUT_DIR / f'{cam}.html'
        path.write_text(html, encoding='utf-8')
        print(f'  {cam}: {path.stat().st_size // 1024} KB → {path.name}')


def _chart_block(div_id: str, chart: dict) -> str:
    data_json   = json.dumps(chart.get('data', []))
    layout_json = json.dumps(chart.get('layout', {}))
    return (
        f'<div id="{div_id}" style="margin-bottom:20px"></div>'
        f'<script>Plotly.newPlot("{div_id}",{data_json},{layout_json},'
        f'{{responsive:true,displayModeBar:false}});</script>'
    )


def build_moisture_trends():
    print('Building moisture trends HTML…')
    d = api.load_data()

    charts = {
        'achute_fill':     api.chart_chute_fill(d['achute'], label='Chute A'),
        'achute_moisture': api.chart_chute_moisture(d['achute'], label='Chute A'),
        'chuteb_fill':     api.chart_chute_fill(d['chuteb']),
        'chuteb_moisture': api.chart_chute_moisture(d['chuteb']),
    }

    generated = datetime.now().strftime('%Y-%m-%d %H:%M UTC')
    blocks    = '\n'.join(_chart_block(k, v) for k, v in charts.items())

    html = f"""\
<!DOCTYPE html><html><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reworld Haverhill — Moisture &amp; Fill Trends</title>
<script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0f1117;color:#e2e8f0;font-family:sans-serif;padding:20px}}
h1{{font-size:16px;font-weight:700;margin-bottom:4px}}
p{{font-size:11px;color:#64748b;margin-bottom:20px}}
</style></head>
<body>
<h1>Reworld Haverhill — Moisture &amp; Fill Trends</h1>
<p>Generated {generated}</p>
{blocks}
</body></html>"""

    path = OUT_DIR / 'moisture_trends.html'
    path.write_text(html, encoding='utf-8')
    print(f'  moisture_trends: {path.stat().st_size // 1024} KB → {path.name}')


if __name__ == '__main__':
    build_camera_html()
    print()
    build_moisture_trends()
    print(f'\nDone. Files are in: {OUT_DIR}')
    print('Upload them to Google Drive → "Reworld Haverhill — Live Dashboard Data"')
    for f in sorted(OUT_DIR.glob('*.html')):
        print(f'  {f.name}')
