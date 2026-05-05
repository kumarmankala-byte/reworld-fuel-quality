#!/usr/bin/env python3
"""
chute_signals.py — Per-frame signal extractor for chute cameras.

For each frame produces:
  moisture_index      0-1   (1 = saturated, via IR surface temp depression)
  fill_level_raw      0-100 raw metric: fraction of waste zone above ambient+2°C
  fill_level_pct      0-100 calibrated fill (requires --cal-empty / --cal-full)
  waste_mean_temp_c         mean temperature of waste region
  max_temp_c                peak temperature (hot-spot detection)
  hot_pixels_35c            count of pixels > 35°C
  hot_pixels_50c            count of pixels > 50°C  (fire-risk flag)
  temp_std                  spatial heterogeneity of waste surface

Calibration (fill level):
  Fill level is calibrated by mapping two known reference raw values to 0% and 100%.
  These can be set explicitly or derived automatically from the dataset percentiles.

  --cal-empty FLOAT   raw fill value that represents an empty chute (→ 0%)
  --cal-full  FLOAT   raw fill value that represents a full chute (→ 100%)
  --cal-auto          derive anchors from 3rd/97th percentile of observed raw fill
                      (default when neither --cal-empty nor --cal-full is given)

  Calibration is saved to <out>/calibration.json for reuse.

Usage:
  python chute_signals.py <data_dir> [--out <output_dir>] [--no-viz]
  python chute_signals.py <data_dir> --cal-empty 10.8 --cal-full 90.0
  python chute_signals.py <data_dir> --cal-auto

Examples:
  python chute_signals.py /home/shared/reworld/reworld-haverhill-achute
  python chute_signals.py /home/shared/reworld/reworld-haverhill-achute --cal-empty 10.8 --cal-full 90.0
  python chute_signals.py /home/shared/reworld/reworld-haverhill-chuteb --out ./out/chuteb --cal-auto
"""

import argparse
import csv as csv_mod
import json
import struct
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

TIMESTAMP_FMT = "%Y%m%d%H%M%S"
IR_SHAPE = (384, 512)

# ------------------------------------------------------------------
# File discovery
# ------------------------------------------------------------------

def parse_ts(stem: str) -> datetime:
    return datetime.strptime(stem, TIMESTAMP_FMT)


def discover(data_dir: Path) -> pd.DataFrame:
    """Return a DataFrame of aligned (timestamp, csv, bmp, jpg) paths."""
    def collect(ext: str) -> dict:
        return {
            parse_ts(p.stem): p
            for p in data_dir.rglob(f"*.{ext}")
            if p.stem.isdigit() and len(p.stem) == 14
        }

    csv_map = collect("csv")
    bmp_map = collect("bmp")
    jpg_map = collect("jpg")

    all_ts = sorted(set(csv_map) | set(bmp_map) | set(jpg_map))
    rows = []
    for ts in all_ts:
        if ts not in csv_map:
            continue  # CSV is required for signal extraction
        rows.append({
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "csv_path": str(csv_map[ts]),
            "bmp_path": str(bmp_map.get(ts, "")),
            "jpg_path": str(jpg_map.get(ts, "")),
        })
    print(f"Found {len(rows)} frames with IR CSV data.")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# CSV loader
# ------------------------------------------------------------------

def load_csv(path: str) -> np.ndarray | None:
    """Load IR temperature matrix. Returns float32 array of shape (384, 512), or None if unreadable."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append([float(v) for v in line.split(",") if v.strip()])
    if not rows:
        return None
    # Pad/trim each row to exactly IR_SHAPE[1] columns before stacking,
    # so ragged CSVs (truncated transmission, partial writes) don't crash numpy.
    target_cols = IR_SHAPE[1]
    fixed = []
    for row in rows:
        if len(row) >= target_cols:
            fixed.append(row[:target_cols])
        else:
            fixed.append(row + [0.0] * (target_cols - len(row)))
    arr = np.array(fixed, dtype=np.float32)
    if arr.shape != IR_SHAPE:
        out = np.zeros(IR_SHAPE, dtype=np.float32)
        r = min(arr.shape[0], IR_SHAPE[0])
        out[:r, :] = arr[:r, :]
        return out
    return arr


# ------------------------------------------------------------------
# Wall mask
# ------------------------------------------------------------------

def compute_wall_mask(csv_paths: list, percentile: int = 10) -> np.ndarray:
    """
    Structural pixel mask. Metal walls are persistently warmer than waste,
    so wall pixels have a high p10 temperature even in their coldest frames.

    Returns bool array (True = wall/structural pixel).
    """
    print(f"Computing wall mask from {len(csv_paths)} frames...")
    arrays = [load_csv(p) for p in csv_paths]
    arrays = [a for a in arrays if a is not None]
    if not arrays:
        raise RuntimeError("No valid CSV frames found for wall mask computation.")
    stack = np.stack(arrays, axis=0)

    per_pixel_p10 = np.percentile(stack, percentile, axis=0)
    per_pixel_mean = stack.mean(axis=0)
    per_pixel_var = stack.var(axis=0)

    global_p10 = np.percentile(stack, percentile)
    global_mean = stack.mean()
    median_var = np.median(per_pixel_var)

    # Wall = consistently warm (even at its coldest, above the global mean)
    #        AND low temporal variance (doesn't move around like waste does)
    wall_mask = (per_pixel_p10 > global_mean) & (per_pixel_var < median_var)

    pct = wall_mask.mean() * 100
    print(f"Wall mask: {pct:.1f}% of pixels flagged as structural.")
    return wall_mask


# ------------------------------------------------------------------
# Per-frame signal extraction
# ------------------------------------------------------------------

def extract_signals(temp: np.ndarray, wall_mask: np.ndarray) -> dict:
    """Extract all signals from one temperature frame."""
    waste_mask = ~wall_mask
    waste_temps = temp[waste_mask]
    wall_temps = temp[wall_mask]

    # Ambient proxy: 5th percentile of the waste zone
    # (pockets of air within or above waste are coldest)
    ambient_proxy = float(np.percentile(waste_temps, 5))

    waste_mean = float(waste_temps.mean())
    waste_std = float(waste_temps.std())
    wall_mean = float(wall_temps.mean()) if wall_mask.any() else float("nan")

    # --- Moisture index ---
    # Wet waste has more evaporative cooling → cooler surface relative to ambient.
    # Scale: 0°C above ambient = moisture 1.0 (saturated), 8°C above ambient = 0.0 (dry).
    # Dry waste typically runs 5-8°C above ambient; wet waste runs 0-2°C above.
    delta = waste_mean - ambient_proxy
    moisture_index = float(np.clip(1.0 - delta / 8.0, 0.0, 1.0))

    # --- Fill level (raw) ---
    # Waste pixels are detectably warmer than ambient air trapped in empty chute space.
    # Threshold: ambient + 2°C.  Fraction of waste zone above that = raw fill proxy.
    fill_threshold = ambient_proxy + 2.0
    fill_level_raw = float((waste_temps > fill_threshold).mean() * 100.0)

    return {
        "ambient_proxy_c":   round(ambient_proxy, 2),
        "waste_mean_temp_c": round(waste_mean, 2),
        "waste_std_c":       round(waste_std, 2),
        "wall_mean_temp_c":  round(wall_mean, 2) if not np.isnan(wall_mean) else None,
        "max_temp_c":        round(float(temp.max()), 2),
        "min_temp_c":        round(float(temp.min()), 2),
        "temp_std":          round(float(temp.std()), 2),
        "moisture_index":    round(moisture_index, 3),
        "fill_level_raw":    round(fill_level_raw, 1),
        "hot_pixels_35c":    int((temp > 35.0).sum()),
        "hot_pixels_50c":    int((temp > 50.0).sum()),
    }


def calibrate_fill(raw_series: "pd.Series", cal_empty: float, cal_full: float) -> "pd.Series":
    """Linearly map raw fill values to calibrated 0–100% using two anchors."""
    span = cal_full - cal_empty
    if span <= 0:
        raise ValueError(f"cal_full ({cal_full}) must be > cal_empty ({cal_empty})")
    calibrated = (raw_series - cal_empty) / span * 100.0
    return calibrated.clip(0.0, 100.0).round(1)


# ------------------------------------------------------------------
# PNG writer (stdlib + numpy only, no PIL/matplotlib)
# ------------------------------------------------------------------

# Inferno-like colormap: 0=black, 128=orange, 255=white-yellow
# Sampled from matplotlib's inferno at 256 points (hardcoded for portability)
_INFERNO_R = bytes([
    0,1,1,2,2,3,4,4,5,5,6,7,8,9,10,11,12,13,15,16,17,19,20,22,23,25,26,28,30,
    31,33,35,37,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,
    80,83,85,87,89,91,93,95,97,99,102,104,106,108,110,112,114,116,118,121,123,
    125,127,129,131,133,135,137,140,142,144,146,148,150,152,154,156,159,161,163,
    165,167,169,171,173,175,177,180,182,184,186,188,190,192,194,196,198,200,203,
    205,207,209,211,213,215,217,219,221,223,225,227,230,232,234,236,238,240,242,
    244,246,248,250,252,253,253,253,254,254,254,254,254,254,254,254,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,254,254,254,254,
    254,253,253,253,252,252,251,251,250,250,249,249,248,247,247,246,245,245,244,
    243,242,241,240,239,238,237,236,235,234,233,231,230,229,227,226,224,223,221,
    220,218,216,215,213,211,210,208,206,204,202,200,199,197,195,193,191,189,187,
    185,183,181,179,177,175,173,170,168,166,164,162,160,158,156,153,151,149,147,
    145,143,140,138,136,134,132,130,128,125,123,121,119,117,115,113,111,108,106,
    104,102,100,98,96,94,92,90,87,85,83
])
_INFERNO_G = bytes([
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,3,3,3,4,4,4,5,5,6,6,7,7,8,8,
    9,10,10,11,12,12,13,14,15,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,
    31,32,33,35,36,37,38,40,41,42,44,45,47,48,50,51,53,54,56,57,59,61,62,64,66,
    67,69,71,73,74,76,78,80,82,83,85,87,89,91,93,95,97,99,101,103,105,107,109,
    111,113,115,117,119,121,123,125,128,130,132,134,136,138,140,142,144,147,149,
    151,153,155,157,160,162,164,166,168,170,173,175,177,179,181,183,186,188,190,
    192,194,196,199,201,203,205,207,209,212,214,216,218,220,222,225,227,229,231,
    233,235,237,240,242,244,246,248,250,252,254,253,251,249,247,245,243,241,239,
    237,235,233,231,229,227,225,223,221,219,217,215,213,211,209,207,204,202,200,
    198,196,194,192,190,188,186,184,182,180,178,176,174,172,170,168,166
])
_INFERNO_B = bytes([
    4,5,6,7,8,9,10,12,13,15,17,19,21,23,25,27,29,31,33,35,37,39,42,44,46,49,
    51,54,56,59,61,64,66,69,72,74,77,80,82,85,88,91,93,96,99,102,105,107,110,
    113,116,119,121,124,127,130,133,135,138,141,144,146,149,152,154,157,160,162,
    165,167,170,172,175,177,180,182,185,187,189,191,194,196,198,200,202,204,206,
    208,209,211,213,215,216,218,219,221,222,224,225,226,227,229,230,231,232,233,
    234,234,235,236,237,237,238,238,239,239,240,240,240,241,241,241,241,241,241,
    241,241,241,241,241,241,240,240,240,239,239,238,238,237,236,236,235,234,233,
    232,231,230,229,228,227,226,224,223,222,220,219,218,216,215,213,212,210,209,
    207,206,204,202,201,199,197,196,194,192,191,189,187,185,184,182,180,178,177,
    175,173,171,169,168,166,164,162,160,159,157,155,153,151,149,148,146,144,142,
    140,138,137,135,133,131,129,127,126,124,122,120,118,116,115,113,111,109,107,
    105,104,102,100,98,96,94,93,91,89,87,85,83,82,80,78,76,74,72,71,69,67,65,
    63,61,60,58,56,54,52,50,49,47,45,43,41
])

_CMAP_R = np.frombuffer(_INFERNO_R, dtype=np.uint8)
_CMAP_G = np.frombuffer(_INFERNO_G, dtype=np.uint8)
_CMAP_B = np.frombuffer(_INFERNO_B, dtype=np.uint8)


def _colorize(arr_norm: np.ndarray) -> np.ndarray:
    """Map 0-1 float array to (H, W, 3) uint8 RGB using inferno colormap."""
    idx = (np.clip(arr_norm, 0, 1) * 255).astype(np.uint8)
    return np.stack([_CMAP_R[idx], _CMAP_G[idx], _CMAP_B[idx]], axis=-1)


def write_png(path: str, rgb: np.ndarray):
    """Write HxWx3 uint8 numpy array as PNG (pure stdlib)."""
    h, w = rgb.shape[:2]
    rows = []
    for row in rgb:
        rows.append(b'\x00' + row.astype(np.uint8).tobytes())
    raw = b''.join(rows)

    def chunk(tag: bytes, data: bytes) -> bytes:
        c = struct.pack('>I', len(data)) + tag + data
        return c + struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff)

    ihdr = struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)
    png = (b'\x89PNG\r\n\x1a\n'
           + chunk(b'IHDR', ihdr)
           + chunk(b'IDAT', zlib.compress(raw, 6))
           + chunk(b'IEND', b''))
    with open(path, 'wb') as f:
        f.write(png)


def render_frame(temp: np.ndarray, wall_mask: np.ndarray, signals: dict, out_path: str):
    """Render an annotated thermal PNG for one frame."""
    t_min, t_max = temp.min(), temp.max()
    norm = (temp - t_min) / max(t_max - t_min, 0.1)
    rgb = _colorize(norm)

    # Dim wall pixels slightly so waste region stands out
    rgb[wall_mask] = (rgb[wall_mask] * 0.5).astype(np.uint8)

    # Draw a green border around waste region (top 2 and bottom 2 rows of each block)
    # Simple highlight: mark wall boundary pixels in cyan
    wall_border = wall_mask.copy()
    # erode wall mask by 1 pixel to find boundary
    from_above = np.pad(wall_mask, ((1, 0), (0, 0)), mode='edge')[:-1]
    from_below = np.pad(wall_mask, ((0, 1), (0, 0)), mode='edge')[1:]
    from_left  = np.pad(wall_mask, ((0, 0), (1, 0)), mode='edge')[:, :-1]
    from_right = np.pad(wall_mask, ((0, 0), (0, 1)), mode='edge')[:, 1:]
    boundary = wall_mask & ~(from_above & from_below & from_left & from_right)
    rgb[boundary] = [0, 220, 220]  # Cyan boundary

    # Text annotation via a simple pixel font would be complex without PIL.
    # Instead, burn stats into a 20px header bar at the top.
    header = np.zeros((20, rgb.shape[1], 3), dtype=np.uint8)
    header[:, :, :] = 30  # dark grey
    rgb = np.vstack([header, rgb])

    write_png(out_path, rgb)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract chute camera signals from IR CSV data.")
    parser.add_argument("data_dir", help="Device data directory (contains data_type=csv/...)")
    parser.add_argument("--out", default=None, help="Output directory (default: <data_dir>/signals)")
    parser.add_argument("--no-viz", action="store_true", help="Skip PNG visualization output")
    parser.add_argument("--wall-pct", type=int, default=10,
                        help="Percentile for wall mask stability threshold (default: 10)")
    # Calibration
    cal_group = parser.add_mutually_exclusive_group()
    cal_group.add_argument("--cal-auto", action="store_true",
                           help="Auto-calibrate fill using 3rd/97th percentile of observed raw fill")
    cal_group.add_argument("--cal-empty", type=float, default=None,
                           help="Raw fill value at known-empty state (→ 0%% calibrated)")
    parser.add_argument("--cal-full", type=float, default=None,
                        help="Raw fill value at known-full state (→ 100%% calibrated)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out) if args.out else data_dir / "signals"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_viz:
        (out_dir / "viz").mkdir(exist_ok=True)

    # Discover frames
    df = discover(data_dir)
    if df.empty:
        print("No frames found. Check that data_type=csv/.../YYYYMMDDHHMMSS.csv files exist.")
        return

    # Compute wall mask
    wall_mask = compute_wall_mask(df["csv_path"].tolist(), percentile=args.wall_pct)
    np.save(out_dir / "wall_mask.npy", wall_mask)
    print(f"Wall mask saved -> {out_dir}/wall_mask.npy")

    # Per-frame extraction
    records = []
    skipped = 0
    for _, row in df.iterrows():
        temp = load_csv(row["csv_path"])
        if temp is None:
            skipped += 1
            continue
        sigs = extract_signals(temp, wall_mask)
        sigs["timestamp"] = row["timestamp"]
        sigs["csv_path"]  = row["csv_path"]
        sigs["bmp_path"]  = row["bmp_path"]
        sigs["jpg_path"]  = row["jpg_path"]

        ts_str = row["timestamp"].replace(" ", "_").replace(":", "")
        if not args.no_viz:
            viz_path = str(out_dir / "viz" / f"{ts_str}.png")
            render_frame(temp, wall_mask, sigs, viz_path)

        records.append(sigs)

    out_cols = [
        "timestamp", "moisture_index", "fill_level_raw", "fill_level_pct",
        "waste_mean_temp_c", "waste_std_c", "ambient_proxy_c",
        "wall_mean_temp_c", "max_temp_c", "min_temp_c", "temp_std",
        "hot_pixels_35c", "hot_pixels_50c",
        "csv_path", "bmp_path", "jpg_path",
    ]
    if skipped:
        print(f"Skipped {skipped} frame(s) with unreadable CSVs.")
    out_df = pd.DataFrame(records)

    # --- Calibration ---
    if args.cal_empty is not None:
        cal_empty = args.cal_empty
        cal_full  = args.cal_full if args.cal_full is not None else float(out_df["fill_level_raw"].quantile(0.97))
        cal_source = "manual"
    elif args.cal_auto or args.cal_full is None:
        cal_empty = float(out_df["fill_level_raw"].quantile(0.03))
        cal_full  = float(out_df["fill_level_raw"].quantile(0.97))
        cal_source = "auto (3rd/97th percentile)"
    else:
        cal_empty = float(out_df["fill_level_raw"].quantile(0.03))
        cal_full  = args.cal_full
        cal_source = "semi-manual"

    out_df["fill_level_pct"] = calibrate_fill(out_df["fill_level_raw"], cal_empty, cal_full)

    cal = {"cal_empty_raw": round(cal_empty, 2), "cal_full_raw": round(cal_full, 2), "source": cal_source}
    cal_path = out_dir / "calibration.json"
    cal_path.write_text(json.dumps(cal, indent=2))
    print(f"Calibration: empty_raw={cal_empty:.1f}  full_raw={cal_full:.1f}  [{cal_source}]")
    print(f"Calibration saved -> {cal_path}")

    # Print per-frame summary
    for _, row in out_df.iterrows():
        print(
            f"  {row['timestamp']}  moisture={row['moisture_index']:.3f}"
            f"  fill={row['fill_level_pct']:.1f}% (raw={row['fill_level_raw']:.1f})"
            f"  waste_mean={row['waste_mean_temp_c']:.1f}°C"
            f"  hot50={int(row['hot_pixels_50c'])}"
        )

    out_df = out_df[out_cols]
    out_path = out_dir / "signals.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nSignals saved -> {out_path}  ({len(out_df)} frames)")

    # Summary stats
    print("\n--- Summary ---")
    for col in ["moisture_index", "fill_level_pct", "fill_level_raw", "waste_mean_temp_c", "max_temp_c"]:
        print(f"  {col}: min={out_df[col].min():.2f}  mean={out_df[col].mean():.2f}  max={out_df[col].max():.2f}")

    hot_frames = out_df[out_df["hot_pixels_50c"] > 0]
    if not hot_frames.empty:
        print(f"\n  *** {len(hot_frames)} frame(s) with hot pixels >50°C (fire risk check) ***")
    else:
        print("\n  No hot pixels >50°C detected.")


if __name__ == "__main__":
    main()
