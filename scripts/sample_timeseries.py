#!/usr/bin/env python3
"""
Stratified random sampler for timeseries.csv.

Strata: (site_name, device_name, time_of_day_bucket)
For each stratum, N samples are drawn. For each sample:
  - ir_file  (.bmp)  → 8-bit grayscale rendered IR image
  - rgb_file (.jpg)  → visible RGB image
  - csv_file (.csv)  → 384×512 per-pixel temperature map (°C)

Output per sample: a 2-panel PNG saved to sample_images/
  Left:  RGB image with IR zone footprint outlined in cyan and IR
         heatmap alpha-blended inside it
  Right: IR temperature heatmap with °C colorbar

IR→RGB alignment uses per-device homography estimated via NMI (Normalized
Mutual Information) translation-only grid search. Results are cached to
ir_homographies.json so re-runs don't recompute.

Usage:
    python sample_timeseries.py [--csv timeseries.csv] [--n 2] [--seed 42]
                                [--out sample_images] [--ir-alpha 0.5]
                                [--recalibrate]
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from PIL import Image
from scipy.ndimage import shift as nd_shift
from skimage.metrics import normalized_mutual_information


# ---------------------------------------------------------------------------
# Time-of-day bucketing
# ---------------------------------------------------------------------------

TIME_BUCKETS = {
    "night":     (0,  6),
    "morning":   (6,  12),
    "afternoon": (12, 18),
    "evening":   (18, 24),
}


def time_bucket(hour: int) -> str:
    for name, (start, end) in TIME_BUCKETS.items():
        if start <= hour < end:
            return name
    return "night"


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def s3_download(s3_uri: str, dest_dir: Path) -> Path | None:
    if not s3_uri or not isinstance(s3_uri, str) or not s3_uri.startswith("s3://"):
        return None
    local = dest_dir / Path(s3_uri).name
    if local.exists():
        return local
    result = subprocess.run(
        ["aws", "s3", "cp", s3_uri, str(local)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  [warn] could not download {s3_uri}: {result.stderr.strip()[:120]}")
        return None
    return local


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_rgb(path: Path) -> np.ndarray | None:
    try:
        return np.array(Image.open(path).convert("RGB"))
    except Exception as e:
        print(f"  [warn] rgb open failed {path}: {e}")
        return None


def load_ir_bmp(path: Path) -> np.ndarray | None:
    """Returns (H, W) uint8 grayscale — the rendered IR image."""
    try:
        return np.array(Image.open(path).convert("L"))
    except Exception as e:
        print(f"  [warn] ir open failed {path}: {e}")
        return None


def load_temp_csv(path: Path) -> np.ndarray | None:
    try:
        return pd.read_csv(path, header=None).values.astype(np.float32)
    except Exception as e:
        print(f"  [warn] csv read failed {path}: {e}")
        return None


# ---------------------------------------------------------------------------
# IR→RGB homography: geometry + NMI translation refinement
# ---------------------------------------------------------------------------

H_CACHE_FILE = Path("ir_homographies.json")

# MC200 / NC200 camera specs (datasheet values)
# IR output is 512×384 (2× upsampled from 256×192 physical sensor at 12μm pitch, 3.2mm lens)
# RGB sensor: 1/2.7" CMOS, 2.8mm lens, full 4:3 FOV = 65.5°H × 49.2°V
# 16:9 RGB outputs are a center-vertical crop of the 4:3 sensor, so FOV_H stays the same
# and FOV_V scales as: 49.2° × (output_height / (output_width × 3/4))
IR_FOV_H_DEG  = 52.0   # ±1° per datasheet
IR_FOV_V_DEG  = 40.0   # ±1° per datasheet
RGB_FOV_H_DEG = 65.5   # ±5° per datasheet; constant regardless of output resolution
RGB_FOV_V_43  = 49.2   # at full 4:3 sensor readout


def _load_h_cache() -> dict:
    if H_CACHE_FILE.exists():
        raw = json.loads(H_CACHE_FILE.read_text())
        return {k: np.array(v, dtype=np.float32) for k, v in raw.items()}
    return {}


def _save_h_cache(cache: dict):
    H_CACHE_FILE.write_text(
        json.dumps({k: v.tolist() for k, v in cache.items()}, indent=2)
    )


def _initial_H(ir_shape, rgb_shape) -> np.ndarray:
    """
    Compute IR→RGB homography from camera FOV specs.

    For 16:9 RGB outputs the vertical FOV is smaller than 4:3 (center crop of the
    sensor), so the IR's 40° V FOV actually slightly exceeds the frame — ty comes
    out negative and warpPerspective will simply clip the out-of-frame rows.

    Scales are non-uniform (~1.985 H vs ~2.032 V) because the IR sensor pixels are
    square (12 μm) while the RGB lens covers a slightly different angular extent
    per pixel in each axis at its typical output resolutions.
    """
    ir_h, ir_w = ir_shape[:2]
    rgb_h, rgb_w = rgb_shape[:2]

    # Effective vertical FOV of the RGB output:
    # 4:3 uses full sensor; 16:9 (and other) is a vertical center-crop.
    full_4_3_height = rgb_w * 3 / 4          # e.g. 960 for 1280-wide
    rgb_fov_v = RGB_FOV_V_43 * (rgb_h / full_4_3_height)

    # Angular size per pixel for each sensor
    ir_dpp_h  = IR_FOV_H_DEG / ir_w
    ir_dpp_v  = IR_FOV_V_DEG / ir_h
    rgb_dpp_h = RGB_FOV_H_DEG / rgb_w
    rgb_dpp_v = rgb_fov_v     / rgb_h

    sx = ir_dpp_h / rgb_dpp_h   # ≈ 1.985
    sy = ir_dpp_v / rgb_dpp_v   # ≈ 2.032

    # Center the IR footprint; ty can be negative for 16:9 (IR taller than frame)
    tx = (rgb_w - ir_w * sx) / 2
    ty = (rgb_h - ir_h * sy) / 2

    return np.array([[sx, 0, tx], [0, sy, ty], [0, 0, 1]], dtype=np.float32)


def estimate_ir_to_rgb_H(
    ir_bmp: np.ndarray,
    rgb_arr: np.ndarray,
    device_name: str,
    h_cache: dict,
    search_px: int = 100,
    downsample: int = 4,
) -> tuple[np.ndarray, str]:
    """
    Return (H, method_label) where H maps IR pixel coords to RGB pixel coords.

    Strategy:
      1. If device already in cache, return cached H.
      2. Compute geometry-based initial H (scale + center).
      3. Refine with a coarse NMI translation grid search, then a fine pass
         around the best coarse hit.
      4. Cache and return the result.
    """
    if device_name in h_cache:
        return h_cache[device_name], "cached"

    ir_h, ir_w = ir_bmp.shape[:2]
    rgb_h, rgb_w = rgb_arr.shape[:2]

    H0 = _initial_H((ir_h, ir_w), (rgb_h, rgb_w))

    # Pre-warp IR onto RGB canvas
    ir_f = ir_bmp.astype(np.float32) / 255.0
    ir_pre = cv2.warpPerspective(ir_f, H0, (rgb_w, rgb_h))

    rgb_g = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    # Downsample for speed
    ir_d  = cv2.resize(ir_pre, (rgb_w // downsample, rgb_h // downsample))
    rgb_d = cv2.resize(rgb_g,  (rgb_w // downsample, rgb_h // downsample))

    half = search_px // downsample
    coarse_step = max(1, half // 10)

    best_nmi, best_dy, best_dx = -np.inf, 0, 0
    for dy in range(-half, half + 1, coarse_step):
        for dx in range(-half, half + 1, coarse_step):
            shifted = nd_shift(ir_d, [dy, dx], mode="constant", cval=0)
            mask = shifted > 0.01
            if mask.sum() < 200:
                continue
            nmi = normalized_mutual_information(rgb_d[mask], shifted[mask])
            if nmi > best_nmi:
                best_nmi, best_dy, best_dx = nmi, dy, dx

    # Fine pass ±coarse_step around best coarse hit
    for dy in range(best_dy - coarse_step, best_dy + coarse_step + 1):
        for dx in range(best_dx - coarse_step, best_dx + coarse_step + 1):
            shifted = nd_shift(ir_d, [dy, dx], mode="constant", cval=0)
            mask = shifted > 0.01
            if mask.sum() < 200:
                continue
            nmi = normalized_mutual_information(rgb_d[mask], shifted[mask])
            if nmi > best_nmi:
                best_nmi, best_dy, best_dx = nmi, dy, dx

    dy_rgb = best_dy * downsample
    dx_rgb = best_dx * downsample

    H_t = np.array([[1, 0, dx_rgb], [0, 1, dy_rgb], [0, 0, 1]], dtype=np.float32)
    H_full = H_t @ H0

    h_cache[device_name] = H_full
    _save_h_cache(h_cache)

    label = f"NMI  shift=({dx_rgb:+d},{dy_rgb:+d})px  NMI={best_nmi:.3f}"
    print(f"  [align] {device_name}: {label}")
    return H_full, label


# ---------------------------------------------------------------------------
# Overlay renderer
# ---------------------------------------------------------------------------

def render_overlay(
    rgb_arr: np.ndarray,
    ir_bmp: np.ndarray,
    temp: np.ndarray | None,
    H: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Return an RGB image with the IR heatmap blended inside the IR footprint
    polygon and the polygon border drawn in cyan.
    """
    rgb_h, rgb_w = rgb_arr.shape[:2]
    ir_h, ir_w = ir_bmp.shape[:2]

    # Use temperature CSV for the colormap if available, else use BMP
    if temp is not None:
        t_norm = (temp - temp.min()) / (temp.max() - temp.min() + 1e-9)
        ir_colored = (cm.inferno(t_norm)[:, :, :3] * 255).astype(np.uint8)
    else:
        ir_colored = np.stack([cm.inferno(ir_bmp.astype(float) / 255)[:, :, i] * 255
                                for i in range(3)], axis=-1).astype(np.uint8)

    ir_warped = cv2.warpPerspective(ir_colored, H, (rgb_w, rgb_h))
    coverage  = cv2.warpPerspective(
        np.ones((ir_h, ir_w), dtype=np.float32), H, (rgb_w, rgb_h)
    )[:, :, None].clip(0, 1)

    base   = rgb_arr.astype(np.float32)
    result = (base * (1 - alpha * coverage) + ir_warped.astype(np.float32) * alpha * coverage)
    result = result.clip(0, 255).astype(np.uint8)

    # IR footprint polygon
    corners = np.array([[0, 0], [ir_w - 1, 0], [ir_w - 1, ir_h - 1], [0, ir_h - 1]],
                       dtype=np.float32)
    ch = np.hstack([corners, np.ones((4, 1), dtype=np.float32)])
    tc = (H @ ch.T).T
    tc = (tc[:, :2] / tc[:, 2:3]).astype(np.int32)
    cv2.polylines(result, [tc.reshape(-1, 1, 2)], isClosed=True, color=(0, 255, 255), thickness=3)

    return result


# ---------------------------------------------------------------------------
# Collage builder
# ---------------------------------------------------------------------------

def build_collage(
    row: pd.Series,
    rgb_path: Path | None,
    ir_path:  Path | None,
    csv_path: Path | None,
    out_path: Path,
    h_cache:  dict,
    ir_alpha: float,
):
    rgb  = load_rgb(rgb_path)     if rgb_path  else None
    ir   = load_ir_bmp(ir_path)   if ir_path   else None
    temp = load_temp_csv(csv_path) if csv_path  else None

    if rgb is None and ir is None:
        print(f"  [skip] no usable images for {row['Timestamp']}")
        return

    device = str(row.get("device_name", ""))

    # --- Overlay panel (left) ---
    align_label = ""
    if rgb is not None and ir is not None:
        H, align_label = estimate_ir_to_rgb_H(ir, rgb, device, h_cache)
        overlay = render_overlay(rgb, ir, temp, H, alpha=ir_alpha)
    else:
        overlay = rgb  # fallback: just show RGB

    # --- Heatmap panel (right) ---
    heatmap_data = temp if temp is not None else (
        ir.astype(np.float32) / 255.0 if ir is not None else None
    )
    heatmap_label = "IR temp (°C)" if temp is not None else "IR (normalized)"
    vmin = float(temp.min()) if temp is not None else None
    vmax = float(temp.max()) if temp is not None else None

    # --- Layout ---
    has_heatmap = heatmap_data is not None
    n_cols = 2 if has_heatmap else 1
    fig = plt.figure(figsize=(10 * n_cols, 6))
    gs  = gridspec.GridSpec(1, n_cols, wspace=0.06, width_ratios=[2, 1] if has_heatmap else [1])

    ts_str = row["Timestamp"].strftime("%Y-%m-%d %H:%M:%S") if hasattr(row["Timestamp"], "strftime") else str(row["Timestamp"])
    title  = f"{device}  |  {row.get('site_name','')}  |  {ts_str}  |  {row.get('time_bucket','')}"
    if align_label:
        title += f"\nalignment: {align_label}"
    fig.suptitle(title, fontsize=8, y=1.02)

    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(overlay, aspect="auto")
    ax0.set_title("RGB with IR zone overlay", fontsize=9)
    ax0.axis("off")

    if has_heatmap:
        ax1 = fig.add_subplot(gs[1])
        im  = ax1.imshow(heatmap_data, cmap="inferno", vmin=vmin, vmax=vmax, aspect="auto")
        cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        cbar.set_label("°C" if temp is not None else "", fontsize=8)
        if temp is not None:
            mean_t = float(temp.mean())
            ax1.set_title(
                f"{heatmap_label}\nmin={vmin:.1f}  max={vmax:.1f}  mean={mean_t:.1f} °C",
                fontsize=8,
            )
        else:
            ax1.set_title(heatmap_label, fontsize=9)
        ax1.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stratified sampler for timeseries.csv")
    parser.add_argument("--csv",         default="timeseries.csv", help="Input timeseries CSV")
    parser.add_argument("--n",           type=int,   default=2,     help="Samples per stratum")
    parser.add_argument("--seed",        type=int,   default=42,    help="Random seed")
    parser.add_argument("--out",         default="sample_images",   help="Output directory")
    parser.add_argument("--ir-alpha",    type=float, default=0.5,   help="IR overlay opacity [0-1]")
    parser.add_argument("--recalibrate", action="store_true",
                        help="Ignore cached homographies and re-estimate for all devices")
    args = parser.parse_args()

    h_cache = {} if args.recalibrate else _load_h_cache()
    if h_cache:
        print(f"Loaded {len(h_cache)} cached homographies from {H_CACHE_FILE}")

    df = pd.read_csv(args.csv, parse_dates=["Timestamp"])
    df["hour"]        = df["Timestamp"].dt.hour
    df["time_bucket"] = df["hour"].apply(time_bucket)
    df["site_name"]   = df["site_name"].fillna("unknown")

    for col in ["csv_file", "ir_file", "rgb_file"]:
        df[col] = df[col].replace("", pd.NA)

    df_valid = df[df["ir_file"].notna() | df["rgb_file"].notna()].copy()
    strata   = df_valid.groupby(["site_name", "device_name", "time_bucket"])
    print(f"Dataset: {len(df)} rows, {len(df_valid)} with images, {len(strata)} strata")

    rng = np.random.default_rng(args.seed)
    chunks = []
    for (site, device, bucket), group in strata:
        picked = group.sample(min(args.n, len(group)), random_state=int(rng.integers(1 << 31)))
        picked = picked.copy()
        picked["site_name"]   = site
        picked["device_name"] = device
        picked["time_bucket"] = bucket
        chunks.append(picked)
    sample_rows = pd.concat(chunks, ignore_index=True)
    print(f"Sampling {len(sample_rows)} rows total ({args.n} per stratum)\n")

    out_dir = Path(args.out)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        for _, row in sample_rows.iterrows():
            ts_str  = row["Timestamp"].strftime("%Y%m%d_%H%M%S")
            site    = str(row["site_name"]).replace("/", "-")
            device  = str(row["device_name"]).replace("/", "-")
            bucket  = row["time_bucket"]
            out_path = out_dir / f"{site}__{device}__{bucket}__{ts_str}.png"

            print(f"[{site} | {device} | {bucket} | {ts_str}]")

            rgb_path = s3_download(row.get("rgb_file"), tmp)
            ir_path  = s3_download(row.get("ir_file"),  tmp)
            csv_path = s3_download(row.get("csv_file"), tmp)

            build_collage(row, rgb_path, ir_path, csv_path, out_path, h_cache, args.ir_alpha)

    print(f"\nDone. Collages saved to: {out_dir}/")


if __name__ == "__main__":
    main()
