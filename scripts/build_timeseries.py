#!/usr/bin/env python3
"""
Build a timeseries CSV matching csv/ir/rgb files by timestamp.

Usage:
    python build_timeseries.py <s3_path> [--local-dir <dir>] [--tolerance <seconds>]
    python build_timeseries.py <s3_path> --no-download [--output <file>] [--tolerance <seconds>]

Examples:
    # Download and scan a specific device prefix
    python build_timeseries.py s3://bai-stl128tetra-uw2-data-field-data/reworld-haverhill-achute//site=haverhill/facility=a/device_id=reworld-haverhill-achute/
    python build_timeseries.py s3://bai-stl128tetra-uw2-data-field-data/device_id=reworld-haverhill-tipping1/ --local-dir ./tipping1

    # Scan entire bucket (or any prefix) without downloading anything
    python build_timeseries.py s3://bai-stl128tetra-uw2-data-field-data/ --no-download
    python build_timeseries.py s3://bai-stl128tetra-uw2-data-field-data/ --no-download --output ./all_devices.csv
"""

import csv
import bisect
import argparse
import subprocess
import re
from datetime import datetime
from pathlib import Path

TOLERANCE_SECS = 60
EXT_MAP = {"csv": ".csv", "ir": ".bmp", "rgb": ".jpg"}


def parse_ts(stem):
    return datetime.strptime(stem, "%Y%m%d%H%M%S")


def nearest(ts, sorted_ts, ts_map, tolerance_secs: int):
    if not sorted_ts:
        return ""
    idx = bisect.bisect_left(sorted_ts, ts)
    candidates = []
    if idx < len(sorted_ts):
        candidates.append(sorted_ts[idx])
    if idx > 0:
        candidates.append(sorted_ts[idx - 1])
    best = min(candidates, key=lambda t: abs((t - ts).total_seconds()))
    if abs((best - ts).total_seconds()) > tolerance_secs:
        return ""
    return ts_map[best]


def parse_s3_metadata(s3_path: str) -> tuple:
    path = s3_path.rstrip("/")
    m = re.search(r"device_id=([^/]+)", path)
    device_name = m.group(1) if m else ""
    m = re.search(r"site=([^/]+)", path)
    site_name = f"bai-{m.group(1)}" if m else ""
    return device_name, site_name


def s3_path_to_local_name(s3_path: str) -> str:
    path = s3_path.rstrip("/")
    m = re.search(r"device_id=([^/]+)", path)
    if m:
        return m.group(1)
    segments = [s for s in path.replace("s3://", "").split("/") if s]
    return segments[-1] if segments else "data"


# ---------------------------------------------------------------------------
# Download-based path
# ---------------------------------------------------------------------------

def collect_local(base: Path, data_type: str, ext: str) -> dict:
    files = {}
    pattern = f"data_type={data_type}"
    for p in base.rglob(f"*.{ext}"):
        if pattern in str(p):
            try:
                files[parse_ts(p.stem)] = str(p)
            except ValueError:
                pass
    return files


def sync_from_s3(s3_path: str, local_dir: Path):
    print(f"Syncing {s3_path} -> {local_dir} ...")
    local_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(["aws", "s3", "sync", s3_path, str(local_dir)], capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"aws s3 sync failed with exit code {result.returncode}")


def build_timeseries_local(local_dir: Path, tolerance_secs: int, device_name: str = "", site_name: str = "") -> Path:
    csv_map = collect_local(local_dir, "csv", "csv")
    ir_map  = collect_local(local_dir, "ir",  "bmp")
    rgb_map = collect_local(local_dir, "rgb", "jpg")

    all_ts = sorted(set(csv_map) | set(ir_map) | set(rgb_map))
    csv_ts = sorted(csv_map)
    ir_ts  = sorted(ir_map)
    rgb_ts = sorted(rgb_map)

    out_path = local_dir / "timeseries.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "device_name", "site_name", "csv_file", "ir_file", "rgb_file"])
        for ts in all_ts:
            writer.writerow([
                ts.strftime("%Y-%m-%d %H:%M:%S"),
                device_name,
                site_name,
                nearest(ts, csv_ts, csv_map, tolerance_secs),
                nearest(ts, ir_ts,  ir_map,  tolerance_secs),
                nearest(ts, rgb_ts, rgb_map, tolerance_secs),
            ])

    print(f"Wrote {len(all_ts)} rows to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Listing-based path (no download)
# ---------------------------------------------------------------------------

def list_s3_keys(s3_path: str) -> list:
    """Return full S3 URIs for every object under s3_path using aws s3 ls --recursive."""
    print(f"Listing {s3_path} ...")
    result = subprocess.run(
        ["aws", "s3", "ls", "--recursive", s3_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"aws s3 ls failed: {result.stderr.strip()}")

    m = re.match(r"s3://([^/]+)", s3_path)
    bucket = m.group(1) if m else ""

    uris = []
    for line in result.stdout.splitlines():
        # Format: "2023-01-01 00:00:00      12345 key/path/to/file"
        parts = line.split(None, 3)
        if len(parts) == 4:
            uris.append(f"s3://{bucket}/{parts[3]}")
    print(f"Found {len(uris)} objects.")
    return uris


def collect_from_listing(uris: list) -> dict:
    """
    Group S3 URIs by (device_name, site_name) then by data_type.
    Returns {(device_name, site_name): {"csv": {ts: uri}, "ir": {ts: uri}, "rgb": {ts: uri}}}
    """
    groups = {}
    for uri in uris:
        data_type = next((dt for dt in EXT_MAP if f"data_type={dt}" in uri), None)
        if not data_type:
            continue
        if not uri.endswith(EXT_MAP[data_type]):
            continue

        stem = uri.split("/")[-1].rsplit(".", 1)[0]
        try:
            ts = parse_ts(stem)
        except ValueError:
            continue

        device_name, site_name = parse_s3_metadata(uri)
        key = (device_name, site_name)
        if key not in groups:
            groups[key] = {"csv": {}, "ir": {}, "rgb": {}}
        groups[key][data_type][ts] = uri

    return groups


def build_timeseries_s3(s3_path: str, tolerance_secs: int, out_path: Path) -> Path:
    uris = list_s3_keys(s3_path)
    groups = collect_from_listing(uris)

    if not groups:
        print("No matching files found (expected data_type=csv/ir/rgb in paths).")
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "device_name", "site_name", "csv_file", "ir_file", "rgb_file"])
        for (device_name, site_name), data in sorted(groups.items()):
            csv_map = data["csv"]
            ir_map  = data["ir"]
            rgb_map = data["rgb"]

            all_ts = sorted(set(csv_map) | set(ir_map) | set(rgb_map))
            csv_ts = sorted(csv_map)
            ir_ts  = sorted(ir_map)
            rgb_ts = sorted(rgb_map)

            for ts in all_ts:
                writer.writerow([
                    ts.strftime("%Y-%m-%d %H:%M:%S"),
                    device_name,
                    site_name,
                    nearest(ts, csv_ts, csv_map, tolerance_secs),
                    nearest(ts, ir_ts,  ir_map,  tolerance_secs),
                    nearest(ts, rgb_ts, rgb_map, tolerance_secs),
                ])
                total += 1

    print(f"Wrote {total} rows across {len(groups)} device(s) to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build timeseries CSV from S3 field data.")
    parser.add_argument("s3_path", help="S3 path to scan (e.g. s3://bucket/ or s3://bucket/prefix/)")
    parser.add_argument("--local-dir", help="Local directory to sync into, or output directory for --no-download")
    parser.add_argument("--output", help="Output CSV path (only used with --no-download; default: timeseries.csv)")
    parser.add_argument("--tolerance", type=int, default=TOLERANCE_SECS,
                        help=f"Max seconds between timestamps to count as a match (default: {TOLERANCE_SECS})")
    parser.add_argument("--no-sync", action="store_true",
                        help="Skip S3 sync and use existing local files only")
    parser.add_argument("--no-download", action="store_true",
                        help="List S3 files via aws s3 ls without downloading; output CSV contains S3 paths")
    args = parser.parse_args()

    if args.no_download:
        if args.output:
            out_path = Path(args.output)
        elif args.local_dir:
            out_path = Path(args.local_dir) / "timeseries.csv"
        else:
            out_path = Path("timeseries.csv")
        build_timeseries_s3(args.s3_path, args.tolerance, out_path)
    else:
        local_dir = Path(args.local_dir) if args.local_dir else Path.home() / s3_path_to_local_name(args.s3_path)
        device_name, site_name = parse_s3_metadata(args.s3_path)
        if not args.no_sync:
            sync_from_s3(args.s3_path, local_dir)
        build_timeseries_local(local_dir, args.tolerance, device_name, site_name)


if __name__ == "__main__":
    main()
