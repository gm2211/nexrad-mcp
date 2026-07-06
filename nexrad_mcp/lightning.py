"""
GOES-R series GLM (Geostationary Lightning Mapper) flash detection.

Source: public NOAA "Open Data Dissemination" S3 buckets, anonymous access
(no credentials, no API key):
  - noaa-goes19: GOES-East (GOES-19), covers the Americas + Atlantic.
  - noaa-goes18: GOES-West (GOES-18), covers the Pacific + western N. America.

Product: GLM-L2-LCFA (Level 2 Lightning Cluster-Filter Algorithm), one
20-second granule per file, laid out as:

    GLM-L2-LCFA/{year}/{day_of_year:03d}/{hour:02d}/
        OR_GLM-L2-LCFA_G{sat}_s{start}_e{end}_c{created}.nc

where {start} etc. are YYYYDDDHHMMSSt (t = tenths of a second) timestamps.
Verified empirically against both buckets before writing this module.

Each granule is a small NetCDF4 file with flash-level variables (flash_lat,
flash_lon, flash_energy, flash_time_offset_of_first_event, ...) alongside
finer-grained group/event tables we don't need for a simple "is there
lightning near me" query.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Optional

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError

from .radar import bearing_deg, compass, haversine_km

_MAX_MINUTES = 30
_MAX_GRANULES_TO_FETCH = 90  # sample evenly if more than this would apply
_CACHE_MAX_AGE_SECONDS = 2 * 3600  # evict cached granules older than 2 hours

_DL_DIR = os.path.join(tempfile.gettempdir(), "nexrad_mcp_cache", "glm")
os.makedirs(_DL_DIR, exist_ok=True)


def _evict_stale_granules() -> None:
    """Delete cached granule files older than _CACHE_MAX_AGE_SECONDS so the
    cache dir doesn't grow unbounded. Cleanup failure never breaks a query."""
    try:
        cutoff = datetime.now(timezone.utc).timestamp() - _CACHE_MAX_AGE_SECONDS
        for fn in os.listdir(_DL_DIR):
            path = os.path.join(_DL_DIR, fn)
            if os.path.getmtime(path) < cutoff:
                os.remove(path)
    except OSError:
        pass


_s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED),
                    region_name="us-east-1")


def _bucket_for(lon: float) -> tuple[str, str]:
    """Pick GOES-East vs GOES-West coverage by longitude, per the task spec:
    lon >= -105 -> GOES-East (noaa-goes19), else GOES-West (noaa-goes18)."""
    if lon >= -105:
        return "noaa-goes19", "G19"
    return "noaa-goes18", "G18"


def _hour_prefixes(minutes: int) -> list[tuple[str, datetime]]:
    """Return (bucket-key-prefix, hour_start) pairs covering the last
    `minutes` back from now, at hour granularity (GLM keys are bucketed by
    UTC hour)."""
    now = datetime.now(timezone.utc)
    start = now - timedelta(minutes=minutes)
    hours = []
    cur = start.replace(minute=0, second=0, microsecond=0)
    while cur <= now:
        doy = cur.timetuple().tm_yday
        prefix = f"GLM-L2-LCFA/{cur.year}/{doy:03d}/{cur.hour:02d}/"
        hours.append((prefix, cur))
        cur += timedelta(hours=1)
    return hours


def _parse_granule_start(key: str) -> Optional[datetime]:
    """Parse the s{YYYYDDDHHMMSSt} start timestamp out of a GLM key/filename."""
    fn = key.split("/")[-1]
    try:
        s_part = fn.split("_s")[1].split("_")[0]  # YYYYDDDHHMMSSt
        year = int(s_part[0:4])
        doy = int(s_part[4:7])
        hh = int(s_part[7:9])
        mm = int(s_part[9:11])
        ss = int(s_part[11:13])
        tenths = int(s_part[13:14]) if len(s_part) > 13 else 0
        base = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy - 1)
        return base.replace(hour=hh, minute=mm, second=ss) + timedelta(
            milliseconds=tenths * 100
        )
    except (IndexError, ValueError):
        return None


def _list_granules(bucket: str, minutes: int) -> list[str]:
    """List GLM granule keys in `bucket` covering the last `minutes`,
    filtered to those whose start time actually falls in the window."""
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(minutes=minutes)
    keys: list[str] = []
    for prefix, _hour_start in _hour_prefixes(minutes):
        try:
            paginator = _s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    keys.append(obj["Key"])
        except ClientError:
            continue

    def in_window(key: str) -> bool:
        t = _parse_granule_start(key)
        return t is not None and cutoff <= t <= now

    keys = [k for k in keys if in_window(k)]
    keys.sort()
    return keys


def _sample_evenly(items: list, n: int) -> list:
    if len(items) <= n:
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def _download_granule(bucket: str, key: str) -> str:
    fn = key.split("/")[-1]
    local_path = os.path.join(_DL_DIR, fn)
    if not os.path.exists(local_path):
        _s3.download_file(bucket, key, local_path)
    return local_path


def get_lightning_activity(lat: float, lon: float, radius_km: float = 50,
                           minutes: int = 10) -> dict:
    """Check for recent lightning (GOES GLM flash detections) near a point.

    Args:
        lat, lon: point of interest (decimal degrees).
        radius_km: search radius around the point (default 50).
        minutes: how far back to look, in minutes (default 10, capped at 30).
            GLM granules are ~20 seconds each, so this can mean scanning up
            to ~90 small files for the full 30-minute window.

    Source: NOAA's GOES-19 (GOES-East) or GOES-18 (GOES-West) Geostationary
    Lightning Mapper, chosen automatically by longitude. GLM detects total
    lightning (both in-cloud and cloud-to-ground flashes, not just
    cloud-to-ground like older ground-based networks), with roughly 1-2
    minutes of latency from real time, and full-disk coverage of the
    Americas and adjacent oceans — it will NOT see lightning over e.g.
    Europe or Asia.

    Returns: flash_count (within radius_km and the time window),
    flashes_per_minute, nearest_flash (distance_km + compass bearing FROM
    the point), most_recent_flash_utc, granules_checked, satellite used, and
    a plain-language summary. A flash_count of 0 with granules_checked > 0
    is a normal, good result (no lightning nearby), not an error.
    """
    minutes = max(1, min(minutes, _MAX_MINUTES))
    _evict_stale_granules()
    bucket, sat_label = _bucket_for(lon)

    try:
        keys = _list_granules(bucket, minutes)
    except Exception as e:
        return {"error": f"could not list GLM granules from {bucket}: {e}"}

    if not keys:
        return {"error": f"no GLM granules found in {bucket} for the last "
                          f"{minutes} minute(s) — the satellite feed may be "
                          f"temporarily behind; try again shortly."}

    keys_to_fetch = _sample_evenly(keys, _MAX_GRANULES_TO_FETCH)

    flashes = []  # (lat, lon, energy, abs_time)
    granules_ok = 0
    granules_failed = 0

    for key in keys_to_fetch:
        try:
            local_path = _download_granule(bucket, key)
            granule_start = _parse_granule_start(key) or datetime.now(timezone.utc)
            import netCDF4  # lazy import: slow-ish, only needed here
            ds = netCDF4.Dataset(local_path)
            try:
                flats = ds.variables["flash_lat"][:]
                flons = ds.variables["flash_lon"][:]
                fenergy = ds.variables["flash_energy"][:]
                ftime = ds.variables["flash_time_offset_of_first_event"][:]
            finally:
                ds.close()
            for flat, flon, energy, toff in zip(flats, flons, fenergy, ftime):
                d = haversine_km(lat, lon, float(flat), float(flon))
                if d <= radius_km:
                    abs_time = granule_start + timedelta(seconds=float(toff))
                    flashes.append((float(flat), float(flon), float(energy), abs_time, d))
            granules_ok += 1
        except Exception:
            granules_failed += 1
            continue

    if granules_ok == 0:
        return {"error": f"all {len(keys_to_fetch)} GLM granule(s) failed to "
                          f"download or decode from {bucket}; cannot determine "
                          f"lightning activity right now."}

    flash_count = len(flashes)
    flashes_per_minute = round(flash_count / minutes, 2) if minutes else 0.0

    nearest_flash = None
    most_recent_utc = None
    if flashes:
        nearest = min(flashes, key=lambda f: f[4])
        brg = bearing_deg(lat, lon, nearest[0], nearest[1])
        nearest_flash = {
            "distance_km": round(nearest[4], 1),
            "bearing": f"{round(brg)}° ({compass(brg)})",
        }
        most_recent = max(flashes, key=lambda f: f[3])
        most_recent_utc = most_recent[3].isoformat()

    if flash_count == 0:
        summary = (f"No lightning within {radius_km} km in the last "
                   f"{minutes} min ({sat_label}, {granules_ok} granule(s) checked).")
    else:
        summary = (f"Active lightning: {flash_count} flash(es) in the last "
                   f"{minutes} min within {radius_km} km, nearest "
                   f"{nearest_flash['distance_km']} km to the "
                   f"{nearest_flash['bearing'].split('(')[1].rstrip(')')}.")

    result = {
        "point": {"lat": lat, "lon": lon},
        "radius_km": radius_km,
        "minutes": minutes,
        "satellite": sat_label,
        "granules_checked": granules_ok,
        "flash_count": flash_count,
        "flashes_per_minute": flashes_per_minute,
        "nearest_flash": nearest_flash,
        "most_recent_flash_utc": most_recent_utc,
        "summary": summary,
    }
    if granules_failed:
        result["granules_failed"] = granules_failed
    return result
