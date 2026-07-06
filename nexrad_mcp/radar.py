"""
Core NEXRAD Level II access + analysis.

All functions are plain Python (no MCP dependency) so they can be unit-tested
or reused directly. The MCP server in server.py is a thin wrapper over these.

Data source: NOAA Open Data (NODD) NEXRAD Level II archive on AWS S3,
discovered via the `nexradaws` index. No credentials or API keys required.
"""

from __future__ import annotations

import math
import os
import tempfile
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional

import nexradaws
import numpy as np

warnings.filterwarnings("ignore")

# Lazy import of pyart (slow to import) so `--help` etc. stays snappy.
_pyart = None


def _pa():
    global _pyart
    if _pyart is None:
        import pyart  # noqa
        _pyart = pyart
    return _pyart


_CONN = nexradaws.NexradAwsInterface()
_DL_DIR = os.path.join(tempfile.gettempdir(), "nexrad_mcp_cache")
os.makedirs(_DL_DIR, exist_ok=True)

# Products we expose, mapped to Py-ART field names.
FIELDS = {
    "reflectivity": ("reflectivity", "dBZ"),
    "velocity": ("velocity", "m/s"),
    "cc": ("cross_correlation_ratio", ""),
    "zdr": ("differential_reflectivity", "dB"),
    "spectrum_width": ("spectrum_width", "m/s"),
}


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bearing_deg(lat1, lon1, lat2, lon2) -> float:
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(math.radians(lat2))
    x = (math.cos(math.radians(lat1)) * math.sin(math.radians(lat2))
         - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.cos(dlon))
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def compass(deg: float) -> str:
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return dirs[int((deg + 11.25) % 360 // 22.5)]


# --------------------------------------------------------------------------- #
# Scan discovery + download
# --------------------------------------------------------------------------- #
@dataclass
class ScanRef:
    key: str
    filename: str
    scan_time_utc: datetime


def list_recent_scans(site: str, n: int = 6) -> list[ScanRef]:
    """Return the n most recent available scans for a site (today UTC,
    falling back to yesterday if today's partition is nearly empty)."""
    site = site.upper()
    out: list[ScanRef] = []
    now = datetime.now(timezone.utc)
    for day_offset in (0, 1):  # today, then yesterday (UTC rollover)
        d = now.timestamp() - day_offset * 86400
        dt = datetime.fromtimestamp(d, timezone.utc)
        try:
            scans = _CONN.get_avail_scans(
                str(dt.year), f"{dt.month:02d}", f"{dt.day:02d}", site
            )
        except Exception:
            scans = []
        for s in scans:
            fn = s.key.split("/")[-1]
            if fn.endswith("_MDM"):  # skip metadata objects
                continue
            out.append(ScanRef(key=s.key, filename=fn,
                               scan_time_utc=_parse_time(fn)))
        if len(out) >= n:
            break
    out.sort(key=lambda r: r.scan_time_utc)
    return out[-n:]


def _parse_time(filename: str) -> datetime:
    # KLWX20260705_000847_V06
    stamp = filename.split("_")[0][4:] + filename.split("_")[1]  # date+time
    return datetime.strptime(stamp, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


@lru_cache(maxsize=8)
def _download(key: str) -> str:
    """Download a scan by key, cached so repeated queries on the same
    volume don't re-fetch. Returns local filepath."""
    for s in _all_scan_objs_for_key(key):
        res = _CONN.download(s, _DL_DIR)
        if res.success:
            return res.success[0].filepath
    raise RuntimeError(f"could not download {key}")


def _all_scan_objs_for_key(key: str):
    fn = key.split("/")[-1]
    site = fn[:4]
    t = _parse_time(fn)
    scans = _CONN.get_avail_scans(str(t.year), f"{t.month:02d}",
                                  f"{t.day:02d}", site)
    return [s for s in scans if s.key == key]


@lru_cache(maxsize=4)
def _load_sweep0(key: str):
    """Load a volume and return its lowest (0.5 deg) sweep as a Py-ART radar."""
    fp = _download(key)
    radar = _pa().io.read_nexrad_archive(fp)
    return radar.extract_sweeps([0])


# --------------------------------------------------------------------------- #
# Public analysis functions
# --------------------------------------------------------------------------- #
def get_latest_scan(site: str) -> dict:
    scans = list_recent_scans(site, n=1)
    if not scans:
        return {"site": site.upper(), "error": "no scans found"}
    s = scans[-1]
    age_min = (datetime.now(timezone.utc) - s.scan_time_utc).total_seconds() / 60
    return {
        "site": site.upper(),
        "filename": s.filename,
        "scan_time_utc": s.scan_time_utc.isoformat(),
        "age_minutes": round(age_min, 1),
    }


def query_point(site: str, lat: float, lon: float,
                key: Optional[str] = None) -> dict:
    """Return all dual-pol products at the gate nearest to (lat, lon)."""
    if key is None:
        latest = list_recent_scans(site, n=1)
        if not latest:
            return {"error": "no scans found"}
        key = latest[-1].key
    sweep = _load_sweep0(key)

    glat = sweep.gate_latitude["data"]
    glon = sweep.gate_longitude["data"]
    idx = np.unravel_index(
        np.argmin((glat - lat) ** 2 + (glon - lon) ** 2), glat.shape
    )
    gate_err_km = haversine_km(lat, lon, float(glat[idx]), float(glon[idx]))

    rlat = float(sweep.latitude["data"][0])
    rlon = float(sweep.longitude["data"][0])
    dist = haversine_km(rlat, rlon, lat, lon)
    brg = bearing_deg(rlat, rlon, lat, lon)

    values = {}
    for label, (fld, unit) in FIELDS.items():
        if fld in sweep.fields:
            v = sweep.fields[fld]["data"][idx]
            values[label] = None if np.ma.is_masked(v) else round(float(v), 2)
        else:
            values[label] = None

    return {
        "site": site.upper(),
        "filename": key.split("/")[-1],
        "target": {"lat": lat, "lon": lon},
        "range_km_from_radar": round(dist, 1),
        "bearing_from_radar": f"{round(brg)}\u00b0 ({compass(brg)})",
        "gate_error_km": round(gate_err_km, 2),
        "values": values,
        "interpretation": _interpret(values),
    }


def _interpret(v: dict) -> str:
    dbz = v.get("reflectivity")
    cc = v.get("cc")
    if dbz is None:
        return "No meaningful echo at this location."
    parts = []
    if dbz < 20:
        parts.append("very light/no precip")
    elif dbz < 35:
        parts.append("light rain")
    elif dbz < 50:
        parts.append("moderate rain")
    else:
        parts.append("intense core (heavy rain/hail possible)")
    if cc is not None:
        if cc >= 0.97:
            parts.append("CC high = uniform meteorological scatter (not debris)")
        elif cc < 0.80 and dbz >= 45:
            parts.append("LOW CC with strong echo = possible debris; check velocity")
        elif cc < 0.90:
            parts.append("lowered CC = mixed/non-uniform targets (often clutter/bio)")
    return "; ".join(parts)


def check_storms_near(site: str, lat: float, lon: float,
                      radius_km: float = 60,
                      dbz_threshold: float = 45) -> dict:
    """Find storm cores (>= dbz_threshold) within radius_km of a point,
    reporting their distance and bearing FROM the point."""
    latest = list_recent_scans(site, n=1)
    if not latest:
        return {"error": "no scans found"}
    key = latest[-1].key
    sweep = _load_sweep0(key)

    glat = sweep.gate_latitude["data"]
    glon = sweep.gate_longitude["data"]
    refl = sweep.fields["reflectivity"]["data"]

    mask = (~np.ma.getmaskarray(refl)) & (refl >= dbz_threshold)
    if not mask.any():
        return {"site": site.upper(), "filename": key.split("/")[-1],
                "cells": [], "summary": f"No echoes >= {dbz_threshold} dBZ found."}

    # Coarse distance filter
    ii = np.where(mask)
    cells = []
    for r, g in zip(*ii):
        plat, plon = float(glat[r, g]), float(glon[r, g])
        d = haversine_km(lat, lon, plat, plon)
        if d <= radius_km:
            cells.append((d, bearing_deg(lat, lon, plat, plon),
                          float(refl[r, g]), plat, plon))
    if not cells:
        return {"site": site.upper(), "filename": key.split("/")[-1],
                "cells": [],
                "summary": f"No cores >= {dbz_threshold} dBZ within {radius_km} km."}

    # Cluster crudely by rounding to ~0.1 deg and keeping the strongest gate
    seen = {}
    for d, b, z, plat, plon in cells:
        bucket = (round(plat, 1), round(plon, 1))
        if bucket not in seen or z > seen[bucket][2]:
            seen[bucket] = (d, b, z, plat, plon)
    clusters = sorted(seen.values(), key=lambda c: c[0])[:8]

    return {
        "site": site.upper(),
        "filename": key.split("/")[-1],
        "point": {"lat": lat, "lon": lon},
        "radius_km": radius_km,
        "dbz_threshold": dbz_threshold,
        "cells": [
            {"distance_km": round(d, 1),
             "bearing": f"{round(b)}\u00b0 ({compass(b)})",
             "max_dbz": round(z, 1),
             "lat": round(plat, 3), "lon": round(plon, 3)}
            for d, b, z, plat, plon in clusters
        ],
        "summary": (f"{len(clusters)} core(s) >= {dbz_threshold} dBZ within "
                    f"{radius_km} km; nearest {round(clusters[0][0],1)} km "
                    f"to the {compass(clusters[0][1])}."),
    }


def estimate_motion(site: str, lat: float, lon: float,
                    radius_km: float = 60,
                    dbz_threshold: float = 40) -> dict:
    """Compare the two most recent volumes to estimate whether the nearest
    core is approaching the point. Uses centroid displacement of strong echo."""
    scans = list_recent_scans(site, n=2)
    if len(scans) < 2:
        return {"error": "need at least two scans to estimate motion"}

    def centroid(key):
        sweep = _load_sweep0(key)
        glat = sweep.gate_latitude["data"]
        glon = sweep.gate_longitude["data"]
        refl = sweep.fields["reflectivity"]["data"]
        mask = (~np.ma.getmaskarray(refl)) & (refl >= dbz_threshold)
        # restrict to radius
        if not mask.any():
            return None
        plat = glat[mask]; plon = glon[mask]; z = refl[mask]
        d = np.array([haversine_km(lat, lon, a, b) for a, b in zip(plat, plon)])
        near = d <= radius_km
        if not near.any():
            return None
        w = z[near]
        return (float(np.average(plat[near], weights=w)),
                float(np.average(plon[near], weights=w)))

    c_old = centroid(scans[0].key)
    c_new = centroid(scans[1].key)
    if c_old is None or c_new is None:
        return {"site": site.upper(),
                "summary": "No qualifying echo near point in one or both scans; "
                           "nothing to track."}

    dt_min = (scans[1].scan_time_utc - scans[0].scan_time_utc).total_seconds() / 60
    d_old = haversine_km(lat, lon, *c_old)
    d_new = haversine_km(lat, lon, *c_new)
    closing = d_old - d_new  # positive = getting closer
    speed = (closing / dt_min * 60) if dt_min else 0.0
    # Only compute an ETA if it's meaningfully approaching (> 3 km/h closing);
    # tiny closing rates are noise and produce absurd ETAs.
    eta = (d_new / (closing / dt_min)) if (closing > 0.1 and speed > 3) else None

    trend = ("APPROACHING" if closing > 1 else
             "MOVING AWAY" if closing < -1 else "roughly stationary/parallel")
    return {
        "site": site.upper(),
        "scans": [scans[0].filename, scans[1].filename],
        "minutes_between": round(dt_min, 1),
        "echo_distance_km": {"previous": round(d_old, 1), "now": round(d_new, 1)},
        "closing_speed_kmh": round(speed, 1),
        "trend": trend,
        "eta_minutes": round(eta, 0) if eta else None,
        "summary": (f"Nearest cluster is {trend.lower()} at "
                    f"~{abs(round(speed,1))} km/h"
                    + (f"; ETA ~{round(eta)} min if it holds course."
                       if eta else ".")),
    }
