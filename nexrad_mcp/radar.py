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
from datetime import datetime, timedelta, timezone
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
# Note: differential_phase is the raw (unfiltered) PhiDP as decoded straight
# from the volume, NOT the specific differential phase (KDP). KDP requires a
# retrieval algorithm (see kdp_at_point()); it is not included in this map
# because it is expensive to compute for an entire sweep just to answer a
# single-point query.
FIELDS = {
    "reflectivity": ("reflectivity", "dBZ"),
    "velocity": ("velocity", "m/s"),
    "cc": ("cross_correlation_ratio", ""),
    "zdr": ("differential_reflectivity", "dB"),
    "spectrum_width": ("spectrum_width", "m/s"),
    "differential_phase": ("differential_phase", "deg"),
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
# Radar site lookup
# --------------------------------------------------------------------------- #
def find_nearest_sites(lat: float, lon: float, n: int = 3) -> list[dict]:
    """Find the n nearest NEXRAD radar sites to a lat/lon, sorted by distance.

    Uses Py-ART's built-in NEXRAD site table (id, lat, lon, elevation).
    """
    from pyart.io.nexrad_common import NEXRAD_LOCATIONS

    out = []
    for site_id, info in NEXRAD_LOCATIONS.items():
        d = haversine_km(lat, lon, info["lat"], info["lon"])
        elev = info["elev"]  # feet in Py-ART's table
        out.append({
            "site": site_id,
            "lat": info["lat"],
            "lon": info["lon"],
            # A handful of sites in Py-ART's table have a -99999 sentinel for
            # unknown elevation; surface that honestly as null.
            "elev_m": round(elev * 0.3048) if elev != -99999 else None,
            "distance_km": round(d, 1),
        })
    out.sort(key=lambda r: r["distance_km"])
    return out[:n]


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


def get_scan_at(site: str, time_utc_iso: str) -> ScanRef:
    """Find the scan whose start time is nearest to the given UTC timestamp.

    Looks at that day's scans and, if the timestamp is within 30 minutes of
    UTC midnight, the adjacent day too (to cover volumes that straddle the
    UTC day boundary).
    """
    site = site.upper()
    target = datetime.fromisoformat(time_utc_iso)
    if target.tzinfo is None:
        target = target.replace(tzinfo=timezone.utc)
    else:
        target = target.astimezone(timezone.utc)

    days = {target}
    minutes_from_midnight = target.hour * 60 + target.minute
    if minutes_from_midnight <= 30:
        days.add(target - timedelta(days=1))
    elif (24 * 60 - minutes_from_midnight) <= 30:
        days.add(target + timedelta(days=1))

    candidates: list[ScanRef] = []
    for d in days:
        try:
            scans = _CONN.get_avail_scans(
                str(d.year), f"{d.month:02d}", f"{d.day:02d}", site
            )
        except Exception:
            scans = []
        for s in scans:
            fn = s.key.split("/")[-1]
            if fn.endswith("_MDM"):
                continue
            candidates.append(ScanRef(key=s.key, filename=fn,
                                       scan_time_utc=_parse_time(fn)))

    if not candidates:
        raise ValueError(f"No scans found for {site} near {time_utc_iso}")

    best = min(candidates, key=lambda r: abs((r.scan_time_utc - target).total_seconds()))
    return best


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


@lru_cache(maxsize=4)
def _load_sweep(key: str, idx: int):
    """Load a volume and return one sweep by index as a Py-ART radar.

    Used for the Doppler split cut: in NEXRAD volume coverage patterns the
    lowest elevation is scanned twice — sweep 0 (surveillance cut) carries
    reflectivity/dual-pol fields, sweep 1 (Doppler cut, same elevation)
    carries velocity and spectrum width.
    """
    fp = _download(key)
    radar = _pa().io.read_nexrad_archive(fp)
    return radar.extract_sweeps([idx])


@lru_cache(maxsize=2)
def _load_volume(key: str):
    """Load and return the full Py-ART radar object (all sweeps/tilts).

    Kept separate from _load_sweep0 (and with a smaller cache) since full
    volumes are ~10-20x the memory of a single sweep and most tools only
    need the lowest tilt.
    """
    fp = _download(key)
    return _pa().io.read_nexrad_archive(fp)


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


def kdp_at_point(sweep, lat: float, lon: float) -> Optional[float]:
    """Compute specific differential phase (KDP, deg/km) at the gate nearest
    (lat, lon) using Py-ART's Maesaka retrieval on the given sweep.

    This is a real retrieval (not just raw differential_phase), so it takes
    a couple of seconds per call \u2014 only run it when a caller actually wants
    KDP, not on every query_point call.
    """
    pa = _pa()
    kdp_field, _, _ = pa.retrieve.kdp_maesaka(sweep)
    glat = sweep.gate_latitude["data"]
    glon = sweep.gate_longitude["data"]
    idx = np.unravel_index(
        np.argmin((glat - lat) ** 2 + (glon - lon) ** 2), glat.shape
    )
    v = kdp_field["data"][idx]
    return None if np.ma.is_masked(v) else round(float(v), 3)


def query_point(site: str, lat: float, lon: float,
                key: Optional[str] = None,
                time_utc: Optional[str] = None,
                storm_motion_deg: Optional[float] = None,
                storm_motion_kts: Optional[float] = None,
                include_kdp: bool = False) -> dict:
    """Return all dual-pol products at the gate nearest to (lat, lon).

    If time_utc is given (and key is not), resolves the volume nearest that
    timestamp instead of the latest one.

    Storm-relative velocity (SRV): NEXRAD radial velocity convention is
    POSITIVE = moving AWAY from the radar, NEGATIVE = moving TOWARD the radar.
    If storm_motion_deg (direction the storm is moving FROM, meteorological
    convention, e.g. 270 = storm moving from the west) and storm_motion_kts
    are both given, this computes the component of the storm's motion along
    the radial line from the radar through the queried gate, and subtracts it
    from the measured radial velocity:

        storm_relative_velocity = measured_velocity - radial_component_of_storm_motion

    A positive SRV still means "away from radar" and negative "toward radar",
    but now with the storm's own translation removed \u2014 e.g. useful for seeing
    rotation or inflow/outflow that would otherwise be masked by fast storm
    motion. Units match the raw velocity field (m/s, as decoded by Py-ART);
    convert storm_motion_kts internally so both terms are in the same units.

    The `differential_phase` value in `values` is the RAW (unfiltered) PhiDP
    as decoded from the volume, not KDP. Set include_kdp=True to additionally
    run a real KDP retrieval (Py-ART's Maesaka method) at this gate \u2014 this
    takes a few extra seconds because it processes the whole sweep, so it's
    opt-in rather than always-on.
    """
    if key is None:
        if time_utc is not None:
            ref = get_scan_at(site, time_utc)
            key = ref.key
        else:
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

    # Split-cut fallback: at the lowest elevation NEXRAD scans twice — the
    # surveillance cut (sweep 0) has no usable velocity, the Doppler cut
    # (sweep 1, same elevation) does. If velocity is missing here, sample
    # the Doppler cut at the same point.
    velocity_source = None
    if values.get("velocity") is None:
        try:
            dop = _load_sweep(key, 1)
            dlat = dop.gate_latitude["data"]
            dlon = dop.gate_longitude["data"]
            didx = np.unravel_index(
                np.argmin((dlat - lat) ** 2 + (dlon - lon) ** 2), dlat.shape
            )
            for label in ("velocity", "spectrum_width"):
                fld = FIELDS[label][0]
                if values.get(label) is None and fld in dop.fields:
                    v = dop.fields[fld]["data"][didx]
                    if not np.ma.is_masked(v):
                        values[label] = round(float(v), 2)
                        velocity_source = ("Doppler split cut (sweep 1, "
                                           "same elevation)")
        except Exception:
            pass  # volume with a single sweep, or download hiccup

    result = {
        "site": site.upper(),
        "filename": key.split("/")[-1],
        "target": {"lat": lat, "lon": lon},
        "range_km_from_radar": round(dist, 1),
        "bearing_from_radar": f"{round(brg)}\u00b0 ({compass(brg)})",
        "gate_error_km": round(gate_err_km, 2),
        "values": values,
        "interpretation": _interpret(values),
    }
    if velocity_source:
        result["velocity_source"] = velocity_source

    if storm_motion_deg is not None and storm_motion_kts is not None:
        vel = values.get("velocity")
        if vel is None:
            result["storm_relative_velocity"] = None
        else:
            # Convert "direction FROM" to the vector's direction of travel
            # (TO), then project onto the radar->gate radial (brg).
            travel_deg = (storm_motion_deg + 180) % 360
            storm_kts_to_ms = storm_motion_kts * 0.514444
            angle_diff = math.radians(travel_deg - brg)
            radial_component_ms = storm_kts_to_ms * math.cos(angle_diff)
            srv = vel - radial_component_ms
            result["storm_relative_velocity"] = round(srv, 2)
            result["storm_relative_velocity_note"] = (
                "Same convention as velocity: positive = net motion away from "
                "radar after removing storm translation, negative = toward. "
                "Large |SRV| near where raw velocity is small can indicate "
                "rotation the raw field alone would obscure."
            )

    if include_kdp:
        try:
            result["kdp_deg_per_km"] = kdp_at_point(sweep, lat, lon)
        except Exception as e:
            result["kdp_deg_per_km"] = None
            result["kdp_error"] = str(e)

    return result


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


def get_vertical_profile(site: str, lat: float, lon: float,
                         time_utc: Optional[str] = None) -> dict:
    """Build a vertical profile through the radar volume at a single point.

    Loads every elevation tilt in the volume and, for each one, finds the
    gate nearest (lat, lon) and reports its height above ground and all
    dual-pol product values there. Because higher tilts sample higher above
    the ground the farther you are from the radar, this approximates a
    vertical slice through the storm at that location (like a single column
    out of a cross-section), which is how you check e.g. whether a hail core
    aloft is over a location before it reaches the surface, or whether a
    reflectivity core is elevated (aloft) vs. surface-based.

    Also computes, across the whole profile:
      - composite_reflectivity: the strongest reflectivity (dBZ) seen at this
        lat/lon across ALL tilts (not just the lowest one) — a quick way to
        see if there's a strong core aloft even if the surface tilt looks weak.
      - echo_top_km: the highest beam altitude (km above sea level) at this
        point where reflectivity was still >= 18 dBZ (a common "echo top"
        threshold). Null if no tilt reached 18 dBZ at this point.

    Args:
        site: 4-letter radar ID, e.g. "KLWX".
        lat, lon: point of interest (decimal degrees).
        time_utc: optional ISO-8601 UTC timestamp to pick a specific past
            volume instead of the latest one.
    """
    if time_utc is not None:
        ref = get_scan_at(site, time_utc)
        key = ref.key
    else:
        latest = list_recent_scans(site, n=1)
        if not latest:
            return {"error": "no scans found"}
        key = latest[-1].key

    radar = _load_volume(key)

    rlat = float(radar.latitude["data"][0])
    rlon = float(radar.longitude["data"][0])
    dist = haversine_km(rlat, rlon, lat, lon)
    brg = bearing_deg(rlat, rlon, lat, lon)

    profile = []
    best_dbz = None
    echo_top_km = None

    for sweep_idx in range(radar.nsweeps):
        s0, s1 = radar.get_start_end(sweep_idx)
        glat = radar.gate_latitude["data"][s0:s1 + 1]
        glon = radar.gate_longitude["data"][s0:s1 + 1]
        galt = radar.gate_altitude["data"][s0:s1 + 1]

        idx = np.unravel_index(
            np.argmin((glat - lat) ** 2 + (glon - lon) ** 2), glat.shape
        )
        gate_err_km = haversine_km(lat, lon, float(glat[idx]), float(glon[idx]))
        beam_height_km = float(galt[idx]) / 1000.0

        values = {}
        for label, (fld, unit) in FIELDS.items():
            if fld in radar.fields:
                v = radar.fields[fld]["data"][s0:s1 + 1][idx]
                values[label] = None if np.ma.is_masked(v) else round(float(v), 2)
            else:
                values[label] = None

        dbz = values.get("reflectivity")
        if dbz is not None:
            if best_dbz is None or dbz > best_dbz:
                best_dbz = dbz
            if dbz >= 18.0:
                if echo_top_km is None or beam_height_km > echo_top_km:
                    echo_top_km = beam_height_km

        profile.append({
            "elevation_deg": round(float(radar.fixed_angle["data"][sweep_idx]), 2),
            "beam_height_km": round(beam_height_km, 2),
            "gate_error_km": round(gate_err_km, 2),
            "values": values,
        })

    return {
        "site": site.upper(),
        "filename": key.split("/")[-1],
        "target": {"lat": lat, "lon": lon},
        "range_km_from_radar": round(dist, 1),
        "bearing_from_radar": f"{round(brg)}° ({compass(brg)})",
        "profile": profile,
        "composite_reflectivity": best_dbz,
        "echo_top_km": round(echo_top_km, 2) if echo_top_km is not None else None,
    }


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
