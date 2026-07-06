"""
NEXRAD Level 3 (derived) product access.

Data source: the public `unidata-nexrad-level3` S3 bucket (no credentials;
anonymous/unsigned access). Key layout (verified empirically against the live
bucket): a flat namespace of keys shaped `SSS_PPP_YYYY_MM_DD_HH_MM_SS`, where
SSS is the 3-letter site id (the 4-letter ICAO id minus its leading K/P/T,
e.g. KLWX -> LWX) and PPP is the product code (e.g. DVL, EET, NST). There are
no directory delimiters, so listings must always use a `SSS_PPP_YYYY_MM_DD`
day prefix — anything broader pages through a multi-year archive.

Decoding uses MetPy's Level3File for every product: it handles both the
gridded radial products (DVL, EET, DAA, DTA, HHC) and the storm-track
product's symbology/tabular blocks (NST), which Py-ART cannot parse
(`read_nexrad_level3` returns empty fields for DVL/EET and raises
NotImplementedError for NST/NMD). Physical scaling was verified empirically
by cross-checking MetPy's decode against Py-ART's on products both can read:
DAA/DTA arrive from MetPy in hundredths of inches (Py-ART: inches), HHC
arrives as NWS classification code / 10.

Note on legacy products: the NWS retired the standalone TVS (NTV) and hail
index (NHI) Level 3 products — no data has been published for them since
before 2025, fleet-wide (verified against multiple sites in the bucket), so
they cannot be surfaced. Storm tracks (NST) and mesocyclone (NMD) are still
produced; NMD only contains detections while rotation is actually being
identified.
"""

from __future__ import annotations

import math
import os
import re
import tempfile
from datetime import datetime, timedelta, timezone
from functools import lru_cache

import boto3
import numpy as np
from botocore import UNSIGNED
from botocore.config import Config

from .radar import bearing_deg, compass, haversine_km

BUCKET = "unidata-nexrad-level3"

_DL_DIR = os.path.join(tempfile.gettempdir(), "nexrad_mcp_cache_l3")
os.makedirs(_DL_DIR, exist_ok=True)

# Curated gridded product set (RadarScope's derived-data classes).
# code -> (units, description)
GRID_PRODUCTS = {
    "DVL": ("kg/m^2", "Digital vertically integrated liquid (VIL). High VIL "
                      "(>~50) suggests a storm holding lots of water/hail aloft."),
    "EET": ("kft", "Enhanced echo tops: height of the highest 18 dBZ echo. "
                   "Taller tops = deeper, usually stronger storm."),
    "DAA": ("in", "Digital one-hour precipitation accumulation (dual-pol QPE)."),
    "DTA": ("in", "Digital storm-total precipitation accumulation (dual-pol QPE)."),
    "HHC": ("class", "Hybrid hydrometeor classification: what the radar thinks "
                     "is falling (rain, snow, hail, biological, ...)."),
}
FEATURE_PRODUCTS = {
    "NST": "Storm tracking information (STI): storm cell positions, motion, "
           "and forecast track.",
    "NMD": "Mesocyclone detections (rotation signatures).",
}
RETIRED_PRODUCTS = {
    "NTV": "Tornado vortex signature (TVS) — retired by the NWS; no data "
           "published fleet-wide since before 2025.",
    "NHI": "Hail index — retired by the NWS; no data published fleet-wide "
           "since before 2025.",
}

# NEXRAD hydrometeor classification (HHC): MetPy's mapped value = NWS code/10
# (verified against Py-ART's decode of the same file).
HHC_CLASSES = {
    0: "no data",
    1: "biological (birds/insects)",
    2: "ground clutter / anomalous propagation",
    3: "ice crystals",
    4: "dry snow",
    5: "wet snow",
    6: "light/moderate rain",
    7: "heavy rain",
    8: "big drops",
    9: "graupel",
    10: "hail (possibly with rain)",
    11: "large hail",
    12: "giant hail",
    14: "unknown",
    15: "range folded",
}

_METPY = None


def _mp_level3():
    """Lazy import: MetPy is slow to import, keep server startup snappy."""
    global _METPY
    if _METPY is None:
        from metpy.io import Level3File
        _METPY = Level3File
    return _METPY


@lru_cache(maxsize=1)
def _s3():
    return boto3.client(
        "s3", config=Config(signature_version=UNSIGNED), region_name="us-east-1"
    )


def _site3(site: str) -> str:
    """KLWX -> LWX, TJUA -> JUA."""
    site = site.upper().strip()
    return site[-3:] if len(site) == 4 else site


def _key_time(key: str) -> datetime:
    # SSS_PPP_YYYY_MM_DD_HH_MM_SS
    parts = key.split("_")
    return datetime(*map(int, parts[2:8]), tzinfo=timezone.utc)


def _list_keys(prefix: str) -> list[str]:
    keys, kwargs = [], dict(Bucket=BUCKET, Prefix=prefix)
    while True:
        r = _s3().list_objects_v2(**kwargs)
        keys += [o["Key"] for o in r.get("Contents", [])]
        if r.get("IsTruncated"):
            kwargs["ContinuationToken"] = r["NextContinuationToken"]
        else:
            break
    return keys


def _latest_key(site3: str, code: str) -> str | None:
    """Newest key for a site/product, looking at today then yesterday (UTC).

    Keys sort lexicographically by timestamp, so the last key of the day
    prefix is the newest.
    """
    now = datetime.now(timezone.utc)
    for day in (now, now - timedelta(days=1)):
        prefix = f"{site3}_{code}_{day:%Y_%m_%d}"
        keys = _list_keys(prefix)
        if keys:
            return keys[-1]
    return None


@lru_cache(maxsize=16)
def _download(key: str) -> str:
    fp = os.path.join(_DL_DIR, key)
    if not os.path.exists(fp):
        _s3().download_file(BUCKET, key, fp)
    return fp


def _parse(key: str):
    return _mp_level3()(_download(key))


def _age_minutes(t: datetime) -> float:
    return round((datetime.now(timezone.utc) - t).total_seconds() / 60, 1)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def list_l3_products(site: str) -> dict:
    """List which RadarScope-class Level 3 products have fresh data for a site."""
    s3id = _site3(site)
    products = []
    for code in list(GRID_PRODUCTS) + list(FEATURE_PRODUCTS):
        if code in GRID_PRODUCTS:
            units, desc = GRID_PRODUCTS[code]
            kind = "grid"
        else:
            units, desc, kind = None, FEATURE_PRODUCTS[code], "features"
        key = _latest_key(s3id, code)
        entry = {"product": code, "kind": kind, "description": desc}
        if units:
            entry["units"] = units
        if key:
            t = _key_time(key)
            entry.update(available=True,
                         latest_time_utc=t.isoformat(),
                         age_minutes=_age_minutes(t))
        else:
            entry["available"] = False
        products.append(entry)
    retired = [{"product": c, "available": False, "note": note}
               for c, note in RETIRED_PRODUCTS.items()]
    return {"site": site.upper(), "products": products,
            "retired_products": retired}


def _radial_layer(f):
    """Return the radial-image layer dict from a MetPy Level3File."""
    if not getattr(f, "sym_block", None):
        raise ValueError("product file contains no symbology (data) block")
    for layer in f.sym_block:
        for item in layer:
            if isinstance(item, dict) and "data" in item and "start_az" in item:
                return item
    raise ValueError("no radial data layer found in product")


def get_l3_grid_value(site: str, product: str, lat: float, lon: float) -> dict:
    """Sample a gridded Level 3 product at a point.

    Returns the physical value, units, product valid time, and data age.
    """
    product = product.upper()
    if product not in GRID_PRODUCTS:
        return {"error": f"unknown/unsupported gridded product {product!r}; "
                         f"supported: {sorted(GRID_PRODUCTS)}"}
    s3id = _site3(site)
    key = _latest_key(s3id, product)
    if key is None:
        return {"site": site.upper(), "product": product,
                "error": "no recent data (today/yesterday UTC) in archive"}

    f = _parse(key)
    try:
        layer = _radial_layer(f)
    except (ValueError, KeyError):
        return {"error": f"product {product} for {site.upper()} has no "
                         "decodable radial data"}
    raw = np.array(layer["data"])
    mapped = f.map_data(raw)
    topped = None
    if isinstance(mapped, tuple):  # EET returns (value, "topped" flag)
        mapped, topped = mapped
    mapped = np.asarray(mapped, dtype=float)

    # Radar/product center: prod_desc lat/lon are scaled by 1000.
    clat = f.prod_desc.lat / 1000.0
    clon = f.prod_desc.lon / 1000.0
    rng_km = haversine_km(clat, clon, lat, lon)
    brg = bearing_deg(clat, clon, lat, lon)

    n_rad, n_bins = mapped.shape
    if rng_km > f.max_range:
        return {"site": site.upper(), "product": product,
                "error": f"point is {rng_km:.0f} km from radar, beyond the "
                         f"{f.max_range:.0f} km product range"}
    start_az = np.asarray(layer["start_az"][:n_rad], dtype=float) % 360.0
    rad_idx = int(np.argmin(np.abs((start_az - brg + 180) % 360 - 180)))
    bin_km = f.max_range / n_bins
    bin_idx = min(int(rng_km / bin_km), n_bins - 1)

    v = float(mapped[rad_idx, bin_idx])
    units, desc = GRID_PRODUCTS[product]

    # Product-specific physical rescaling (verified against Py-ART's decode).
    value: float | str | None
    if math.isnan(v):
        value = None
    elif product in ("DAA", "DTA"):
        value = round(v / 100.0, 2)  # MetPy yields hundredths of inches
    elif product == "HHC":
        value = HHC_CLASSES.get(int(round(v)), f"class {int(round(v))}")
    else:
        value = round(v, 2)

    t = _key_time(key)
    out = {
        "site": site.upper(),
        "product": product,
        "description": desc,
        "target": {"lat": lat, "lon": lon},
        "range_km_from_radar": round(rng_km, 1),
        "bearing_from_radar": f"{round(brg)}° ({compass(brg)})",
        "value": value,
        "units": "classification" if product == "HHC" else units,
        "product_time_utc": t.isoformat(),
        "age_minutes": _age_minutes(t),
    }
    if topped is not None and value is not None:
        out["echo_top_capped"] = bool(np.asarray(topped)[rad_idx, bin_idx])
    return out


# --------------------------------------------------------------------------- #
# Storm features (tracks / meso)
# --------------------------------------------------------------------------- #
def _km_offset_to_latlon(clat: float, clon: float, x_km: float, y_km: float):
    """Convert km east/north of the radar to lat/lon (small-angle approx)."""
    dlat = y_km / 111.2
    dlon = x_km / (111.2 * math.cos(math.radians(clat)))
    return round(clat + dlat, 4), round(clon + dlon, 4)


_MVT_RE = re.compile(
    r"^\s{1,4}(\w\d)\s+(\d+)/\s*(\d+)\s+(NEW|\d+/\s*\d+)", re.MULTILINE
)


def _parse_sti_tab(pages) -> dict[str, dict]:
    """Parse the STORM POSITION/FORECAST tabular pages into {storm_id: attrs}."""
    out: dict[str, dict] = {}
    for page in pages or []:
        text = page if isinstance(page, str) else "\n".join(page)
        for m in _MVT_RE.finditer(text):
            sid, az, rng_nm, mvt = m.groups()
            attrs = {"azimuth_deg_from_radar": int(az),
                     "range_nm_from_radar": int(rng_nm)}
            if mvt == "NEW":
                attrs["movement"] = "NEW (first detection, no track yet)"
            else:
                d, s = mvt.split("/")
                attrs["movement_from_deg"] = int(d)
                attrs["movement_kts"] = int(s)
            out[sid] = attrs
    return out


def _parse_sti_sym(f) -> list[dict]:
    """Parse storm markers/tracks from the NST symbology block.

    The block is a flat item stream per storm cell: a 'current storm position'
    marker (x/y km east/north of the radar), a Storm ID text item, then
    optional past-track and forecast-track polylines.
    """
    cells: list[dict] = []
    clat = f.prod_desc.lat / 1000.0
    clon = f.prod_desc.lon / 1000.0
    current: dict | None = None
    for layer in getattr(f, "sym_block", None) or []:
        for item in layer:
            if not isinstance(item, dict):
                continue
            if "current storm position" in item:
                if current:
                    cells.append(current)
                x, y = item["current storm position"]
                la, lo = _km_offset_to_latlon(clat, clon, x, y)
                current = {"lat": la, "lon": lo}
            elif item.get("type") == "Storm ID" and current is not None:
                current["id"] = item.get("id")
            elif "track" in item and current is not None:
                track = [list(_km_offset_to_latlon(clat, clon, x, y))
                         for x, y in item["track"]]
                markers = str(item.get("markers", ""))
                kind = ("forecast_track" if "forecast" in markers
                        else "past_track")
                current[kind] = track
    if current:
        cells.append(current)
    return cells


def get_l3_storm_features(site: str) -> dict:
    """Combined storm-feature report: storm cells/tracks (NST) + mesocyclone
    detections (NMD), with lat/lon and age. TVS and hail-index products were
    retired by the NWS and are reported as permanently unavailable."""
    s3id = _site3(site)
    out: dict = {"site": site.upper()}

    # --- storm tracks (NST / STI) ---
    key = _latest_key(s3id, "NST")
    if key is None:
        out["storm_cells"] = []
        out["storm_tracks_note"] = "no recent NST (storm track) product"
    else:
        f = _parse(key)
        cells = _parse_sti_sym(f)
        attrs = _parse_sti_tab(getattr(f, "tab_pages", None))
        for c in cells:
            extra = attrs.get(c.get("id"))
            if extra:
                c.update(extra)
        t = _key_time(key)
        out["storm_tracks_time_utc"] = t.isoformat()
        out["storm_tracks_age_minutes"] = _age_minutes(t)
        out["storm_cells"] = cells
        out["storm_cells_note"] = (
            "movement_from_deg is the direction the cell is moving FROM "
            "(meteorological convention); forecast_track points are "
            "[lat, lon] at +15/+30/+45/+60 min."
        )

    # --- mesocyclone (NMD) ---
    key = _latest_key(s3id, "NMD")
    meso: dict = {"detections": []}
    if key is None:
        meso["note"] = "no recent NMD (mesocyclone) product"
    else:
        f = _parse(key)
        t = _key_time(key)
        meso["time_utc"] = t.isoformat()
        meso["age_minutes"] = _age_minutes(t)
        found = []
        clat = f.prod_desc.lat / 1000.0
        clon = f.prod_desc.lon / 1000.0
        for layer in getattr(f, "sym_block", None) or []:
            for item in layer:
                if isinstance(item, dict) and "x" in item and "y" in item:
                    la, lo = _km_offset_to_latlon(clat, clon,
                                                  item["x"], item["y"])
                    d = {"lat": la, "lon": lo}
                    for k in ("id", "type", "radius"):
                        if k in item:
                            d[k] = item[k]
                    found.append(d)
        meso["detections"] = found
        if not found:
            meso["note"] = ("product present but contains no detections "
                            "(no mesocyclones currently identified)")
    out["mesocyclones"] = meso

    out["tvs"] = {"available": False, "note": RETIRED_PRODUCTS["NTV"]}
    out["hail"] = {"available": False, "note": RETIRED_PRODUCTS["NHI"]}
    return out
