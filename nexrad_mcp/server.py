"""
NEXRAD radar MCP server.

Exposes raw dual-pol radar analysis (Level II), derived Level 3 products,
and active NWS warnings as MCP tools so an AI client (Claude Desktop, Claude
Code, etc.) can query the actual radar data — not just pre-rendered imagery.

Run directly:   python -m nexrad_mcp.server
Or via the console script:   nexrad-mcp

Tools:
  find_nearest_radar(lat, lon)              -> closest radar sites
  get_latest_scan(site)                     -> newest volume + age
  query_point(site, lat, lon, ...)          -> all products at a point (+SRV/KDP)
  get_vertical_profile(site, lat, lon)      -> all tilts at a point + echo tops
  check_storms_near(site, lat, lon, ...)    -> cores within a radius
  estimate_motion(site, lat, lon, ...)      -> approaching / moving away
  list_l3_products(site)                    -> Level 3 product freshness
  get_l3_value_at_point(site, product, ...) -> VIL / echo tops / precip / HCA
  get_storm_features(site)                  -> storm tracks + mesocyclones
  get_active_warnings(lat, lon)             -> active NWS warnings at a point
  get_hourly_forecast(lat, lon, ...)        -> NWS hourly forecast
  get_thunder_outlook(lat, lon, ...)        -> thunder probability + gust/precip series
  get_lightning_activity(lat, lon, ...)     -> recent GOES GLM flash detections

Sites are 4-letter ICAO IDs, e.g. KLWX (Sterling VA), KOKX (NYC), KFWS (Dallas).
All data is public (NOAA/Unidata/NWS); no API key required.
"""

from __future__ import annotations

from typing import Optional

from mcp.server.fastmcp import FastMCP

from . import radar as R


mcp = FastMCP("nexrad-radar")


@mcp.tool()
def find_nearest_radar(lat: float, lon: float, n: int = 3) -> list[dict]:
    """Find the NEXRAD radar sites nearest to a lat/lon.

    Use this first when you only know a location: it tells you which `site`
    id to pass to the other tools. Returns up to n sites sorted by distance,
    each with its 4-letter id, lat/lon, ground elevation in meters (null for
    a few sites where the table has no value), and distance_km from your
    point. Prefer the closest WSR-88D (ids starting with K/P); ids starting
    with T are TDWR terminal radars near airports — fine for close-range
    detail but they are not the source for the Level 3 tools here.

    Args:
        lat: Latitude of the point of interest (decimal degrees).
        lon: Longitude (decimal degrees, negative for west).
        n: How many sites to return (default 3).
    """
    return R.find_nearest_sites(lat, lon, n)


@mcp.tool()
def get_latest_scan(site: str) -> dict:
    """Get the most recent available Level II volume scan for a radar site.

    Args:
        site: 4-letter radar ID, e.g. "KLWX". Case-insensitive.

    Returns the filename, scan start time (UTC), and how many minutes old it is.
    Use this to confirm data freshness before interpreting other results.
    """
    return R.get_latest_scan(site)


@mcp.tool()
def query_point(site: str, lat: float, lon: float,
                time_utc: Optional[str] = None,
                storm_motion_deg: Optional[float] = None,
                storm_motion_kts: Optional[float] = None,
                include_kdp: bool = False) -> dict:
    """Read all dual-pol radar products at the gate nearest a lat/lon.

    Returns reflectivity (dBZ), radial velocity (m/s), correlation coefficient
    (CC), differential reflectivity (ZDR), spectrum width, and raw differential
    phase (PhiDP, deg), plus a plain-language interpretation. High CC (>=0.97)
    with strong reflectivity is ordinary precipitation; low CC (<0.80)
    co-located with a strong core can indicate a tornado debris signature and
    warrants checking velocity.

    Velocity sign convention: positive = air moving AWAY from the radar,
    negative = TOWARD it.

    Args:
        site: 4-letter radar ID, e.g. "KLWX".
        lat: Latitude of the point of interest (decimal degrees).
        lon: Longitude (decimal degrees, negative for west).
        time_utc: optional ISO-8601 UTC timestamp (e.g. "2026-07-05T21:30:00")
            to query the archived volume nearest that time instead of the
            latest scan — for "what was over my house at 5:30?" questions.
        storm_motion_deg: optional storm motion direction, meteorological
            convention = the direction the storm is moving FROM (e.g. 240
            means moving from the WSW toward the ENE).
        storm_motion_kts: optional storm motion speed in knots. When both
            motion args are given, the result gains storm_relative_velocity
            (m/s): the measured radial velocity minus the storm motion's
            component along the radar-to-point direction. Same sign
            convention as velocity. This unmasks rotation/inflow that fast
            storm translation would otherwise hide.
        include_kdp: if True, also run a KDP (specific differential phase,
            deg/km) retrieval at this gate — adds a few seconds. High KDP
            (>~1-2 deg/km) indicates heavy liquid rain loading.
    """
    return R.query_point(site, lat, lon, time_utc=time_utc,
                         storm_motion_deg=storm_motion_deg,
                         storm_motion_kts=storm_motion_kts,
                         include_kdp=include_kdp)


@mcp.tool()
def get_vertical_profile(site: str, lat: float, lon: float,
                         time_utc: Optional[str] = None) -> dict:
    """Sample every elevation tilt of the radar volume above one point.

    Think of it as drilling a vertical column through the storm over a
    location. For each tilt you get the beam's height (km above sea level)
    and all dual-pol values there; the result also includes:
      - composite_reflectivity: strongest dBZ found at ANY height over the
        point — reveals strong cores aloft that the lowest tilt misses.
      - echo_top_km: highest altitude still >= 18 dBZ (null if none). Tall
        echo tops (>10-12 km in summer) usually mean a vigorous updraft.

    Interpretation tips for non-meteorologists: reflectivity increasing with
    height (strong aloft, weak below) often means hail/heavy rain that has
    not reached the ground yet or an elevated storm; ZDR near 0 dB inside a
    >55 dBZ core aloft is a classic hail signal. Note the beam only samples
    discrete heights — at long range the lowest sample may already be several
    km up. NEXRAD volumes use "split cuts": some tilts carry only
    reflectivity-family fields and adjacent near-identical tilts carry only
    velocity-family fields, so null velocity on one tilt is normal — check
    its twin at nearly the same elevation_deg.

    Args:
        site: 4-letter radar ID, e.g. "KLWX".
        lat, lon: the point of interest (decimal degrees).
        time_utc: optional ISO-8601 UTC timestamp for an archived volume.
    """
    return R.get_vertical_profile(site, lat, lon, time_utc)


@mcp.tool()
def check_storms_near(site: str, lat: float, lon: float,
                      radius_km: float = 60,
                      dbz_threshold: float = 45) -> dict:
    """Find storm cores near a point and report their distance and bearing.

    Scans the lowest tilt for gates at or above dbz_threshold within radius_km
    of the point, clusters them, and lists each core's distance, compass
    bearing FROM the point, and peak dBZ. Good for "is there a storm near me
    and which direction is it?"

    Args:
        site: 4-letter radar ID.
        lat, lon: The point to search around (decimal degrees).
        radius_km: Search radius in km (default 60).
        dbz_threshold: Minimum reflectivity to count as a core (default 45;
            50+ is a strong/severe core, 40 catches moderate rain).
    """
    return R.check_storms_near(site, lat, lon, radius_km, dbz_threshold)


@mcp.tool()
def estimate_motion(site: str, lat: float, lon: float,
                    radius_km: float = 60,
                    dbz_threshold: float = 40) -> dict:
    """Estimate whether the nearest storm is approaching a point or moving away.

    Compares the two most recent volumes, tracking the reflectivity-weighted
    centroid of qualifying echo near the point, and reports closing speed,
    a trend label (APPROACHING / MOVING AWAY / roughly stationary), and an
    ETA when it is meaningfully closing. For NWS-quality cell tracks and
    motion vectors, also see get_storm_features.

    Args:
        site: 4-letter radar ID.
        lat, lon: The point of interest (decimal degrees).
        radius_km: How far out to consider echo (default 60).
        dbz_threshold: Minimum reflectivity to track (default 40).
    """
    return R.estimate_motion(site, lat, lon, radius_km, dbz_threshold)


@mcp.tool()
def list_l3_products(site: str) -> dict:
    """List which derived Level 3 products have fresh data for a radar site.

    Level 3 products are the NWS's precomputed analyses (the same layers
    RadarScope shows): digital VIL, echo tops, rainfall accumulations,
    hydrometeor classification, storm tracks, and mesocyclone detections.
    Each entry reports availability, the latest product time (UTC), and its
    age in minutes; ages under ~10 minutes mean the radar is actively
    producing it. Retired products (TVS, hail index — discontinued by the
    NWS) are listed separately so their absence isn't mistaken for calm
    weather.

    Args:
        site: 4-letter radar ID, e.g. "KLWX".
    """
    from . import level3 as L3
    return L3.list_l3_products(site)


@mcp.tool()
def get_l3_value_at_point(site: str, product: str,
                          lat: float, lon: float) -> dict:
    """Sample a gridded Level 3 product at a lat/lon.

    Products and how to read them:
      - "DVL" digital VIL (kg/m^2): vertically integrated liquid; >~50 often
        signals hail potential, higher = more water/ice suspended aloft.
      - "EET" enhanced echo tops (kft, thousands of feet): storm depth;
        summer thunderstorms commonly 30-45 kft, severe ones 50+ kft. The
        result may include echo_top_capped=true when the true top exceeded
        what the radar could measure.
      - "DAA" one-hour rainfall (inches) and "DTA" storm-total rainfall
        (inches): radar-estimated accumulations (dual-pol QPE).
      - "HHC" hydrometeor classification: a text label of what is falling
        (rain / heavy rain / dry snow / hail / biological / ground clutter
        etc.) rather than a number.

    A null value means no echo/data at that spot (normal in clear air).
    Also returns product time (UTC) and age_minutes — check freshness before
    trusting it.

    Args:
        site: 4-letter radar ID, e.g. "KLWX".
        product: one of "DVL", "EET", "DAA", "DTA", "HHC".
        lat, lon: the point of interest (decimal degrees).
    """
    from . import level3 as L3
    return L3.get_l3_grid_value(site, product, lat, lon)


@mcp.tool()
def get_storm_features(site: str) -> dict:
    """Get NWS-algorithm storm features for a site: tracked cells + mesocyclones.

    storm_cells comes from the storm tracking (STI) product: each detected
    cell has an id (e.g. "Q5"), current lat/lon, its motion (direction it is
    moving FROM in degrees + speed in knots), and past/forecast track points
    ([lat, lon] pairs; forecasts at +15/+30/+45/+60 minutes). This is the
    authoritative "where is that storm going" answer — better than eyeballing
    two frames.

    mesocyclones comes from the mesocyclone detection product: rotation
    signatures with lat/lon. An empty detections list with a fresh product
    time means the algorithm is running and simply found no rotation — a
    genuinely good sign. Any detection near your location during a
    thunderstorm deserves attention (and a check of get_active_warnings).

    TVS (tornado vortex signature) and hail index appear in the response
    only as permanently-unavailable notes: the NWS retired those standalone
    products (no data fleet-wide since before 2025). For tornado threat, use
    mesocyclones + low CC in query_point + get_active_warnings; for hail
    threat, use DVL/EET/HHC via get_l3_value_at_point.

    Args:
        site: 4-letter radar ID, e.g. "KLWX".
    """
    from . import level3 as L3
    return L3.get_l3_storm_features(site)


@mcp.tool()
def get_active_warnings(lat: float, lon: float,
                        all_events: bool = False) -> dict:
    """Get active NWS warnings/watches covering a point (api.weather.gov).

    Returns each alert's event type, headline, severity (Minor/Moderate/
    Severe/Extreme), onset and expiry times, and the counties/zones covered.
    By default only radar-relevant alerts are returned: Tornado Warning,
    Severe Thunderstorm Warning, Flash Flood Warning, Special Weather
    Statement, Tornado Watch, Severe Thunderstorm Watch. An empty list is a
    normal "nothing active here" result, not an error.

    This is the official warning source — radar tools here are analysis
    aids, but act on NWS warnings.

    Args:
        lat, lon: the point to check (decimal degrees).
        all_events: set True to include every active alert type (winter,
            heat, marine, air quality, ...), not just the radar-relevant set.
    """
    from . import warnings as W
    return W.get_active_warnings(lat, lon, all_events)


@mcp.tool()
def get_hourly_forecast(lat: float, lon: float, hours: int = 12) -> dict:
    """Get the NWS hourly forecast for a point (api.weather.gov).

    Returns up to `hours` hourly periods (capped at 156, the full length of
    the NWS hourly series), each with: start_time (ISO-8601), temperature
    (+ unit, typically F), wind_speed, wind_direction, wind_gust (only
    present when the forecast office publishes it on this endpoint — most
    don't; use get_thunder_outlook for a dedicated gust time series),
    probability_of_precipitation (percent, may be null), and short_forecast
    (a plain-language one-liner, e.g. "Chance Showers And Thunderstorms").

    This is a general-purpose forecast, not a radar analysis tool — pair it
    with query_point / check_storms_near for "is it about to storm right
    now" vs. this for "what's expected over the next several hours."

    Args:
        lat, lon: point of interest (decimal degrees).
        hours: how many hourly periods to return (default 12, max 156).
    """
    from . import forecast as F
    return F.get_hourly_forecast(lat, lon, hours)


@mcp.tool()
def get_thunder_outlook(lat: float, lon: float, hours: int = 12) -> dict:
    """Get thunderstorm probability + wind gust + precip probability series.

    Pulls the raw NWS gridpoint forecast (forecastGridData) rather than the
    prose hourly forecast, exposing:
      - probability_of_thunder (percent, hourly buckets): NOT every NWS
        office publishes this element. If absent, it is omitted from the
        result and thunder_probability_available is False — that means
        "unavailable for this office," not "no thunderstorm risk." As a
        rule of thumb for non-meteorologists: >=30% in the next few hours
        is worth flagging, >=60% means thunderstorms are more likely than
        not.
      - wind_gust (km/h, hourly buckets).
      - probability_of_precipitation (percent, hourly buckets).

    Args:
        lat, lon: point of interest (decimal degrees).
        hours: how many hourly buckets to return from now (default 12, max 156).
    """
    from . import forecast as F
    return F.get_thunder_outlook(lat, lon, hours)


@mcp.tool()
def get_lightning_activity(lat: float, lon: float, radius_km: float = 50,
                           minutes: int = 10) -> dict:
    """Check for recent lightning near a point via GOES GLM satellite data.

    Source: NOAA's GOES-19 (GOES-East) or GOES-18 (GOES-West) Geostationary
    Lightning Mapper (GLM), chosen automatically by longitude. GLM detects
    TOTAL lightning — both in-cloud and cloud-to-ground flashes, not just
    cloud-to-ground like older ground-based networks — with roughly 1-2
    minutes of latency from real time. Its field of view covers the Americas
    and adjacent oceans; it will NOT see lightning over e.g. Europe or Asia.

    Returns: flash_count and flashes_per_minute within radius_km over the
    last `minutes`, nearest_flash (distance_km + compass bearing FROM the
    point), most_recent_flash_utc, granules_checked, which satellite was
    used, and a plain-language summary (e.g. "no lightning within 50 km in
    last 10 min" or "active lightning: 12 flashes, nearest 8 km to the SW").
    A flash_count of 0 with granules_checked > 0 is a normal, good result,
    not an error — it means the area was checked and is quiet.

    Args:
        lat, lon: point of interest (decimal degrees).
        radius_km: search radius around the point (default 50).
        minutes: how far back to look, in minutes (default 10, max 30).
    """
    from . import lightning as LT
    return LT.get_lightning_activity(lat, lon, radius_km, minutes)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
