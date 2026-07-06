"""
NEXRAD Level II MCP server.

Exposes raw dual-pol radar analysis as MCP tools so an AI client (Claude
Desktop, Claude Code, etc.) can query the actual radar volume — not just
pre-rendered imagery.

Run directly:   python -m nexrad_mcp.server
Or via the console script:   nexrad-mcp

Tools:
  get_latest_scan(site)                     -> newest volume + age
  query_point(site, lat, lon)               -> all products at a point
  check_storms_near(site, lat, lon, ...)    -> cores within a radius
  estimate_motion(site, lat, lon, ...)      -> approaching / moving away

Sites are 4-letter ICAO IDs, e.g. KLWX (Sterling VA), KOKX (NYC), KFWS (Dallas).
Data is public NOAA NEXRAD Level II on AWS; no API key required.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from . import radar as R

mcp = FastMCP("nexrad-level2")


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
def query_point(site: str, lat: float, lon: float) -> dict:
    """Read all dual-pol radar products at the gate nearest a lat/lon.

    Returns reflectivity (dBZ), radial velocity (m/s), correlation coefficient
    (CC), differential reflectivity (ZDR), and spectrum width, plus a plain-
    language interpretation. High CC (>=0.97) with strong reflectivity is
    ordinary precipitation; low CC (<0.80) co-located with a strong core can
    indicate a tornado debris signature and warrants checking velocity.

    Args:
        site: 4-letter radar ID, e.g. "KLWX".
        lat: Latitude of the point of interest (decimal degrees).
        lon: Longitude (decimal degrees, negative for west).
    """
    return R.query_point(site, lat, lon)


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
    ETA when it is meaningfully closing.

    Args:
        site: 4-letter radar ID.
        lat, lon: The point of interest (decimal degrees).
        radius_km: How far out to consider echo (default 60).
        dbz_threshold: Minimum reflectivity to track (default 40).
    """
    return R.estimate_motion(site, lat, lon, radius_km, dbz_threshold)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
