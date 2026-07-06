"""
NWS active alerts (warnings/watches) lookup.

Source: api.weather.gov, the National Weather Service's public API. No API
key required, but it DOES require a descriptive User-Agent header — requests
without one are rejected.
"""

from __future__ import annotations

import requests

_USER_AGENT = "nexrad-mcp (github.com/gm2211/nexrad-mcp)"
_BASE_URL = "https://api.weather.gov/alerts/active"

# Event types most relevant to someone interrogating radar for
# thunderstorm/tornado threats. `all_events=True` bypasses this filter.
_RADAR_RELEVANT_EVENTS = {
    "Tornado Warning",
    "Severe Thunderstorm Warning",
    "Flash Flood Warning",
    "Special Weather Statement",
    "Tornado Watch",
    "Severe Thunderstorm Watch",
}


def get_active_warnings(lat: float, lon: float, all_events: bool = False) -> dict:
    """Get active NWS alerts (warnings/watches/statements) covering a point.

    Args:
        lat, lon: point to check (decimal degrees).
        all_events: if False (default), only returns the event types most
            relevant to radar interrogation: Tornado Warning, Severe
            Thunderstorm Warning, Flash Flood Warning, Special Weather
            Statement, Tornado Watch, Severe Thunderstorm Watch. If True,
            returns every active alert type the NWS API reports for the
            point (winter weather, marine, heat, etc.).

    Returns a list of alerts, each with: event (alert type), headline
    (human-readable one-liner), severity (Extreme/Severe/Moderate/Minor/
    Unknown), onset (when it started/starts, ISO-8601), expires (ISO-8601),
    and areaDesc (the counties/zones it covers). An empty list means no
    active alerts of the requested kind cover this point right now — that is
    a normal, good result, not an error.
    """
    try:
        resp = requests.get(
            _BASE_URL,
            params={"point": f"{lat},{lon}"},
            headers={"User-Agent": _USER_AGENT},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as e:
        return {"error": f"could not fetch alerts from api.weather.gov: {e}"}

    alerts = []
    for feature in data.get("features", []):
        p = feature.get("properties", {})
        event = p.get("event")
        if not all_events and event not in _RADAR_RELEVANT_EVENTS:
            continue
        alerts.append({
            "event": event,
            "headline": p.get("headline"),
            "severity": p.get("severity"),
            "onset": p.get("onset"),
            "expires": p.get("expires"),
            "areaDesc": p.get("areaDesc"),
        })

    return {
        "point": {"lat": lat, "lon": lon},
        "all_events": all_events,
        "count": len(alerts),
        "alerts": alerts,
    }
