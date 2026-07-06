"""
NWS point forecasts (hourly + thunderstorm/gridpoint outlook).

Source: api.weather.gov, the National Weather Service's public API. No API
key required, but it DOES require a descriptive User-Agent header — requests
without one are rejected (same as warnings.py).

Flow: /points/{lat},{lon} resolves a lat/lon to the office (gridId) and grid
cell (gridX, gridY) that covers it, and returns the forecastHourly and
forecastGridData URLs for that cell. Those are then fetched directly.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import requests

_USER_AGENT = "nexrad-mcp (github.com/gm2211/nexrad-mcp)"
_POINTS_URL = "https://api.weather.gov/points/{lat},{lon}"
_TIMEOUT = 15

_MAX_HOURS = 156  # matches the length of the NWS hourly forecast series


def _get_json(url: str, params: dict | None = None) -> dict:
    resp = requests.get(
        url, params=params, headers={"User-Agent": _USER_AGENT}, timeout=_TIMEOUT
    )
    resp.raise_for_status()
    return resp.json()


def _resolve_point(lat: float, lon: float) -> dict:
    """Look up the forecast office/grid cell and product URLs for a point."""
    data = _get_json(_POINTS_URL.format(lat=lat, lon=lon))
    props = data.get("properties", {})
    return {
        "forecastHourly": props.get("forecastHourly"),
        "forecastGridData": props.get("forecastGridData"),
        "gridId": props.get("gridId"),
        "gridX": props.get("gridX"),
        "gridY": props.get("gridY"),
    }


# ISO-8601 duration validTime, e.g. "2026-07-06T04:00:00+00:00/PT3H"
_VALID_TIME_RE = re.compile(
    r"^(?P<start>[^/]+)/P(?:T?)(?P<dur>.+)$"
)
_DURATION_RE = re.compile(
    r"(?:(?P<days>\d+)D)?T?(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?"
)


def _parse_iso8601_duration(dur: str) -> timedelta:
    """Parse the duration portion of an ISO-8601 interval, e.g. 'T3H' -> 3h,
    '3H' -> 3h, '1DT2H' -> 1 day 2 hours."""
    m = re.match(
        r"^(?:(?P<days>\d+)D)?T?(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?"
        r"(?:(?P<seconds>\d+)S)?$",
        dur,
    )
    if not m or not any(m.groups()):
        return timedelta(hours=1)
    parts = m.groupdict(default="0")
    return timedelta(
        days=int(parts["days"]),
        hours=int(parts["hours"]),
        minutes=int(parts["minutes"]),
        seconds=int(parts["seconds"]),
    )


def _expand_value_series(values: list[dict]) -> list[dict]:
    """Expand a gridpoint "values" series (each entry covering a validTime
    interval like "2026-07-06T00:00:00+00:00/PT3H") into one entry per hour.

    Each returned entry: {"time_utc": iso, "value": ...}.
    """
    out = []
    for entry in values:
        vt = entry.get("validTime", "")
        if "/" not in vt:
            continue
        start_str, dur_str = vt.split("/", 1)
        try:
            start = datetime.fromisoformat(start_str)
        except ValueError:
            continue
        # Duration strings look like "PT3H", "PT1H", "P0DT3H", etc.
        dur_body = dur_str[1:] if dur_str.startswith("P") else dur_str
        duration = _parse_iso8601_duration(dur_body)
        hours = max(1, round(duration.total_seconds() / 3600))
        value = entry.get("value")
        for h in range(hours):
            out.append({
                "time_utc": (start + timedelta(hours=h)).isoformat(),
                "value": value,
            })
    return out


def get_hourly_forecast(lat: float, lon: float, hours: int = 12) -> dict:
    """Get the NWS hourly forecast for a point.

    Args:
        lat, lon: point of interest (decimal degrees).
        hours: how many hourly periods to return, starting from the current
            period (default 12). Capped at 156 (the length of the NWS
            hourly series).

    Returns up to `hours` periods, each with: start_time (ISO-8601, local
    office time zone as reported by NWS), temperature (+ unit, typically F),
    wind_speed, wind_direction, wind_gust (only present if the office
    publishes it on this endpoint — most don't; use get_thunder_outlook for
    a gust time series), probability_of_precipitation (percent, may be null),
    and short_forecast (a plain-language one-liner, e.g. "Chance Showers And
    Thunderstorms").
    """
    hours = max(1, min(hours, _MAX_HOURS))
    try:
        point = _resolve_point(lat, lon)
        hourly_url = point.get("forecastHourly")
        if not hourly_url:
            return {"error": "NWS did not return a forecastHourly URL for this point "
                              "(likely outside NWS coverage, e.g. open ocean)."}
        data = _get_json(hourly_url)
    except (requests.RequestException, ValueError) as e:
        return {"error": f"could not fetch forecast from api.weather.gov: {e}"}

    periods = data.get("properties", {}).get("periods", [])[:hours]
    out_periods = []
    for p in periods:
        pop = p.get("probabilityOfPrecipitation", {})
        out_periods.append({
            "start_time": p.get("startTime"),
            "temperature": p.get("temperature"),
            "temperature_unit": p.get("temperatureUnit"),
            "wind_speed": p.get("windSpeed"),
            "wind_direction": p.get("windDirection"),
            "wind_gust": p.get("windGust"),
            "probability_of_precipitation": pop.get("value") if isinstance(pop, dict) else pop,
            "short_forecast": p.get("shortForecast"),
        })

    return {
        "point": {"lat": lat, "lon": lon},
        "grid": {"office": point.get("gridId"), "gridX": point.get("gridX"),
                 "gridY": point.get("gridY")},
        "count": len(out_periods),
        "periods": out_periods,
    }


def get_thunder_outlook(lat: float, lon: float, hours: int = 12) -> dict:
    """Get thunderstorm probability + related gridpoint series for a point.

    Args:
        lat, lon: point of interest (decimal degrees).
        hours: how many hourly buckets to return from now (default 12).
            Capped at 156.

    Pulls the raw NWS gridpoint forecast (forecastGridData) and extracts:
      - probability_of_thunder: percent chance of thunder each hour. NOT
        every forecast office publishes this element — if absent, the field
        is omitted and thunder_probability_available is set to False so
        callers don't mistake missing data for "no thunder risk."
      - wind_gust: forecast wind gust (mph) time series.
      - probability_of_precipitation: percent chance of any precipitation.

    As a rule of thumb, probability_of_thunder >= 30% in the next few hours
    is worth flagging to a non-meteorologist as a real chance of storms;
    >= 60% suggests thunderstorms are more likely than not.
    """
    hours = max(1, min(hours, _MAX_HOURS))
    try:
        point = _resolve_point(lat, lon)
        grid_url = point.get("forecastGridData")
        if not grid_url:
            return {"error": "NWS did not return a forecastGridData URL for this point "
                              "(likely outside NWS coverage, e.g. open ocean)."}
        data = _get_json(grid_url)
    except (requests.RequestException, ValueError) as e:
        return {"error": f"could not fetch gridpoint data from api.weather.gov: {e}"}

    props = data.get("properties", {})
    now = datetime.now(timezone.utc)

    def series_for(key: str) -> list[dict]:
        raw = props.get(key, {})
        values = raw.get("values", []) if isinstance(raw, dict) else []
        expanded = _expand_value_series(values)
        # keep only buckets from the current hour onward, capped at `hours`
        future = [e for e in expanded
                  if datetime.fromisoformat(e["time_utc"]) >= now - timedelta(hours=1)]
        return future[:hours]

    thunder_series = series_for("probabilityOfThunder")
    gust_series = series_for("windGust")
    pop_series = series_for("probabilityOfPrecipitation")

    result = {
        "point": {"lat": lat, "lon": lon},
        "grid": {"office": point.get("gridId"), "gridX": point.get("gridX"),
                 "gridY": point.get("gridY")},
        "thunder_probability_available": bool(thunder_series),
        "wind_gust": {
            "unit": "km/h",
            "series": gust_series,
        },
        "probability_of_precipitation": {
            "unit": "percent",
            "series": pop_series,
        },
    }
    if thunder_series:
        result["probability_of_thunder"] = {
            "unit": "percent",
            "series": thunder_series,
        }
        max_p = max((e["value"] for e in thunder_series if e["value"] is not None),
                    default=None)
        result["summary"] = (
            f"Peak thunder probability in the next {len(thunder_series)}h window: "
            f"{max_p}%." if max_p is not None
            else "Thunder probability series present but empty."
        )
    else:
        result["summary"] = (
            "This forecast office does not publish probabilityOfThunder on "
            "the gridpoint product; only wind gust and precipitation "
            "probability are available here."
        )

    return result
