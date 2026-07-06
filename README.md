# nexrad-mcp

An MCP server that gives an AI client (Claude Desktop, Claude Code, any MCP
client) direct access to **raw NEXRAD radar data** — the same data classes a
human interrogates in RadarScope:

- **Level II** dual-pol moments (reflectivity, velocity, CC, ZDR, spectrum
  width, differential phase, opt-in KDP retrieval), queried at any point, at
  any tilt, for the latest volume or any archived timestamp;
- **Level 3** derived products (digital VIL, enhanced echo tops, 1-hour and
  storm-total rainfall, hydrometeor classification, storm cell tracks,
  mesocyclone detections);
- **Active NWS warnings** for a point;
- **NWS hourly forecasts and thunderstorm outlook** for a point;
- **Lightning** detection via GOES GLM satellite data.

Unlike weather MCPs that return pre-rendered radar *images*, this decodes the
actual data with [Py-ART](https://arm-doe.github.io/pyart/) and
[MetPy](https://unidata.github.io/MetPy/), so you can ask questions like
*"what's the CC over my house right now, and is that a debris signature or
just clutter?"* — the same interrogation you'd do by hand in RadarScope.

## Installation

Requires [uv](https://docs.astral.sh/uv/). The first run is slow (a minute or
two): `uvx` resolves and installs Py-ART/MetPy, which pull in scipy and
friends. Subsequent runs start in seconds from uv's cache.

**Claude Code**

```bash
claude mcp add nexrad -- uvx --from git+https://github.com/gm2211/nexrad-mcp nexrad-mcp
```

**Claude Desktop** — add to `claude_desktop_config.json`
(macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "nexrad": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/gm2211/nexrad-mcp", "nexrad-mcp"]
    }
  }
}
```

**Any other MCP client** — configure the server command as:

```bash
uvx --from git+https://github.com/gm2211/nexrad-mcp nexrad-mcp
```

(stdio transport; no API keys or credentials — all data sources are public.)

For development, clone and `uv sync`, then run `uv run nexrad-mcp`.

## Tools

| Tool | What it answers | Source |
|------|-----------------|--------|
| `find_nearest_radar(lat, lon)` | Which radar covers my location? | Py-ART site table |
| `get_latest_scan(site)` | What's the newest volume and how old is it? | Level II |
| `query_point(site, lat, lon, ...)` | What are all products at this exact spot? Optional: at a past time (`time_utc`), storm-relative velocity (`storm_motion_deg/kts`), KDP retrieval (`include_kdp`) | Level II |
| `get_vertical_profile(site, lat, lon, ...)` | What does the storm look like at every height above this spot? Includes composite reflectivity and 18 dBZ echo top | Level II |
| `check_storms_near(site, lat, lon, ...)` | Any cores near me, and which direction? | Level II |
| `estimate_motion(site, lat, lon, ...)` | Is the nearest storm coming toward me? | Level II |
| `list_l3_products(site)` | Which derived products are fresh at this site? | Level 3 |
| `get_l3_value_at_point(site, product, lat, lon)` | VIL / echo tops / rainfall / precip type at this spot | Level 3 |
| `get_storm_features(site)` | NWS-tracked cells with motion + forecast tracks, mesocyclone detections | Level 3 |
| `get_active_warnings(lat, lon)` | Any tornado/severe/flood warnings here right now? | api.weather.gov |
| `get_hourly_forecast(lat, lon, hours=12)` | What's the hourly forecast (temp, wind, precip chance, short description)? | api.weather.gov |
| `get_thunder_outlook(lat, lon, hours=12)` | What's the hourly thunderstorm probability, wind gust, and precip probability outlook? | api.weather.gov |
| `get_lightning_activity(lat, lon, radius_km=50, minutes=10)` | Any lightning near me in the last few minutes? | GOES GLM (NOAA, public) |

`site` is a 4-letter radar ID (e.g. `KLWX` = Sterling VA) — use
`find_nearest_radar` if you don't know it.

## RadarScope data-parity coverage

| RadarScope data class | Surfaced here | Source | Notes |
|---|---|---|---|
| Base reflectivity / velocity / CC / ZDR / spectrum width | ✅ `query_point`, all tilts via `get_vertical_profile` | Level II | velocity auto-falls back to the Doppler split cut |
| Differential phase / KDP | ✅ raw PhiDP always; KDP via `include_kdp=True` (Maesaka retrieval, ~3 s) | Level II | |
| Storm-relative velocity | ✅ `query_point(storm_motion_deg=…, storm_motion_kts=…)` | Level II (computed) | supply storm motion, e.g. from `get_storm_features` |
| Composite reflectivity, echo tops | ✅ `get_vertical_profile` (computed) + `EET` product | Level II + Level 3 | |
| Digital VIL (DVL) | ✅ `get_l3_value_at_point("DVL")` | Level 3 | kg/m² |
| Enhanced echo tops (EET) | ✅ `get_l3_value_at_point("EET")` | Level 3 | kft, with "capped" flag |
| 1-hour / storm-total precip (DAA/DTA) | ✅ `get_l3_value_at_point` | Level 3 | inches, dual-pol QPE |
| Hydrometeor classification (HHC) | ✅ `get_l3_value_at_point("HHC")` | Level 3 | text class (rain/hail/snow/…) |
| Storm tracks (STI) | ✅ `get_storm_features` | Level 3 (NST) | position, motion vector, past + forecast track |
| Mesocyclone (MD) | ✅ `get_storm_features` | Level 3 (NMD) | detections with lat/lon |
| TVS (tornado vortex signature) | ❌ product retired by the NWS | — | no fleet-wide NTV data published since before 2025 (verified in-bucket); use mesocyclones + low-CC debris checks + warnings instead |
| Hail index (HI) | ❌ product retired by the NWS | — | same; hail potential via DVL/EET/HHC |
| NWS warnings/watches | ✅ `get_active_warnings` | api.weather.gov | radar-relevant filter by default |
| Lightning | ✅ `get_lightning_activity` | GOES GLM (NOAA, public) | total lightning (in-cloud + cloud-to-ground), ~1-2 min latency, Americas + adjacent oceans only |

Beyond RadarScope parity — forecasting (RadarScope doesn't do this at all):

| Feature | Surfaced here | Source |
|---|---|---|
| Hourly forecast (temp, wind, precip chance) | ✅ `get_hourly_forecast` | api.weather.gov |
| Thunderstorm probability outlook | ✅ `get_thunder_outlook` | api.weather.gov (gridpoint data) |

## Data sources

- **Level II**: NOAA Open Data volumes on AWS S3, discovered via the
  `nexradaws` index. Decoded with Py-ART.
- **Level 3**: the public `unidata-nexrad-level3` S3 bucket (anonymous
  access). Decoded with MetPy; scaling cross-checked against Py-ART.
- **Warnings, hourly forecast, thunderstorm outlook**: `api.weather.gov`
  (requires only a User-Agent header).
- **Lightning**: GOES GLM (Geostationary Lightning Mapper) Level 2 data on
  the public `noaa-goes19` (GOES-East) and `noaa-goes18` (GOES-West) S3
  buckets, anonymous access. Decoded with netCDF4.

## Try it without MCP

```bash
uv run python -c "from nexrad_mcp import radar as R; \
import json; print(json.dumps(R.query_point('KLWX', 38.905, -78.235), indent=2))"
```

## How the polling works

Each call lists the current UTC day's volumes for the site via the `nexradaws`
index (the main archive bucket blocks anonymous listing, so we go through the
index rather than raw S3), grabs the newest `*_V06` key, downloads it, and
decodes it. A completed volume in VCP 212 lands every ~4–6 minutes, so
re-querying more often than ~30–60s just returns the same file. Volume loads
are LRU-cached so repeated queries on one volume don't re-decode. Level 3
products update on a similar cadence and are fetched by day-prefixed key
listing (the bucket is a flat namespace, `SSS_PPP_YYYY_MM_DD_HH_MM_SS`).

## Caveats

- **Not for life-safety.** This is an analysis aid. For warnings, always use
  NWS / official sources (`get_active_warnings` surfaces exactly those). The
  tools return numbers; interpreting a debris signature vs. clutter still
  needs judgment (which is exactly why it pairs well with an LLM that can
  weigh reflectivity + velocity + CC together).
- Beam height rises with range: at 100+ km the lowest tilt is well above
  ground, so a clean surface reading is best within ~60–80 km of the radar.
  `get_vertical_profile` reports the actual beam height of every sample.
- Radar-estimated rainfall (DAA/DTA) is an estimate; gauges beat radar.

## License

MIT. NEXRAD data is public domain (NOAA); Level 3 mirror courtesy of Unidata.
