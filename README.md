# nexrad-mcp

An MCP server that gives an AI client (Claude Desktop, Claude Code) direct
access to **raw NEXRAD Level II dual-pol radar data** — reflectivity, velocity,
correlation coefficient (CC), differential reflectivity (ZDR), spectrum width —
queried at any point, straight from NOAA's free public archive on AWS.

Unlike weather MCPs that return pre-rendered radar *images*, this decodes the
actual volume with [Py-ART](https://arm-doe.github.io/pyart/), so you can ask
questions like *"what's the CC over my house right now, and is that a debris
signature or just clutter?"* — the same interrogation you'd do by hand in
RadarScope.

## Why it exists

Public radar apps render the data beautifully but don't let a model reason over
the numbers. The raw feed is free and keyless; the only missing piece was a
tool layer. That's this.

## Tools

| Tool | What it answers |
|------|-----------------|
| `get_latest_scan(site)` | What's the newest volume and how old is it? |
| `query_point(site, lat, lon)` | What are all products at this exact spot? |
| `check_storms_near(site, lat, lon, radius_km, dbz_threshold)` | Any cores near me, and which direction? |
| `estimate_motion(site, lat, lon, radius_km, dbz_threshold)` | Is the nearest storm coming toward me? |

`site` is a 4-letter radar ID (e.g. `KLWX` = Sterling VA). Find yours at
<https://www.roc.noaa.gov/branches/RDA/wsr88d.php>.

## Install

```bash
git clone <your-repo-url> nexrad-mcp
cd nexrad-mcp
pip install -e .
```

First call downloads a ~10–15 MB volume and imports Py-ART, so it takes a few
seconds; subsequent calls on the same volume are cached.

## Wire it into Claude Desktop

Add to `claude_desktop_config.json`
(macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "nexrad": {
      "command": "nexrad-mcp"
    }
  }
}
```

Or with an explicit interpreter path:

```json
{
  "mcpServers": {
    "nexrad": {
      "command": "python",
      "args": ["-m", "nexrad_mcp.server"]
    }
  }
}
```

Restart Claude Desktop and the four tools appear.

## Try it without MCP

```bash
python -c "from nexrad_mcp import radar as R; \
import json; print(json.dumps(R.query_point('KLWX', 38.905, -78.235), indent=2))"
```

## How the polling works

Each call lists the current UTC day's volumes for the site via the `nexradaws`
index (the main archive bucket blocks anonymous listing, so we go through the
index rather than raw S3), grabs the newest `*_V06` key, downloads it, and
decodes the 0.5° sweep. A completed volume in VCP 212 lands every ~4–6 minutes,
so re-querying more often than ~30–60s just returns the same file. `_load_sweep0`
is LRU-cached so repeated queries on one volume don't re-decode.

For sub-minute liveness you'd move to the real-time *chunk* feed
(`unidata-nexrad-level2-chunks`) plus SNS push, which reassembles partial sweeps
mid-scan — more complex, and not needed for point queries.

## Caveats

- **Not for life-safety.** This is an analysis aid. For warnings, always use
  NWS / official sources. The tool surfaces numbers; interpreting a debris
  signature vs. clutter still needs judgment (which is exactly why it pairs well
  with an LLM that can weigh reflectivity + velocity + CC together).
- 0.5° tilt only. Higher tilts and derived products (SRV, rotation) are a
  straightforward extension.
- Beam height rises with range: at 100+ km the beam is well above ground, so a
  clean surface reading is best within ~60–80 km of the radar.

## License

MIT. NEXRAD data is public domain (NOAA).
