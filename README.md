# nba-deep-learning

A point-in-time NBA **game-totals** model. The design is in [`docs/pipeline.html`](docs/pipeline.html): 13 stages from raw data to a calibrated P(over), all evaluated walk-forward against the closing line.

This branch adds the **data layer (stages 1 to 3)**.

| Stage | Module | What it does |
| --- | --- | --- |
| 1 · Sources | `nbadl/ingest/nba_stats.py`, `nbadl/ingest/sbr.py` | Pulls games, schedule, play-by-play and officials from stats.nba.com (via `nba_api`), and game totals from SportsbookReview. Throttled and resumable. |
| 2 · Store | `nbadl/lake.py`, `nbadl/warehouse.py`, `nbadl/build.py` | An append-only raw lake (gzipped JSON plus a manifest) and a DuckDB warehouse where every row has `available_at`. Features read through `point_in_time(con, t)`, which only exposes rows with `available_at < t`. |
| 3 · Possessions | `nbadl/possessions.py`, `nbadl/pbp_v2.py` | Turns play-by-play into possessions (based on which team has the ball) and 10-player stints. Handles and-ones, dead-ball rebounds, and technical and flagrant free throws. Checks its point totals against the official score. |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest            # 14 tests: possession parser, leakage invariant, parsers, lake
```

## Pulling data

Run these from a home connection. stats.nba.com usually refuses cloud and datacenter IPs.

```bash
nbadl pull-games --first 2013 --last 2025          # ~40 requests
nbadl pull-games-detail --season 2013-14            # ~2 requests per game, about 1 hour per season
nbadl pull-odds --start 2013-10-29 --end 2014-06-20 # one SBR page per date
nbadl build                                         # prints data-quality stats
```

Run `nbadl snapshot-odds` on a schedule on game days (for example every 30 minutes from 10:00 ET). Each run saves a timestamped copy of the current lines, which builds our own intraday line history from now on.

## Not yet checked against live data

The container this was written in can't reach the NBA or SBR sites. All the logic is unit-tested on synthetic data, but these points need checking on the first real pull:

- **SBR page shape.** The JSON path `props.pageProps.oddsTables[].oddsTableModel.gameRows[]` comes from open-source scrapers. The parser raises `SbrShapeError` if it doesn't match, rather than returning nothing.
- **PlayByPlayV2 availability.** V2 has explicit substitution in/out columns. If stats.nba.com has retired it for some seasons, we'll add a V3 adapter.
- **Period starters.** A player who plays a whole period without recording any event can't be seen in play-by-play. `build` reports `incomplete_lineups`, and we'll fill those from box scores by period if the count is high.
- **SBR team names.** `build` reports `unmatched_lines`.
