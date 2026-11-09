"""Project-wide settings. Paths are relative to the repo root unless NBADL_DATA is set."""
from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("NBADL_DATA", REPO_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
WAREHOUSE_PATH = DATA_DIR / "warehouse" / "nba.duckdb"

FIRST_SEASON = 2013  # 2013-14: first season with player tracking data
HOLDOUT_SEASONS = 2  # the last N completed seasons are the final test set


def season_label(start_year: int) -> str:
    """2013 -> '2013-14' (the format stats.nba.com expects)."""
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def seasons(last_start_year: int, first: int = FIRST_SEASON) -> list[str]:
    return [season_label(y) for y in range(first, last_start_year + 1)]


# --- point-in-time conventions (see docs/pipeline.html, stage 2) ---------------
# When a piece of information becomes knowable, relative to the scheduled tip-off.
# Conservative on purpose: if unsure, make data available LATER, never earlier.
RESULT_DELAY = timedelta(hours=3)  # box score / play-by-play final ~ tip + 3h
OFFICIALS_POSTED_HOUR_ET = 9  # NBA posts referee crews ~9:00 ET on game day
INJURY_PUBLISH_LAG = timedelta(minutes=15)  # report stamped HH:30 is assumed usable 15 minutes later
HISTORICAL_OPEN_LINE_LEAD = timedelta(hours=12)  # SBR history has no timestamp for the opener

# Throttle for stats.nba.com; it bans aggressive clients.
NBA_REQUEST_INTERVAL_S = 1.2
