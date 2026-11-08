"""Stage 2: the point-in-time warehouse (DuckDB).

Every table carries `available_at` (UTC): the earliest moment we could have known
that row. Feature code never reads base tables directly. It opens a
`PointInTime(con, t)` context, which exposes `pit_<table>` views filtered to
`available_at < t`. That is the system's one invariant, enforced in one place.

Timestamp conventions (conservative: when unsure, make data available LATER):
  games / possessions / stints   tip + RESULT_DELAY, or the next morning if no tip time
  schedule (tip times)           known well in advance: tip - 30 days
  officials                      09:00 ET on game day (when the NBA posts crews)
  odds 'open' (SBR history)      tip - HISTORICAL_OPEN_LINE_LEAD (true post time unknown)
  odds 'close' (SBR history)     tip (never usable before the game starts)
  odds 'snapshot'                the moment we fetched it
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator
from zoneinfo import ZoneInfo

import duckdb

from . import config

ET = ZoneInfo("America/New_York")

SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    game_id VARCHAR PRIMARY KEY, season VARCHAR, season_type VARCHAR, game_date DATE,
    home_team_id BIGINT, away_team_id BIGINT, home_pts INTEGER, away_pts INTEGER,
    tip_utc TIMESTAMPTZ, available_at TIMESTAMPTZ NOT NULL
);
CREATE TABLE IF NOT EXISTS team_games (
    game_id VARCHAR, team_id BIGINT, opp_team_id BIGINT, is_home BOOLEAN,
    pts INTEGER, fga INTEGER, fta INTEGER, oreb INTEGER, tov INTEGER,
    available_at TIMESTAMPTZ NOT NULL
);
CREATE TABLE IF NOT EXISTS schedule (
    game_id VARCHAR PRIMARY KEY, game_date DATE, home_team_id BIGINT, away_team_id BIGINT,
    tip_utc TIMESTAMPTZ, available_at TIMESTAMPTZ NOT NULL
);
CREATE TABLE IF NOT EXISTS officials (
    game_id VARCHAR, official_id BIGINT, name VARCHAR, available_at TIMESTAMPTZ NOT NULL
);
CREATE TABLE IF NOT EXISTS odds_lines (
    game_id VARCHAR, sbr_game_id VARCHAR, sportsbook VARCHAR, line_type VARCHAR,
    total DOUBLE, over_odds INTEGER, under_odds INTEGER,
    observed_at TIMESTAMPTZ, available_at TIMESTAMPTZ NOT NULL
);
CREATE TABLE IF NOT EXISTS possessions (
    game_id VARCHAR, period INTEGER, idx INTEGER, offense_team_id BIGINT, defense_team_id BIGINT,
    start_clock_s DOUBLE, end_clock_s DOUBLE, points INTEGER, fga INTEGER, fta INTEGER,
    turnovers INTEGER, is_empty BOOLEAN, offense_lineup BIGINT[], defense_lineup BIGINT[],
    available_at TIMESTAMPTZ NOT NULL
);
CREATE TABLE IF NOT EXISTS stints (
    game_id VARCHAR, period INTEGER, team_a BIGINT, team_b BIGINT,
    lineup_a BIGINT[], lineup_b BIGINT[], start_clock_s DOUBLE, end_clock_s DOUBLE,
    seconds DOUBLE, points_a INTEGER, points_b INTEGER, poss_a INTEGER, poss_b INTEGER,
    available_at TIMESTAMPTZ NOT NULL
);
"""

GUARDED_TABLES = ("games", "team_games", "schedule", "officials", "odds_lines", "possessions", "stints")


def connect(path: Path | str | None = None) -> duckdb.DuckDBPyConnection:
    path = Path(path) if path else config.WAREHOUSE_PATH
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute("SET TimeZone='UTC'")
    con.execute(SCHEMA)
    return con


# --- timestamp rules ------------------------------------------------------------
def result_available_at(game_date, tip_utc: datetime | None) -> datetime:
    if tip_utc is not None:
        return tip_utc + config.RESULT_DELAY
    # no tip time: every NBA game is over by 06:00 ET the next day
    nxt = datetime.combine(game_date + timedelta(days=1), datetime.min.time(), ET).replace(hour=6)
    return nxt.astimezone(timezone.utc)


def officials_available_at(game_date) -> datetime:
    return datetime.combine(game_date, datetime.min.time(), ET).replace(
        hour=config.OFFICIALS_POSTED_HOUR_ET).astimezone(timezone.utc)


def close_available_at(game_date, tip_utc: datetime | None) -> datetime:
    if tip_utc is not None:
        return tip_utc
    return datetime.combine(game_date + timedelta(days=1), datetime.min.time(), ET).astimezone(timezone.utc)


def open_available_at(game_date, tip_utc: datetime | None) -> datetime:
    if tip_utc is not None:
        return tip_utc - config.HISTORICAL_OPEN_LINE_LEAD
    # no tip time: assume the opener is known by noon ET on game day
    return datetime.combine(game_date, datetime.min.time(), ET).replace(hour=12).astimezone(timezone.utc)


# --- point-in-time access -----------------------------------------------------
class LeakageError(RuntimeError):
    pass


@contextmanager
def point_in_time(con: duckdb.DuckDBPyConnection, t: datetime) -> Iterator[duckdb.DuckDBPyConnection]:
    """Expose pit_<table> views containing only rows with available_at < t."""
    if t.tzinfo is None:
        raise ValueError("decision time must be timezone-aware")
    ts = t.astimezone(timezone.utc).isoformat()
    for tbl in GUARDED_TABLES:
        con.execute(f"CREATE OR REPLACE TEMP VIEW pit_{tbl} AS "
                    f"SELECT * FROM {tbl} WHERE available_at < TIMESTAMPTZ '{ts}'")
    try:
        yield con
    finally:
        for tbl in GUARDED_TABLES:
            con.execute(f"DROP VIEW IF EXISTS pit_{tbl}")


def assert_no_future_rows(df, t: datetime, col: str = "available_at") -> None:
    """Belt and braces for feature frames: fail if any row is from the future."""
    if len(df) and (df[col] >= t).any():
        raise LeakageError(f"{int((df[col] >= t).sum())} rows with {col} >= {t.isoformat()}")
