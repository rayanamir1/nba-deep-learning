"""Stage 2 transforms: raw lake -> point-in-time warehouse tables.

Each loader rebuilds its table from scratch (DELETE + INSERT), so a rebuild after a
bug fix is always safe. Loaders never call the network.
"""
from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta, timezone

import duckdb
import pandas as pd

from . import warehouse as wh
from .ingest.nba_stats import result_set
from .ingest.sbr import parse_odds_tables
from .lake import RawLake
from .pbp_v3 import events_from_v3, roster_from_summary_v3, team_ids, unresolved_subs
from .possessions import parse_game

log = logging.getLogger(__name__)


def _replace(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame) -> int:
    con.execute(f"DELETE FROM {table}")
    if len(df):
        con.register("_df", df)
        cols = ", ".join(df.columns)
        con.execute(f"INSERT INTO {table} ({cols}) SELECT {cols} FROM _df")
        con.unregister("_df")
    return len(df)


# --- schedule ---------------------------------------------------------------------
def _walk_games(obj):
    """Yield every dict that has a gameId (ScheduleLeagueV2's nesting varies by season)."""
    if isinstance(obj, dict):
        if "gameId" in obj:
            yield obj
        for v in obj.values():
            yield from _walk_games(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_games(v)


def _parse_utc(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def load_schedule(con, lake: RawLake) -> int:
    rows = []
    for rec in lake.iter("schedule"):
        for g in _walk_games(rec.payload):
            tip = _parse_utc(g.get("gameDateTimeUTC"))
            home = (g.get("homeTeam") or {}).get("teamId")
            away = (g.get("awayTeam") or {}).get("teamId")
            gdate = _parse_utc(g.get("gameDateEst") or g.get("gameDateUTC"))
            if tip is None or gdate is None:
                continue
            rows.append({"game_id": str(g["gameId"]), "game_date": gdate.date(), "home_team_id": home,
                         "away_team_id": away, "tip_utc": tip, "available_at": tip - timedelta(days=30)})
    df = pd.DataFrame(rows).drop_duplicates("game_id") if rows else pd.DataFrame()
    return _replace(con, "schedule", df)


# --- games ------------------------------------------------------------------------
def load_games(con, lake: RawLake) -> int:
    tips = dict(con.execute("SELECT game_id, tip_utc FROM schedule").fetchall())
    rows = []
    for rec in lake.iter("games"):
        season, stype = rec.key.rsplit("_", 1)
        by_game: dict[str, dict] = {}
        for r in result_set(rec.payload, "LeagueGameFinderResults"):
            g = by_game.setdefault(r["GAME_ID"], {"game_id": r["GAME_ID"], "season": season, "season_type": stype,
                                                  "game_date": date.fromisoformat(r["GAME_DATE"][:10])})
            side = "home" if " vs. " in r["MATCHUP"] else "away"
            g[f"{side}_team_id"] = r["TEAM_ID"]
            g[f"{side}_pts"] = r["PTS"]
        for g in by_game.values():
            if "home_team_id" not in g or "away_team_id" not in g:
                log.warning("game %s missing a side; skipped", g["game_id"])
                continue
            tip = tips.get(g["game_id"])
            g["tip_utc"] = tip
            g["available_at"] = wh.result_available_at(g["game_date"], tip)
            rows.append(g)
    return _replace(con, "games", pd.DataFrame(rows))


# --- officials ----------------------------------------------------------------------
def load_officials(con, lake: RawLake) -> int:
    dates = dict(con.execute("SELECT game_id, game_date FROM games").fetchall())
    rows = []
    for rec in lake.iter("summary_v3"):
        gd = dates.get(rec.key)
        if gd is None:
            continue
        for o in (rec.payload.get("boxScoreSummary") or {}).get("officials") or []:
            rows.append({"game_id": rec.key, "official_id": o["personId"],
                         "name": o.get("name") or f'{o.get("firstName", "")} {o.get("familyName", "")}'.strip(),
                         "available_at": wh.officials_available_at(gd)})
    return _replace(con, "officials", pd.DataFrame(rows))


# --- possessions and stints ---------------------------------------------------------
def load_pbp(con, lake: RawLake) -> dict[str, int]:
    games = {r[0]: (r[1], r[2]) for r in con.execute("SELECT game_id, available_at, game_date FROM games").fetchall()}
    poss_rows, stint_rows = [], []
    stats = {"games": 0, "incomplete_lineups": 0, "unresolved_subs": 0, "games_without_roster": 0,
             "score_mismatch": 0}
    for rec in lake.iter("pbp_v3"):
        if rec.key not in games:
            continue
        avail = games[rec.key][0]
        roster = None
        if lake.exists("summary_v3", rec.key):
            roster = roster_from_summary_v3(lake.read("summary_v3", rec.key).payload)
        else:
            stats["games_without_roster"] += 1
        events = events_from_v3(rec.payload, roster)
        if not events:
            continue
        teams = team_ids(events)
        parsed = parse_game(events, teams)
        stats["games"] += 1
        stats["incomplete_lineups"] += int(not parsed.lineup_complete)
        stats["unresolved_subs"] += unresolved_subs(events)
        for p in parsed.possessions:
            poss_rows.append({
                "game_id": p.game_id, "period": p.period, "idx": p.index, "offense_team_id": p.offense_team_id,
                "defense_team_id": next(t for t in teams if t != p.offense_team_id),
                "start_clock_s": p.start_clock_s, "end_clock_s": p.end_clock_s, "points": p.points,
                "fga": p.fga, "fta": p.fta, "turnovers": p.turnovers, "is_empty": p.is_empty,
                "offense_lineup": list(p.offense_lineup), "defense_lineup": list(p.defense_lineup),
                "available_at": avail})
        a, b = teams
        for s in parsed.stints:
            stint_rows.append({
                "game_id": s.game_id, "period": s.period, "team_a": a, "team_b": b,
                "lineup_a": list(s.lineup[a]), "lineup_b": list(s.lineup[b]),
                "start_clock_s": s.start_clock_s, "end_clock_s": s.end_clock_s, "seconds": s.seconds,
                "points_a": s.points.get(a, 0), "points_b": s.points.get(b, 0),
                "poss_a": s.possessions.get(a, 0), "poss_b": s.possessions.get(b, 0), "available_at": avail})
    _replace(con, "possessions", pd.DataFrame(poss_rows))
    _replace(con, "stints", pd.DataFrame(stint_rows))
    # cross-check against the official final score
    stats["score_mismatch"] = con.execute("""
        SELECT count(*) FROM (
          SELECT g.game_id, g.home_pts + g.away_pts AS official, sum(s.points_a + s.points_b) AS parsed
          FROM games g JOIN stints s USING (game_id) GROUP BY ALL) WHERE official <> parsed
    """).fetchone()[0]
    return stats


# --- odds -------------------------------------------------------------------------
def _norm(s: str) -> str:
    s = s.lower().replace("l.a.", "los angeles").replace("la ", "los angeles ")
    return re.sub(r"[^a-z0-9 ]", "", s).strip()


SBR_ABBREV_ALIASES = {"BK": "BKN", "GS": "GSW", "NO": "NOP", "NY": "NYK", "PHO": "PHX", "SA": "SAS"}


def team_lookup() -> dict[str, int]:
    from nba_api.stats.static import teams

    out: dict[str, int] = {}
    for t in teams.get_teams():
        for name in (t["full_name"], t["nickname"], t["abbreviation"], f'{t["city"]} {t["nickname"]}'):
            out[_norm(name)] = t["id"]
        out.setdefault(_norm(t["city"]), t["id"])  # 'los angeles' is ambiguous; nickname wins
    out[_norm("LA Clippers")] = out[_norm("Los Angeles Clippers")]
    # SBR's short codes that differ from the NBA's
    for sbr_code, nba_code in SBR_ABBREV_ALIASES.items():
        out[_norm(sbr_code)] = out[_norm(nba_code)]
    out.pop("los angeles", None)
    return out


def match_game(by_day: dict, by_pair: dict, gd: date, home: int | None, away: int | None):
    """(game_id, tip) for an SBR line. Exact (date, home) first; else the single game between the same two
    teams within one day (SBR keeps a postponed game's original date)."""
    hit = by_day.get((gd, home))
    if hit is not None or home is None or away is None:
        return hit if hit is not None else (None, None)
    near = [v for d, v in by_pair.get((home, away), []) if abs((d - gd).days) <= 1]
    return near[0] if len(near) == 1 else (None, None)


def load_odds_history(con, lake: RawLake) -> dict[str, int]:
    lookup = team_lookup()
    # schedule covers games not played yet (live snapshots); games covers seasons without a schedule pull
    sched = con.execute("""
        SELECT game_id, game_date, home_team_id, away_team_id, tip_utc FROM schedule
        UNION ALL
        SELECT game_id, game_date, home_team_id, away_team_id, tip_utc FROM games
        WHERE game_id NOT IN (SELECT game_id FROM schedule)""").df()
    # .df() yields pandas Timestamps; normalise to date so lookups by datetime.date hit
    key, by_pair = {}, {}
    for r in sched.itertuples():
        d = pd.Timestamp(r.game_date).date()
        key[(d, r.home_team_id)] = (r.game_id, r.tip_utc)
        by_pair.setdefault((r.home_team_id, r.away_team_id), []).append((d, (r.game_id, r.tip_utc)))
    rows, unmatched = [], 0
    for source, line_types in (("sbr_history", ("open", "close")), ("sbr_snapshots", ("snapshot",))):
        for rec in lake.iter(source):
            gd = date.fromisoformat(rec.key[:10])
            for ln in parse_odds_tables(rec.payload):
                home, away = lookup.get(_norm(ln.home_team)), lookup.get(_norm(ln.away_team))
                gid, tip = match_game(key, by_pair, gd, home, away)
                if gid is None:
                    unmatched += 1
                for lt in line_types:
                    if lt == "open":
                        total, o, u, avail, obs = ln.open_total, ln.open_over_odds, ln.open_under_odds, wh.open_available_at(gd, tip), None
                    elif lt == "close":
                        total, o, u, avail, obs = ln.current_total, ln.current_over_odds, ln.current_under_odds, wh.close_available_at(gd, tip), None
                    else:
                        total, o, u, avail, obs = ln.current_total, ln.current_over_odds, ln.current_under_odds, rec.fetched_at, rec.fetched_at
                    if total is None:
                        continue
                    rows.append({"game_id": gid, "sbr_game_id": ln.sbr_game_id, "sportsbook": ln.sportsbook,
                                 "line_type": lt, "total": total, "over_odds": o, "under_odds": u,
                                 "observed_at": obs, "available_at": avail})
    n = _replace(con, "odds_lines", pd.DataFrame(rows))
    return {"rows": n, "unmatched_lines": unmatched}


def build_all(con, lake: RawLake) -> dict:
    return {
        "schedule": load_schedule(con, lake),
        "games": load_games(con, lake),
        "officials": load_officials(con, lake),
        "pbp": load_pbp(con, lake),
        "odds": load_odds_history(con, lake),
    }
