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

from . import config
from . import warehouse as wh
from .ingest.nba_stats import result_set
from .ingest.sbr import parse_odds_tables
from .lake import RawLake
from .ingest.injury import full_name, parse_report
from .pbp_v3 import _Names, events_from_v3, roster_from_summary_v3, team_ids, unresolved_subs
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


def home_team_id(sides: list[dict], scheduled_home: int | None) -> int | None:
    """Home team of a game from its two LeagueGameFinder rows. Neutral-site games (Paris, Mexico City, Cup
    finals) print '@' for both teams, so fall back to the schedule's designated home team."""
    vs = [r["TEAM_ID"] for r in sides if " vs. " in r["MATCHUP"]]
    return vs[0] if len(vs) == 1 else scheduled_home


# --- games ------------------------------------------------------------------------
def load_games(con, lake: RawLake) -> int:
    tips = dict(con.execute("SELECT game_id, tip_utc FROM schedule").fetchall())
    sched_home = dict(con.execute("SELECT game_id, home_team_id FROM schedule").fetchall())
    rows = []
    for rec in lake.iter("games"):
        season, stype = rec.key.rsplit("_", 1)
        by_game: dict[str, list[dict]] = {}
        for r in result_set(rec.payload, "LeagueGameFinderResults"):
            by_game.setdefault(r["GAME_ID"], []).append(r)
        for gid, sides in by_game.items():
            home = home_team_id(sides, sched_home.get(gid))
            if len(sides) != 2 or home is None:
                log.warning("game %s missing a side; skipped", gid)
                continue
            h = next(r for r in sides if r["TEAM_ID"] == home)
            a = next(r for r in sides if r["TEAM_ID"] != home)
            gdate = date.fromisoformat(h["GAME_DATE"][:10])
            tip = tips.get(gid)
            rows.append({"game_id": gid, "season": season, "season_type": stype, "game_date": gdate,
                         "home_team_id": h["TEAM_ID"], "away_team_id": a["TEAM_ID"],
                         "home_pts": h["PTS"], "away_pts": a["PTS"], "tip_utc": tip,
                         "available_at": wh.result_available_at(gdate, tip)})
    return _replace(con, "games", pd.DataFrame(rows).drop_duplicates("game_id"))


# --- team box lines -----------------------------------------------------------------
def load_team_games(con, lake: RawLake) -> int:
    """One row per team per game with the box-score columns possession estimates need."""
    avail = dict(con.execute("SELECT game_id, available_at FROM games").fetchall())
    sched_home = dict(con.execute("SELECT game_id, home_team_id FROM schedule").fetchall())
    rows = []
    for rec in lake.iter("games"):
        by_game: dict[str, list[dict]] = {}
        for r in result_set(rec.payload, "LeagueGameFinderResults"):
            if r["GAME_ID"] in avail and r["PTS"] is not None:
                by_game.setdefault(r["GAME_ID"], []).append(r)
        for gid, sides in by_game.items():
            if len(sides) != 2:
                continue
            home = home_team_id(sides, sched_home.get(gid))
            for r, o in (sides, sides[::-1]):
                rows.append({"game_id": gid, "team_id": r["TEAM_ID"], "opp_team_id": o["TEAM_ID"],
                             "is_home": r["TEAM_ID"] == home, "pts": r["PTS"], "fga": r["FGA"], "fta": r["FTA"],
                             "oreb": r["OREB"], "tov": r["TOV"], "available_at": avail[gid]})
    return _replace(con, "team_games", pd.DataFrame(rows))


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


def roster_index(lake: RawLake) -> tuple[_Names, _Names]:
    """(per-team, league-wide) name indexes from every game's roster. The league-wide one is the fallback for a
    player reported under a team he has not yet played a game for (just traded)."""
    by_team, league = _Names(), _Names()
    for rec in lake.iter("summary_v3"):
        for team, players in roster_from_summary_v3(rec.payload).items():
            for p in players:
                if p.get("personId"):
                    by_team.add_roster_player(team, p)
                    league.add_roster_player(0, p)
    return by_team, league


def load_injuries(con, lake: RawLake) -> dict[str, int]:
    """Parse every stored injury report into one row per player listing, matched to player and game ids."""
    by_team, league = roster_index(lake)
    games = con.execute("SELECT game_id, game_date, home_team_id, away_team_id FROM games").df()
    by_day_team: dict = {}
    for r in games.itertuples(index=False):
        d = pd.Timestamp(r.game_date).date()
        by_day_team[(d, r.home_team_id)] = by_day_team[(d, r.away_team_id)] = r.game_id
    rows, stats = [], {"reports": 0, "rows": 0, "unmatched_players": 0, "unmatched_games": 0}
    for rec in lake.iter("injury"):
        if rec.payload.get("missing"):
            continue
        rep = parse_report(rec.payload["pages"])
        if rep["report_at"] is None:
            continue
        stats["reports"] += 1
        avail = rep["report_at"] + config.INJURY_PUBLISH_LAG
        for x in rep["rows"]:
            name = full_name(x["player"])
            pid = by_team.resolve(x["team_id"], name)
            if pid is None:
                pid = league.resolve(0, name)
            gd = datetime.strptime(x["game_date"], "%m/%d/%Y").date() if x["game_date"] else None
            gid = by_day_team.get((gd, x["team_id"]))
            stats["unmatched_players"] += pid is None
            stats["unmatched_games"] += gid is None
            rows.append((rep["report_at"], gid, gd, x["team_id"], pid, x["player"], x["status"], avail))
    stats["rows"] = len(rows)
    df = pd.DataFrame(rows, columns=["report_at", "game_id", "game_date", "team_id", "player_id", "player_name",
                                     "status", "available_at"])
    df["player_id"] = df.player_id.astype("Int64")
    _replace(con, "injuries", df)
    return stats


def build_all(con, lake: RawLake) -> dict:
    return {
        "schedule": load_schedule(con, lake),
        "games": load_games(con, lake),
        "team_games": load_team_games(con, lake),
        "officials": load_officials(con, lake),
        "pbp": load_pbp(con, lake),
        "odds": load_odds_history(con, lake),
        "injuries": load_injuries(con, lake),
    }
