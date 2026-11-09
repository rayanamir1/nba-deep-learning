"""Command line entry point.

  nbadl pull-games  --first 2013 --last 2025      games + schedule per season
  nbadl pull-games-detail --season 2023-24        play-by-play V3 + summary V3 (officials, roster) per game
  nbadl pull-odds   --start 2013-10-29 --end 2014-06-15   SBR history, one page per date
  nbadl pull-injuries --start 2019-10-01 --end 2026-06-30   hourly injury-report PDFs on game days
  nbadl snapshot-odds                              today's SBR lines, timestamped (run on a schedule)
  nbadl build                                      lake -> warehouse, prints data-quality stats

Every pull is resumable: rerun the same command and finished items are skipped.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime, timedelta

from . import config
from . import warehouse as wh
from .build import build_all
from .ingest.nba_stats import SEASON_TYPES, NbaStatsIngest, result_set
from .ingest.injury import InjuryIngest, game_days_with_eve
from .ingest.sbr import SbrIngest
from .lake import RawLake


def _game_ids(lake: RawLake, season: str) -> list[str]:
    ids: set[str] = set()
    for st in SEASON_TYPES:
        key = f"{season}_{st}"
        if lake.exists("games", key):
            ids |= {r["GAME_ID"] for r in result_set(lake.read("games", key).payload, "LeagueGameFinderResults")}
    return sorted(ids)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="nbadl")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pull-games"); p.add_argument("--first", type=int, default=config.FIRST_SEASON); p.add_argument("--last", type=int, required=True)
    p = sub.add_parser("pull-games-detail"); p.add_argument("--season", required=True); p.add_argument("--limit", type=int)
    p = sub.add_parser("pull-odds"); p.add_argument("--start", required=True); p.add_argument("--end", required=True)
    p = sub.add_parser("pull-injuries"); p.add_argument("--start", required=True); p.add_argument("--end", required=True)
    sub.add_parser("snapshot-odds")
    sub.add_parser("build")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    lake = RawLake()

    if args.cmd == "pull-games":
        nba = NbaStatsIngest(lake)
        for season in config.seasons(args.last, args.first):
            for st in SEASON_TYPES:
                nba.games(season, st)
            nba.schedule(season)
            logging.info("season %s done", season)

    elif args.cmd == "pull-games-detail":
        nba = NbaStatsIngest(lake)
        ids = _game_ids(lake, args.season)[: args.limit]
        failed: list[str] = []
        for i, gid in enumerate(ids, 1):
            for fetch in (nba.pbp, nba.summary):
                try:
                    fetch(gid)
                except Exception as e:  # one bad game must not stop a season; rerun retries it
                    logging.error("%s %s failed: %s", fetch.__name__, gid, e)
                    failed.append(gid)
            if i % 50 == 0:
                logging.info("%s: %d/%d games", args.season, i, len(ids))
        logging.info("%s done: %d games, %d failed fetches %s", args.season, len(ids), len(failed),
                     sorted(set(failed))[:20])

    elif args.cmd == "pull-odds":
        sbr = SbrIngest(lake)
        d, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
        while d <= end:
            sbr.history(d)
            d += timedelta(days=1)

    elif args.cmd == "pull-injuries":
        start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
        con = wh.connect()
        game_dates = [r[0] for r in con.execute("SELECT DISTINCT game_date FROM games WHERE game_date BETWEEN ? AND ?",
                                                [start, end]).fetchall()]
        days = game_days_with_eve(game_dates, start, end)
        logging.info("injury reports: %d days x hourly", len(days))
        InjuryIngest(lake).pull_days(days)

    elif args.cmd == "snapshot-odds":
        key = SbrIngest(lake).snapshot(date.today())
        logging.info("stored snapshot %s", key)

    elif args.cmd == "build":
        con = wh.connect()
        print(json.dumps(build_all(con, lake), indent=2, default=str))


if __name__ == "__main__":
    main()
