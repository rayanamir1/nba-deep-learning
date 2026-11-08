"""Stage 1 ingest of NBA game totals from SportsbookReview (SBR).

SBR's odds pages are Next.js apps: the full odds table is embedded as JSON in
<script id="__NEXT_DATA__">. We store that JSON (only the odds tables) in the lake,
then parse it into one row per game x sportsbook with opening and current lines.

Two modes, two lake sources:
  sbr_history/<YYYY-MM-DD>               one fetch per past date. Its "current" line is
                                          the closing line, but there is no timestamp
                                          for when the opener was posted.
  sbr_snapshots/<YYYY-MM-DD>__<stamp>     repeated fetches on game day. Each snapshot is
                                          a timestamped observation of the line, which
                                          builds our own intraday line history over time.

ASSUMPTION TO VERIFY ON FIRST REAL FETCH: the JSON path
props.pageProps.oddsTables[*].oddsTableModel.gameRows[*] and the field names below
follow what open-source SBR scrapers read. `parse_odds_tables` fails loudly if the
shape differs, rather than silently returning nothing.

Use is subject to SBR's terms of service; keep request rates low (one page per date).
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, asdict
from datetime import date, datetime, timezone
from typing import Any

import requests

from ..lake import RawLake, utcnow

TOTALS_URL = "https://www.sportsbookreview.com/betting-odds/nba-basketball/totals/full-game/?date={d}"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}
_NEXT_DATA_RE = re.compile(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', re.S)


class SbrShapeError(ValueError):
    """The page did not have the structure we expect; the scraper needs updating."""


def extract_odds_tables(html: str) -> list[dict]:
    m = _NEXT_DATA_RE.search(html)
    if not m:
        raise SbrShapeError("no __NEXT_DATA__ script found on page")
    data = json.loads(m.group(1))
    try:
        tables = data["props"]["pageProps"]["oddsTables"]
    except (KeyError, TypeError) as e:
        raise SbrShapeError(f"missing props.pageProps.oddsTables ({e})") from e
    if not isinstance(tables, list):
        raise SbrShapeError("oddsTables is not a list")
    return tables


@dataclass(frozen=True)
class TotalLine:
    sbr_game_id: str
    start_utc: str | None
    away_team: str
    home_team: str
    away_score: int | None
    home_score: int | None
    sportsbook: str
    open_total: float | None
    open_over_odds: int | None
    open_under_odds: int | None
    current_total: float | None
    current_over_odds: int | None
    current_under_odds: int | None


def _team(t: dict | None) -> str:
    t = t or {}
    return t.get("shortName") or t.get("name") or t.get("fullName") or ""


def _line(view: dict, key: str) -> tuple[Any, Any, Any]:
    ln = view.get(key) or {}
    return ln.get("total"), ln.get("overOdds"), ln.get("underOdds")


def parse_odds_tables(tables: list[dict]) -> list[TotalLine]:
    out: list[TotalLine] = []
    for table in tables:
        try:
            rows = table["oddsTableModel"]["gameRows"]
        except (KeyError, TypeError) as e:
            raise SbrShapeError(f"odds table without oddsTableModel.gameRows ({e})") from e
        for row in rows:
            gv = row.get("gameView")
            if not gv:
                raise SbrShapeError("game row without gameView")
            for view in row.get("oddsViews") or []:
                if not view:  # SBR leaves null slots for books with no line
                    continue
                o_t, o_over, o_under = _line(view, "openingLine")
                c_t, c_over, c_under = _line(view, "currentLine")
                out.append(
                    TotalLine(
                        sbr_game_id=str(gv.get("gameId")),
                        start_utc=gv.get("startDate"),
                        away_team=_team(gv.get("awayTeam")),
                        home_team=_team(gv.get("homeTeam")),
                        away_score=gv.get("awayTeamScore"),
                        home_score=gv.get("homeTeamScore"),
                        sportsbook=str(view.get("sportsbook", "")),
                        open_total=o_t,
                        open_over_odds=o_over,
                        open_under_odds=o_under,
                        current_total=c_t,
                        current_over_odds=c_over,
                        current_under_odds=c_under,
                    )
                )
    return out


class SbrIngest:
    def __init__(self, lake: RawLake | None = None, interval_s: float = 3.0, session: requests.Session | None = None):
        self.lake = lake or RawLake()
        self.interval = interval_s
        self.session = session or requests.Session()
        self._last = 0.0

    def _get(self, d: date) -> list[dict]:
        wait = self.interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()
        r = self.session.get(TOTALS_URL.format(d=d.isoformat()), headers=HEADERS, timeout=30)
        r.raise_for_status()
        return extract_odds_tables(r.text)

    def history(self, d: date) -> bool:
        """Fetch a past date once. Returns True if a request was made."""
        key = d.isoformat()
        if self.lake.exists("sbr_history", key):
            return False
        self.lake.write("sbr_history", key, self._get(d))
        return True

    def snapshot(self, d: date, now: datetime | None = None) -> str:
        """Fetch today's lines and store a timestamped snapshot. Run this on a schedule."""
        now = now or utcnow()
        key = f"{d.isoformat()}__{now.astimezone(timezone.utc):%Y%m%dT%H%M%SZ}"
        self.lake.write("sbr_snapshots", key, self._get(d), fetched_at=now)
        return key


def lines_as_dicts(tables: list[dict]) -> list[dict]:
    return [asdict(x) for x in parse_odds_tables(tables)]
