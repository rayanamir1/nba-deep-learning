"""Stage 1 ingest from stats.nba.com via nba_api. Writes raw responses to the lake.

Sources and lake names:
  games/<season>_<type>     LeagueGameFinder: one row per team per game (scores, date)
  schedule/<season>         ScheduleLeagueV2: scheduled tip-off times (gameDateTimeEst)
  pbp_v2/<game_id>          PlayByPlayV2: events with clear substitution in/out columns
  summary/<game_id>         BoxScoreSummaryV2: officials, inactive players, line score

Every fetch is idempotent: an existing lake key is skipped, so the job can be
killed and resumed at any point. Requests are throttled; stats.nba.com bans
clients that hammer it, and it often refuses cloud/datacenter IPs entirely, so
large pulls are best run from a home connection.
"""
from __future__ import annotations

import logging
import time
from typing import Callable

from .. import config
from ..lake import RawLake

log = logging.getLogger(__name__)

SEASON_TYPES = {"regular": "Regular Season", "playoffs": "Playoffs"}


class Throttle:
    def __init__(self, interval_s: float = config.NBA_REQUEST_INTERVAL_S):
        self.interval = interval_s
        self._last = 0.0

    def wait(self) -> None:
        delta = time.monotonic() - self._last
        if delta < self.interval:
            time.sleep(self.interval - delta)
        self._last = time.monotonic()


def _with_retries(fn: Callable[[], dict], what: str, tries: int = 4) -> dict:
    delay = 5.0
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except Exception as e:  # nba_api raises plain requests/JSON errors
            if attempt == tries:
                raise
            log.warning("%s failed (%s), retry %d in %.0fs", what, e, attempt, delay)
            time.sleep(delay)
            delay *= 2
    raise AssertionError("unreachable")


class NbaStatsIngest:
    def __init__(self, lake: RawLake | None = None, throttle: Throttle | None = None, timeout: int = 60):
        self.lake = lake or RawLake()
        self.throttle = throttle or Throttle()
        self.timeout = timeout

    def _fetch(self, source: str, key: str, call: Callable[[], dict]) -> bool:
        """Fetch and store unless already in the lake. Returns True if a request was made."""
        if self.lake.exists(source, key):
            return False
        self.throttle.wait()
        payload = _with_retries(call, f"{source}/{key}")
        self.lake.write(source, key, payload)
        return True

    # --- season-level -------------------------------------------------------
    def games(self, season: str, season_type: str = "regular") -> None:
        from nba_api.stats.endpoints import leaguegamefinder

        self._fetch(
            "games",
            f"{season}_{season_type}",
            lambda: leaguegamefinder.LeagueGameFinder(
                player_or_team_abbreviation="T",
                season_nullable=season,
                season_type_nullable=SEASON_TYPES[season_type],
                league_id_nullable="00",
                timeout=self.timeout,
            ).get_dict(),
        )

    def schedule(self, season: str) -> None:
        from nba_api.stats.endpoints import scheduleleaguev2

        self._fetch(
            "schedule",
            season,
            lambda: scheduleleaguev2.ScheduleLeagueV2(season=season, timeout=self.timeout).get_dict(),
        )

    # --- game-level ---------------------------------------------------------
    def pbp(self, game_id: str) -> bool:
        from nba_api.stats.endpoints import playbyplayv2

        return self._fetch(
            "pbp_v2",
            game_id,
            lambda: playbyplayv2.PlayByPlayV2(game_id=game_id, timeout=self.timeout).get_dict(),
        )

    def summary(self, game_id: str) -> bool:
        from nba_api.stats.endpoints import boxscoresummaryv2

        return self._fetch(
            "summary",
            game_id,
            lambda: boxscoresummaryv2.BoxScoreSummaryV2(game_id=game_id, timeout=self.timeout).get_dict(),
        )


def result_set(payload: dict, name: str) -> list[dict]:
    """Turn an nba_api result set {headers, rowSet} into a list of dicts."""
    for rs in payload.get("resultSets", []):
        if rs.get("name") == name:
            return [dict(zip(rs["headers"], row)) for row in rs["rowSet"]]
    return []
