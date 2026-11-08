"""Adapter: stats.nba.com PlayByPlayV2 rows -> normalized possessions.Event.

EVENTMSGTYPE codes: 1 made FG, 2 missed FG, 3 free throw, 4 rebound, 5 turnover,
6 foul, 7 violation, 8 substitution (PLAYER1 out, PLAYER2 in), 9 timeout,
10 jump ball, 11 ejection, 12 period start, 13 period end, 18 instant replay.

PERSON1TYPE: 4/5 = home/visitor player, 2/3 = home/visitor TEAM (team rebounds and
team turnovers put the team id in PLAYER1_ID and leave PLAYER1_TEAM_ID empty).

Free-throw type is read from the description text ("Technical", "Flagrant") rather
than from EVENTMSGACTIONTYPE codes, which are less consistent across seasons.
"""
from __future__ import annotations

from .ingest.nba_stats import result_set
from .possessions import Event, Kind

_KIND = {
    1: Kind.FG_MADE, 2: Kind.FG_MISSED, 4: Kind.REBOUND, 5: Kind.TURNOVER, 6: Kind.FOUL,
    7: Kind.VIOLATION, 8: Kind.SUB, 9: Kind.TIMEOUT, 10: Kind.JUMP_BALL,
    12: Kind.PERIOD_START, 13: Kind.PERIOD_END,
}


def _clock(s: str | None) -> float:
    if not s:
        return 0.0
    m, sec = s.split(":")
    return int(m) * 60 + float(sec)


def _desc(r: dict) -> str:
    return " ".join(x for x in (r.get("HOMEDESCRIPTION"), r.get("NEUTRALDESCRIPTION"), r.get("VISITORDESCRIPTION")) if x)


def _pid(x) -> int | None:
    try:
        x = int(x)
    except (TypeError, ValueError):
        return None
    return x or None


def _team(r: dict) -> int | None:
    t = _pid(r.get("PLAYER1_TEAM_ID"))
    if t is not None:
        return t
    if r.get("PERSON1TYPE") in (2, 3):  # team-credited event: PLAYER1_ID holds the team id
        return _pid(r.get("PLAYER1_ID"))
    return None


def events_from_v2(payload: dict) -> list[Event]:
    rows = result_set(payload, "PlayByPlay")
    out: list[Event] = []
    for r in rows:
        mt = r["EVENTMSGTYPE"]
        desc = _desc(r)
        team = _team(r)
        player = _pid(r.get("PLAYER1_ID")) if r.get("PERSON1TYPE") not in (2, 3) else None
        kw: dict = {}
        if mt == 3:
            made = "MISS" not in desc.upper()
            kind = Kind.FT_MADE if made else Kind.FT_MISSED
            low = desc.lower()
            kw["ft_type"] = "technical" if "technical" in low else "flagrant" if "flagrant" in low else "regular"
            kw["points"] = 1 if made else 0
        elif mt == 1:
            kind = Kind.FG_MADE
            kw["points"] = 3 if "3PT" in desc.upper() else 2
        elif mt == 8:
            kind = Kind.SUB
            kw["player_in_id"] = _pid(r.get("PLAYER2_ID"))
        else:
            kind = _KIND.get(mt, Kind.OTHER)
        out.append(Event(
            game_id=str(r["GAME_ID"]), seq=int(r["EVENTNUM"]), period=int(r["PERIOD"]),
            clock_s=_clock(r.get("PCTIMESTRING")), kind=kind, team_id=team, player_id=player, **kw,
        ))
    return out


def team_ids(events: list[Event]) -> tuple[int, int]:
    teams = sorted({e.team_id for e in events if e.team_id is not None and e.player_id is not None})
    if len(teams) != 2:
        raise ValueError(f"expected 2 teams in play-by-play, found {teams}")
    return teams[0], teams[1]
