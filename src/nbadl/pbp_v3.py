"""Adapter: stats.nba.com PlayByPlayV3 actions -> normalized possessions.Event.

PlayByPlayV2 stopped returning data in 2025 (it answers with an empty body), so V3
is the play-by-play source. Differences from V2 that matter here:

  * order: `actionNumber` is NOT unique (a block or steal shares it with the shot or
    turnover it belongs to); `actionId` is unique and increasing, so it is `seq`.
  * clock is ISO-8601 ("PT11M39.00S").
  * team-credited events (team rebound, shot-clock turnover) have teamId=0 and the
    team id in personId, the same convention as V2's PERSON1TYPE 2/3 rows.
  * blocks and steals have an empty actionType; they map to OTHER but still carry
    the player, which helps infer who started a period.
  * a substitution carries only the player going OUT (personId). The player coming
    IN is only named in the text, "SUB: <in> FOR <out>", as a family name or as
    "F. Family" when two teammates share one. It is resolved against the game's
    roster (BoxScoreSummaryV3 players) and the names seen in the play-by-play
    itself. An unresolved name leaves player_in_id=None and is counted by the caller.
"""
from __future__ import annotations

import re
import unicodedata

from .possessions import Event, Kind

_KIND = {
    "Made Shot": Kind.FG_MADE, "Missed Shot": Kind.FG_MISSED, "Rebound": Kind.REBOUND,
    "Turnover": Kind.TURNOVER, "Foul": Kind.FOUL, "Violation": Kind.VIOLATION,
    "Substitution": Kind.SUB, "Timeout": Kind.TIMEOUT, "Jump Ball": Kind.JUMP_BALL,
}
_CLOCK = re.compile(r"PT(\d+)M([\d.]+)S")
_SUB = re.compile(r"^SUB:\s*(.+?)\s+FOR\s+(.+?)\s*$")


def _clock(s: str | None) -> float:
    m = _CLOCK.match(s or "")
    return int(m[1]) * 60 + float(m[2]) if m else 0.0


def _norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s.lower().replace("'", "").replace("’", "")).strip()


def actions(payload: dict) -> list[dict]:
    return (payload.get("game") or {}).get("actions") or []


def roster_from_summary_v3(payload: dict) -> dict[int, list[dict]]:
    """{team_id: [{personId, nameI, familyName, name}, ...]} from BoxScoreSummaryV3."""
    s = payload.get("boxScoreSummary") or {}
    out: dict[int, list[dict]] = {}
    for side in ("homeTeam", "awayTeam"):
        t = s.get(side) or {}
        if t.get("teamId"):
            out[int(t["teamId"])] = list(t.get("players") or [])
    return out


def _collapse(key: str) -> str:
    """Make 'poeltl' and 'poltl' (two transliterations of Pöltl) the same key."""
    return re.sub(r"(oe|ue|ae)", lambda m: m[1][0], key)


_SUFFIX = re.compile(r"\s+(jr\.?|sr\.?|ii|iii|iv|v)$")


def _variants(name: str) -> list[str]:
    """Keys from exact to loose: exact, transliteration-collapsed, suffix-stripped, both."""
    k = _norm(name)
    stripped = _SUFFIX.sub("", k)
    return [k, _collapse(k), stripped, _collapse(stripped)]


class _Names:
    """Per-team name -> player id index. Ambiguous names map to None.

    Each looseness level (see _variants) is its own layer and lookup goes exact-first, so a loose form of one
    player ('Carter Jr.' -> 'carter') can never steal or spoil another player's exact name ('Carter').
    Weak names (first name alone, for sources that print 'Hansen' for Hansen Yang) are the last resort.
    """

    def __init__(self) -> None:
        self._idx: dict[int, list[dict[str, int | None]]] = {}
        self._weak: dict[int, dict[str, int | None]] = {}

    @staticmethod
    def _set(d: dict, k: str, pid: int) -> None:
        d[k] = None if (k in d and d[k] != pid) else pid  # two teammates share it

    def add(self, team: int, pid: int, *names: str | None) -> None:
        layers = self._idx.setdefault(team, [{} for _ in range(4)])
        for n in names:
            if n:
                for layer, k in zip(layers, _variants(n)):
                    self._set(layer, k, pid)

    def add_weak(self, team: int, pid: int, *names: str | None) -> None:
        d = self._weak.setdefault(team, {})
        for n in names:
            if n:
                self._set(d, _norm(n), pid)

    def add_roster_player(self, team: int, p: dict) -> None:
        pid = int(p["personId"])
        self.add(team, pid, p.get("nameI"), p.get("familyName"), p.get("name"))
        first, family = p.get("firstName") or "", p.get("familyName") or ""
        for k in range(1, len(first)):  # 'Jay. Williams' separates Jaylin from Jalen
            self.add(team, pid, f"{first[:k]}. {family}")
        self.add_weak(team, pid, first)

    def resolve(self, team: int, name: str) -> int | None:
        for layer, k in zip(self._idx.get(team, []), _variants(name)):
            if k in layer:
                return layer[k]
        return self._weak.get(team, {}).get(_norm(name))


def _names(acts: list[dict], roster: dict[int, list[dict]] | None) -> _Names:
    names = _Names()
    for team, players in (roster or {}).items():
        for p in players:
            if p.get("personId"):
                names.add_roster_player(team, p)
    for a in acts:
        if a.get("teamId") and a.get("personId") and a.get("actionType") != "Substitution":
            names.add(int(a["teamId"]), int(a["personId"]), a.get("playerNameI"), a.get("playerName"))
    for a in acts:  # the player going out of a sub is identified by id: their name is reliable too
        if a.get("actionType") == "Substitution" and a.get("teamId") and a.get("personId"):
            names.add(int(a["teamId"]), int(a["personId"]), a.get("playerNameI"), a.get("playerName"))
    return names


def events_from_v3(payload: dict, roster: dict[int, list[dict]] | None = None) -> list[Event]:
    acts = actions(payload)
    gid = str((payload.get("game") or {}).get("gameId", ""))
    names = _names(acts, roster)
    out: list[Event] = []
    for a in acts:
        at = a.get("actionType") or ""
        sub = a.get("subType") or ""
        desc = a.get("description") or ""
        team = int(a.get("teamId") or 0) or None
        player = int(a.get("personId") or 0) or None
        if team is None and player is not None and at in ("Rebound", "Turnover", "Foul", "Violation", "Timeout"):
            team, player = player, None  # team-credited event: personId holds the team id
        kw: dict = {}
        if at == "Free Throw":
            made = not desc.startswith("MISS ")  # case-sensitive: "Missi" (a player) is a make
            kind = Kind.FT_MADE if made else Kind.FT_MISSED
            low = (sub + " " + desc).lower()
            kw["ft_type"] = "technical" if "technical" in low else "flagrant" if "flagrant" in low else "regular"
            kw["points"] = 1 if made else 0
        elif at == "Made Shot":
            kind = Kind.FG_MADE
            kw["points"] = int(a.get("shotValue") or (3 if "3PT" in desc.upper() else 2))
        elif at == "period":
            kind = Kind.PERIOD_START if sub == "start" else Kind.PERIOD_END if sub == "end" else Kind.OTHER
        elif at == "Substitution":
            kind = Kind.SUB
            m = _SUB.match(desc)
            if m and team is not None:
                kw["player_in_id"] = names.resolve(team, m[1])
        else:
            kind = _KIND.get(at, Kind.OTHER)
        out.append(Event(
            game_id=gid, seq=int(a["actionId"]), period=int(a["period"]), clock_s=_clock(a.get("clock")),
            kind=kind, team_id=team, player_id=player, **kw,
        ))
    return out


def unresolved_subs(events: list[Event]) -> int:
    return sum(1 for e in events if e.kind == Kind.SUB and e.player_in_id is None)


def team_ids(events: list[Event]) -> tuple[int, int]:
    teams = sorted({e.team_id for e in events if e.team_id is not None and e.player_id is not None})
    if len(teams) != 2:
        raise ValueError(f"expected 2 teams in play-by-play, found {teams}")
    return teams[0], teams[1]
