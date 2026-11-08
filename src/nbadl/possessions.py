"""Stage 3: turn play-by-play into possessions and stints.

Approach ("who has the ball" runs)
-----------------------------------
Each event that only the team WITH the ball can produce marks that team as the
offense: a field goal attempt, a regular free throw, a turnover, and a rebound
(a defensive rebound hands the ball over, and an offensive one keeps it). A
possession is a maximal run of consecutive offensive events by one team inside a
period. The rules that fall out of this:

  * and-one: the made FG and its free throw belong to the same team -> one possession
  * a missed non-final FT is followed by a dead-ball team rebound credited to the
    shooting team, which keeps it in the same run
  * technical FTs can be shot by the team WITHOUT the ball, so they are excluded
    from run detection; their points go to `points_outside_runs`
  * flagrant FTs are shot by the fouled team, which then keeps the ball, so they
    stay in the run
  * fouls, subs, timeouts, violations and jump balls never decide who has the ball

A run with no FG attempt, FT trip or turnover (for example a defensive rebound
as the period horn sounds) is kept but flagged `is_empty`; pace counts exclude it.

Invariants (tested): points in runs plus points outside runs equal the final
score, and stint seconds sum to the game length.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Kind(str, Enum):
    FG_MADE = "fg_made"
    FG_MISSED = "fg_missed"
    FT_MADE = "ft_made"
    FT_MISSED = "ft_missed"
    REBOUND = "rebound"
    TURNOVER = "turnover"
    FOUL = "foul"
    SUB = "sub"
    TIMEOUT = "timeout"
    JUMP_BALL = "jump_ball"
    VIOLATION = "violation"
    PERIOD_START = "period_start"
    PERIOD_END = "period_end"
    OTHER = "other"


@dataclass(frozen=True)
class Event:
    game_id: str
    seq: int  # order within the game
    period: int
    clock_s: float  # seconds remaining in the period
    kind: Kind
    team_id: int | None = None
    player_id: int | None = None  # actor; for SUB the player going OUT
    player_in_id: int | None = None  # SUB only: the player coming IN
    points: int = 0
    ft_type: str | None = None  # "regular" | "technical" | "flagrant" (FT only)


OFFENSIVE_KINDS = {Kind.FG_MADE, Kind.FG_MISSED, Kind.FT_MADE, Kind.FT_MISSED, Kind.TURNOVER, Kind.REBOUND}
SCORING_KINDS = {Kind.FG_MADE, Kind.FT_MADE}


def period_length_s(period: int) -> float:
    return 720.0 if period <= 4 else 300.0


@dataclass
class Possession:
    game_id: str
    period: int
    index: int  # 0-based within the game
    offense_team_id: int
    start_clock_s: float
    end_clock_s: float
    points: int = 0
    fga: int = 0
    fta: int = 0
    turnovers: int = 0
    offense_lineup: tuple[int, ...] = ()
    defense_lineup: tuple[int, ...] = ()
    last_clock_s: float = 0.0

    @property
    def is_empty(self) -> bool:
        return self.fga == 0 and self.fta == 0 and self.turnovers == 0

    @property
    def seconds(self) -> float:
        return self.start_clock_s - self.end_clock_s


@dataclass
class Stint:
    game_id: str
    period: int
    lineup: dict[int, tuple[int, ...]]  # team_id -> sorted 5 player ids
    start_clock_s: float
    end_clock_s: float
    points: dict[int, int] = field(default_factory=dict)
    possessions: dict[int, int] = field(default_factory=dict)  # non-empty possessions by offense team

    @property
    def seconds(self) -> float:
        return self.start_clock_s - self.end_clock_s


@dataclass
class GameParse:
    possessions: list[Possession]
    stints: list[Stint]
    points_by_team: dict[int, int]
    points_outside_runs: dict[int, int]
    lineup_complete: bool  # False if any period had fewer than 5 known starters per team


def is_offensive(ev: Event) -> bool:
    if ev.kind not in OFFENSIVE_KINDS or ev.team_id is None:
        return False
    if ev.kind in (Kind.FT_MADE, Kind.FT_MISSED) and ev.ft_type == "technical":
        return False
    return True


def infer_period_starters(events: list[Event], team_ids: tuple[int, int]) -> dict[int, set[int]]:
    """Players who were on court when the period began.

    A player is a starter if their first appearance in the period is anything other
    than being subbed IN: they recorded an event, or they were subbed OUT first.
    Requires knowing each event's team. Players who play the whole period without
    recording anything are invisible here; the caller sees len < 5 and must fill
    them from a box-score-by-period source (lineup_complete=False).
    """
    seen: set[int] = set()
    starters: dict[int, set[int]] = {t: set() for t in team_ids}
    for ev in events:
        if ev.kind == Kind.SUB:
            if ev.player_in_id is not None:
                seen.add(ev.player_in_id)
            pid = ev.player_id
        else:
            pid = ev.player_id
        if pid is None or pid in seen or ev.team_id not in starters:
            continue
        seen.add(pid)
        starters[ev.team_id].add(pid)
    return starters


def parse_game(
    events: list[Event],
    team_ids: tuple[int, int],
    starters_override: dict[int, dict[int, set[int]]] | None = None,
) -> GameParse:
    """Parse one game's events (any order) into possessions and stints.

    starters_override: {period: {team_id: {player ids}}}, used when play-by-play alone
    cannot identify all 10 players on court at a period's start.
    """
    events = sorted(events, key=lambda e: e.seq)
    by_period: dict[int, list[Event]] = {}
    for ev in events:
        by_period.setdefault(ev.period, []).append(ev)

    possessions: list[Possession] = []
    stints: list[Stint] = []
    points = {t: 0 for t in team_ids}
    outside = {t: 0 for t in team_ids}
    lineup_complete = True
    prev_end: dict[int, set[int]] | None = None  # who was on court when the previous period ended

    for period in sorted(by_period):
        pev = by_period[period]
        plen = period_length_s(period)
        starters = (starters_override or {}).get(period)
        if not starters:
            # A lineup can only change through a logged sub, and period-break subs are logged inside the
            # new period, so the previous period's end lineup is exact whenever it is complete.
            if prev_end is not None and all(len(prev_end[t]) == 5 for t in team_ids):
                starters = prev_end
            else:
                starters = infer_period_starters(pev, team_ids)
        on_court = {t: set(starters.get(t, set())) for t in team_ids}
        if any(len(v) != 5 for v in on_court.values()):
            lineup_complete = False

        def lineup(t: int) -> tuple[int, ...]:
            return tuple(sorted(on_court[t]))

        stint = Stint(ev_game(pev), period, {t: lineup(t) for t in team_ids}, plen, plen)
        cur: Possession | None = None

        def close_possession(at_clock: float) -> None:
            nonlocal cur
            if cur is None:
                return
            cur.end_clock_s = at_clock
            possessions.append(cur)
            if not cur.is_empty:
                stint.possessions[cur.offense_team_id] = stint.possessions.get(cur.offense_team_id, 0) + 1
            cur = None

        for ev in pev:
            if ev.kind == Kind.SUB and ev.team_id in on_court:
                # close the stint at this clock, swap players, open a new stint
                stint.end_clock_s = ev.clock_s
                if stint.seconds > 0 or stint.points or stint.possessions:
                    stints.append(stint)
                    carry_points, carry_poss = {}, {}
                else:  # zero-length stint (several subs at one dead ball): drop it
                    carry_points, carry_poss = stint.points, stint.possessions
                on_court[ev.team_id].discard(ev.player_id)
                if ev.player_in_id is not None:
                    on_court[ev.team_id].add(ev.player_in_id)
                stint = Stint(ev.game_id, period, {t: lineup(t) for t in team_ids}, ev.clock_s, ev.clock_s,
                              dict(carry_points), dict(carry_poss))
                # a possession can span a substitution (e.g. between free throws); it stays
                # credited to the lineups that started it, as most RAPM pipelines do
                continue

            if is_offensive(ev):
                if cur is None or cur.offense_team_id != ev.team_id:
                    # the previous possession ended at its own last offensive event
                    close_possession(cur.last_clock_s if cur is not None else plen)
                    start = possessions[-1].end_clock_s if possessions_in_period(possessions, period) else plen
                    defense = next(t for t in team_ids if t != ev.team_id)
                    cur = Possession(ev.game_id, period, len(possessions), ev.team_id, start, start,
                                     offense_lineup=lineup(ev.team_id), defense_lineup=lineup(defense))
                cur.last_clock_s = ev.clock_s
                if ev.kind in (Kind.FG_MADE, Kind.FG_MISSED):
                    cur.fga += 1
                elif ev.kind in (Kind.FT_MADE, Kind.FT_MISSED):
                    cur.fta += 1
                elif ev.kind == Kind.TURNOVER:
                    cur.turnovers += 1

            if ev.kind in SCORING_KINDS and ev.team_id in points:
                points[ev.team_id] += ev.points
                stint.points[ev.team_id] = stint.points.get(ev.team_id, 0) + ev.points
                if cur is not None and is_offensive(ev) and cur.offense_team_id == ev.team_id:
                    cur.points += ev.points
                else:
                    outside[ev.team_id] += ev.points

        close_possession(0.0)
        stint.end_clock_s = 0.0
        stints.append(stint)
        prev_end = {t: set(on_court[t]) for t in team_ids}

    return GameParse(possessions, stints, points, outside, lineup_complete)


def ev_game(evs: list[Event]) -> str:
    return evs[0].game_id if evs else ""


def possessions_in_period(poss: list[Possession], period: int) -> bool:
    return bool(poss) and poss[-1].period == period


def count_possessions(parse: GameParse) -> dict[int, int]:
    """Non-empty possessions per offense team: the pace numerator."""
    out: dict[int, int] = {}
    for p in parse.possessions:
        if not p.is_empty:
            out[p.offense_team_id] = out.get(p.offense_team_id, 0) + 1
    return out
