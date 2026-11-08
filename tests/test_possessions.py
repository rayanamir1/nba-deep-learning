from nbadl.possessions import Event, Kind, count_possessions, parse_game

A, B = 100, 200
A_START = [1, 2, 3, 4, 5]
B_START = [11, 12, 13, 14, 15]


class G:
    """Tiny builder for a synthetic game."""

    def __init__(self):
        self.evs: list[Event] = []

    def add(self, period, clock, kind, team=None, player=None, **kw):
        self.evs.append(Event("g1", len(self.evs), period, clock, kind, team, player, **kw))
        return self


def starters_touch(g: G, period: int, clock: float):
    """Make every starter record a harmless event so they are inferred as starters."""
    for p in A_START:
        g.add(period, clock, Kind.FOUL, A, p)
    for p in B_START:
        g.add(period, clock, Kind.FOUL, B, p)


def build_game() -> G:
    g = G()
    g.add(1, 720, Kind.PERIOD_START)
    starters_touch(g, 1, 720)
    # A possession 1: made 2 with and-one free throw (one possession, 3 points)
    g.add(1, 700, Kind.FG_MADE, A, 1, points=2)
    g.add(1, 700, Kind.FOUL, B, 11)
    g.add(1, 700, Kind.FT_MADE, A, 1, points=1, ft_type="regular")
    # B possession 2: missed first FT, dead-ball team rebound to B, made second FT
    g.add(1, 680, Kind.FT_MISSED, B, 12, ft_type="regular")
    g.add(1, 680, Kind.REBOUND, B, None)
    g.add(1, 680, Kind.FT_MADE, B, 12, points=1, ft_type="regular")
    # technical FT by B while A has the ball next: must NOT split A's possession
    # A possession 3: miss, offensive rebound, sub, made 3
    g.add(1, 660, Kind.FG_MISSED, A, 2)
    g.add(1, 658, Kind.REBOUND, A, 3)
    g.add(1, 650, Kind.FT_MADE, B, 13, points=1, ft_type="technical")
    g.add(1, 650, Kind.SUB, A, 5, player_in_id=6)
    g.add(1, 640, Kind.FG_MADE, A, 6, points=3)
    # B possession 4: turnover
    g.add(1, 620, Kind.TURNOVER, B, 14)
    # A possession 5: miss, B defensive rebound
    g.add(1, 600, Kind.FG_MISSED, A, 4)
    g.add(1, 598, Kind.REBOUND, B, 15)
    # B possession 6: made 2
    g.add(1, 580, Kind.FG_MADE, B, 15, points=2)
    # A possession 7: last-second miss, B rebound as the horn sounds -> B empty possession 8
    g.add(1, 2, Kind.FG_MISSED, A, 1)
    g.add(1, 0, Kind.REBOUND, B, 11)
    g.add(1, 0, Kind.PERIOD_END)
    return g


def test_possession_runs():
    parse = parse_game(build_game().evs, (A, B))
    offs = [p.offense_team_id for p in parse.possessions]
    assert offs == [A, B, A, B, A, B, A, B]
    assert parse.possessions[-1].is_empty
    assert count_possessions(parse) == {A: 4, B: 3}


def test_points_reconcile():
    parse = parse_game(build_game().evs, (A, B))
    assert parse.points_by_team == {A: 3 + 3, B: 1 + 1 + 2}
    in_runs = sum(p.points for p in parse.possessions)
    outside = sum(parse.points_outside_runs.values())
    assert in_runs + outside == sum(parse.points_by_team.values())
    assert parse.points_outside_runs[B] == 1  # the technical free throw
    assert parse.possessions[0].points == 3  # and-one counted in one possession


def test_stints_and_lineups():
    parse = parse_game(build_game().evs, (A, B))
    assert parse.lineup_complete
    assert len(parse.stints) == 2
    assert sum(s.seconds for s in parse.stints) == 720
    first, second = parse.stints
    assert first.lineup[A] == (1, 2, 3, 4, 5)
    assert second.lineup[A] == (1, 2, 3, 4, 6)
    assert first.points[A] + second.points[A] == 6
    assert sum(s.points.get(A, 0) + s.points.get(B, 0) for s in parse.stints) == 10


def test_incomplete_lineup_is_flagged():
    g = G()
    g.add(1, 700, Kind.FG_MADE, A, 1, points=2)
    g.add(1, 680, Kind.FG_MADE, B, 11, points=2)
    parse = parse_game(g.evs, (A, B))
    assert not parse.lineup_complete


def test_starters_override_fixes_lineup():
    g = G()
    g.add(1, 700, Kind.FG_MADE, A, 1, points=2)
    g.add(1, 680, Kind.FG_MADE, B, 11, points=2)
    parse = parse_game(g.evs, (A, B), starters_override={1: {A: set(A_START), B: set(B_START)}})
    assert parse.lineup_complete
    assert parse.stints[0].lineup[B] == tuple(B_START)
