from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from nbadl import warehouse as wh
from nbadl.features.referees import RefereeEffects

T0 = datetime(2023, 10, 20, 23, tzinfo=timezone.utc)
EFFECT = {1: 4.0, 2: -4.0}  # refs 1 and 2 shift the total; 3..8 are neutral


def world(n=400, seed=0):
    rng = np.random.default_rng(seed)
    con = wh.connect(":memory:")
    for g in range(n):
        tip = T0 + timedelta(hours=g * 6)
        av = wh.result_available_at(tip.date(), tip)
        crew = [int(x) for x in rng.choice(np.arange(1, 9), 3, replace=False)]
        miss = sum(EFFECT.get(r, 0.0) for r in crew) + rng.normal(0, 8)
        con.execute("INSERT INTO games VALUES (?,?,?,?,?,?,?,?,?,?)",
                    [f"g{g}", "2023-24", "regular", tip.date(), 1, 2, 110 + int(miss), 110, tip, av])
        con.execute("INSERT INTO odds_lines VALUES (?,?,?,?,?,?,?,?,?)",
                    [f"g{g}", "s", "book", "close", 220.0, -110, -110, tip, tip])
        for r in crew:
            con.execute("INSERT INTO officials VALUES (?,?,?,?)", [f"g{g}", r, "x", wh.officials_available_at(tip.date())])
    return con


def fit(con, days, **kw):
    t = T0 + timedelta(days=days)
    with wh.point_in_time(con, t) as c:
        return RefereeEffects(**kw).fit(c, t)


def test_recovers_signs_and_shrinks_neutral_officials():
    m = fit(world(), 120, half_life_days=1e6, alpha=20.0)
    assert m.coef[1] > 1.5 and m.coef[2] < -1.5
    assert max(abs(m.coef[r]) for r in range(3, 9)) < 1.0
    assert m.crew_effect([1, 3, 4]) > m.crew_effect([3, 4, 5]) > m.crew_effect([2, 3, 4])


def test_heavy_penalty_and_unseen_official():
    m = fit(world(), 120, half_life_days=1e6, alpha=1e7)
    assert max(abs(c) for c in m.coef.values()) < 0.05
    assert m.crew_effect([999]) == 0.0


def test_future_games_do_not_change_earlier_effects():
    con = world()
    before = fit(con, 40, alpha=50.0).crew_effect([1, 2, 3])
    con.execute("UPDATE games SET home_pts = 300 WHERE game_id = 'g399'")
    assert fit(con, 40, alpha=50.0).crew_effect([1, 2, 3]) == pytest.approx(before, abs=1e-9)
