from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from nbadl import warehouse
from nbadl.features.player_skills import PlayerSkills
from nbadl.features.rapm import Rapm

T0 = datetime(2023, 10, 1, 23, tzinfo=timezone.utc)
TEAM_A, TEAM_B = list(range(1, 9)), list(range(11, 19))  # 8-man pools


def simulate(off, dfn, *, n_games=120, stints=40, seed=0, pair=None):
    """Random 5-man lineups; points ~ Poisson around 24 + sum of offense effects + opponent defense effects.

    pair=(star, scrub): the two are on the floor together 85% of the time (the confounded case).
    """
    rng = np.random.default_rng(seed)
    con = warehouse.connect(":memory:")

    def lineup(pool):
        if pair and pair[0] in pool:
            if rng.random() < 0.85:
                rest = [p for p in pool if p not in pair]
                return sorted([*pair, *rng.choice(rest, 3, replace=False)])
            solo = pair[int(rng.random() < 0.5)]
            rest = [p for p in pool if p not in pair]
            return sorted([solo, *rng.choice(rest, 4, replace=False)])
        return sorted(rng.choice(pool, 5, replace=False).tolist())

    for g in range(n_games):
        tip = T0 + timedelta(days=g)
        av = warehouse.result_available_at(tip.date(), tip)
        con.execute("INSERT INTO games VALUES (?,?,?,?,?,?,?,?,?,?)",
                    [f"g{g}", "2023-24", "regular", tip.date(), 1, 2, 0, 0, tip, av])
        for _ in range(stints):
            la, lb = lineup(TEAM_A), lineup(TEAM_B)
            poss = 12
            ra = 24 + sum(off[p] for p in la) / 100 * poss + sum(dfn[p] for p in lb) / 100 * poss
            rb = 24 + sum(off[p] for p in lb) / 100 * poss + sum(dfn[p] for p in la) / 100 * poss
            con.execute("INSERT INTO stints VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        [f"g{g}", 1, 1, 2, [int(x) for x in la], [int(x) for x in lb], 720, 0, 180,
                         int(rng.poisson(max(ra, 1))), int(rng.poisson(max(rb, 1))), poss, poss, av])
    return con


def fit(model, con, t=T0 + timedelta(days=130)):
    with warehouse.point_in_time(con, t) as c:
        return model.fit(c, t)


def effects(seed=1):
    rng = np.random.default_rng(seed)
    pl = TEAM_A + TEAM_B
    return {p: float(rng.normal(0, 3)) for p in pl}, {p: float(rng.normal(0, 3)) for p in pl}


def test_recovers_known_player_effects():
    off, dfn = effects()
    m = fit(Rapm(half_life_days=1e6, alpha=50.0), simulate(off, dfn))
    ps = sorted(off)
    assert np.corrcoef([m.get(p)[0] for p in ps], [off[p] for p in ps])[0, 1] > 0.8
    assert np.corrcoef([m.get(p)[1] for p in ps], [dfn[p] for p in ps])[0, 1] > 0.6


def test_separates_a_star_from_the_scrub_who_always_plays_with_him():
    off = {p: 0.0 for p in TEAM_A + TEAM_B}
    dfn = dict(off)
    off[1] = 8.0  # star; player 2 is average but on the floor with him 85% of the time
    con = simulate(off, dfn, pair=(1, 2), n_games=200)
    rapm = fit(Rapm(half_life_days=1e6, alpha=50.0), con, T0 + timedelta(days=210))
    skill = fit(PlayerSkills(half_life_days=1e6, k_off=10.0), con, T0 + timedelta(days=210))
    rapm_gap = rapm.get(1)[0] - rapm.get(2)[0]
    skill_gap = skill.get(1)[0] - skill.get(2)[0]
    assert rapm_gap > 2.5 * skill_gap  # regression pulls the pair apart; on-court rating can't
    assert rapm_gap > 3


def test_heavy_penalty_shrinks_toward_zero_and_unseen_is_average():
    off, dfn = effects()
    con = simulate(off, dfn, n_games=30)
    light = fit(Rapm(half_life_days=1e6, alpha=10.0), con, T0 + timedelta(days=40))
    heavy = fit(Rapm(half_life_days=1e6, alpha=1e6), con, T0 + timedelta(days=40))
    assert heavy.table.off.abs().mean() < 0.1 * light.table.off.abs().mean()
    assert light.get(999) == (0.0, 0.0)


def test_future_stints_do_not_change_earlier_ratings():
    off, dfn = effects()
    con = simulate(off, dfn, n_games=40)
    t = T0 + timedelta(days=30)
    before = fit(Rapm(alpha=100.0), con, t).get(1)
    con.execute("UPDATE stints SET points_a = 500 WHERE game_id = 'g39'")
    after = fit(Rapm(alpha=100.0), con, t).get(1)
    assert before == pytest.approx(after, abs=1e-6)


def test_stints_with_incomplete_lineups_are_skipped():
    off, dfn = effects()
    con = simulate(off, dfn, n_games=10)
    av = warehouse.result_available_at(T0.date(), T0)
    con.execute("INSERT INTO stints VALUES ('g0',2,1,2,[1,2,3,4],[11,12,13,14,15],720,0,60,2,2,5,5,?)", [av])
    m = fit(Rapm(alpha=100.0), con, T0 + timedelta(days=20))
    assert len(m.table) > 0
