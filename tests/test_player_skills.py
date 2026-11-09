from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from nbadl import warehouse
from nbadl.features.player_skills import PlayerSkills, estimate_k

T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)


def fit(sk, con, t):
    with warehouse.point_in_time(con, t) as c:
        return sk.fit(c, t)


def build(rows):
    """rows: (game_id, day_offset, lineup_a, lineup_b, pts_a, pts_b, poss). One stint per game."""
    con = warehouse.connect(":memory:")
    for gid, day, la, lb, pa, pb, poss in rows:
        tip = datetime(2023, 10, 1, 23, tzinfo=timezone.utc) + timedelta(days=day)
        av = warehouse.result_available_at(tip.date(), tip)
        con.execute("INSERT INTO games VALUES (?,?,?,?,?,?,?,?,?,?)",
                    [gid, "2023-24", "regular", tip.date(), 1, 2, pa, pb, tip, av])
        con.execute("INSERT INTO stints VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    [gid, 1, 1, 2, la, lb, 720, 0, 2880, pa, pb, poss, poss, av])
    return con


A = [1, 2, 3, 4, 5]
B = [11, 12, 13, 14, 15]


def test_few_possessions_are_shrunk_toward_league():
    # one hot game for lineup A against a normal league of many B-vs-C games
    rows = [(f"g{i}", i, B, [21, 22, 23, 24, 25], 100, 100, 100) for i in range(60)]
    rows.append(("hot", 61, A, B, 150, 100, 100))  # A scores 150 on 100 possessions
    con = build(rows)
    sk = fit(PlayerSkills(half_life_days=1e6, k_off=300.0), con, T0)
    raw = 100 * (150 / 100 - sk.league["ppp"])
    assert 0 < sk.get(1)[0] < raw / 3  # nudged up, nowhere near the raw +35


def test_more_data_means_less_shrinkage():
    rows = [(f"g{i}", i, B, [21, 22, 23, 24, 25], 100, 100, 100) for i in range(40)]
    rows += [(f"a{i}", 50 + i, A, B, 120, 100, 100) for i in range(3)]
    rows += [(f"c{i}", 50 + i, [31, 32, 33, 34, 35], B, 120, 100, 100) for i in range(30)]
    con = build(rows)
    sk = fit(PlayerSkills(half_life_days=1e6, k_off=300.0), con, T0)
    assert sk.get(31)[0] > sk.get(1)[0] > 0


def test_unseen_player_is_average_and_decay_forgets():
    rows = [(f"old{i}", i, A, B, 130, 100, 100) for i in range(20)]
    rows += [(f"new{i}", 400 + i, A, B, 100, 100, 100) for i in range(20)]
    con = build(rows)
    slow = fit(PlayerSkills(half_life_days=1e6, k_off=50.0), con, T0)
    fast = fit(PlayerSkills(half_life_days=30.0, k_off=50.0), con, T0)
    assert slow.get(999) == (0.0, 0.0, 0.0)
    assert fast.get(1)[0] < slow.get(1)[0]  # short memory forgets the old hot stretch


def test_estimate_k_recovers_noise_to_talent_ratio():
    rng = np.random.default_rng(0)
    n = np.full(500, 2000.0)
    tau, sigma2 = 0.03, 1.4
    true = 1.15 + rng.normal(0, tau, 500)
    obs = true + rng.normal(0, np.sqrt(sigma2 / n))
    k = estimate_k(obs, n, 1.15, sigma2)
    assert k == pytest.approx(sigma2 / tau ** 2, rel=0.25)


def test_future_games_do_not_change_earlier_skills():
    rows = [(f"g{i}", i, A, B, 110, 100, 100) for i in range(30)]
    con = build(rows)
    t = datetime(2023, 10, 20, tzinfo=timezone.utc)
    before = fit(PlayerSkills(k_off=100.0), con, t).get(1)
    con.execute("UPDATE stints SET points_a = 300, points_b = 0 WHERE game_id = 'g29'")
    after = fit(PlayerSkills(k_off=100.0), con, t).get(1)
    assert before == pytest.approx(after, abs=1e-9)
