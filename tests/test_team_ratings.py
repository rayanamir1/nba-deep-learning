from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from nbadl import warehouse as wh
from nbadl.backtest.replay import replay
from nbadl.models.team_ratings import TeamRatingsModel

T0 = datetime(2022, 10, 20, 23, tzinfo=timezone.utc)
N_TEAMS = 8


def world(seed=0, days=160, drift=0.0):
    """Eight teams with known pace and offense/defense effects; games simulated from the model's own form."""
    rng = np.random.default_rng(seed)
    pace = rng.normal(0, 0.04, N_TEAMS)
    off = rng.normal(0, 0.05, N_TEAMS)
    dfn = rng.normal(0, 0.05, N_TEAMS)
    games, tgs = [], []
    for d in range(days):
        tip = T0 + timedelta(days=d)
        order = rng.permutation(N_TEAMS)
        for k in range(0, N_TEAMS, 2):
            h, a = int(order[k]), int(order[k + 1])
            poss = np.exp(np.log(98) + pace[h] + pace[a]) * rng.normal(1, 0.01)
            lift = 1 + drift * d / days  # league-wide scoring drifts upward over time
            ph = rng.poisson(lift * poss * np.exp(np.log(1.12) + off[h] + dfn[a] + 0.02))
            pa = rng.poisson(lift * poss * np.exp(np.log(1.12) + off[a] + dfn[h]))
            gid = f"g{d}-{k}"
            av = wh.result_available_at(tip.date(), tip)
            games.append({"game_id": gid, "season": "2022-23", "season_type": "regular", "game_date": tip.date(),
                          "home_team_id": h, "away_team_id": a, "home_pts": ph, "away_pts": pa, "tip_utc": tip,
                          "available_at": av})
            for tm, opp, home, pts in ((h, a, True, ph), (a, h, False, pa)):
                # fabricate a box line whose possession estimate equals poss: fga + .44 fta - oreb + tov
                tgs.append({"game_id": gid, "team_id": tm, "opp_team_id": opp, "is_home": home, "pts": pts,
                            "fga": int(round(poss)) - 10, "fta": 0, "oreb": 0, "tov": 10, "available_at": av})
    return pace, off, dfn, pd.DataFrame(games), pd.DataFrame(tgs)


def load(con, games, tgs, odds=None):
    for name, df in (("games", games), ("team_games", tgs)):
        con.register("df", df)
        con.execute(f"INSERT INTO {name} SELECT * FROM df")
        con.unregister("df")


@pytest.fixture(scope="module")
def fitted():
    pace, off, dfn, games, tgs = world()
    con = wh.connect(":memory:")
    load(con, games, tgs)
    t = T0 + timedelta(days=161)
    m = TeamRatingsModel(half_life_days=1e6, alpha_pace=1e-4, alpha_eff=1e-4)  # no decay, light shrinkage
    with wh.point_in_time(con, t) as c:
        m.fit(c, t)
    return m, pace, off, dfn


def test_recovers_known_team_effects(fitted):
    m, pace, off, dfn = fitted
    order = [m.idx[i] for i in range(N_TEAMS)]
    assert np.corrcoef(m.pace[order], pace)[0, 1] > 0.9
    assert np.corrcoef(m.off[order], off)[0, 1] > 0.8
    assert np.corrcoef(m.dfn[order], dfn)[0, 1] > 0.8
    assert m.home_edge == pytest.approx(0.02, abs=0.02)


def test_prediction_is_possessions_times_ppp(fitted):
    m, *_ = fitted
    f = m.predict(None, {"home_team_id": 0, "away_team_id": 1}, None)
    assert 190 < f.mean < 260 and 5 < f.sigma < 30


def test_heavier_ridge_shrinks_ratings_toward_zero():
    _, _, _, games, tgs = world()
    con = wh.connect(":memory:")
    load(con, games, tgs)
    t = T0 + timedelta(days=161)
    spread = {}
    for alpha in (1e-4, 1.0):
        m = TeamRatingsModel(half_life_days=1e6, alpha_eff=alpha, alpha_pace=alpha)
        with wh.point_in_time(con, t) as c:
            m.fit(c, t)
        spread[alpha] = float(np.std(m.off))
    assert spread[1.0] < 0.25 * spread[1e-4]


def test_recent_games_outweigh_old_ones():
    m = TeamRatingsModel(half_life_days=30)
    _, _, _, games, tgs = world()
    con = wh.connect(":memory:")
    load(con, games, tgs)
    t = T0 + timedelta(days=161)
    with wh.point_in_time(con, t) as c:
        df = m._training_frame(c, t)
    newest = df.loc[df.tip_utc.idxmax(), "w"]
    assert newest > 0.95 and df.w.min() < 0.1 and (df.w >= m.min_weight).all()


def test_model_passes_the_harness_leak_check():
    _, _, _, games, tgs = world(days=140)
    con = wh.connect(":memory:")
    load(con, games, tgs)
    # odds so the harness has a line and a close to score against
    odds = []
    for g in games.itertuples():
        for lt, tot, av in (("open", 220.0, wh.open_available_at(g.game_date, g.tip_utc)),
                            ("close", 221.0, wh.close_available_at(g.game_date, g.tip_utc))):
            odds.append({"game_id": g.game_id, "sbr_game_id": g.game_id, "sportsbook": "a", "line_type": lt,
                         "total": tot, "over_odds": -110, "under_odds": -110, "observed_at": None, "available_at": av})
    con.register("o", pd.DataFrame(odds))
    con.execute("INSERT INTO odds_lines SELECT * FROM o")
    con.execute("INSERT INTO games SELECT 'p'||i, '2021-22', 'playoffs', DATE '2022-06-01', 1,2,100,100, TIMESTAMPTZ '2022-06-01 00:00:00+00', TIMESTAMPTZ '2022-06-02 00:00:00+00' FROM range(2) t(i)")
    con.execute("INSERT INTO games SELECT 'q'||i, '2022-23', 'playoffs', DATE '2023-06-01', 1,2,100,100, TIMESTAMPTZ '2023-06-01 00:00:00+00', TIMESTAMPTZ '2023-06-02 00:00:00+00' FROM range(2) t(i)")
    model = TeamRatingsModel(half_life_days=1e6, min_games=40)
    # rows after day 60 only; leak check on every 20th game proves nothing future leaks into fit/predict
    df = replay(con, model, ["2022-23"], final=True, leak_check_every=20)
    assert len(df) > 200 and df.p_over.between(0, 1).all()


def test_level_offset_tracks_league_drift():
    _, _, _, games, tgs = world(days=200, drift=0.10)
    con = wh.connect(":memory:")
    load(con, games, tgs)
    t = T0 + timedelta(days=201)
    recent_actual = (games.tail(40).home_pts + games.tail(40).away_pts).mean()
    bias = {}
    for name, level_hl in (("tracking", 30.0), ("stale", 1e6)):
        m = TeamRatingsModel(half_life_days=1e6, level_half_life_days=level_hl, alpha_pace=1e-3, alpha_eff=1e-3)
        with wh.point_in_time(con, t) as c:
            m.fit(c, t)
        bias[name] = abs(np.mean([m._mean_total(h, a) for h, a in zip(games.tail(40).home_team_id, games.tail(40).away_team_id)])
                         - recent_actual)
    assert bias["tracking"] < 0.4 * bias["stale"] and bias["tracking"] < 6
