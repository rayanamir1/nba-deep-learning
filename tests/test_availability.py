from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from nbadl import warehouse as wh
from nbadl.features.availability import AvailabilityModel, build_frame

T0 = datetime(2023, 10, 20, 23, tzinfo=timezone.utc)
TEAM, OPP = 1610612738, 1610612747
STAR, BENCH = 101, 102
STARTERS = [101, 103, 104, 105, 106]


def world(n_games=60, seed=0, star_out_games=(), report_late=()):
    """One team with a star and four other starters; star misses `star_out_games` and the report says so
    (unless the game is in report_late, where the report only appears after the decision time)."""
    rng = np.random.default_rng(seed)
    con = wh.connect(":memory:")
    for g in range(n_games):
        tip = T0 + timedelta(days=g)
        av = wh.result_available_at(tip.date(), tip)
        gid = f"g{g}"
        con.execute("INSERT INTO games VALUES (?,?,?,?,?,?,100,100,?,?)",
                    [gid, "2023-24", "regular", tip.date(), TEAM, OPP, tip, av])
        star_out = g in star_out_games
        lineup = [p for p in STARTERS if not (star_out and p == STAR)] + ([107] if star_out else [])
        lineup = lineup[:5]
        con.execute("INSERT INTO stints VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    [gid, 1, TEAM, OPP, lineup, [201, 202, 203, 204, 205], 720, 0, 2880, 100, 100, 100, 100, av])
        if star_out:
            when = tip - timedelta(hours=1 if g in report_late else 20)
            con.execute("INSERT INTO injuries VALUES (?,?,?,?,?,?,?,?)",
                        [when, gid, tip.date(), TEAM, STAR, "Star, A", "Out", when + timedelta(minutes=15)])
    return con


def frame(con, t, **kw):
    with wh.point_in_time(con, t) as c:
        return build_frame(c, **kw)


def test_history_features_use_only_earlier_games():
    con = world()
    f = frame(con, T0 + timedelta(days=100))
    first = f[(f.game_id == "g0")]
    assert len(first) == 0  # nobody has history before the first game
    later = f[(f.game_id == "g20") & (f.player_id == STAR)].iloc[0]
    assert later.app_rate == 1.0 and later.start_rate == 1.0 and abs(later.m_recent - 48.0) < 1e-6


def test_report_status_is_what_was_known_before_decision_time():
    con = world(star_out_games={40}, report_late={})
    f = frame(con, T0 + timedelta(days=100))
    row = f[(f.game_id == "g40") & (f.player_id == STAR)].iloc[0]
    assert row.status == "Out" and row.played == 0.0
    con = world(star_out_games={40}, report_late={40})  # report posted 1h before tip: after the decision time
    row = frame(con, T0 + timedelta(days=100))
    row = row[(row.game_id == "g40") & (row.player_id == STAR)].iloc[0]
    assert row.status == "None"


def test_out_status_predicts_not_playing_and_minutes_rescale_to_240():
    outs = set(range(10, 50, 3))
    con = world(n_games=70, star_out_games=outs)
    f = frame(con, T0 + timedelta(days=100))
    m = AvailabilityModel(c=100.0, min_rows=50).fit(f)
    star_out = f[(f.player_id == STAR) & (f.status == "Out")].head(1)
    star_in = f[(f.player_id == STAR) & (f.status == "None") & (f.game_id == "g60")]
    assert m.predict(star_out).p_play.iloc[0] < 0.2
    assert m.predict(star_in).p_play.iloc[0] > 0.8
    pred = m.predict(f[f.game_id == "g60"])
    assert np.allclose(pred.groupby('team_id').exp_minutes.sum(), 240.0)


def test_target_game_rows_have_no_outcome_and_never_leak_it():
    con = world(n_games=40)
    t = T0 + timedelta(days=30)  # games g31.. are in the future
    target = pd.DataFrame([{"game_id": "g35", "tip_utc": T0 + timedelta(days=35),
                            "home_team_id": TEAM, "away_team_id": OPP}])
    f = frame(con, t, target=target)
    row = f[f.game_id == "g35"]
    assert len(row) > 0 and row.played.isna().all() and row.minutes.isna().all()
    # rewriting the future must not change the target rows' features
    con.execute("UPDATE stints SET lineup_a = [1,2,3,4,5] WHERE game_id = 'g35'")
    g = frame(con, t, target=target)
    pd.testing.assert_frame_equal(f[f.game_id == "g35"].reset_index(drop=True),
                                  g[g.game_id == "g35"].reset_index(drop=True))
