from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from nbadl import warehouse as wh
from nbadl.backtest.baselines import LeagueAverageModel
from nbadl.backtest.replay import (Forecast, HoldoutLockedError, LeakageDetected, holdout_seasons, p_over,
                                   replay, summarize)


def seed(con, season, start, days, playoffs_last=3):
    games, odds = [], []
    for d in range(days):
        tip = start + timedelta(days=d)
        for k in range(2):
            gid = f"{season}-{d}-{k}"
            total = 215 + (d * 3 + k * 5) % 15
            stype = "playoffs" if d >= days - playoffs_last else "regular"
            games.append({"game_id": gid, "season": season, "season_type": stype, "game_date": tip.date(),
                          "home_team_id": 1 + k, "away_team_id": 10 + k, "home_pts": total // 2,
                          "away_pts": total - total // 2, "tip_utc": tip,
                          "available_at": wh.result_available_at(tip.date(), tip)})
            for book in ("a", "b"):
                for lt, tot, av in (("open", 218.0, wh.open_available_at(tip.date(), tip)),
                                    ("close", 219.5, wh.close_available_at(tip.date(), tip))):
                    odds.append({"game_id": gid, "sbr_game_id": gid, "sportsbook": book, "line_type": lt,
                                 "total": tot, "over_odds": -110, "under_odds": -110, "observed_at": None,
                                 "available_at": av})
    for name, rows in (("games", games), ("odds_lines", odds)):
        df = pd.DataFrame(rows)
        con.register("df", df)
        con.execute(f"INSERT INTO {name} SELECT * FROM df")
        con.unregister("df")


@pytest.fixture
def con():
    c = wh.connect(":memory:")
    seed(c, "2021-22", datetime(2021, 10, 20, 23, tzinfo=timezone.utc), 120)
    seed(c, "2022-23", datetime(2022, 10, 20, 23, tzinfo=timezone.utc), 120)
    seed(c, "2023-24", datetime(2023, 10, 20, 23, tzinfo=timezone.utc), 120)
    return c


def test_p_over_is_half_when_line_equals_mean():
    assert p_over(220.0, Forecast(220.0, 18.0)) == pytest.approx(0.5)
    assert p_over(238.0, Forecast(220.0, 18.0)) == pytest.approx(0.1587, abs=1e-3)  # one sigma above


def test_holdout_is_last_two_completed_seasons(con):
    assert holdout_seasons(con) == ["2022-23", "2023-24"]


def test_holdout_is_locked_without_final_flag(con):
    with pytest.raises(HoldoutLockedError):
        replay(con, LeagueAverageModel(), ["2023-24"])
    assert len(replay(con, LeagueAverageModel(), ["2023-24"], final=True, leak_check_every=0)) > 0


def test_replay_scores_non_holdout_season_and_passes_leak_checks(con):
    df = replay(con, LeagueAverageModel(), ["2021-22"], leak_check_every=40)
    assert len(df) == 240
    assert (df.decision_t < df.tip_utc).all()
    s = summarize(df)
    assert s["n_games"] == 240 - df.push.sum() and df.push.sum() > 0 and 0 <= s["brier"] <= 1 and s["mae_close"] > 0


def test_bet_line_is_the_open_line_not_the_close(con):
    df = replay(con, LeagueAverageModel(), ["2021-22"], leak_check_every=0)
    assert (df.bet_total == 218.0).all() and (df.close_total == 219.5).all()
    assert (df.clv == df.side * 1.5).all()


def test_refits_weekly_not_every_game(con):
    class Counting(LeagueAverageModel):
        fits = 0

        def fit(self, con, t):
            Counting.fits += 1
            super().fit(con, t)

    replay(con, Counting(), ["2021-22"], refit_days=7, leak_check_every=0)
    assert 15 <= Counting.fits <= 20  # ~120 days / 7


def test_leaky_model_is_caught(con):
    class Leaky:
        def fit(self, con, t):
            pass

        def predict(self, con, game, t):  # reads the BASE table, not pit_games
            m = con.execute("SELECT avg(home_pts + away_pts) FROM games").fetchone()[0]
            return Forecast(float(m), 18.0)

    with pytest.raises(LeakageDetected):
        replay(con, Leaky(), ["2021-22"], leak_check_every=10)
