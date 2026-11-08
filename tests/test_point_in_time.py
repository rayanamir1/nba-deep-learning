"""The system's core invariant: a feature computed for decision time t must not change
when anything with available_at >= t changes."""
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from nbadl import warehouse as wh

T0 = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)


def seed(con):
    rows = []
    for i in range(20):
        tip = T0 + timedelta(days=i)
        rows.append({"game_id": f"g{i}", "season": "2023-24", "season_type": "regular",
                     "game_date": tip.date(), "home_team_id": 1, "away_team_id": 2,
                     "home_pts": 110 + i, "away_pts": 100 + i, "tip_utc": tip,
                     "available_at": wh.result_available_at(tip.date(), tip)})
    df = pd.DataFrame(rows)
    con.register("df", df)
    con.execute("INSERT INTO games SELECT * FROM df")


def avg_total_feature(con, t):
    with wh.point_in_time(con, t) as c:
        return c.execute("SELECT avg(home_pts + away_pts), count(*) FROM pit_games").fetchone()


def test_feature_ignores_future_rows():
    con = wh.connect(":memory:")
    seed(con)
    t = T0 + timedelta(days=10)
    before = avg_total_feature(con, t)
    # corrupt everything that was not knowable at t
    con.execute("UPDATE games SET home_pts = home_pts + 999 WHERE available_at >= ?", [t])
    assert avg_total_feature(con, t) == before


def test_game_result_not_visible_at_tip():
    con = wh.connect(":memory:")
    seed(con)
    tip = T0 + timedelta(days=5)
    with wh.point_in_time(con, tip) as c:
        ids = {r[0] for r in c.execute("SELECT game_id FROM pit_games").fetchall()}
    assert "g5" not in ids and "g4" in ids


def test_closing_line_never_before_tip():
    tip = datetime(2024, 1, 10, 0, 30, tzinfo=timezone.utc)
    assert wh.close_available_at(date(2024, 1, 9), tip) == tip
    assert wh.open_available_at(date(2024, 1, 9), tip) < tip


def test_naive_time_rejected():
    con = wh.connect(":memory:")
    with pytest.raises(ValueError):
        with wh.point_in_time(con, datetime(2024, 1, 1)):
            pass


def test_assert_no_future_rows():
    t = T0
    df = pd.DataFrame({"available_at": [T0 - timedelta(hours=1), T0]})
    with pytest.raises(wh.LeakageError):
        wh.assert_no_future_rows(df, t)
