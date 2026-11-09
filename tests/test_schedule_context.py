import pandas as pd

from nbadl.features.schedule_context import context_features, miles

BOS, LAL, DEN = 1610612738, 1610612747, 1610612743


def sched(rows):
    return pd.DataFrame(rows, columns=["game_id", "home_team_id", "away_team_id", "tip_utc"])


def test_rest_b2b_travel_and_altitude():
    s = sched([
        ("g1", BOS, LAL, "2024-01-01 00:00"),   # LAL's first game: cap rest, no travel
        ("g2", DEN, LAL, "2024-01-02 00:00"),   # LAL on the road again next day: back-to-back, BOS->DEN
        ("g3", LAL, BOS, "2024-01-05 00:00"),   # 3 days off
    ])
    f = context_features(s)
    assert f.loc["g1", "away_rest"] == 5 and f.loc["g1", "away_travel"] == 0
    assert f.loc["g2", "away_b2b"] == 1 and f.loc["g2", "altitude"] == 1
    assert f.loc["g2", "away_travel"] == miles(BOS, DEN) / 1000
    assert f.loc["g3", "home_rest"] == 3 and f.loc["g3", "home_b2b"] == 0
    assert f.loc["g3", "home_travel"] == miles(DEN, LAL) / 1000  # LAL flew in from Denver's game


def test_games_in_last_four_days_and_no_future_peeking():
    s = sched([
        ("a", BOS, LAL, "2024-01-01 00:00"),
        ("b", LAL, BOS, "2024-01-02 00:00"),
        ("c", BOS, DEN, "2024-01-04 00:00"),
    ])
    f = context_features(s)
    assert f.loc["c", "home_g4"] == 2          # BOS played on Jan 1 and Jan 2 within 4 days
    # dropping the later game must not change an earlier game's features
    f2 = context_features(s.iloc[:2])
    assert f2.loc[["a", "b"]].equals(f.loc[["a", "b"]])


def test_miles_sanity():
    assert 2400 < miles(BOS, LAL) < 2700
