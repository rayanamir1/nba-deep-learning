import numpy as np
import pandas as pd
import pytest

from nbadl.backtest.calibrate import as_scored, walk_forward_calibrate
from nbadl.backtest.metrics import brier, reliability


def synth(n=2500, signal=0.0, seed=0, overconfident=1.0):
    """Games where the market line is the truth plus noise and the model adds `signal` of the true residual."""
    rng = np.random.default_rng(seed)
    line = rng.normal(222, 6, n)
    truth_gap = rng.normal(0, 6, n)             # what the market missed
    actual = line + truth_gap + rng.normal(0, 17, n)
    model = line + signal * truth_gap + rng.normal(0, 6, n) * overconfident  # signal plus pure noise
    tip = pd.Timestamp("2021-10-20", tz="UTC") + pd.to_timedelta(np.arange(n) * 0.25, unit="D")
    return pd.DataFrame({"game_id": [f"g{i}" for i in range(n)], "tip_utc": tip, "decision_t": tip - pd.Timedelta(hours=12),
                         "bet_total": line, "close_total": line, "actual": actual, "mean": model, "sigma": 18.0,
                         "p_over": 0.5, "push": False})


def test_pure_noise_gets_k_near_zero_and_probabilities_near_half():
    cal = walk_forward_calibrate(synth(signal=0.0))
    a = cal[cal.active]
    assert a.k.iloc[-1] < 0.12 and abs(a.p_cal.mean() - 0.5) < 0.03 and a.p_cal.std() < 0.08


def test_real_signal_gets_a_larger_k():
    noisy = walk_forward_calibrate(synth(signal=0.0)).k.dropna().iloc[-1]
    real = walk_forward_calibrate(synth(signal=1.0)).k.dropna().iloc[-1]
    assert real > noisy + 0.15 and real < 1.0


def test_calibration_repairs_overconfident_probabilities():
    df = synth(signal=0.3, overconfident=2.0)
    from nbadl.backtest.replay import p_over, Forecast
    df["p_over"] = [p_over(l, Forecast(m, 18.0)) for l, m in zip(df.bet_total, df["mean"])]
    cal = as_scored(walk_forward_calibrate(df))
    y = (cal.actual > cal.bet_total).astype(int)
    assert brier(cal.p_over, y) < brier(cal.p_over_raw, y)
    r = reliability(cal.p_over, y, bins=5)
    assert (abs(r.mean_p - r.freq_over)[r.n > 100] < 0.06).all()


def test_nothing_is_scored_before_enough_results_are_known():
    cal = walk_forward_calibrate(synth(n=1000), min_games=300)
    assert not cal.active.iloc[:300].any() and cal.active.iloc[-1]


def test_future_outcomes_cannot_change_earlier_calibration():
    df = synth(n=1500, signal=1.0)
    base = walk_forward_calibrate(df)
    tampered = df.copy()
    tampered.loc[tampered.index[-300:], "actual"] += 40  # rewrite the last 300 results
    again = walk_forward_calibrate(tampered)
    # every game decided before those results were knowable is untouched
    cutoff = tampered.tip_utc.iloc[-300] + pd.Timedelta(hours=3)
    early = base.decision_t < cutoff
    assert early.sum() > 1000
    pd.testing.assert_series_equal(base.p_cal[early], again.p_cal[early])
