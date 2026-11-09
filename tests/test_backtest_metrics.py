import numpy as np
import pytest

from nbadl.backtest import metrics as m

P = [0.60, 0.60, 0.40, 0.55]
Y = [1, 0, 0, 1]


def test_brier_matches_hand_calculation():
    assert m.brier(P, Y) == pytest.approx(0.2206, abs=1e-4)
    assert m.brier([0.5] * 4, Y) == pytest.approx(0.25)


def test_log_loss_matches_hand_calculation():
    assert m.log_loss(P, Y) == pytest.approx(0.6342, abs=1e-3)
    assert m.log_loss([0.5] * 4, Y) == pytest.approx(np.log(2))


def test_log_loss_survives_certainty_that_is_wrong():
    assert np.isfinite(m.log_loss([1.0], [0]))


def test_clv_sign_convention():
    # over at 228.5, closes 230 -> +1.5 ; under at 228.5, closes 230 -> -1.5
    assert m.clv_points([1], [228.5], [230.0])[0] == 1.5
    assert m.clv_points([-1], [228.5], [230.0])[0] == -1.5


def test_reliability_bins():
    r = m.reliability([0.61, 0.62, 0.99, 1.0], [1, 0, 1, 1])
    assert r.n.sum() == 4 and r.iloc[-1].n == 2  # p=1.0 lands in the top bin, not bin 10
