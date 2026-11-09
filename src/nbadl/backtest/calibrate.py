"""Stage 11: shrink the model toward the market line, then calibrate the probability.

Works on the replay() output, walk-forward: a game's k and Platt curve are fitted only on earlier games whose
results were already knowable at that game's decision time (tip + RESULT_DELAY). Nothing here sees the future.

  d = model_mean - line            how far we disagree with the market
  e = actual - line                how far the game actually landed from the market
  k = sum(d*e) / sum(d*d)          regression slope through the origin: the share of our disagreement that was real
  shrunk mean = line + k*d,        sigma = rms(e - k*d)
  P(over) = Phi(k*d / sigma)       then a Platt (logistic) map fixes any remaining systematic over/under-confidence
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

from .. import config

MIN_SIGMA = 1.0


def _fit(train: pd.DataFrame, platt: bool) -> dict:
    d = (train["mean"] - train.bet_total).to_numpy()
    e = (train.actual - train.bet_total).to_numpy()
    k = float(np.clip((d * e).sum() / max((d * d).sum(), 1e-9), 0.0, 1.0))
    sigma = max(float(np.sqrt(np.mean((e - k * d) ** 2))), MIN_SIGMA)
    fit = {"k": k, "sigma": sigma, "platt": None}
    if platt:
        y = (e > 0).astype(int)
        z = logit(np.clip(norm.cdf(k * d / sigma), 1e-6, 1 - 1e-6)).reshape(-1, 1)
        if 0 < y.sum() < len(y):
            lr = LogisticRegression(C=1e6).fit(z, y)  # C huge = effectively unregularised
            fit["platt"] = (float(lr.intercept_[0]), float(lr.coef_[0, 0]))
    return fit


def _apply(fit: dict, mean: float, line: float) -> tuple[float, float, float]:
    shrunk = line + fit["k"] * (mean - line)
    p = float(norm.cdf((shrunk - line) / fit["sigma"]))  # P(total > line) with total ~ N(shrunk, sigma)
    if fit["platt"] is not None:
        a, b = fit["platt"]
        p = float(expit(a + b * logit(min(max(p, 1e-6), 1 - 1e-6))))
    return shrunk, fit["sigma"], p


def walk_forward_calibrate(df: pd.DataFrame, *, platt: bool = True, refit_days: int = 7, min_games: int = 300,
                           result_delay=config.RESULT_DELAY) -> pd.DataFrame:
    """Return df (sorted by decision time) with k, mean_shrunk, sigma_shrunk, p_cal and `active`.
    Rows before min_games results are known stay inactive (p_cal = 0.5) and should be excluded from scoring."""
    df = df.sort_values("decision_t").reset_index(drop=True)
    settled_at = (pd.to_datetime(df.tip_utc, utc=True) + result_delay).to_numpy()
    decided = pd.to_datetime(df.decision_t, utc=True).to_numpy()
    usable = ~df.push.to_numpy()
    out = {c: np.full(len(df), np.nan) for c in ("k", "mean_shrunk", "sigma_shrunk", "p_cal")}
    active = np.zeros(len(df), bool)
    fit, last = None, None
    for i in range(len(df)):
        if last is None or (decided[i] - last) >= np.timedelta64(refit_days, "D"):
            known = (settled_at < decided[i]) & usable
            if known.sum() >= min_games:
                fit, last = _fit(df[known], platt), decided[i]
        if fit is None:
            out["p_cal"][i] = 0.5
            continue
        shrunk, sigma, p = _apply(fit, df["mean"].iat[i], df.bet_total.iat[i])
        out["k"][i], out["mean_shrunk"][i], out["sigma_shrunk"][i], out["p_cal"][i] = fit["k"], shrunk, sigma, p
        active[i] = True
    res = df.copy()
    for c, v in out.items():
        res[c] = v
    res["active"] = active
    return res


def as_scored(cal: pd.DataFrame) -> pd.DataFrame:
    """A frame summarize() can score: only active rows, with the calibrated probability and shrunk forecast."""
    a = cal[cal.active].copy()
    a["p_over_raw"], a["mean_raw"], a["sigma_raw"] = a.p_over, a["mean"], a.sigma
    a["p_over"], a["mean"], a["sigma"] = a.p_cal, a.mean_shrunk, a.sigma_shrunk
    return a
