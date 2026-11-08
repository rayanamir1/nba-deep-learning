"""Stage 13 scoring. Every function takes plain arrays so it can be tested without a database."""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12  # keeps log(0) finite if a model ever says exactly 0% or 100%


def brier(p, y) -> float:
    """Mean squared gap between predicted probability and the 0/1 outcome. 0.25 = always saying 50%."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def log_loss(p, y) -> float:
    """Average -log(probability given to what actually happened). 0.693 = always saying 50%."""
    p, y = np.clip(np.asarray(p, float), EPS, 1 - EPS), np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def mae(pred, actual) -> float:
    return float(np.mean(np.abs(np.asarray(pred, float) - np.asarray(actual, float))))


def clv_points(side, bet_total, close_total):
    """Points of value vs the closing total. Over bettor wants a LOWER number than the close, so
    over gains (close - bet); an under bettor gains (bet - close). Independent of the game result."""
    side, bet, close = np.asarray(side, float), np.asarray(bet_total, float), np.asarray(close_total, float)
    return side * (close - bet)  # side = +1 over, -1 under


def reliability(p, y, bins: int = 10) -> pd.DataFrame:
    """When the model says ~60%, how often is it over? Rows with count 0 are dropped."""
    df = pd.DataFrame({"p": np.asarray(p, float), "y": np.asarray(y, float)})
    df["bin"] = np.minimum((df.p * bins).astype(int), bins - 1)
    out = df.groupby("bin").agg(n=("y", "size"), mean_p=("p", "mean"), freq_over=("y", "mean")).reset_index()
    return out
