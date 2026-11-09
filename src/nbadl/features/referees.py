"""Stage 6: referee effects on the total, shrunk and time-decayed.

Target is the market's miss, e = actual total - consensus closing total, so an official's effect is only what the
market has NOT already priced. Each game has ~3 officials; the design matrix has one column per official (1 if he
worked the game) and ridge regression splits the game's miss among its crew. The ridge penalty (alpha, in pretend
games) pulls officials with few games to 0.

crew_effect(officials) = sum of their coefficients, in points of total.
Only finished games with a visible closing line enter a fit (everything comes through the pit_* views).
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge


class RefereeEffects:
    def __init__(self, half_life_days: float = 730.0, alpha: float = 600.0, min_weight: float = 0.01):
        self.half_life_days, self.alpha, self.min_weight = half_life_days, alpha, min_weight
        self.coef: dict[int, float] = {}
        self.intercept = 0.0

    def fit(self, con, t: datetime) -> "RefereeEffects":
        df = con.execute("""
            SELECT o.game_id, o.official_id, g.tip_utc, g.home_pts + g.away_pts - c.close_total AS e
            FROM pit_officials o JOIN pit_games g USING (game_id)
            JOIN (SELECT game_id, avg(total) AS close_total FROM pit_odds_lines
                  WHERE line_type = 'close' AND game_id IS NOT NULL GROUP BY 1) c USING (game_id)
            WHERE g.tip_utc IS NOT NULL AND g.home_pts > 0""").df()
        self.coef = {}
        if df.empty:
            return self
        games = df.groupby("game_id").agg(tip=("tip_utc", "first"), e=("e", "first"))
        age = (pd.Timestamp(t) - pd.to_datetime(games.tip, utc=True)).dt.total_seconds() / 86400
        w = 0.5 ** (age.clip(lower=0) / self.half_life_days)
        keep = w[w >= self.min_weight].index
        df = df[df.game_id.isin(keep)]
        gi = {g: i for i, g in enumerate(keep)}
        refs = sorted(df.official_id.unique())
        ri = {r: i for i, r in enumerate(refs)}
        X = sparse.csr_matrix((np.ones(len(df)), (df.game_id.map(gi), df.official_id.map(ri))),
                              shape=(len(gi), len(refs)))
        y, sw = games.loc[keep, "e"].to_numpy(), w.loc[keep].to_numpy()
        fit = Ridge(alpha=self.alpha).fit(X, y, sample_weight=sw)
        self.intercept = float(fit.intercept_)
        self.coef = {r: float(c) for r, c in zip(refs, fit.coef_)}
        return self

    def crew_effect(self, official_ids) -> float:
        """Points added to the total by this crew (unseen officials contribute 0)."""
        return float(sum(self.coef.get(int(o), 0.0) for o in official_ids))
