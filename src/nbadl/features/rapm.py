"""Stage 5: regularized adjusted plus-minus (RAPM) from stints, with recency decay.

Every stint gives two observations, one per team on offense:

    points per 100 possessions = intercept + sum(off[p] for p in offense lineup)
                                           + sum(dfn[p] for p in defense lineup)

solved for all players at once by ridge regression. That is what separates two players who share most of their
minutes: only the stints where one plays without the other can tell them apart, and the regression finds them.
The ridge penalty pulls everyone toward 0, so players with little evidence stay near average.

Rows are weighted by possessions (a 30-possession stint says more than a 2-possession one) times
0.5 ** (age / half_life). `alpha` is in "pretend possessions" of weight on the zero prior, so it plays the same
role as k in player_skills.

Output units: points per 100 possessions relative to an average player. dfn follows the Stage 4 convention:
positive = allows more points = worse defense.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge


def stint_frame(con) -> pd.DataFrame:
    """One row per stint and offensive side: offense lineup, defense lineup, points, possessions, tip time."""
    return con.execute("""
        SELECT s.game_id, g.tip_utc, s.lineup_a AS off, s.lineup_b AS dfn, s.points_a AS pts, s.poss_a AS poss
        FROM pit_stints s JOIN pit_games g USING (game_id) WHERE s.poss_a > 0 AND g.tip_utc IS NOT NULL
          AND len(s.lineup_a) = 5 AND len(s.lineup_b) = 5
        UNION ALL
        SELECT s.game_id, g.tip_utc, s.lineup_b, s.lineup_a, s.points_b, s.poss_b
        FROM pit_stints s JOIN pit_games g USING (game_id) WHERE s.poss_b > 0 AND g.tip_utc IS NOT NULL
          AND len(s.lineup_a) = 5 AND len(s.lineup_b) = 5""").df()


class Rapm:
    def __init__(self, half_life_days: float = 365.0, alpha: float = 2000.0, min_weight: float = 0.01):
        self.half_life_days, self.alpha, self.min_weight = half_life_days, alpha, min_weight
        self.table = pd.DataFrame(columns=["off", "def", "n_poss"])
        self.intercept = 0.0

    def fit(self, con, t: datetime) -> "Rapm":
        df = stint_frame(con)
        if df.empty:
            return self
        age = (pd.Timestamp(t) - pd.to_datetime(df.tip_utc, utc=True)).dt.total_seconds() / 86400
        df["w"] = 0.5 ** (age.clip(lower=0) / self.half_life_days)
        df = df[df.w >= self.min_weight].reset_index(drop=True)
        players = sorted({p for lu in df.off for p in lu} | {p for lu in df.dfn for p in lu})
        idx = {p: i for i, p in enumerate(players)}
        n, m = len(players), len(df)
        rows = np.repeat(np.arange(m), 5)
        off_cols = np.fromiter((idx[p] for lu in df.off for p in lu), dtype=np.int64, count=5 * m)
        def_cols = np.fromiter((n + idx[p] for lu in df.dfn for p in lu), dtype=np.int64, count=5 * m)
        X = sparse.csr_matrix((np.ones(10 * m), (np.r_[rows, rows], np.r_[off_cols, def_cols])), shape=(m, 2 * n))
        y = 100.0 * df.pts.to_numpy() / df.poss.to_numpy()
        w = df.poss.to_numpy() * df.w.to_numpy()
        # sklearn's ridge objective is sum(w * err^2) + alpha * |coef|^2 : alpha is in units of possession-weight
        fit = Ridge(alpha=self.alpha, fit_intercept=True, solver="sparse_cg", tol=1e-5, max_iter=2000).fit(X, y, sample_weight=w)
        poss = pd.Series(np.bincount(off_cols, weights=np.repeat(w, 5), minlength=n), index=players)
        self.intercept = float(fit.intercept_)
        self.table = pd.DataFrame({"off": fit.coef_[:n], "def": fit.coef_[n:], "n_poss": poss}, index=players)
        return self

    def get(self, player_id: int) -> tuple[float, float]:
        """(offense, defense) relative to an average player; an unseen player is exactly average."""
        if player_id in self.table.index:
            r = self.table.loc[player_id]
            return float(r["off"]), float(r["def"])
        return 0.0, 0.0
