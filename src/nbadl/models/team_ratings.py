"""Stage 9 baseline: total = possessions x points per possession, with additive team ratings.

    log(possessions)        = pace0 + pace[home] + pace[away]                       (ridge regression)
    log(points per poss)    = eff0 + off[team] + def[opponent] + home_edge * is_home  (ridge Poisson GLM)
    expected total          = possessions * (ppp[home] + ppp[away])

Both fits use every prior game, each down-weighted by 0.5 ** (age / half_life), so last month matters more than
three seasons ago. The ridge penalty shrinks every rating toward the league average, which is what stops a
team that scored 130 in one game from being rated as a 130-point team.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import PoissonRegressor, Ridge

from ..backtest.replay import Forecast

OWN_POSSESSIONS = "tg.fga + 0.44 * tg.fta - tg.oreb + tg.tov"  # the standard box-score estimate


class TeamRatingsModel:
    def __init__(self, half_life_days: float = 365.0, alpha_pace: float = 1e-3, alpha_eff: float = 1e-3,
                 sigma_scale: float = 1.0, min_games: int = 100, min_weight: float = 0.01,
                 default: tuple[float, float] = (220.0, 19.0)):
        self.half_life_days, self.alpha_pace, self.alpha_eff = half_life_days, alpha_pace, alpha_eff
        self.sigma_scale, self.min_games, self.min_weight, self.default = sigma_scale, min_games, min_weight, default
        self.ready = False

    # --- training data ----------------------------------------------------------------
    def _training_frame(self, con, t: datetime) -> pd.DataFrame:
        df = con.execute(f"""
            SELECT tg.game_id, tg.team_id, tg.opp_team_id, tg.is_home, tg.pts, g.tip_utc,
                   {OWN_POSSESSIONS} AS own_poss
            FROM pit_team_games tg JOIN pit_games g USING (game_id)
            WHERE g.tip_utc IS NOT NULL""").df()
        if df.empty:
            return df
        # both teams share the game's possessions; average the two estimates
        df["poss"] = df.groupby("game_id").own_poss.transform("mean")
        age_days = (pd.Timestamp(t) - pd.to_datetime(df.tip_utc, utc=True)).dt.total_seconds() / 86400
        df["w"] = 0.5 ** (age_days.clip(lower=0) / self.half_life_days)
        return df[df.w >= self.min_weight].reset_index(drop=True)

    # --- fit -----------------------------------------------------------------------------
    def fit(self, con, t: datetime) -> None:
        df = self._training_frame(con, t)
        self.ready = False
        if df.game_id.nunique() < self.min_games:
            return
        teams = sorted(set(df.team_id) | set(df.opp_team_id))
        self.idx = {tm: i for i, tm in enumerate(teams)}
        n = len(teams)
        ti, oi = df.team_id.map(self.idx).to_numpy(), df.opp_team_id.map(self.idx).to_numpy()

        # pace: one row per game, a +1 for each of the two teams
        games = df[df.is_home].reset_index(drop=True)  # exactly one home row per game
        hi = games.team_id.map(self.idx).to_numpy()
        ai = games.opp_team_id.map(self.idx).to_numpy()
        r = np.arange(len(games))
        Xp = sparse.csr_matrix((np.ones(2 * len(games)), (np.r_[r, r], np.r_[hi, ai])), shape=(len(games), n))
        pace = Ridge(alpha=self.alpha_pace, fit_intercept=True).fit(Xp, np.log(games.poss), sample_weight=games.w)
        self.pace0, self.pace = float(pace.intercept_), pace.coef_

        # efficiency: one row per team-game; weight = exposure (possessions) x recency, target = points / possession
        rows = np.arange(len(df))
        home_flag = df.is_home.astype(float).to_numpy()
        Xe = sparse.csr_matrix(
            (np.r_[np.ones(2 * len(df)), home_flag], (np.r_[rows, rows, rows], np.r_[ti, n + oi, np.full(len(df), 2 * n)])),
            shape=(len(df), 2 * n + 1))
        glm = PoissonRegressor(alpha=self.alpha_eff, max_iter=500).fit(
            Xe, df.pts / df.poss, sample_weight=df.poss * df.w)
        self.eff0, self.off, self.dfn, self.home_edge = (
            float(glm.intercept_), glm.coef_[:n], glm.coef_[n:2 * n], float(glm.coef_[2 * n]))
        self.ready = True

        # game-level spread from the last year of training residuals (in-sample, so slightly optimistic)
        recent = games[games.tip_utc >= pd.Timestamp(t) - pd.Timedelta(days=365)]
        total = df.groupby("game_id").pts.sum()
        pred = np.array([self._mean_total(h, a) for h, a in zip(recent.team_id, recent.opp_team_id)])
        resid = total.reindex(recent.game_id).to_numpy() - pred
        self.sigma = float(np.sqrt(np.mean(resid ** 2))) if len(resid) > 30 else self.default[1]

    # --- predict -------------------------------------------------------------------------
    def _get(self, arr, team) -> float:
        i = self.idx.get(team)
        return float(arr[i]) if i is not None else 0.0  # a team never seen gets the league average

    def _mean_total(self, home: int, away: int) -> float:
        poss = np.exp(self.pace0 + self._get(self.pace, home) + self._get(self.pace, away))
        ppp_h = np.exp(self.eff0 + self._get(self.off, home) + self._get(self.dfn, away) + self.home_edge)
        ppp_a = np.exp(self.eff0 + self._get(self.off, away) + self._get(self.dfn, home))
        return float(poss * (ppp_h + ppp_a))

    def predict(self, con, game: dict, t: datetime) -> Forecast:
        if not self.ready:
            return Forecast(*self.default)
        return Forecast(self._mean_total(game["home_team_id"], game["away_team_id"]), self.sigma * self.sigma_scale)
