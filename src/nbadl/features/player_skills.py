"""Stage 4: shrunk, time-decayed player skills from stints.

For every player we accumulate, over the games they played, what their team did while they were on the floor:

    points scored / own possessions      -> offense   (higher is better)
    points allowed / opponent possessions -> defense   (lower is better)
    possessions per 48 minutes (both teams) -> pace

Each game counts 0.5 ** (age / half_life), so last month matters more than two seasons ago. Raw rates are noisy
when the sample is small (a player with 38 possessions is mostly luck), so each is shrunk toward the league rate:

    shrunk = (weighted_points + k * league_rate) / (weighted_possessions + k)

k is the number of "pretend league-average possessions" added as a prior. Left as None it is estimated from the data
(empirical Bayes): the spread of players' rates splits into real talent (tau^2) and sampling noise (sigma^2 / n),
and k = sigma^2 / tau^2.

Output units: offense / defense in points per 100 possessions relative to the league (defense: positive = allows
more = worse), pace in possessions per 48 minutes relative to the league.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

MIN_PRIOR_POSS = 200.0  # players with fewer decayed possessions don't inform the prior estimate


def player_games(con) -> pd.DataFrame:
    """One row per (player, game) from the point-in-time stints view."""
    return con.execute("""
        WITH s AS (
            SELECT game_id, unnest(lineup_a) AS player_id, seconds, points_a AS pf, points_b AS pa, poss_a AS qf, poss_b AS qa
            FROM pit_stints
            UNION ALL
            SELECT game_id, unnest(lineup_b), seconds, points_b, points_a, poss_b, poss_a FROM pit_stints)
        SELECT s.player_id, s.game_id, g.tip_utc,
               SUM(pf) AS pf, SUM(pa) AS pa, SUM(qf) AS qf, SUM(qa) AS qa, SUM(seconds) AS seconds
        FROM s JOIN pit_games g USING (game_id) WHERE g.tip_utc IS NOT NULL
        GROUP BY s.player_id, s.game_id, g.tip_utc""").df()


def estimate_k(rate: np.ndarray, n: np.ndarray, league: float, sigma2: float) -> float:
    """Method-of-moments prior strength: E[(rate - league)^2] = tau^2 + sigma^2 / n."""
    ok = n >= MIN_PRIOR_POSS
    if ok.sum() < 20:
        return 500.0
    tau2 = float(np.mean((rate[ok] - league) ** 2 - sigma2 / n[ok]))
    return float(sigma2 / max(tau2, 1e-9))


class PlayerSkills:
    def __init__(self, half_life_days: float = 365.0, k_off: float | None = None, k_def: float | None = None,
                 k_pace_seconds: float | None = None, min_weight: float = 0.01):
        self.half_life_days, self.min_weight = half_life_days, min_weight
        self.k_off, self.k_def, self.k_pace_seconds = k_off, k_def, k_pace_seconds
        self.table = pd.DataFrame(columns=["off", "def", "pace", "n_poss", "n_games"])
        self.league = {"ppp": 1.15, "pace": 100.0}

    def fit(self, con, t: datetime) -> "PlayerSkills":
        df = player_games(con)
        if df.empty:
            return self
        age = (pd.Timestamp(t) - pd.to_datetime(df.tip_utc, utc=True)).dt.total_seconds() / 86400
        df["w"] = 0.5 ** (age.clip(lower=0) / self.half_life_days)
        df = df[df.w >= self.min_weight]
        for c in ("pf", "pa", "qf", "qa", "seconds"):
            df[f"w_{c}"] = df.w * df[c]
        lg_off = df.w_pf.sum() / max(df.w_qf.sum(), 1e-9)
        lg_def = df.w_pa.sum() / max(df.w_qa.sum(), 1e-9)
        lg_pace = 2880.0 * (df.w_qf.sum() + df.w_qa.sum()) / 2 / max(df.w_seconds.sum(), 1e-9)

        p = df.groupby("player_id").agg(pf=("w_pf", "sum"), pa=("w_pa", "sum"), qf=("w_qf", "sum"), qa=("w_qa", "sum"),
                                         sec=("w_seconds", "sum"), n_games=("game_id", "nunique"))
        # sampling noise of a points-per-possession rate: variance of (points - league_rate * possessions) per possession
        sig_off = float(((df.w * (df.pf - lg_off * df.qf) ** 2).sum()) / max(df.w_qf.sum(), 1e-9))
        sig_def = float(((df.w * (df.pa - lg_def * df.qa) ** 2).sum()) / max(df.w_qa.sum(), 1e-9))
        k_off = self.k_off if self.k_off is not None else estimate_k((p.pf / p.qf.clip(lower=1e-9)).to_numpy(), p.qf.to_numpy(), lg_off, sig_off)
        k_def = self.k_def if self.k_def is not None else estimate_k((p.pa / p.qa.clip(lower=1e-9)).to_numpy(), p.qa.to_numpy(), lg_def, sig_def)
        k_sec = self.k_pace_seconds if self.k_pace_seconds is not None else 2880.0 * 5  # 5 games of prior

        off = (p.pf + k_off * lg_off) / (p.qf + k_off)
        dfn = (p.pa + k_def * lg_def) / (p.qa + k_def)
        # pace: possessions per team per 48 min, shrunk by adding k_sec seconds at league pace
        pace_obs = 2880.0 * (p.qf + p.qa) / 2
        pace = (pace_obs + k_sec * lg_pace) / (p.sec + k_sec)
        self.table = pd.DataFrame({
            "off": 100 * (off - lg_off), "def": 100 * (dfn - lg_def), "pace": pace - lg_pace,
            "n_poss": p.qf, "n_games": p.n_games})
        self.k = {"off": k_off, "def": k_def, "pace_seconds": k_sec}
        self.league = {"ppp": lg_off, "pace": lg_pace}
        return self

    def get(self, player_id: int) -> tuple[float, float, float]:
        """(offense, defense, pace) relative to league; an unseen player is exactly average."""
        if player_id in self.table.index:
            r = self.table.loc[player_id]
            return float(r["off"]), float(r["def"]), float(r["pace"])
        return 0.0, 0.0, 0.0

    def lineup(self, players) -> tuple[float, float, float]:
        """Average skill of five players (equal minutes)."""
        v = np.array([self.get(p) for p in players])
        return tuple(float(x) for x in v.mean(axis=0))
