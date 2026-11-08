"""Deliberately dumb models. Any real model has to beat these on the scoreboard."""
from __future__ import annotations

from datetime import datetime

from .replay import Forecast


class LeagueAverageModel:
    """Forecast every game as the league's trailing-year average total. No teams, no players."""

    def __init__(self, window_days: int = 365, min_games: int = 30, default: tuple[float, float] = (200.0, 19.0)):
        self.window_days, self.min_games, self.default = window_days, min_games, default
        self.mean, self.sigma = default

    def fit(self, con, t: datetime) -> None:
        n, mean, sd = con.execute(
            "SELECT count(*), avg(home_pts + away_pts), stddev_samp(home_pts + away_pts) FROM pit_games "
            "WHERE tip_utc >= ?::TIMESTAMPTZ - to_days(?)", [t, self.window_days]).fetchone()
        self.mean, self.sigma = (float(mean), float(sd)) if n >= self.min_games else self.default

    def predict(self, con, game: dict, t: datetime) -> Forecast:
        return Forecast(self.mean, self.sigma)
