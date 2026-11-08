"""Stage 13: walk-forward replay.

For every game we pretend the clock is at its decision time t = tip - decision_offset, fit the model on
rows with available_at < t (refitting at most every `refit_days`), ask it for a forecast of the game
total, compare against the line a bettor could actually see at t, and score against what happened.
"""
from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol

import numpy as np
import pandas as pd

from .. import config
from .. import warehouse as wh
from . import metrics

log = logging.getLogger(__name__)


class HoldoutLockedError(RuntimeError):
    pass


class LeakageDetected(RuntimeError):
    pass


@dataclass(frozen=True)
class Forecast:
    mean: float   # expected combined points
    sigma: float  # game-specific spread of the total (stage 10)


class Model(Protocol):
    def fit(self, con, t: datetime) -> None: ...
    def predict(self, con, game: dict, t: datetime) -> Forecast: ...
    # `con` always has the pit_* views for time t; models must read only pit_* tables.


def p_over(line: float, f: Forecast) -> float:
    """P(total > line) if the total ~ Normal(mean, sigma)."""
    z = (line - f.mean) / f.sigma
    return 0.5 * math.erfc(z / math.sqrt(2))


def holdout_seasons(con, n: int = 2) -> list[str]:
    """Last n *completed* seasons. A season counts as completed once it has playoff games."""
    rows = con.execute("SELECT DISTINCT season FROM games WHERE season_type='playoffs' ORDER BY season").fetchall()
    return [r[0] for r in rows][-n:]


def _games_to_score(con, seasons: list[str]) -> pd.DataFrame:
    # Labels (final score, consensus close) are read from the base tables on purpose: they are what we
    # score *against*, never what the model sees.
    placeholders = ",".join("?" * len(seasons))
    return con.execute(f"""
        SELECT g.game_id, g.season, g.game_date, g.home_team_id, g.away_team_id, g.tip_utc,
               g.home_pts + g.away_pts AS actual, c.close_total
        FROM games g
        JOIN (SELECT game_id, avg(total) AS close_total FROM odds_lines
              WHERE line_type='close' AND game_id IS NOT NULL GROUP BY 1) c USING (game_id)
        WHERE g.season IN ({placeholders}) AND g.tip_utc IS NOT NULL
        ORDER BY g.tip_utc""", seasons).df()


def _bet_line(con, game_id: str) -> float | None:
    """Consensus of the latest line each book had published at t (call inside point_in_time)."""
    row = con.execute("""
        SELECT avg(total) FROM (SELECT sportsbook, arg_max(total, available_at) AS total
                                FROM pit_odds_lines WHERE game_id = ? GROUP BY 1)""", [game_id]).fetchone()
    return None if row is None or row[0] is None else float(row[0])


# one numeric column per guarded table to corrupt when proving the model cannot see the future
_CORRUPT = {"games": "home_pts = home_pts + 999", "team_games": "pts = pts + 99", "odds_lines": "total = total + 50",
            "possessions": "points = points + 99", "stints": "points_a = points_a + 99",
            "officials": "official_id = official_id + 1"}


def leak_check(con, model: Model, game: dict, t: datetime) -> None:
    """Forecast the game, corrupt every row with available_at >= t inside a transaction, forecast again
    with a freshly fitted copy of the model, roll back. Different answers = the model saw the future."""
    def run() -> Forecast:
        m = copy.deepcopy(model)
        with wh.point_in_time(con, t) as c:
            m.fit(c, t)
            return m.predict(c, game, t)

    before = run()
    con.execute("BEGIN TRANSACTION")
    try:
        for tbl, assignment in _CORRUPT.items():
            con.execute(f"UPDATE {tbl} SET {assignment} WHERE available_at >= ?", [t])
        after = run()
    finally:
        con.execute("ROLLBACK")
    if (before.mean, before.sigma) != (after.mean, after.sigma):
        raise LeakageDetected(f"forecast for {game['game_id']} at {t} changed when future rows changed: "
                              f"{before} -> {after}")


def replay(con, model: Model, seasons: list[str], *, refit_days: int = 7,
           decision_offset: timedelta = config.HISTORICAL_OPEN_LINE_LEAD - timedelta(minutes=1), final: bool = False,
           leak_check_every: int = 250) -> pd.DataFrame:
    # Default decision time = one minute after the opener becomes knowable. Deciding later would let the model see
    # hours of newer information than the price it bets at (history only has an open and a close snapshot).
    locked = sorted(set(seasons) & set(holdout_seasons(con)))
    if locked and not final:
        raise HoldoutLockedError(f"{locked} is the final holdout; pass final=True to score it (once, at the end)")
    if locked:
        log.warning("SCORING THE FINAL HOLDOUT %s", locked)

    games = _games_to_score(con, seasons)
    out, last_fit = [], None
    for i, g in enumerate(games.itertuples(index=False)):
        t = g.tip_utc.to_pydatetime() - decision_offset
        game = g._asdict()
        if last_fit is None or t - last_fit >= timedelta(days=refit_days):
            with wh.point_in_time(con, t) as c:
                model.fit(c, t)
            last_fit = t
        with wh.point_in_time(con, t) as c:
            line = _bet_line(c, g.game_id)
            if line is None:
                continue
            f = model.predict(c, game, t)
        if leak_check_every and i % leak_check_every == 0:
            leak_check(con, model, game, t)
        p = p_over(line, f)
        out.append({"game_id": g.game_id, "season": g.season, "tip_utc": g.tip_utc, "decision_t": t,
                    "bet_total": line, "close_total": g.close_total, "actual": g.actual,
                    "mean": f.mean, "sigma": f.sigma, "p_over": p,
                    "side": 1 if p > 0.5 else -1})
    df = pd.DataFrame(out)
    if len(df):
        df["push"] = df.actual == df.bet_total
        df["clv"] = metrics.clv_points(df.side, df.bet_total, df.close_total)
    return df


def summarize(df: pd.DataFrame) -> dict:
    """The scoreboard. Pushes are excluded from probability scores (neither over nor under won)."""
    d = df[~df.push]
    y = (d.actual > d.bet_total).astype(int)
    won = ((d.side == 1) & (y == 1)) | ((d.side == -1) & (y == 0))
    n = len(d)
    return {
        "n_games": n,
        "brier": metrics.brier(d.p_over, y), "brier_coinflip": 0.25,
        "log_loss": metrics.log_loss(d.p_over, y), "log_loss_coinflip": math.log(2),
        "mae_model": metrics.mae(d["mean"], d.actual), "mae_close": metrics.mae(d.close_total, d.actual),
        "hit_rate": float(won.mean()), "hit_rate_se": math.sqrt(0.25 / n), "breakeven_at_-110": 110 / 210,
        "mean_clv_points": float(d.clv.mean()), "pct_games_positive_clv": float((d.clv > 0).mean()),
    }
