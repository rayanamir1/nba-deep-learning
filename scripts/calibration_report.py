"""Stage 11 report on the non-holdout seasons: raw vs shrink-only vs shrink+Platt, plus CLV by disagreement size.
Usage: calibration_report.py  (replays the team-ratings model, caches data/team_ratings_2019_24.parquet)"""
import os

import numpy as np
import pandas as pd

from nbadl import warehouse as wh
from nbadl.backtest.calibrate import as_scored, walk_forward_calibrate
from nbadl.backtest.replay import replay, summarize
from nbadl.models.team_ratings import TeamRatingsModel

CACHE = "data/team_ratings_2019_24.parquet"
if os.path.exists(CACHE):
    df = pd.read_parquet(CACHE)
else:
    df = replay(wh.connect(), TeamRatingsModel(), ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24"],
                leak_check_every=0)
    df.to_parquet(CACHE)
shrink, full = walk_forward_calibrate(df, platt=False), walk_forward_calibrate(df, platt=True)
ids = set(full.game_id[full.active])
rows = {"raw model": summarize(df[df.game_id.isin(ids)]), "shrink only": summarize(as_scored(shrink)),
        "shrink + Platt": summarize(as_scored(full))}
print(pd.DataFrame(rows).loc[["n_games", "brier", "log_loss", "mae_model", "hit_rate", "mean_clv_points"]]
      .round(4).to_string(), "\n(coin flip: brier 0.2500, log loss 0.6931)\n")
sc = as_scored(full)
sc = sc[~sc.push].copy()
sc["absd"] = (sc.mean_raw - sc.bet_total).abs()
sc["won"] = np.where(sc.mean_raw > sc.bet_total, sc.actual > sc.bet_total, sc.actual < sc.bet_total)
sc["quintile"] = pd.qcut(sc.absd, 5, labels=["1 smallest", "2", "3", "4", "5 largest"])
print(sc.groupby("quintile", observed=True).agg(n=("won", "size"), mean_abs_disagree=("absd", "mean"),
                                                win_rate=("won", "mean"), clv=("clv", "mean")).round(3).to_string())
