"""Grid-search half-life and ridge strength for TeamRatingsModel on the VALIDATION seasons only.
The holdout is locked by the harness; the later non-holdout seasons are kept for one final report."""
import itertools
import logging
import sys
import time

import pandas as pd

from nbadl import warehouse as wh
from nbadl.backtest.replay import replay, summarize
from nbadl.models.team_ratings import TeamRatingsModel

logging.basicConfig(level=logging.WARNING)
VALID = ["2019-20", "2020-21", "2021-22"]
con = wh.connect()
rows = []
for half_life, level_hl, alpha in itertools.product([45, 90, 135, 180], [7, 14, 28], [1e-2, 3e-2]):
    t0 = time.time()
    df = replay(con, TeamRatingsModel(half_life_days=half_life, level_half_life_days=level_hl, alpha_pace=alpha, alpha_eff=alpha), VALID,
                leak_check_every=0)
    s = summarize(df)
    rows.append({"half_life": half_life, "level_hl": level_hl, "alpha": alpha, "n": s["n_games"], "mae": s["mae_model"],
                 "mae_close": s["mae_close"], "brier": s["brier"], "log_loss": s["log_loss"],
                 "hit_rate": s["hit_rate"], "clv": s["mean_clv_points"], "secs": round(time.time() - t0)})
    print(rows[-1], flush=True)
out = pd.DataFrame(rows).sort_values("log_loss")
out.to_csv("data/tuning_team_ratings_2.csv", index=False)
print(out.round(4).to_string(index=False))
