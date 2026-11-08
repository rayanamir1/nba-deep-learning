"""Score the league-average baseline on the given seasons. Usage: run_baseline.py 2019-20 2020-21 [--final]"""
import logging
import sys

import pandas as pd

from nbadl import warehouse as wh
from nbadl.backtest.baselines import LeagueAverageModel
from nbadl.backtest.replay import replay, summarize

logging.basicConfig(level=logging.INFO, format="%(message)s")
args = [a for a in sys.argv[1:] if not a.startswith("--")]
con = wh.connect()
df = replay(con, LeagueAverageModel(), args, final="--final" in sys.argv)
rows = {s: summarize(g) for s, g in df.groupby("season")}
rows["ALL"] = summarize(df)
pd.set_option("display.width", 200)
print(pd.DataFrame(rows).round(4).to_string())
df.to_parquet("data/baseline_league_avg.parquet")
