"""Stage 7 check: do rest / travel / altitude features explain what the market missed?

Target e = actual - close_total (the market's miss). Walk-forward by season: fit OLS on earlier
seasons only, score the held-out season. Reports the out-of-sample R^2 and MAE change.
"""
import duckdb
import numpy as np
import pandas as pd

from nbadl import config
from nbadl.features.schedule_context import FEATURES, context_features

con = duckdb.connect(str(config.WAREHOUSE_PATH), read_only=True)
sched = con.execute("SELECT game_id, home_team_id, away_team_id, tip_utc FROM schedule").df()
feat = context_features(sched)
df = pd.read_parquet("data/team_ratings_2019_24.parquet").join(feat, on="game_id")
df = df[~df.push].dropna(subset=["close_total"]).copy()
df["e"] = df.actual - df.close_total
df["d"] = df["mean"] - df.close_total
X = df[FEATURES].to_numpy(float)
print(f"{len(df)} games; mean |market miss| {df.e.abs().mean():.2f}")

print("\nunivariate: coefficient (points) and t-stat on e = actual - close")
for f in FEATURES:
    x = df[f].to_numpy(float); x = x - x.mean()
    b = (x * df.e).sum() / (x * x).sum()
    se = df.e.std() / np.sqrt((x * x).sum())
    print(f"  {f:13s} {b:+7.2f}  t={b / se:+5.1f}")

seasons = sorted(df.season.unique())
rows = []
for s in seasons[1:]:
    tr, te = df[df.season < s], df[df.season == s]
    A = np.c_[np.ones(len(tr)), tr[FEATURES]]
    lam = 50.0 * np.eye(A.shape[1]); lam[0, 0] = 0
    w = np.linalg.solve(A.T @ A + lam, A.T @ tr.e)
    pred = np.c_[np.ones(len(te)), te[FEATURES]] @ w
    rows.append((s, len(te), (te.e ** 2).mean(), ((te.e - pred) ** 2).mean(),
                 te.e.abs().mean(), (te.e - pred).abs().mean()))
r = pd.DataFrame(rows, columns=["season", "n", "mse_market", "mse_with", "mae_market", "mae_with"])
print("\nwalk-forward (fit on earlier seasons):\n", r.round(2).to_string(index=False))
print(f"\nOOS R^2 vs market: {1 - (r.mse_with * r.n).sum() / (r.mse_market * r.n).sum():+.4f}")
