"""Stage 8 x 5 integration check: does tonight's injury-adjusted rotation explain the market's miss?

For each game, rotation strength S = sum over a team's players of (expected minutes / 48) * RAPM, for offense and
defense, combined into a game total (points per 100 possessions, times pace/100). Two versions of expected minutes:
  with reports   : AvailabilityModel using injury status (what we would know at the open)
  history only   : same model, status ignored (the 'usual rotation' the team ratings already reflect)
x = S(with reports) - S(history only) is the extra information the reports carry.
Target e = actual - consensus close. Walk-forward: month m uses RAPM / availability fitted before m; the slope on x
is fitted on earlier months only. Scored 2022-23 and 2023-24 (2021-22 is warm-up). Holdout untouched.
"""
import warnings
from datetime import timezone

import duckdb
import numpy as np
import pandas as pd

from nbadl import config, warehouse as wh
from nbadl.features import availability as av
from nbadl.features.rapm import Rapm

warnings.filterwarnings("ignore")
con = duckdb.connect(str(config.WAREHOUSE_PATH), read_only=True)
tr = pd.read_parquet("data/team_ratings_2019_24.parquet").set_index("game_id")
tr = tr[~tr.push].dropna(subset=["close_total"])
tr["e"] = tr.actual - tr.close_total
games = con.execute("SELECT game_id, home_team_id, away_team_id FROM games").df().set_index("game_id")

with wh.point_in_time(con, pd.Timestamp("2024-07-01", tz="UTC").to_pydatetime()) as c:
    frame = av.build_frame(c)
frame["tip"] = pd.to_datetime(frame.tip_utc, utc=True)
frame["month"] = frame.tip.dt.strftime("%Y-%m")
months = sorted(m for m in frame.month.unique() if "2021-10" <= m <= "2024-06")
tr_ids = set(tr.index)

out = []
for m in months:
    t = pd.Timestamp(m + "-01", tz="UTC")
    with wh.point_in_time(con, t.to_pydatetime()) as c:
        rapm = Rapm().fit(c, t.to_pydatetime())
    past = frame[frame.tip < t]
    here = frame[(frame.month == m) & frame.game_id.isin(tr_ids)].copy()
    if here.empty:
        continue
    have = av.AvailabilityModel().fit(past)
    hist = av.AvailabilityModel()
    h2, p2 = here.copy(), past.copy()
    for d in (h2, p2):
        d["status"], d["others_out"] = "None", 0.0
    hist.fit(p2)
    a, b = have.predict(here), hist.predict(h2)
    for tag, p in (("rep", a), ("hist", b)):
        p["off"] = p.player_id.map(rapm.table["off"]).fillna(0.0) * p.exp_minutes / 48.0
        p["dfn"] = p.player_id.map(rapm.table["def"]).fillna(0.0) * p.exp_minutes / 48.0
    s = a.groupby(["game_id", "team_id"])[["off", "dfn"]].sum().join(
        b.groupby(["game_id", "team_id"])[["off", "dfn"]].sum(), rsuffix="_h")
    s["d_off"], s["d_dfn"] = s.off - s.off_h, s.dfn - s.dfn_h
    s = s.reset_index().join(games, on="game_id")
    home = s[s.team_id == s.home_team_id].set_index("game_id")
    away = s[s.team_id == s.away_team_id].set_index("game_id")
    x = (home.d_off + away.d_dfn + away.d_off + home.d_dfn).rename("x")  # points per 100 poss, total
    out.append(pd.DataFrame({"x": x, "month": m}))
df = pd.concat(out).join(tr[["e", "season", "close_total"]], how="inner")
df["x"] = df.x * 0.99  # ~99 possessions per team
print(f"{len(df)} games, x sd {df.x.std():.2f} pts, e sd {df.e.std():.2f}")
print("corr(x, e) all:", round(df.x.corr(df.e), 4))
print("x by size: |x|>1.5 games:", int((df.x.abs() > 1.5).sum()))
ts = df[df.season.isin(["2022-23", "2023-24"])]
tr_ = df[df.season == "2021-22"]
print("univariate slope (2022-24): %.2f  t=%.1f" % ((lambda xx: ((xx * ts.e).sum() / (xx * xx).sum()))(ts.x - ts.x.mean()),
      ((ts.x - ts.x.mean()) * ts.e).sum() / (ts.e.std() * np.sqrt(((ts.x - ts.x.mean()) ** 2).sum()))))
# walk-forward by month, slope from earlier months only
df = df.sort_values("month")
pred = pd.Series(0.0, index=df.index)
for m in sorted(df.month.unique()):
    a, b = df[df.month < m], df[df.month == m]
    if len(a) > 300:
        sl = (a.x * a.e).sum() / (a.x ** 2).sum()
        pred[b.index] = sl * b.x
sc = df[df.season.isin(["2022-23", "2023-24"])]
mse0, mse1 = (sc.e ** 2).mean(), ((sc.e - pred[sc.index]) ** 2).mean()
print(f"OOS R2 vs market: {1 - mse1 / mse0:+.4f}   MAE market {sc.e.abs().mean():.3f} -> {(sc.e - pred[sc.index]).abs().mean():.3f}")
