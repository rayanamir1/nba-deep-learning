"""Stage 6 check: do shrunk referee effects explain the market's miss out of sample?

Refit monthly on earlier games only (walk-forward), predict e = actual - close for the month's games from the
crew alone, and compare squared error with the market (predicting e = 0). Grid is tuned on seasons before 2022-23.
"""
import warnings
from datetime import datetime, timezone

import duckdb
import numpy as np
import pandas as pd

from nbadl import config, warehouse as wh
from nbadl.features.referees import RefereeEffects

warnings.filterwarnings("ignore")
con = duckdb.connect(str(config.WAREHOUSE_PATH), read_only=True)
d = pd.read_parquet("data/team_ratings_2019_24.parquet")
d = d[~d.push].copy(); d["e"] = d.actual - d.close_total
crew = con.execute("SELECT game_id, list(official_id) AS refs FROM officials GROUP BY 1").df().set_index("game_id").refs
d = d[d.game_id.isin(crew.index)].copy()
d["month"] = pd.to_datetime(d.tip_utc, utc=True).dt.strftime("%Y-%m")


def run(half_life, alpha, lo="2020-10", hi="2024-12"):
    se_m = se_r = n = 0.0
    for m, g in d[(d.month >= lo) & (d.month <= hi)].groupby("month"):
        t = datetime.fromisoformat(m + "-01").replace(tzinfo=timezone.utc)
        with wh.point_in_time(con, t) as c:
            re_ = RefereeEffects(half_life, alpha).fit(c, t)
        pred = np.array([re_.crew_effect(crew[gid]) for gid in g.game_id])
        se_m += (g.e ** 2).sum(); se_r += ((g.e - pred) ** 2).sum(); n += len(g)
    return 1 - se_r / se_m, n


if __name__ == "__main__":
    print("tuning on 2020-10..2022-06 (earlier seasons only)")
    best = None
    for hl in (365, 1000, 1e6):
        for a in (200, 600, 1500):
            r2, n = run(hl, a, "2020-10", "2022-06")
            print(f"  half_life={hl:>9.0f} alpha={a:>5}: OOS R2 {r2:+.5f} (n={n:.0f})")
            if best is None or r2 > best[0]: best = (r2, hl, a)
    print("best", best)
    r2, n = run(best[1], best[2], "2022-10", "2024-12")
    print(f"frozen on 2022-23..2023-24: OOS R2 {r2:+.5f} over {n:.0f} games")
