"""Stages 4-5 tuning: does a lineup's player-skill sum predict the team's points per possession in unseen games?

Walk-forward by month: fit skills / RAPM on games before the month, then for each team-game in the month predict
ppp - league_ppp from the players' actual minutes (oracle minutes: this tests the skills, not the minutes
forecast). Score = weighted R^2 of (actual - league) on the prediction after one fitted scale factor.
Scored seasons are 2020-21 and 2021-22 only; 2019-20 is warm-up. The holdout is never touched.
"""
import sys
import warnings
from datetime import datetime, timezone

import duckdb
import numpy as np
import pandas as pd

from nbadl import config, warehouse as wh
from nbadl.features.player_skills import PlayerSkills
from nbadl.features.rapm import Rapm

warnings.filterwarnings("ignore")
con = duckdb.connect(str(config.WAREHOUSE_PATH), read_only=True)
SEASONS = ("2020-21", "2021-22")

minutes = con.execute(f"""
    WITH s AS (SELECT game_id, team_a AS team_id, team_b AS opp, unnest(lineup_a) AS player_id, seconds FROM stints
               UNION ALL SELECT game_id, team_b, team_a, unnest(lineup_b), seconds FROM stints)
    SELECT s.game_id, s.team_id, s.opp AS opp_id, s.player_id, SUM(seconds) / 60.0 AS m
    FROM s JOIN games g USING (game_id)
    WHERE g.season IN {SEASONS} AND g.season_type LIKE 'reg%' GROUP BY ALL""").df()
tgames = con.execute(f"""
    WITH s AS (SELECT game_id, team_a AS team_id, points_a AS pts, poss_a AS poss FROM stints
               UNION ALL SELECT game_id, team_b, points_b, poss_b FROM stints)
    SELECT s.game_id, s.team_id, g.tip_utc, SUM(pts) AS pts, SUM(poss) AS poss
    FROM s JOIN games g USING (game_id) WHERE g.season IN {SEASONS} AND g.season_type LIKE 'reg%'
    GROUP BY ALL HAVING SUM(poss) > 0""").df()
tgames["ppp"] = tgames.pts / tgames.poss
tgames["month"] = pd.to_datetime(tgames.tip_utc, utc=True).dt.strftime("%Y-%m")
league = (tgames.pts.sum() / tgames.poss.sum())


def evaluate(make, kind):
    xs, ys, ws = [], [], []
    for month, g in tgames.groupby("month"):
        t = datetime.fromisoformat(month + "-01").replace(tzinfo=timezone.utc)
        with wh.point_in_time(con, t) as c:
            m = make().fit(c, t)
        tab = m.table
        off, dfn = tab["off"], tab["def"]
        mm = minutes[minutes.game_id.isin(g.game_id)]
        own = mm.assign(v=mm.player_id.map(off).fillna(0.0) * mm.m / 48.0).groupby(["game_id", "team_id"]).v.sum()
        # opponent's defenders: their players' def effect, matched through the opponent's team-game
        opp = mm.assign(v=mm.player_id.map(dfn).fillna(0.0) * mm.m / 48.0).groupby(["game_id", "opp_id"]).v.sum()
        opp.index.names = ["game_id", "team_id"]
        # for the team on offense, the defenders are the OTHER side's players: look up by (game, offense team)
        d = pd.DataFrame({"own": own}).join(opp.rename("dfn"), how="left").fillna(0.0)
        if kind == "skills":  # Stage 4 values are averages that already include teammates: use mean not sum
            d = d / 5.0
        pred = (d.own - d.dfn * (1 if kind == "rapm" else 1)) if False else (d.own + d.dfn)
        gg = g.set_index(["game_id", "team_id"])
        j = gg.join(pred.rename("pred"), how="left").dropna(subset=["pred"])
        xs.append(j.pred.to_numpy()); ys.append((j.ppp - league).to_numpy()); ws.append(j.poss.to_numpy())
    x, y, w = map(np.concatenate, (xs, ys, ws))
    b = (w * x * y).sum() / (w * x * x).sum()
    ss_tot = (w * (y - np.average(y, weights=w)) ** 2).sum()
    return 1 - (w * (y - b * x) ** 2).sum() / ss_tot, b, len(x)


if __name__ == "__main__":
    print("baseline = league average ppp (R2 0 by construction)")
    print("-- Stage 4 player skills (on-court, shrunk)")
    for hl in (120, 365, 1000):
        r2, b, n = evaluate(lambda hl=hl: PlayerSkills(half_life_days=hl), "skills")
        print(f"  half_life={hl:>5}: R2 {r2:+.4f}  scale {b:.2f}  n={n}")
    print("-- Stage 5 RAPM")
    for hl in (180, 365, 1000):
        for a in (500, 2000, 8000):
            r2, b, n = evaluate(lambda hl=hl, a=a: Rapm(half_life_days=hl, alpha=a), "rapm")
            print(f"  half_life={hl:>5} alpha={a:>5}: R2 {r2:+.4f}  scale {b:.2f}  n={n}")
