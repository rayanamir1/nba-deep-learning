"""Stage 7: rest, travel and altitude, computed from the schedule alone.

For every game and each team we ask "what had this team just been through?":
  * rest_days      days since its previous game (capped), first game of a season -> the cap
  * b2b            1 if it played yesterday
  * g_last4/g_last6 how many games it played in the 4 / 6 days before this one
  * travel_miles   great-circle distance from the previous game's arena to this one's
and per game: whether the game is played at altitude (Denver, Utah).

Point-in-time: a team's features use only games that tipped strictly earlier, plus the game's own
fixture, which the schedule publishes weeks ahead. Nothing here reads a score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REST_CAP = 5
ALTITUDE_TEAMS = {1610612743, 1610612762}  # Denver (5,280 ft), Utah (4,226 ft)

# team_id -> (latitude, longitude) of the home arena's city
ARENA = {
    1610612737: (33.76, -84.40), 1610612738: (42.37, -71.06), 1610612739: (41.50, -81.69),
    1610612740: (29.95, -90.08), 1610612741: (41.88, -87.67), 1610612742: (32.79, -96.81),
    1610612743: (39.75, -105.01), 1610612744: (37.77, -122.39), 1610612745: (29.75, -95.36),
    1610612746: (34.04, -118.27), 1610612747: (34.04, -118.27), 1610612748: (25.78, -80.19),
    1610612749: (43.04, -87.92), 1610612750: (44.98, -93.28), 1610612751: (40.68, -73.97),
    1610612752: (40.75, -73.99), 1610612753: (28.54, -81.38), 1610612754: (39.76, -86.16),
    1610612755: (39.90, -75.17), 1610612756: (33.45, -112.07), 1610612757: (45.53, -122.67),
    1610612758: (38.58, -121.50), 1610612759: (29.43, -98.44), 1610612760: (35.46, -97.52),
    1610612761: (43.64, -79.38), 1610612762: (40.77, -111.90), 1610612763: (35.14, -90.05),
    1610612764: (38.90, -77.02), 1610612765: (42.34, -83.05), 1610612766: (35.23, -80.84),
}

FEATURES = ["home_rest", "away_rest", "home_b2b", "away_b2b", "home_g4", "away_g4",
            "home_travel", "away_travel", "altitude"]


def miles(a: int, b: int) -> float:
    """Great-circle (haversine) distance between two teams' arenas."""
    (la1, lo1), (la2, lo2) = map(lambda p: np.radians(p), (ARENA[a], ARENA[b]))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return float(3959 * 2 * np.arcsin(np.sqrt(h)))


def context_features(schedule: pd.DataFrame) -> pd.DataFrame:
    """schedule: game_id, home_team_id, away_team_id, tip_utc. Returns one row per game with FEATURES."""
    s = schedule.dropna(subset=["tip_utc"]).copy()
    s = s[s.home_team_id.isin(ARENA) & s.away_team_id.isin(ARENA)]  # drops All-Star / exhibition sides
    s["tip_utc"] = pd.to_datetime(s.tip_utc, utc=True)
    # one row per team-game, in time order, remembering where the game was played
    tg = pd.concat([
        s.assign(team=s.home_team_id, venue=s.home_team_id)[["game_id", "tip_utc", "team", "venue"]],
        s.assign(team=s.away_team_id, venue=s.home_team_id)[["game_id", "tip_utc", "team", "venue"]],
    ]).sort_values(["team", "tip_utc", "game_id"]).reset_index(drop=True)
    g = tg.groupby("team")
    prev_tip, prev_venue = g.tip_utc.shift(1), g.venue.shift(1)
    gap = (tg.tip_utc - prev_tip).dt.total_seconds() / 86400
    tg["rest"] = np.floor(gap).fillna(REST_CAP).clip(upper=REST_CAP)  # whole days; new season -> cap
    tg.loc[gap > 30, "rest"] = REST_CAP
    tg["b2b"] = (tg.rest <= 1).astype(float) * (gap.notna())
    tg["travel"] = [0.0 if pd.isna(pv) or pv == v else miles(int(pv), int(v)) for pv, v in zip(prev_venue, tg.venue)]
    tg.loc[gap > 30, "travel"] = 0.0
    # games in the 4 days before tip (3-in-4 shows up as g4 == 2)
    g4 = np.zeros(len(tg))
    for _, idx in tg.groupby("team").indices.items():
        t = tg.tip_utc.dt.tz_convert(None).to_numpy()[idx].astype("datetime64[s]").astype(np.int64) / 86400.0
        left = np.searchsorted(t, t - 4.0, side="left")
        g4[idx] = np.arange(len(idx)) - left
    tg["g4"] = g4

    home = tg.merge(s[["game_id", "home_team_id"]], on="game_id")
    home = home[home.team == home.home_team_id].set_index("game_id")
    away = tg.merge(s[["game_id", "away_team_id"]], on="game_id")
    away = away[away.team == away.away_team_id].set_index("game_id")
    out = pd.DataFrame({
        "home_rest": home.rest, "away_rest": away.rest, "home_b2b": home.b2b, "away_b2b": away.b2b,
        "home_g4": home.g4, "away_g4": away.g4, "home_travel": home.travel / 1000, "away_travel": away.travel / 1000,
    })
    out["altitude"] = s.set_index("game_id").home_team_id.isin(ALTITUDE_TEAMS).astype(float)
    return out[FEATURES]
