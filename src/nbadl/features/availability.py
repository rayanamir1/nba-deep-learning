"""Stage 8: who plays tonight, and for how many minutes.

For every (game, team, candidate player) we build one row of information that was knowable at the decision time:

    status        the latest injury-report status published before the decision time ("None" = not listed)
    m_recent      decayed average minutes over the team's previous 10 games (zeros when he did not play)
    app_rate      share of those 10 games he appeared in
    start_rate    share of those 10 games he started
    others_out    summed m_recent of teammates whose status is Out / Doubtful (the minutes that must go elsewhere)

Candidates are players who appeared for the team in the previous 10 games, plus anyone on the report for the game.
Two small models, both fitted on earlier games only:

    P(plays)            logistic regression on status one-hots, app_rate, m_recent
    minutes | plays     ridge on m_recent, start_rate, others_out, status

expected minutes = P(plays) * minutes|plays, then each team is rescaled to 240. The result is NEVER read from the
game's own box score for the game being predicted: "played" and "minutes" exist only for training rows.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

from .. import config

STATUSES = ["None", "Available", "Probable", "Questionable", "Doubtful", "Out"]
WINDOW = 10
DECAY = 0.85
TEAM_MINUTES = 240.0


def _appearances(con) -> pd.DataFrame:
    """One row per (game, team, player): minutes played and whether he started (on court at the opening tip)."""
    return con.execute("""
        WITH s AS (
            SELECT game_id, period, start_clock_s, team_a AS team_id, unnest(lineup_a) AS player_id, seconds FROM pit_stints
            UNION ALL
            SELECT game_id, period, start_clock_s, team_b, unnest(lineup_b), seconds FROM pit_stints)
        SELECT s.game_id, s.team_id, s.player_id, SUM(seconds) / 60.0 AS minutes,
               MAX(CASE WHEN period = 1 AND start_clock_s >= 719 THEN 1 ELSE 0 END) AS started
        FROM s GROUP BY s.game_id, s.team_id, s.player_id HAVING SUM(seconds) > 0""").df()


def _statuses(con, decision_offset: timedelta) -> pd.DataFrame:
    """Latest report status per (game, team, player) published before that game's decision time."""
    return con.execute(f"""
        SELECT i.game_id, i.team_id, i.player_id, ARG_MAX(i.status, i.report_at) AS status
        FROM pit_injuries i JOIN pit_games g USING (game_id)
        WHERE i.player_id IS NOT NULL
          AND i.available_at <= g.tip_utc - INTERVAL '{int(decision_offset.total_seconds())} seconds'
        GROUP BY i.game_id, i.team_id, i.player_id""").df()


def build_frame(con, decision_offset: timedelta = config.HISTORICAL_OPEN_LINE_LEAD - timedelta(minutes=1),
                target: pd.DataFrame | None = None) -> pd.DataFrame:
    """Candidate rows for every finished game in the point-in-time view, plus rows for `target` games.

    target: game_id, tip_utc, home_team_id, away_team_id for games whose results are not known; their rows have
    NaN played/minutes. History features use only the team's previous finished games.
    """
    app = _appearances(con)
    by_game = {k: dict(zip(g.player_id, zip(g.minutes, g.started))) for k, g in app.groupby(["game_id", "team_id"])}
    games = con.execute("SELECT game_id, tip_utc, home_team_id, away_team_id FROM pit_games "
                        "WHERE tip_utc IS NOT NULL").df()
    if target is not None and len(target):
        games = pd.concat([games, target[games.columns]], ignore_index=True).drop_duplicates("game_id", keep="first")
    tg = pd.concat([games.assign(team_id=games.home_team_id), games.assign(team_id=games.away_team_id)])
    tg = tg[["game_id", "tip_utc", "team_id"]].sort_values(["team_id", "tip_utc"])
    status = _statuses(con, decision_offset)
    st = {(r.game_id, r.team_id, r.player_id): r.status for r in status.itertuples(index=False)}
    listed: dict[tuple, set] = {}
    for (gid, team, pid) in st:
        listed.setdefault((gid, team), set()).add(pid)
    finished = set(app.game_id)
    weights = DECAY ** np.arange(1, WINDOW + 1)  # most recent previous game first

    rows = []
    for team, g in tg.groupby("team_id"):
        hist: list[dict] = []  # finished games so far, oldest first
        for gid, tip in zip(g.game_id, g.tip_utc):
            prev = hist[-WINDOW:][::-1]
            pad = WINDOW - len(prev)
            cands = {p for h in prev for p in h} | listed.get((gid, team), set())
            here = by_game.get((gid, team), {}) if gid in finished else None
            for p in cands:
                m = np.array([h.get(p, (0.0, 0))[0] for h in prev] + [0.0] * pad)
                started = np.array([h.get(p, (0.0, 0))[1] for h in prev] + [0] * pad)
                minutes = np.nan if here is None else here.get(p, (0.0, 0))[0]
                rows.append((gid, team, p, tip, (m * weights).sum() / weights.sum(), (m > 0).mean(),
                             started.mean(), st.get((gid, team, p), "None"), minutes))
            if gid in finished:
                hist.append(by_game.get((gid, team), {}))
    df = pd.DataFrame(rows, columns=["game_id", "team_id", "player_id", "tip_utc", "m_recent", "app_rate",
                                     "start_rate", "status", "minutes"])
    df["played"] = np.where(df.minutes.isna(), np.nan, (df.minutes > 0).astype(float))
    gone = df.status.isin(["Out", "Doubtful"])
    total_gone = df[gone].groupby(["game_id", "team_id"]).m_recent.sum().rename("gone_total")
    df = df.join(total_gone, on=["game_id", "team_id"])
    df["others_out"] = df.gone_total.fillna(0.0) - np.where(gone, df.m_recent, 0.0)
    return df.drop(columns="gone_total")


def _design(df: pd.DataFrame, with_status: bool = True) -> np.ndarray:
    cols = [df.m_recent / 48.0, df.app_rate, df.start_rate, df.others_out / 240.0]
    if with_status:
        cols += [(df.status == s).astype(float) for s in STATUSES[1:]]  # "None" is the baseline
    return np.column_stack(cols)


class AvailabilityModel:
    """P(plays) by logistic regression, minutes-given-plays by ridge; both fitted on finished rows only."""

    def __init__(self, c: float = 1.0, alpha: float = 1.0, min_rows: int = 500):
        self.c, self.alpha, self.min_rows = c, alpha, min_rows
        self.ready = False

    def fit(self, frame: pd.DataFrame) -> "AvailabilityModel":
        train = frame[frame.played.notna()]
        self.ready = False
        if len(train) < self.min_rows or train.played.nunique() < 2:
            return self
        self.play = LogisticRegression(C=self.c, max_iter=500).fit(_design(train), train.played.astype(int))
        got = train[train.played == 1]
        self.mins = Ridge(alpha=self.alpha).fit(_design(got), got.minutes)
        self.ready = True
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Adds p_play, minutes_if_play and exp_minutes (rescaled so each team-game sums to 240)."""
        out = frame.copy()
        if not self.ready:  # no model yet: trust recent minutes, ignore reports
            out["p_play"] = out.app_rate.clip(0, 1)
            out["minutes_if_play"] = out.m_recent / out.app_rate.clip(lower=0.1)
        else:
            X = _design(out)
            out["p_play"] = self.play.predict_proba(X)[:, 1]
            out["minutes_if_play"] = np.clip(self.mins.predict(X), 0.0, 48.0)
        raw = out.p_play * out.minutes_if_play
        total = raw.groupby([out.game_id, out.team_id]).transform("sum")
        out["exp_minutes"] = np.where(total > 0, raw * TEAM_MINUTES / total.where(total > 0, 1.0), 0.0)
        return out
