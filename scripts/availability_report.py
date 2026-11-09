"""Stage 8 check: does the injury report improve who-plays / minutes forecasts out of sample?

Train on games before 2022-10-01, score 2022-23..2023-24 (holdout seasons untouched). Compares the full model with
one that ignores report status (history only)."""
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from nbadl import warehouse
from nbadl.features import availability as av

con = warehouse.connect(read_only=True) if "read_only" in warehouse.connect.__code__.co_varnames else warehouse.connect()
with warehouse.point_in_time(con, pd.Timestamp("2024-07-01", tz="UTC").to_pydatetime()) as c:
    f = av.build_frame(c)
f = f[f.played.notna()]
tip = pd.to_datetime(f.tip_utc, utc=True)
tr, te = f[tip < "2022-10-01"], f[(tip >= "2022-10-01") & (tip < "2024-07-01")]
te = te.copy()
print(len(tr), len(te))
print(te.groupby("status").played.agg(["mean", "size"]).round(3))

def run(frame_tr, frame_te, drop_status):
    a, b = frame_tr.copy(), frame_te.copy()
    if drop_status:
        a["status"] = b["status"] = "None"; a["others_out"] = b["others_out"] = 0.0
    m = av.AvailabilityModel(c=1.0, alpha=1.0).fit(a)
    return m.predict(b)

for name, drop in (("history only", True), ("with reports", False)):
    p = run(tr, te, drop)
    rot = p[p.m_recent >= 10]
    mae = (rot.exp_minutes - rot.minutes).abs().mean()
    ll = log_loss(p.played, p.p_play.clip(1e-4, 1 - 1e-4))
    print(f"{name:13s} play logloss {ll:.4f}  rotation(m_recent>=10) minutes MAE {mae:.2f}")
