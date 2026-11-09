"""Stage 1 ingest of the official NBA injury reports (PDF).

URL:  https://ak-static.cms.nba.com/referee/injury/Injury-Report_<YYYY-MM-DD>_<HH><AM|PM>.pdf
A few reports a day in 2019-20, hourly from 2021. Each PDF is a table (game date / time / matchup / team / player /
current status / reason); game, team and date cells are only printed on the first row of their block, so the parser
carries them down.

Lake: injury/<YYYY-MM-DD>_<HH><AP> holds the words of every page as [text, x0, top], which is enough to re-parse
whenever the parser improves. A report that does not exist is stored as {"missing": true} so reruns skip it.

The parser reads columns by x position, anchored on the header row of each report, because layouts differ by era
(extra "Previous Status" column in 2019, spaces stripped from names in 2025).
"""
from __future__ import annotations

import io
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import requests

from ..lake import RawLake

log = logging.getLogger(__name__)

URL = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{d}_{h:02d}{ap}.pdf"
HOURS = [(h, "AM") for h in range(6, 12)] + [(12, "PM")] + [(h, "PM") for h in range(1, 12)] + [(12, "AM")]
STATUSES = {"out", "doubtful", "questionable", "probable", "available"}
ET = ZoneInfo("America/New_York")
RETRIES, BACKOFF_S, TOL = 3, 3.0, 4.0
_MATCHUP = re.compile(r"[A-Z]{2,3}@[A-Z]{2,3}")
_DATE = re.compile(r"\d{2}/\d{2}/\d{4}")
_TIME = re.compile(r"(\d{1,2}):(\d{2})")

# team id -> squashed lowercase names that appear in the Team column
TEAM_NAMES = {
    1610612737: ["atlantahawks"], 1610612738: ["bostonceltics"], 1610612739: ["clevelandcavaliers"],
    1610612740: ["neworleanspelicans"], 1610612741: ["chicagobulls"], 1610612742: ["dallasmavericks"],
    1610612743: ["denvernuggets"], 1610612744: ["goldenstatewarriors"], 1610612745: ["houstonrockets"],
    1610612746: ["laclippers", "losangelesclippers"], 1610612747: ["losangeleslakers", "lalakers"],
    1610612748: ["miamiheat"], 1610612749: ["milwaukeebucks"], 1610612750: ["minnesotatimberwolves"],
    1610612751: ["brooklynnets"], 1610612752: ["newyorkknicks"], 1610612753: ["orlandomagic"],
    1610612754: ["indianapacers"], 1610612755: ["philadelphia76ers"], 1610612756: ["phoenixsuns"],
    1610612757: ["portlandtrailblazers"], 1610612758: ["sacramentokings"], 1610612759: ["sanantoniospurs"],
    1610612760: ["oklahomacitythunder"], 1610612761: ["torontoraptors"], 1610612762: ["utahjazz"],
    1610612763: ["memphisgrizzlies"], 1610612764: ["washingtonwizards"], 1610612765: ["detroitpistons"],
    1610612766: ["charlottehornets"],
}
TEAM_BY_NAME = {n: tid for tid, names in TEAM_NAMES.items() for n in names}


def key_for(d: date, hour: int, ap: str) -> str:
    return f"{d.isoformat()}_{hour:02d}{ap}"


def pdf_words(content: bytes) -> list[list[list]]:
    """Per page, every word as [text, x0, top]."""
    import pdfplumber

    pages = []
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for pg in pdf.pages:
            pages.append([[w["text"], round(w["x0"], 1), round(w["top"], 1)] for w in pg.extract_words()])
    return pages


def _lines(words: list[list]) -> list[list[list]]:
    out: list[list[list]] = []
    for w in sorted(words, key=lambda w: (w[2], w[1])):
        if out and abs(w[2] - out[-1][0][2]) <= 2.5:
            out[-1].append(w)
        else:
            out.append([w])
    return [sorted(l, key=lambda w: w[1]) for l in out]


def _anchors(line: list[list]) -> dict[str, float] | None:
    """Column start x from a header row (the one containing 'Matchup')."""
    texts = [w[0] for w in line]
    if "Matchup" not in texts:
        return None
    cols: dict[str, float] = {}
    for text, x, _ in line:
        t = text.lower()
        if t.startswith("game") and "date" not in cols:
            cols["date"] = x
        elif t.startswith("game") and "time" not in cols:
            cols["time"] = x
        elif t == "matchup":
            cols["matchup"] = x
        elif t == "team":
            cols["team"] = x
        elif t.startswith("player"):
            cols["player"] = x
        elif t.startswith("current"):
            cols["status"] = x
        elif t == "reason":
            cols["reason"] = x
    return cols if {"date", "time", "matchup", "team", "player", "status", "reason"} <= set(cols) else None


def _column(x: float, cols: dict[str, float]) -> str:
    best = "date"
    for name, cx in sorted(cols.items(), key=lambda kv: kv[1]):
        if x >= cx - TOL:
            best = name
    return best


def report_time(pages: list[list[list]]) -> datetime | None:
    """Publish time from the 'Injury Report: MM/DD/YY HH:MM AM' title, as UTC."""
    for line in _lines(pages[0])[:3]:
        txt = " ".join(w[0] for w in line)
        m = re.search(r"Report:\s*(\d{2}/\d{2}/\d{2,4})\s+(\d{1,2}:\d{2})\s*(AM|PM)", txt)
        if m:
            d, t, ap = m.groups()
            fmt = "%m/%d/%y" if len(d) == 8 else "%m/%d/%Y"
            dt = datetime.strptime(f"{d} {t} {ap}", fmt + " %I:%M %p")
            return dt.replace(tzinfo=ET).astimezone(ZoneInfo("UTC"))
    return None


def parse_report(pages: list[list[list]]) -> dict[str, Any]:
    """{'report_at': UTC datetime | None, 'rows': [one dict per player line]}."""
    rows: list[dict] = []
    cols: dict[str, float] | None = None
    state = {"date": None, "time": None, "matchup": None, "team": None}
    for page in pages:
        for line in _lines(page):
            found = _anchors(line)
            if found:
                cols = found
                continue
            if cols is None:
                continue
            cell: dict[str, list[str]] = {}
            for text, x, _ in line:
                cell.setdefault(_column(x, cols), []).append(text)
            joined = {k: " ".join(v) for k, v in cell.items()}
            if "date" in joined and _DATE.search(joined["date"]):
                state["date"] = _DATE.search(joined["date"]).group(0)
            if "time" in joined and _TIME.search(joined["time"]):
                m = _TIME.search(joined["time"])
                state["time"] = f"{int(m.group(1)):02d}:{m.group(2)}"
            if "matchup" in joined and _MATCHUP.search(joined["matchup"]):
                state["matchup"] = _MATCHUP.search(joined["matchup"]).group(0)
            if "team" in joined:
                tid = TEAM_BY_NAME.get(re.sub(r"[^a-z0-9]", "", joined["team"].lower()))
                if tid:
                    state["team"] = tid
            status = joined.get("status", "").strip()
            player = joined.get("player", "").strip()
            if status.lower() in STATUSES and "," in player and state["team"]:
                rows.append({"game_date": state["date"], "game_time_et": state["time"], "matchup": state["matchup"],
                             "team_id": state["team"], "player": player, "status": status.capitalize()})
    return {"report_at": report_time(pages), "rows": rows}


class InjuryIngest:
    def __init__(self, lake: RawLake | None = None, session: requests.Session | None = None, workers: int = 4):
        self.lake = lake or RawLake()
        self.session = session or requests.Session()
        self.workers = workers

    def fetch(self, d: date, hour: int, ap: str) -> str:
        key = key_for(d, hour, ap)
        if self.lake.exists("injury", key):
            return key
        url = URL.format(d=d.isoformat(), h=hour, ap=ap)
        for attempt in range(RETRIES + 1):
            try:
                r = self.session.get(url, timeout=30)
                if r.status_code in (403, 404):  # S3-style "no such report"
                    self.lake.write("injury", key, {"missing": True})
                    return key
                r.raise_for_status()
                self.lake.write("injury", key, {"url": url, "pages": pdf_words(r.content)})
                return key
            except (requests.RequestException, OSError) as e:
                if attempt == RETRIES:
                    raise
                log.warning("injury/%s failed (%s), retry %d", key, e, attempt + 1)
                time.sleep(BACKOFF_S * 2 ** attempt)
        return key

    def pull_days(self, days: list[date]) -> None:
        jobs = [(d, h, ap) for d in days for h, ap in HOURS]
        done = 0
        with ThreadPoolExecutor(self.workers) as ex:
            futs = [ex.submit(self.fetch, *j) for j in jobs]
            for f in futs:
                try:
                    f.result()
                except Exception as e:  # rerun retries whatever failed
                    log.error("injury fetch failed: %s", e)
                done += 1
                if done % 500 == 0:
                    log.info("injury: %d/%d reports", done, len(jobs))


def game_days_with_eve(game_dates: list[date], start: date, end: date) -> list[date]:
    """Each game day plus the day before it (the previous evening's report already lists tomorrow's games)."""
    s = set()
    for d in game_dates:
        for x in (d - timedelta(days=1), d):
            if start <= x <= end:
                s.add(x)
    return sorted(s)
