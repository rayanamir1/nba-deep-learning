import gzip
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from nbadl.ingest.injury import BlockedError, InjuryIngest, key_for, parse_report, game_days_with_eve
from nbadl.lake import RawLake

FIX = Path(__file__).parent / "fixtures"
DET, NYK, GSW, WAS, PHI = 1610612765, 1610612752, 1610612744, 1610612764, 1610612755


def load(name):
    with gzip.open(FIX / f"injury_{name}.json.gz", "rt") as f:
        return json.load(f)


def test_2023_report_time_and_first_rows():
    r = parse_report(load("a"))
    assert r["report_at"] == datetime(2023, 1, 15, 22, 30, tzinfo=timezone.utc)  # 05:30 PM ET
    first = r["rows"][0]
    assert (first["team_id"], first["player"], first["status"], first["matchup"]) == (DET, "Bagley III, Marvin", "Out", "NYK@DET")
    assert {x["status"] for x in r["rows"]} <= {"Out", "Doubtful", "Questionable", "Probable", "Available"}


def test_not_yet_submitted_line_still_sets_the_game_block():
    # Washington's rows sit under a 'GSW@WAS ... NOT YET SUBMITTED' line: they must belong to GSW@WAS
    rows = parse_report(load("a"))["rows"]
    beal = next(x for x in rows if x["player"].startswith("Beal"))
    assert beal["matchup"] == "GSW@WAS" and beal["team_id"] == WAS


def test_2019_layout_with_previous_status_column():
    r = parse_report(load("b"))
    assert r["report_at"] == datetime(2019, 12, 10, 22, 30, tzinfo=timezone.utc)
    bryant = next(x for x in r["rows"] if x["player"].startswith("Bryant"))  # reason wraps onto the next line
    assert bryant["status"] == "Out" and bryant["matchup"] == "WAS@CHA"


def test_2025_layout_has_spaces_stripped():
    rows = parse_report(load("c"))["rows"]
    brunson = next(x for x in rows if x["player"] == "Brunson,Jalen")
    assert (brunson["team_id"], brunson["status"], brunson["game_time_et"]) == (NYK, "Probable", "07:00")
    embiid = next(x for x in rows if x["player"] == "Embiid,Joel")
    assert embiid["team_id"] == PHI  # team label carried down from a mid-block row


def test_every_block_has_at_most_two_teams():
    for name in "abc":
        by = {}
        for x in parse_report(load(name))["rows"]:
            by.setdefault((x["game_date"], x["matchup"]), set()).add(x["team_id"])
        assert all(len(v) <= 2 for v in by.values())


class FakeSession:
    def __init__(self, status):
        self.status, self.calls = status, 0

    def get(self, url, timeout=0):
        self.calls += 1
        return type("R", (), {"status_code": self.status, "content": b"", "raise_for_status": lambda s: None})()


def test_missing_report_is_recorded_once(tmp_path):
    lake = RawLake(tmp_path)
    s = FakeSession(404)
    ing = InjuryIngest(lake, session=s, interval_s=0)
    ing.fetch(date(2020, 2, 10), 4, "AM")
    ing.fetch(date(2020, 2, 10), 4, "AM")
    assert s.calls == 1
    assert lake.read("injury", key_for(date(2020, 2, 10), 4, "AM")).payload == {"missing": True}


def test_days_include_the_eve_of_each_game_day():
    days = game_days_with_eve([date(2020, 2, 10), date(2020, 2, 12)], date(2020, 2, 1), date(2020, 3, 1))
    assert days == [date(2020, 2, 9), date(2020, 2, 10), date(2020, 2, 11), date(2020, 2, 12)]


def test_full_name_handles_suffixes_and_squashed_names():
    from nbadl.ingest.injury import full_name
    assert full_name("Bagley III, Marvin") == "Marvin Bagley III"
    assert full_name("McCullarJr.,Kevin") == "Kevin McCullar Jr."
    assert full_name("Brunson,Jalen") == "Jalen Brunson"
    assert full_name("Alexander-Walker, Nickeil") == "Nickeil Alexander-Walker"


def test_build_matches_players_and_games_and_delays_availability(tmp_path):
    from nbadl import config, warehouse
    from nbadl.build import load_injuries

    lake = RawLake(tmp_path)
    lake.write("injury", "2023-01-15_05PM", {"pages": load("a")})
    roster = {"boxScoreSummary": {"homeTeam": {"teamId": DET, "players": [
        {"personId": 1630217, "firstName": "Marvin", "familyName": "Bagley III", "nameI": "M. Bagley III",
         "name": "Marvin Bagley III"}]}, "awayTeam": {"teamId": NYK, "players": []}}}
    lake.write("summary_v3", "0022200600", roster)
    con = warehouse.connect(":memory:")
    tip = datetime(2023, 1, 15, 18, tzinfo=timezone.utc)
    con.execute("INSERT INTO games VALUES ('0022200600','2022-23','regular','2023-01-15',?,?,100,100,?,?)",
                [DET, NYK, tip, tip])
    stats = load_injuries(con, lake)
    row = con.execute("SELECT player_id, game_id, status, report_at, available_at FROM injuries "
                      "WHERE player_name LIKE 'Bagley%'").fetchone()
    assert row[0] == 1630217 and row[1] == "0022200600" and row[2] == "Out"
    assert row[4] - row[3] == config.INJURY_PUBLISH_LAG
    assert stats["reports"] == 1 and stats["rows"] > 100


class ScriptedSession:
    """403 for everything except the canary (and, if blocked=False, the canary succeeds)."""

    def __init__(self, blocked):
        self.blocked, self.urls = blocked, []

    def get(self, url, timeout=0):
        self.urls.append(url)
        canary = "2025-01-15_05PM" in url
        status = 403 if (self.blocked or not canary) else 200
        return type("R", (), {"status_code": status, "content": b"", "raise_for_status": lambda s: None})()


def test_403_while_blocked_records_nothing_and_stops(tmp_path):
    lake = RawLake(tmp_path)
    ing = InjuryIngest(lake, session=ScriptedSession(blocked=True), interval_s=0, block_wait_s=0, max_waits=2)
    with pytest.raises(BlockedError):
        ing.fetch(date(2023, 1, 15), 5, "PM")
    assert not lake.exists("injury", key_for(date(2023, 1, 15), 5, "PM"))  # never mislabeled as "missing"


def test_403_while_canary_works_means_the_report_does_not_exist(tmp_path):
    lake = RawLake(tmp_path)
    ing = InjuryIngest(lake, session=ScriptedSession(blocked=False), interval_s=0)
    ing.fetch(date(2020, 2, 10), 4, "AM")
    assert lake.read("injury", key_for(date(2020, 2, 10), 4, "AM")).payload == {"missing": True}
