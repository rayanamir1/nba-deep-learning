import json

import pytest

from nbadl.ingest.sbr import SbrShapeError, extract_odds_tables, parse_odds_tables
from nbadl.lake import RawLake
from nbadl.pbp_v2 import events_from_v2, team_ids
from nbadl.possessions import Kind


def sbr_html(tables):
    blob = json.dumps({"props": {"pageProps": {"oddsTables": tables}}})
    return f'<html><script id="__NEXT_DATA__" type="application/json">{blob}</script></html>'


def test_sbr_parse():
    tables = [{"oddsTableModel": {"gameRows": [{
        "gameView": {"gameId": 123, "startDate": "2024-01-10T00:30:00+00:00",
                     "awayTeam": {"shortName": "Boston"}, "homeTeam": {"shortName": "L.A. Lakers"},
                     "awayTeamScore": 110, "homeTeamScore": 105},
        "oddsViews": [None, {"sportsbook": "draftkings",
                             "openingLine": {"total": 225.5, "overOdds": -110, "underOdds": -110},
                             "currentLine": {"total": 228.0, "overOdds": -105, "underOdds": -115}}],
    }]}}]
    lines = parse_odds_tables(extract_odds_tables(sbr_html(tables)))
    assert len(lines) == 1
    ln = lines[0]
    assert (ln.open_total, ln.current_total, ln.sportsbook) == (225.5, 228.0, "draftkings")


def test_sbr_shape_change_fails_loudly():
    with pytest.raises(SbrShapeError):
        extract_odds_tables("<html>no data</html>")
    with pytest.raises(SbrShapeError):
        parse_odds_tables([{"somethingElse": {}}])


def test_lake_is_append_only(tmp_path):
    lake = RawLake(tmp_path)
    lake.write("x", "k", {"a": 1})
    assert lake.read("x", "k").payload == {"a": 1}
    with pytest.raises(FileExistsError):
        lake.write("x", "k", {"a": 2})


def v2(rows):
    headers = ["GAME_ID", "EVENTNUM", "EVENTMSGTYPE", "EVENTMSGACTIONTYPE", "PERIOD", "PCTIMESTRING",
               "HOMEDESCRIPTION", "NEUTRALDESCRIPTION", "VISITORDESCRIPTION", "PERSON1TYPE",
               "PLAYER1_ID", "PLAYER1_TEAM_ID", "PLAYER2_ID"]
    return {"resultSets": [{"name": "PlayByPlay", "headers": headers, "rowSet": rows}]}


def test_v2_adapter():
    rows = [
        ["g", 1, 1, 1, 1, "11:40", "Tatum 26' 3PT Jump Shot (3 PTS)", None, None, 4, 7, 100, 0],
        ["g", 2, 3, 11, 1, "11:20", None, None, "MISS James Free Throw 1 of 2", 5, 9, 200, 0],
        ["g", 3, 4, 0, 1, "11:20", None, None, "Lakers Rebound", 3, 200, None, 0],
        ["g", 4, 3, 16, 1, "11:10", "Brown Free Throw Technical (1 PTS)", None, None, 4, 8, 100, 0],
        ["g", 5, 8, 0, 1, "11:00", "SUB: White FOR Brown", None, None, 4, 8, 100, 10],
    ]
    evs = events_from_v2(v2(rows))
    assert [e.kind for e in evs] == [Kind.FG_MADE, Kind.FT_MISSED, Kind.REBOUND, Kind.FT_MADE, Kind.SUB]
    assert evs[0].points == 3 and evs[0].clock_s == 700
    assert evs[2].team_id == 200 and evs[2].player_id is None  # team rebound
    assert evs[3].ft_type == "technical"
    assert (evs[4].player_id, evs[4].player_in_id) == (8, 10)
    assert team_ids(evs) == (100, 200)
