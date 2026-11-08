import gzip
import json
from pathlib import Path

import pytest

from nbadl.ingest.sbr import SbrShapeError, extract_odds_tables, parse_odds_tables
from nbadl.lake import RawLake
from nbadl.pbp_v3 import events_from_v3, roster_from_summary_v3, team_ids, unresolved_subs
from nbadl.possessions import Kind, count_possessions, parse_game

FIX = Path(__file__).parent / "fixtures"


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


def _trimmed():
    return json.loads((FIX / "game_0021300001_v3_trimmed.json").read_text())


IND, ORL = 1610612754, 1610612753


def test_v3_adapter_on_real_sample():
    """Trimmed real PlayByPlayV3 + BoxScoreSummaryV3 payloads, ORL @ IND 2013-10-29."""
    f = _trimmed()
    evs = {e.seq: e for e in events_from_v3(f["pbp_v3"], roster_from_summary_v3(f["summary_v3"]))}
    assert evs[1].kind == Kind.PERIOD_START and evs[1].clock_s == 720
    assert evs[14].kind == Kind.FG_MADE and evs[14].points == 3 and evs[14].team_id == IND
    assert evs[10].kind == Kind.OTHER and evs[10].team_id == ORL  # block: shares actionNumber, own actionId
    assert (evs[16].kind, evs[16].team_id, evs[16].player_id) == (Kind.REBOUND, ORL, None)  # team rebound
    assert (evs[24].kind, evs[24].team_id, evs[24].player_id) == (Kind.REBOUND, IND, None)
    assert (evs[236].kind, evs[236].team_id, evs[236].player_id) == (Kind.TURNOVER, IND, None)  # shot clock
    assert evs[23].kind == Kind.FT_MISSED and evs[23].ft_type == "regular"
    assert evs[289].kind == Kind.FT_MISSED and evs[289].ft_type == "technical"
    assert evs[365].kind == Kind.FT_MADE and evs[365].ft_type == "technical" and evs[365].points == 1
    # sub-in names: accent-stripped ("Vucevic" -> Vučević), initial-qualified ("G. Hill" vs
    # S. Hill), and a player (Butler) who appears nowhere else in the play-by-play
    assert (evs[182].player_id, evs[182].player_in_id) == (203094, 202696)
    assert (evs[184].player_id, evs[184].player_in_id) == (203524, 201588)
    assert (evs[453].player_id, evs[453].player_in_id) == (201579, 201588)
    assert (evs[493].player_id, evs[493].player_in_id) == (202331, 2446)


def test_v3_sub_without_roster_is_counted_unresolved():
    f = _trimmed()
    evs = events_from_v3(f["pbp_v3"])  # no summary roster: Butler cannot be named
    assert unresolved_subs(evs) >= 1


def test_v3_full_game_matches_official_score():
    payload = json.loads(gzip.open(FIX / "pbp_v3_0021300001_full.json.gz", "rt").read())
    roster = roster_from_summary_v3(_trimmed()["summary_v3"])
    evs = events_from_v3(payload, roster)
    assert unresolved_subs(evs) == 0
    teams = team_ids(evs)
    assert teams == (ORL, IND)
    g = parse_game(evs, teams)
    assert g.points_by_team == {IND: 97, ORL: 87}
    assert g.lineup_complete
    poss = count_possessions(g)
    assert all(88 <= n <= 100 for n in poss.values()), poss
    assert sum(s.seconds for s in g.stints) == 48 * 60
