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


def test_team_lookup_resolves_sbr_abbreviations():
    from nbadl.build import _norm, team_lookup

    look = team_lookup()
    sbr_names = ["BK", "GS", "NO", "NY", "PHO", "SA", "LAL", "LAC", "PHI", "UTA", "WAS"]
    assert [n for n in sbr_names if _norm(n) not in look] == []
    assert look[_norm("NY")] == look[_norm("NYK")]


class _FakeResp:
    def __init__(self, status, text=""):
        self.status_code, self.text = status, text

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(str(self.status_code))


class _FakeSession:
    def __init__(self, statuses, text):
        self.statuses, self.text, self.calls = list(statuses), text, 0

    def get(self, *a, **k):
        self.calls += 1
        return _FakeResp(self.statuses.pop(0), self.text)


def _sbr(statuses, monkeypatch, tmp_path):
    from datetime import date

    from nbadl.ingest import sbr

    monkeypatch.setattr(sbr.time, "sleep", lambda s: None)
    page = '<script id="__NEXT_DATA__" type="application/json">{"props":{"pageProps":{"oddsTables":[]}}}</script>'
    sess = _FakeSession(statuses, page)
    return sbr.SbrIngest(RawLake(tmp_path), interval_s=0, session=sess), sess, date(2019, 10, 5)


def test_sbr_retries_transient_5xx(monkeypatch, tmp_path):
    ing, sess, d = _sbr([502, 502, 200], monkeypatch, tmp_path)
    assert ing.history(d) is True and sess.calls == 3


def test_sbr_does_not_retry_404(monkeypatch, tmp_path):
    import requests

    ing, sess, d = _sbr([404, 200], monkeypatch, tmp_path)
    with pytest.raises(requests.HTTPError):
        ing.history(d)
    assert sess.calls == 1


def test_sbr_gives_up_after_retries(monkeypatch, tmp_path):
    import requests

    ing, sess, d = _sbr([503] * 10, monkeypatch, tmp_path)
    with pytest.raises(requests.HTTPError):
        ing.history(d)
    assert sess.calls == 5


def _ft(desc, aid):
    return {"actionId": aid, "actionNumber": aid, "period": 1, "clock": "PT10M00.00S", "actionType": "Free Throw",
            "subType": "Free Throw 1 of 2", "description": desc, "teamId": 1, "personId": 7,
            "scoreHome": "0", "scoreAway": "0"}


def test_free_throw_by_a_player_whose_name_starts_with_miss_is_made():
    from nbadl.possessions import Kind

    payload = {"game": {"gameId": "x", "actions": [
        _ft("Missi Free Throw 1 of 2 (2 PTS)", 1),         # Yves Missi made it
        _ft("MISS Missi Free Throw 2 of 2", 2),            # and then missed one
        _ft("MISS Fears Free Throw 1 of 2", 3)]}}
    kinds = [(e.kind, e.points) for e in events_from_v3(payload)]
    assert kinds == [(Kind.FT_MADE, 1), (Kind.FT_MISSED, 0), (Kind.FT_MISSED, 0)]


def test_odds_match_falls_back_to_one_day_off_but_not_when_ambiguous():
    from datetime import date

    from nbadl.build import match_game

    g = date(2021, 3, 3)
    by_day = {(g, 1): ("G1", "tipA")}
    by_pair = {(1, 2): [(g, ("G1", "tipA"))]}
    assert match_game(by_day, by_pair, g, 1, 2) == ("G1", "tipA")                 # exact
    assert match_game(by_day, by_pair, date(2021, 3, 2), 1, 2) == ("G1", "tipA")  # SBR one day early
    assert match_game(by_day, by_pair, date(2021, 3, 6), 1, 2) == (None, None)    # too far
    assert match_game(by_day, by_pair, date(2021, 3, 2), 1, None) == (None, None) # unknown team
    by_pair[(1, 2)].append((date(2021, 3, 1), ("G0", "tipB")))                   # two candidates: refuse
    assert match_game(by_day, by_pair, date(2021, 3, 2), 1, 2) == (None, None)
