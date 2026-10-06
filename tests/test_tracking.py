"""Tracker integrity: picks stay as published and every pick gets settled.

Pins the fixes made after the first three tracked days (2026-09-10..12):

  - a pick whose game is over but who never batted is "did not play", which
    settles the day without counting as a miss;
  - daily totals only count picks that actually batted in a finished game;
  - a rebuild after first pitch keeps the picks in games already underway;
  - a slate for a date never sees plate appearances from that date;
  - HR probabilities are recalibrated without changing their order.

Run: python -m pytest tests/test_tracking.py -v
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlb_hr.results import attach_results  # noqa: E402
from mlb_hr.slate import choose_hr_picks, data_before, lock_started_picks  # noqa: E402
from mlb_hr.snapshot import pick_rows, summarize_rows  # noqa: E402


def _hitter(name, team, pk, prob, bid):
    return {"batter": name, "team": team, "game_pk": pk, "prob_hr": prob,
            "batter_id": bid,
            "hits_proj": {"projected_hits": prob * 4, "prob_at_least_one": 0.6}}


def _slate(hitters_by_game):
    return {
        "date": "2026-09-12",
        "games": [{"game_pk": pk, "venue": f"Park {pk}", "hitters": hs}
                  for pk, hs in hitters_by_game.items()],
    }


def test_did_not_play_settles_without_a_miss():
    judge = _hitter("Aaron Judge", "NYY", 1, 0.22, 10)
    seager = _hitter("Corey Seager", "TEX", 2, 0.24, 20)
    slate = _slate({1: [judge], 2: [seager]})
    slate["picks_6"], slate["hit_picks"] = [judge, seager], []
    results = {
        "fetched_at": "", "final_games": 2, "live_games": 0, "upcoming_games": 0,
        "games": {1: {"state": "Final", "detailed": "Final"},
                  2: {"state": "Final", "detailed": "Final"}},
        "batters": {20: {"game_pk": 2, "pa": 4, "ab": 4, "hits": 1, "hr": 1,
                         "rbi": 1, "summary": "1-4 | HR"}},
    }
    attach_results(slate, results)
    assert slate["results"]["picks"] == {"hit": 1, "scored": 1, "dnp": 1, "pending": 0}

    rows = pick_rows(slate)
    summary = summarize_rows(rows, day="2026-09-12", games=2, built_at="")
    assert summary["hr_picks_final"] == 1 and summary["hr_picks_hit"] == 1
    assert summary["picks_dnp"] == 1 and summary["settled"] is True
    # Expected is over the picks that batted only: Seager's 0.24.
    assert summary["hr_expected"] == 0.24


def test_rebuild_keeps_picks_in_started_games():
    old_pick = _hitter("Kyle Schwarber", "PHI", 1, 0.23, 1)
    previous = {"picks_6": [old_pick], "hit_picks": []}
    # The refit now prefers someone else in the started game, and a new name
    # in a game that has not begun.
    fresh = _slate({
        1: [_hitter("Bryce Harper", "PHI", 1, 0.30, 2), _hitter("Kyle Schwarber", "PHI", 1, 0.20, 1)],
        2: [_hitter("Cal Raleigh", "SEA", 2, 0.25, 3)],
    })
    lock_started_picks(previous, fresh, started={1})
    names = [p["batter"] for p in fresh["picks_6"]]
    assert names[0] == "Kyle Schwarber" and fresh["picks_6"][0]["prob_hr"] == 0.23
    assert "Bryce Harper" not in names, "a started game must not gain a pick"
    assert "Cal Raleigh" in names


def test_published_pick_missing_from_refit_is_kept(tmp_path):
    """A morning pick who is not in tonight's lineup stays on the record."""
    from mlb_hr.snapshot import PICK_COLUMNS, published_picks

    with open(tmp_path / "picks.csv", "w", encoding="utf-8", newline="") as fh:
        import csv
        w = csv.DictWriter(fh, fieldnames=PICK_COLUMNS)
        w.writeheader()
        w.writerow({"date": "2026-09-12", "kind": "hr", "rank": 1, "batter": "Ben Rice",
                    "team": "New York Yankees", "projected": 0.2248, "season_hr": 36,
                    "season_pa": 604})
    feed = tmp_path / "feed.jsonl"
    feed.write_text(json.dumps({"batter": "Ben Rice", "batter_id": 700250}) + "\n")
    fresh = {"games": [{"game_pk": 9, "venue": "Yankee Stadium", "home": "New York Yankees",
                        "away": "Boston Red Sox", "hitters": []}]}

    previous = published_picks("2026-09-12", fresh, tmp_path, str(feed))
    (rice,) = previous["picks_6"]
    assert rice["game_pk"] == 9 and rice["batter_id"] == 700250 and rice["prob_hr"] == 0.2248
    lock_started_picks(previous, fresh, started={9})
    assert [p["batter"] for p in fresh["picks_6"]] == ["Ben Rice"]


def test_hr_picks_one_per_team():
    games = _slate({
        1: [_hitter("A", "NYY", 1, 0.3, 1), _hitter("B", "NYY", 1, 0.29, 2)],
        2: [_hitter("C", "BOS", 2, 0.2, 3)],
    })["games"]
    assert [p["batter"] for p in choose_hr_picks(games)] == ["A", "C"]


def test_slate_data_excludes_its_own_date(tmp_path):
    feed = tmp_path / "season_pa_v2.jsonl"
    rows = [{"game_pk": 1, "date": "2026-09-11", "is_hr": 0},
            {"game_pk": 2, "date": "2026-09-12", "is_hr": 1},
            {"game_pk": 3, "date": "2026-09-10", "is_hr": 0}]
    feed.write_text("".join(json.dumps(r) + "\n" for r in rows))

    path = data_before(str(feed), date(2026, 9, 12))
    kept = [json.loads(line)["date"] for line in open(path)]
    assert kept == ["2026-09-11", "2026-09-10"]
    # Nothing to strip: the original file is used as-is.
    assert data_before(str(feed), date(2026, 9, 13)) == str(feed)


def test_hr_calibration_softens_without_reordering():
    """Replay's top 2%: 24.9% projected, 21.8% actual. The top calibration
    pulls the very top down, leaves the middle alone, and never reorders."""
    from mlb_hr.ensemble import HR_TOP_CAL, MAX_GAME_PROB, top_calibrate

    raw = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, MAX_GAME_PROB, 0.9]
    cal = [top_calibrate(p, HR_TOP_CAL) for p in raw]
    assert cal == sorted(cal), "calibration must never reorder hitters"
    assert 0.215 < top_calibrate(0.25, HR_TOP_CAL) < 0.24
    assert abs(top_calibrate(0.12, HR_TOP_CAL) - 0.12) < 0.01


if __name__ == "__main__":
    import tempfile

    for fn in (test_did_not_play_settles_without_a_miss,
               test_rebuild_keeps_picks_in_started_games,
               test_hr_picks_one_per_team,
               test_hr_calibration_softens_without_reordering):
        fn()
        print(f"PASS {fn.__name__}")
    for fn in (test_slate_data_excludes_its_own_date,
               test_published_pick_missing_from_refit_is_kept):
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
            print(f"PASS {fn.__name__}")


def test_schedule_follows_the_postseason(monkeypatch):
    """The page froze on 2026-09-27: the schedule was asked for regular-season
    games only, so the wild card round read as "no games"."""
    from mlb_hr import fetch

    asked = []
    game = {"gamePk": 849844, "officialDate": "2026-10-01", "gameType": "F",
            "venue": {"name": "Truist Park"}, "status": {"abstractGameState": "Preview"},
            "teams": {"home": {"team": {"name": "Atlanta Braves", "id": 144}},
                      "away": {"team": {"name": "Philadelphia Phillies", "id": 143}}}}

    def fake_get(url, *a, **k):
        asked.append(url)
        return {"dates": [{"date": "2026-10-01", "games": [game]}]}

    monkeypatch.setattr(fetch, "_get", fake_get)
    games = fetch.schedule(date(2026, 10, 1), date(2026, 10, 1))
    types = asked[0].split("gameType=")[1].split("&")[0].split(",")
    assert {"R", "F", "D", "L", "W"} <= set(types)
    assert "S" not in types and "A" not in types       # no spring training, no All-Star Game
    assert [g.game_pk for g in games] == [849844] and games[0].state == "Preview"


def test_a_hitter_faces_the_other_club_s_starter(monkeypatch):
    """Each lineup used to be handed its own starter's hand, so a righty-vs-lefty
    game gave both lineups the wrong platoon split."""
    from types import SimpleNamespace

    from mlb_hr import slate as slate_mod

    game = SimpleNamespace(game_pk=7, date="2026-10-01", venue="Truist Park", state="Preview",
                           home="Atlanta Braves", away="Philadelphia Phillies",
                           home_id=144, away_id=143)
    monkeypatch.setattr(slate_mod, "schedule", lambda start, end: [game])
    monkeypatch.setattr(slate_mod, "probable_pitchers", lambda day: [{
        "game_pk": 7, "home_sp_id": 1, "home_sp": "Lefty Home", "away_sp_id": 2,
        "away_sp": "Righty Away"}])
    monkeypatch.setattr(slate_mod, "pitcher_hands", lambda ids: {1: "L", 2: "R"})
    monkeypatch.setattr(slate_mod, "active_rosters", lambda ids, season: {
        144: [{"id": 10, "name": "Home Bat", "pos": "RF"}],
        143: [{"id": 20, "name": "Away Bat", "pos": "1B"}]})

    def batter(bid, name):
        return SimpleNamespace(batter_id=bid, batter=name, pa_total=400, hr_total=20, hand="R",
                               hr_vs_r=14, pa_vs_r=280, hr_vs_l=6, pa_vs_l=120)

    seen = {}

    class Model:
        park_factors = {}

        def get_batter(self, bid):
            return {10: batter(10, "Home Bat"), 20: batter(20, "Away Bat")}.get(bid)

        def pr_hr_today(self, b, _pa, venue, sp_hand):
            seen[b.batter] = sp_hand
            return 0.1

    built = slate_mod.GameSlate(date(2026, 10, 1), Model())
    facing = {h["batter"]: h["facing_hand"] for h in built.games[0]["hitters"]}
    assert facing == seen == {"Home Bat": "R", "Away Bat": "L"}
    home_bat = next(h for h in built.games[0]["hitters"] if h["batter"] == "Home Bat")
    assert (home_bat["hr_vs_facing"], home_bat["pa_vs_facing"]) == (14, 280)


def test_a_one_game_slate_picks_one_bat_per_team():
    games = _slate({
        1: [_hitter("A", "ATL", 1, 0.3, 1), _hitter("B", "ATL", 1, 0.29, 2),
            _hitter("C", "PHI", 1, 0.2, 3), _hitter("D", "PHI", 1, 0.1, 4)],
    })["games"]
    assert [p["batter"] for p in choose_hr_picks(games)] == ["A", "C"]
    # A started game still cannot gain a pick.
    assert choose_hr_picks(games, closed={1}) == []
