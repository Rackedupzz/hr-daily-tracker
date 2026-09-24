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
    """Backtest top decile: 22.2% projected, 17.7% actual; bottom 3.4% vs 5.6%."""
    from mlb_hr.ensemble import calibrate_game_prob

    raw = [0.034, 0.10, 0.176, 0.222, 0.31]
    cal = [calibrate_game_prob(p) for p in raw]
    assert cal == sorted(cal), "calibration must never reorder hitters"
    assert 0.16 < cal[3] < 0.20 and 0.045 < cal[0] < 0.065


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
