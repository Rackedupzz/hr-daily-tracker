"""The 2026-09-30 accuracy fixes, each pinned to the behaviour it corrects.

Run: python -m pytest tests/test_model_fixes.py
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _pa(**kw) -> dict:
    base = {"game_pk": 1, "date": "2026-06-01", "venue": "Home Park", "batting_team": "Home",
            "pitching_team": "Away", "is_home": 1, "batter_id": 1, "batter": "A",
            "bat_side": "R", "pitcher_id": 9, "pitch_hand": "R", "pa_index": 0,
            "event_type": "field_out", "is_hr": 0, "launch_speed": None, "launch_angle": None,
            "lineup_slot": 1}
    base.update(kw)
    return base


def _write(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(path)


# ------------------------------------------------------------------- parks

def test_park_factor_compares_both_lineups_home_and_away(tmp_path):
    """A neutral park is neutral even when its club out-slugs everyone.

    'Home' hitters homer at 6%, everyone else at 2%; no park affects anything.
    At Home Park the rate is 4% (half Home hitters, half visitors); in Home's
    road games it is also 4% (half Home hitters, half the hosts). Counting only
    Home's own hitters on the road (6%) used to read Home Park as 0.67 raw.
    """
    from mlb_hr.parks import build_park_factors

    rows, pk = [], 0
    clubs = ["Home", "A", "B", "C"]
    for g in range(60):
        pk += 1
        host = "Home" if g % 2 == 0 else clubs[1 + g % 3]
        guest = clubs[1 + g % 3] if host == "Home" else "Home"
        for is_home, bat, pit in ((1, host, guest), (0, guest, host)):
            for i in range(50):
                rate = 0.06 if bat == "Home" else 0.02
                rows.append(_pa(game_pk=pk, venue=f"{host} Park", batting_team=bat,
                                pitching_team=pit, is_home=is_home, pa_index=i,
                                is_hr=int((i % 50) < rate * 50)))
    factors = build_park_factors(_write(tmp_path / "feed.jsonl", rows))
    home = factors["Home Park"]
    assert home.measured
    assert home.raw_factor == pytest.approx(1.0, abs=1e-9)
    assert home.factor == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------- switch hitters

def test_switch_hitter_bats_from_both_sides_and_gets_no_platoon_penalty(tmp_path):
    from mlb_hr.features import build_profiles, platoon_factors
    from mlb_hr.model import HRModel

    rows = []
    for i in range(60):
        vs_lefty = i % 3 == 0
        rows.append(_pa(pa_index=i, game_pk=i // 4, batter_id=7, pitch_hand="L" if vs_lefty else "R",
                        bat_side="R" if vs_lefty else "L"))
    rows.append(_pa(pa_index=99, game_pk=99, batter_id=7, pitch_hand="R", bat_side="L"))
    path = _write(tmp_path / "feed.jsonl", rows)
    profile = build_profiles(path)[7]
    assert profile.bat_side == "L"                  # the last plate appearance...
    assert profile.bats == "S"                      # ...is not his batting hand
    assert profile.side_vs("L") == "R" and profile.side_vs("R") == "L"
    factors = platoon_factors({7: profile})
    assert factors[("S", "L")] == factors[("S", "R")] == 1.0
    assert HRModel(path).get_batter(7).hand == "S"


def test_one_sided_hitter_is_not_a_switch_hitter(tmp_path):
    from mlb_hr.features import build_profiles

    rows = [_pa(pa_index=i, batter_id=8, bat_side="L") for i in range(80)]
    rows += [_pa(pa_index=80 + i, batter_id=8, bat_side="R") for i in range(3)]
    assert build_profiles(_write(tmp_path / "feed.jsonl", rows))[8].bats == "L"


# ------------------------------------------------------------ starters

def test_only_the_first_hitter_in_a_slot_is_a_starter(tmp_path):
    """A pinch hitter inherits the slot in the feed; he did not start."""
    from mlb_hr.features import recent_lineups

    rows = []
    for g in range(5):
        day = f"2026-06-0{g + 1}"
        rows.append(_pa(game_pk=g, date=day, batter_id=1, lineup_slot=3, pa_index=2))
        rows.append(_pa(game_pk=g, date=day, batter_id=2, lineup_slot=9, pa_index=8))
        rows.append(_pa(game_pk=g, date=day, batter_id=3, lineup_slot=9, pa_index=70))  # PH
    starts, usual = recent_lineups(_write(tmp_path / "feed.jsonl", rows), "2026-06-30")
    assert starts.get(1) == 5 and starts.get(2) == 5
    assert 3 not in starts
    assert usual[1] == 3 and usual[2] == 9


def test_slot_exposure_is_the_starter_s_not_the_slot_s():
    from mlb_hr.features import SLOT_PA

    values = [SLOT_PA[s] for s in range(1, 10)]
    assert values == sorted(values, reverse=True)
    assert SLOT_PA[1] < 4.6 and SLOT_PA[9] < 3.5


# ------------------------------------------------------------ expected HR

def test_expected_home_runs_score_contact_at_league_rates(tmp_path):
    from mlb_hr.features import build_profiles

    rows = []
    for i in range(40):     # two hitters, same contact; only one of them homers
        for b in (1, 2):
            hr = int(b == 1 and i % 4 == 0)
            rows.append(_pa(pa_index=i * 2 + b, batter_id=b, launch_speed=104.0,
                            launch_angle=28.0, is_hr=hr,
                            event_type="home_run" if hr else "field_out"))
    profiles = build_profiles(_write(tmp_path / "feed.jsonl", rows))
    # Same batted balls, so the same expected home runs, despite 10 HR vs 0.
    assert profiles[1].xhr == pytest.approx(profiles[2].xhr)
    assert 0 < profiles[1].xhr < 10
    assert profiles[1].xhit == pytest.approx(profiles[2].xhit)


# ------------------------------------------------------------ the stack

def test_stacked_probability_moves_the_right_way_with_each_term():
    from mlb_hr.ensemble import HR_STACK, stacked_game_prob

    base = dict(r_hand=0.035, r_all=0.035, park=1.0, weather=1.0, sp_index=1.0,
                pen_index=1.0, expected_pa=4.2, xhr_rate=0.035, league_rate=0.031)
    p0 = stacked_game_prob(**base)
    assert 0.05 < p0 < 0.3
    for term, bump in (("r_all", 0.05), ("park", 1.2), ("weather", 1.1), ("sp_index", 1.3),
                       ("pen_index", 1.3), ("expected_pa", 4.5), ("xhr_rate", 0.05)):
        up = stacked_game_prob(**{**base, term: bump, **({"r_hand": bump} if term == "r_all" else {})})
        assert up > p0, term
    assert all(v > 0 for k, v in HR_STACK.items() if k != "intercept")
    # The tracker's level is an odds multiplier: it moves everyone, reorders no one.
    p_up = stacked_game_prob(**base, level=1.2)
    assert p_up / (1 - p_up) == pytest.approx(1.2 * p0 / (1 - p0))


def test_model_fit_file_overrides_defaults_only_when_complete(tmp_path):
    from mlb_hr.ensemble import HR_STACK, load_model_fit

    good = {k: v + 0.01 for k, v in HR_STACK.items()}
    path = tmp_path / "fit.json"
    path.write_text(json.dumps({"hr_stack": {"coef": good}}), encoding="utf-8")
    fit = load_model_fit(path)
    assert fit["hr_stack"]["coef"] == good
    assert "source" in fit["hit_stack"]          # missing stack: code default
    path.write_text(json.dumps({"hr_stack": {"coef": {"intercept": 0}}}), encoding="utf-8")
    assert load_model_fit(path)["hr_stack"]["coef"] == HR_STACK
    path.write_text("not json", encoding="utf-8")
    assert load_model_fit(path)["hr_stack"]["coef"] == HR_STACK


def test_hit_projection_uses_the_stack_when_it_has_the_inputs():
    from mlb_hr.pitching import LeagueRates, PitcherProfile, project_hits

    league = LeagueRates(hit_rate=0.225, k_rate=0.22)
    sp = PitcherProfile(pitcher_id=1, bf=500, hits_allowed=110, strikeouts=110)
    old = project_hits(0.25, 400, sp, None, league, 4.2)
    assert old["model"] == "log5"
    new = project_hits(0.25, 400, sp, None, league, 4.2, batter_xhit=100.0, batter_k=80)
    assert new["model"] == "stack"
    assert 0.5 < new["projected_hits_raw"] < 1.6
    assert 0.3 < new["prob_at_least_one"] < 0.9
    coors = project_hits(0.25, 400, sp, None, league, 4.2, batter_xhit=100.0, batter_k=80,
                         park_hit=1.15)
    contact = project_hits(0.25, 400, sp, None, league, 4.2, batter_xhit=100.0, batter_k=140)
    assert coors["projected_hits_raw"] > new["projected_hits_raw"]
    assert contact["projected_hits_raw"] < new["projected_hits_raw"]   # more strikeouts


# ------------------------------------------------------- doubleheaders

def _doubleheader_results() -> dict:
    """Batter 7 went 1-for-4 in game 1 and homered in game 2."""
    final = {"state": "Final", "detailed": "Final"}
    g1 = {"pa": 4, "ab": 4, "hits": 1, "hr": 0, "rbi": 0, "summary": "1-4"}
    g2 = {"pa": 4, "ab": 4, "hits": 2, "hr": 1, "rbi": 2, "summary": "2-4, HR"}
    combined = {"batter_id": 7, "batter": "A", "pa": 8, "ab": 8, "hits": 3, "hr": 1, "rbi": 2,
                "summary": "1-4; 2-4, HR", "game_pk": 2, "by_game": {1: g1, 2: g2}}
    return {"fetched_at": "now", "games": {1: final, 2: final}, "batters": {7: combined},
            "final_games": 2, "live_games": 0, "upcoming_games": 0}


def test_a_doubleheader_pick_is_scored_on_his_own_game():
    from mlb_hr.results import attach_results

    pick1 = {"batter": "A", "batter_id": 7, "team": "Home", "game_pk": 1, "prob_hr": 0.2}
    pick2 = dict(pick1, game_pk=2)
    slate = {"games": [{"game_pk": 1, "hitters": [dict(pick1)]},
                       {"game_pk": 2, "hitters": [dict(pick2)]}],
             "picks_6": [pick1], "hit_picks": [dict(pick2, hits_proj={"projected_hits": 1})]}
    attach_results(slate, _doubleheader_results())
    assert pick1["result"]["hr"] == 0 and pick1["result"]["hits"] == 1   # not game 2's HR
    assert slate["results"]["picks"]["hit"] == 0
    assert slate["hit_picks"][0]["result"]["hits"] == 2
    assert slate["games"][0]["hitters"][0]["result"]["hr"] == 0
    assert slate["games"][1]["hitters"][0]["result"]["hr"] == 1


def test_homer_grades_a_doubleheader_pick_on_his_own_game():
    from mlb_hr.homer import grade_card, results_from_slate
    from mlb_hr.results import attach_results

    card = {"hr_picks": [{"batter_id": 7, "game_pk": 1}], "game_picks": [],
            "hit_picks": [{"batter_id": 7, "game_pk": 2}], "parlays": []}
    grade_card(card, _doubleheader_results())
    assert card["record"]["hr"]["hit"] == 0 and card["hr_picks"][0]["result"]["hr"] == 0
    assert card["hit_picks"][0]["result"]["hits"] == 2
    # Rebuilt from a saved slate, too.
    slate = {"games": [{"game_pk": 1, "live": {"state": "Final"},
                        "hitters": [{"batter_id": 7, "game_pk": 1}]},
                       {"game_pk": 2, "live": {"state": "Final"},
                        "hitters": [{"batter_id": 7, "game_pk": 2}]}]}
    attach_results(slate, _doubleheader_results())
    card = {"hr_picks": [{"batter_id": 7, "game_pk": 1}], "game_picks": [],
            "hit_picks": [], "parlays": []}
    grade_card(card, results_from_slate(slate))
    assert card["hr_picks"][0]["result"]["hr"] == 0


# -------------------------------------------------------- tracker level

def _summary(path: Path, rows: list[dict]) -> None:
    from mlb_hr.snapshot import SUMMARY_COLUMNS

    with open(path / "daily_summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in SUMMARY_COLUMNS})


def test_level_follows_recent_actual_over_projected(tmp_path):
    from mlb_hr.snapshot import LEVEL_CLAMP, hr_level

    assert hr_level(tmp_path, date(2026, 6, 10), "v1") == 1.0      # no record yet
    hot = [{"date": f"2026-06-0{d}", "settled": "True", "slate_hr_base": "28.0",
            "slate_hr_actual": "40", "model_version": "v1"} for d in range(1, 10)]
    _summary(tmp_path, hot)
    level = hr_level(tmp_path, date(2026, 6, 10), "v1")
    assert 1.0 < level <= LEVEL_CLAMP[1]
    # A different model's days say nothing about this one.
    assert hr_level(tmp_path, date(2026, 6, 10), "v2") == 1.0
    # Only days before the slate's date count.
    assert hr_level(tmp_path, date(2026, 6, 1), "v1") == 1.0


def test_slate_calibration_counts_starters_who_batted_in_final_games():
    from mlb_hr.snapshot import slate_calibration

    final = {"final": True, "pa": 4, "hr": 1, "hits": 2}
    slate = {"model_version": "v1", "hr_level": 1.1, "games": [{"hitters": [
        {"lineup_slot": 1, "prob_hr": 0.2, "prob_ensemble_base": 0.18,
         "hits_proj": {"projected_hits_raw": 1.1}, "result": final},
        {"lineup_slot": 2, "prob_hr": 0.1, "prob_ensemble_base": 0.09,
         "hits_proj": {"projected_hits_raw": 0.9}, "result": {**final, "hr": 0, "hits": 0}},
        {"lineup_slot": None, "prob_hr": 0.3, "result": final},                  # no lineup
        {"lineup_slot": 3, "prob_hr": 0.3, "result": {**final, "final": False}},  # not final
        {"lineup_slot": 4, "prob_hr": 0.3, "result": {**final, "pa": 0}},         # scratched
    ]}]}
    s = slate_calibration(slate)
    assert s["slate_starters"] == 2
    assert s["slate_hr_expected"] == pytest.approx(0.3)
    assert s["slate_hr_base"] == pytest.approx(0.27)
    assert s["slate_hr_actual"] == 1
    assert s["slate_hits_expected"] == pytest.approx(2.0) and s["slate_hits_actual"] == 2
    assert s["model_version"] == "v1" and s["hr_level"] == 1.1
