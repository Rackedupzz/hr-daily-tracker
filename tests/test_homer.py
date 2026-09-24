"""HOMER and the Results tab.

Pins the rules HOMER's record depends on:

  - his book never reads plate appearances from the day he is handicapping;
  - picks in games that have started are carried over, never re-handicapped;
  - one HR pick per game, and hitters below the PA floor are skipped;
  - grading: a homer is a hit, a hitless final is a miss, a final with no line
    is a no-result, and a tied/postponed game is no decision;
  - both new pages render from saved data.

Run: python -m pytest tests/test_homer.py -v
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlb_hr import homer  # noqa: E402
from mlb_hr.render_review import render_homer_html, render_results_html  # noqa: E402


def _pa(day, game, batter, bid, team, opp, pitcher, pid, hr=0, ev=None, la=None,
        idx=0, home=1, venue="Park A"):
    return {
        "game_pk": game, "date": day, "venue": venue, "batting_team": team,
        "pitching_team": opp, "is_home": home, "batter_id": bid, "batter": batter,
        "bat_side": "R", "pitcher_id": pid, "pitcher": pitcher, "pitch_hand": "R",
        "pa_index": idx, "event_type": "home_run" if hr else "field_out", "is_hr": hr,
        "launch_speed": ev, "launch_angle": la, "trajectory": "fly_ball" if la and la > 20 else None,
    }


def _write_feed(tmp_path, rows):
    path = tmp_path / "pa.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(path)


def test_book_excludes_the_day_being_handicapped(tmp_path):
    rows = [_pa("2026-09-10", 1, "Slugger", 10, "A", "B", "Arm", 99, hr=1, ev=108, la=30)
            for _ in range(5)]
    # Same hitter, the day being handicapped: must be invisible.
    rows += [_pa("2026-09-11", 2, "Slugger", 10, "A", "B", "Arm", 99, hr=1, ev=108, la=30)
             for _ in range(50)]
    book = homer.build_book(_write_feed(tmp_path, rows), date(2026, 9, 11))
    assert book["batters"][10]["season"]["pa"] == 5
    assert book["league"]["pa"] == 5


def test_unposted_lineup_uses_his_usual_spot(tmp_path):
    rows = [dict(_pa("2026-09-0" + str(d), d, "Leadoff", 7, "A", "B", "Arm", 99, idx=0),
                 lineup_slot=1) for d in range(1, 6)]
    book = homer.build_book(_write_feed(tmp_path, rows), date(2026, 9, 10))
    assert book["batters"][7]["usual_slot"] == 1
    projected = homer.hr_features(book, 7, "R", 99, "R", "B", "Park A", None, None)
    posted = homer.hr_features(book, 7, "R", 99, "R", "B", "Park A", None, 9)
    assert projected["exp_pa"] == 4.65 and posted["exp_pa"] == 3.78


def test_log5_is_neutral_against_a_league_average_pitcher():
    assert abs(homer.log5(0.05, 0.03, 0.03) - 0.05) < 1e-9
    assert homer.log5(0.05, 0.045, 0.03) > 0.05


def test_calibration_pulls_toward_league_but_keeps_order():
    lg = 0.03
    a, b = homer.calibrate(0.06, lg), homer.calibrate(0.045, lg)
    assert a < 0.06 and a > b


def test_wind_out_to_pull_field_helps_most_and_domes_are_neutral():
    out_lf, _ = homer.weather_factor({"temp_f": 70, "wind_mph": 10, "wind_dir": "Out To LF"}, "R")
    out_rf, _ = homer.weather_factor({"temp_f": 70, "wind_mph": 10, "wind_dir": "Out To RF"}, "R")
    wind_in, _ = homer.weather_factor({"temp_f": 70, "wind_mph": 10, "wind_dir": "In From CF"}, "R")
    dome, _ = homer.weather_factor({"temp_f": 95, "wind_mph": 20, "wind_dir": "Out To CF",
                                    "condition": "Dome"}, "R")
    assert out_lf > out_rf > 1.0 > wind_in
    assert dome == 1.0


def _book_and_slate(tmp_path):
    rows = []
    for g in range(1, 31):
        for i, (name, bid, team, opp) in enumerate([
            ("Masher", 1, "A", "B"), ("Punch", 2, "A", "B"),
            ("Bopper", 3, "C", "D"), ("Rookie", 4, "C", "D"),
        ]):
            power = name in ("Masher", "Bopper")
            for k in range(4):
                hit = power and k == 0 and g % 3 == 0
                rows.append(_pa("2026-09-01", g * 10 + i, name, bid, team, opp, "Arm", 90 + i,
                                hr=int(hit), ev=106 if power else 85, la=28 if power else 5,
                                idx=k))
    book = homer.build_book(_write_feed(tmp_path, rows), date(2026, 9, 12))

    def hitter(name, bid, team, side, prob):
        return {"batter": name, "batter_id": bid, "team": team, "side": side,
                "bat_side": "R", "lineup_slot": 3, "prob_hr": prob}

    slate = {
        "date": "2026-09-12",
        "games": [
            {"game_pk": 100, "home": "A", "away": "B", "venue": "Park A",
             "home_sp": "Arm", "away_sp": "Arm", "home_sp_hand": "R", "away_sp_hand": "R",
             "hitters": [hitter("Masher", 1, "A", "home", 0.2), hitter("Punch", 2, "A", "home", 0.05)]},
            {"game_pk": 200, "home": "C", "away": "D", "venue": "Park A",
             "home_sp": "Arm", "away_sp": "Arm", "home_sp_hand": "R", "away_sp_hand": "R",
             "hitters": [hitter("Bopper", 3, "C", "home", 0.18), hitter("Rookie", 4, "C", "home", 0.04)]},
        ],
    }
    return book, slate


def test_one_pick_per_game_and_pa_floor(tmp_path, monkeypatch):
    book, slate = _book_and_slate(tmp_path)
    card = homer.build_card(slate, book)
    games = [p["game_pk"] for p in card["hr_picks"]]
    assert len(games) == len(set(games)) == 2
    assert {p["batter"] for p in card["hr_picks"]} == {"Masher", "Bopper"}

    monkeypatch.setattr(homer, "MIN_SEASON_PA", 1000)
    assert homer.build_card(slate, book)["hr_picks"] == []


def test_started_games_keep_their_published_picks(tmp_path):
    book, slate = _book_and_slate(tmp_path)
    first = homer.build_card(slate, book)
    published = next(p for p in first["hr_picks"] if p["game_pk"] == 100)
    published["prob"] = 0.99  # anything a refit would not reproduce
    results = {"games": {100: {"state": "Live"}, 200: {"state": "Preview"}}, "batters": {}}
    again = homer.build_card(slate, book, previous=first, results=results)
    kept = next(p for p in again["hr_picks"] if p["game_pk"] == 100)
    assert kept["prob"] == 0.99
    assert again["locked"] == [100]


def test_grading_hits_misses_no_results_and_no_decisions(tmp_path):
    book, slate = _book_and_slate(tmp_path)
    card = homer.build_card(slate, book)
    masher = next(p for p in card["hr_picks"] if p["batter"] == "Masher")
    bopper = next(p for p in card["hr_picks"] if p["batter"] == "Bopper")
    results = {
        "fetched_at": "x",
        "games": {100: {"state": "Final", "home_score": 5, "away_score": 2},
                  200: {"state": "Final", "home_score": 3, "away_score": 3}},
        "batters": {1: {"game_pk": 100, "hr": 1, "summary": "1-4 | HR"}},
    }
    homer.grade_card(card, results)
    assert masher["result"]["hr"] == 1
    assert bopper["result"].get("dnp") is True
    assert card["record"]["hr"] == {"hit": 1, "scored": 1, "dnp": 1, "pending": 0}
    tied = next(g for g in card["game_picks"] if g["game_pk"] == 200)
    assert tied["result"].get("no_decision")
    assert card["record"]["games"]["won"] + card["record"]["games"]["lost"] == 1
    assert card["record"]["complete"] is True


def test_selection_skips_injured_and_hitters_missing_from_a_posted_lineup():
    cands = [
        {"game_pk": 1, "prob": 0.30, "season_pa": 500, "injury_note": "DTD"},
        {"game_pk": 1, "prob": 0.25, "season_pa": 500, "lineup_posted": True,
         "lineup_confirmed": False},
        {"game_pk": 1, "prob": 0.20, "season_pa": 500, "lineup_posted": True,
         "lineup_confirmed": True},
        {"game_pk": 2, "prob": 0.22, "season_pa": 40},
        {"game_pk": 2, "prob": 0.10, "season_pa": 300},
    ]
    picks = homer.select_hr_picks(cands)
    assert [p["prob"] for p in picks] == [0.20, 0.10]


def test_learned_weights_contributions_add_up():
    model = {"features": ["xhr", "park"], "coef": [0.5, -0.2], "intercept": -2.0,
             "mean": [0.0, 0.0], "scale": [1.0, 0.5]}
    prob, contrib = homer.apply_model(model, {"xhr": 1.0, "park": 0.1})
    import math
    assert abs(contrib["xhr"] - 0.5) < 1e-9 and abs(contrib["park"] + 0.04) < 1e-9
    assert abs(prob - 1 / (1 + math.exp(-(-2.0 + 0.46)))) < 1e-9


def test_walk_forward_never_trains_on_the_week_it_predicts(monkeypatch):
    import numpy as np

    from mlb_hr import homer_fit

    seen = []
    real_fit = homer_fit._fit

    def spy(X, y):
        seen.append(len(y))
        return real_fit(X, y)

    monkeypatch.setattr(homer_fit, "_fit", spy)
    weeks = np.array(sorted(["w1", "w2", "w3", "w4", "w5"] * 40))
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    y = (rng.random(200) < 0.3).astype(int)
    preds = homer_fit.walk_forward(X, y, weeks, [0, 1])
    # First MIN_TRAIN_WEEKS weeks have no prediction; later weeks train only on the past.
    assert np.isnan(preds[weeks < "w4"]).all() and not np.isnan(preds[weeks >= "w4"]).any()
    assert seen == [120, 160]


def _cand(pk, bid, prob, pa=400):
    return {"game_pk": pk, "batter_id": bid, "batter": f"B{bid}", "prob": prob, "season_pa": pa}


def test_parlays_use_one_leg_per_game_and_quote_honest_odds():
    hr = [_cand(1, 10, 0.25), _cand(2, 20, 0.22), _cand(3, 30, 0.10)]
    hits = [_cand(1, 11, 0.80), _cand(2, 21, 0.78), _cand(4, 41, 0.75), _cand(5, 51, 0.74),
            _cand(6, 61, 0.73), _cand(7, 71, 0.72), _cand(8, 81, 0.60, pa=20)]
    plays = {p["key"]: p for p in homer.build_parlays(hr, hits)}
    assert set(plays) == {"hit3", "hit3e1", "hit3e2", "hit4e1", "hit3e3"}
    for play in plays.values():
        games = [leg["game_pk"] for leg in play["leg_list"]]
        assert len(games) == len(set(games)) == play["legs"]
        assert sum(leg["type"] == "hr" for leg in play["leg_list"]) == play["hr"]
        assert all(leg["batter_id"] != 81 for leg in play["leg_list"])  # below PA floor
        expected = 1.0
        for leg in play["leg_list"]:
            expected *= leg["prob"]
        assert abs(play["prob"] - expected) < 1e-12
    # Built to cash takes the likeliest hits in order, one game each.
    assert [l["batter_id"] for l in plays["hit3"]["leg_list"]] == [11, 21, 41]
    assert homer.fair_odds(0.5) == "-100" and homer.fair_odds(0.2) == "+400"
    assert homer.fair_odds(0.75) == "-300"


def test_every_play_can_clear_the_cash_bar():
    """No home run legs: a parlay cannot beat its weakest leg, and HOMER's best
    home run bat cashes about 21%, well under the three-days-in-ten bar."""
    assert all(p["hr"] == 0 for p in homer.PARLAY_PLAYS)
    assert all(3 <= p["legs"] <= 4 for p in homer.PARLAY_PLAYS)
    assert all(0 <= p["value"] <= p["legs"] for p in homer.PARLAY_PLAYS)
    featured = [p for p in homer.PARLAY_PLAYS if p.get("featured")]
    assert len(featured) == 1 and featured[0]["key"] == "hit3"
    assert featured[0]["legs"] == 3 and featured[0]["value"] == 0
    # A key names one exact shape, so no two plays may share one.
    shapes = {(p["legs"], p["hr"], p["value"]) for p in homer.PARLAY_PLAYS}
    assert len(shapes) == len(homer.PARLAY_PLAYS)
    assert len({p["key"] for p in homer.PARLAY_PLAYS}) == len(homer.PARLAY_PLAYS)


def test_a_mixed_play_takes_its_likely_legs_before_its_value_legs():
    hits = [homer.price_leg(_cand(pk, 100 + pk, 0.78, 400), "hits", 0.77, None)
            for pk in range(1, 4)]                       # likely, no edge
    hits += [homer.price_leg(_cand(pk, 100 + pk, 0.70, 400), "hits", 0.50, None)
             for pk in range(4, 7)]                      # big edge, still contenders
    spec = [{"key": "mix", "name": "Mix", "legs": 3, "hr": 0, "value": 1, "blurb": ""}]
    legs = homer.build_parlays([], hits, spec)[0]["leg_list"]
    # Two likeliest first, then the biggest edge from a game they did not claim.
    assert [l["batter_id"] for l in legs] == [101, 102, 104]


def test_book_price_compounds_margin_and_value_plays_chase_edge():
    fit = {"hits": {"market": {"x": [0.5, 0.8], "y": [0.5, 0.8]}}}
    leg = homer.price_leg({"prob": 0.70}, "hits", 0.60, fit)
    assert abs(leg["book_implied"] - 0.60 * homer.BOOK_MARGIN["hits"]) < 1e-9
    assert leg["book_edge"] > 0 and leg["book_price"].startswith("-")
    assert homer.american(3.0) == "+200" and homer.american(1.5) == "-200"

    def priced(pk, bid, prob, season, kind="hits"):
        return homer.price_leg(_cand(pk, bid, prob), kind, season, None)

    hr = [priced(1, 10, 0.24, 0.23, "hr"), priced(2, 20, 0.20, 0.12, "hr"),
          priced(3, 30, 0.12, 0.04, "hr")]
    hits = [priced(pk, 100 + pk, 0.78, 0.76) for pk in range(4, 9)]
    hits += [priced(pk, 100 + pk, 0.70, 0.58) for pk in range(9, 14)]
    hits += [priced(14, 114, 0.55, 0.30)]  # huge edge, but below the value floor
    plays = {p["key"]: p for p in homer.build_parlays(hr, hits)}
    cash, value = plays["hit3"], plays["hit3e3"]
    # Built to cash: the three most likely hits, one game each.
    assert {l["game_pk"] for l in cash["leg_list"]} == set(range(4, 7))
    # Full value: every leg the book underrates most, among real contenders.
    assert {l["game_pk"] for l in value["leg_list"]} <= set(range(9, 14))
    for play in plays.values():
        decimal = 1.0
        for l in play["leg_list"]:
            decimal /= l["book_implied"]
        assert abs(play["book_decimal"] - decimal) < 1e-9
        assert abs(play["ev"] - (play["prob"] * decimal - 1)) < 1e-9


def test_a_reshaped_play_cannot_inherit_the_old_shape_s_track_record():
    """The replay files records by key, so the key has to mean one exact shape."""
    play = {"key": "hit3", "legs": 3, "hr": 0, "value": 0}
    measured = {"hit3": {"legs": 3, "hr": 0, "value": 0, "days": 122, "won": 51}}
    assert homer.play_evidence(measured, play)["won"] == 51
    # Same key, four legs now: the three-leg record is no longer this play's.
    assert homer.play_evidence(measured, {**play, "legs": 4}) is None
    assert homer.play_evidence(measured, {**play, "hr": 1}) is None
    assert homer.play_evidence(measured, {**play, "value": 1}) is None
    assert homer.play_evidence({}, play) is None
    # A record written before the shape was stored is taken at its word.
    assert homer.play_evidence({"hit3": {"days": 122, "won": 51}}, play)["won"] == 51


def test_parlay_grading_voids_a_leg_who_never_batted():
    card = {"hr_picks": [], "game_picks": [], "hit_picks": [], "parlays": [
        {"key": "a", "leg_list": [{"type": "hit", "game_pk": 1, "batter_id": 10, "prob": 0.8},
                                  {"type": "hit", "game_pk": 2, "batter_id": 20, "prob": 0.8}]},
        {"key": "b", "leg_list": [{"type": "hr", "game_pk": 1, "batter_id": 10, "prob": 0.2},
                                  {"type": "hit", "game_pk": 3, "batter_id": 30, "prob": 0.8}]},
        {"key": "c", "leg_list": [{"type": "hit", "game_pk": 3, "batter_id": 30, "prob": 0.8},
                                  {"type": "hit", "game_pk": 4, "batter_id": 40, "prob": 0.8}]},
    ]}
    results = {"fetched_at": "t",
               "games": {1: {"state": "Final"}, 2: {"state": "Final"},
                         3: {"state": "Live"}, 4: {"state": "Final"}},
               "batters": {10: {"hits": 2, "hr": 0}, 30: {"hits": 0, "hr": 0},
                           40: {"hits": 0, "hr": 0}}}
    homer.grade_card(card, results)
    a, b, c = card["parlays"]
    assert a["status"] == "won" and a["leg_list"][1]["status"] == "void"
    assert b["status"] == "lost"          # no homer in a final game sinks it
    assert c["status"] == "lost"          # one final miss is enough, even with a leg live
    assert card["record"]["parlays"] == {"won": 1, "lost": 2, "void": 0, "pending": 0}


def test_hit_features_reward_contact_and_penalize_strikeouts(tmp_path):
    rows = []
    for g in range(30):
        rows.append(dict(_pa("2026-09-01", g, "Contact", 1, "A", "B", "Arm", 90, idx=0),
                         event_type="single", launch_speed=95, launch_angle=12))
        rows.append(dict(_pa("2026-09-01", g, "Whiff", 2, "A", "B", "Arm", 90, idx=1),
                         event_type="strikeout"))
        rows.append(dict(_pa("2026-09-01", g, "Mixed", 3, "A", "B", "Arm", 90, idx=2),
                         event_type="field_out" if g % 2 else "double",
                         launch_speed=90, launch_angle=15))
    book = homer.build_book(_write_feed(tmp_path, rows), date(2026, 9, 5))
    contact = homer.hit_features(book, 1, "R", 90, "R", "B", "Park A", 3)
    whiff = homer.hit_features(book, 2, "R", 90, "R", "B", "Park A", 3)
    assert contact["rules_prob"] > whiff["rules_prob"]
    assert contact["x"]["k_rate"] < whiff["x"]["k_rate"]


def test_calibration_is_monotone_and_grades_only_exist_if_they_outperform():
    import numpy as np

    from mlb_hr import homer_fit

    rng = np.random.default_rng(1)
    p = rng.uniform(0.02, 0.35, 4000)
    # Overconfident at the top: the true rate flattens out above 0.18.
    y = (rng.random(4000) < np.minimum(p, 0.18)).astype(int)
    cal = homer_fit.pav_calibration(p, y)
    assert cal["y"] == sorted(cal["y"]) and max(cal["y"]) < 0.25

    samples = [{"calibrated": float(v), "season_pa": 400, "y": int(t)} for v, t in zip(p, y)]
    bands = homer_fit.grade_bands(samples)
    rates = [b["rate"] for b in bands]
    assert rates == sorted(rates, reverse=True) and len(set(rates)) == len(rates)
    assert bands[-1]["min_prob"] == 0.0 and [b["grade"] for b in bands] == list("ABCDE")[:len(bands)]


def test_replay_reads_the_starting_lineup_not_the_subs():
    from mlb_hr import homer_fit

    rows = [homer.Row(r) for r in (
        dict(_pa("2026-09-01", 9, "Starter", 1, "A", "B", "Ace", 50, idx=0), lineup_slot=1, inning=1),
        dict(_pa("2026-09-01", 9, "Sub", 2, "A", "B", "Ace", 50, hr=1, idx=40), lineup_slot=1, inning=8),
        dict(_pa("2026-09-01", 9, "Visitor", 3, "B", "A", "Home Arm", 60, idx=1, home=0), lineup_slot=1),
    )]
    s = homer_fit._setup(rows)
    assert s["lineups"]["A"][1][0] == 1
    assert s["starters"]["B"][0] == 50 and s["home"] == "A" and s["away"] == "B"
    assert s["homered"][2] == 1 and s["homered"][1] == 0


def test_season_record_and_pages_render(tmp_path):
    book, slate = _book_and_slate(tmp_path)
    card = homer.build_card(slate, book)
    for g in slate["games"]:
        g["live"] = {"state": "Final", "detailed": "Final", "home_score": 4, "away_score": 1}
    slate["games"][0]["hitters"][0]["result"] = {
        "pa": 4, "ab": 4, "hits": 1, "hr": 1, "rbi": 1, "summary": "1-4 | HR",
        "state": "Final", "detailed": "Final", "final": True}
    slate["picks_6"] = [slate["games"][0]["hitters"][0]]
    slate["hit_picks"] = []
    slate["results"] = {"fetched_at": "t", "final_games": 2, "live_games": 0,
                        "upcoming_games": 0, "picks": {"hit": 1, "scored": 1, "dnp": 0,
                                                       "pending": 0},
                        "hit_picks": {"hit": 0, "scored": 0}, "any": True}
    homer.grade_card(card, homer.results_from_slate(slate))
    homer.save_card(card, tmp_path)

    record = homer.season_record(tmp_path)
    assert record["hr_scored"] >= 1 and record["days"][0]["date"] == "2026-09-12"

    page = render_homer_html(card, record, homer.card_days(tmp_path), "2026-09-12")
    assert "HOMER" in page and "Masher" in page and 'href="/results"' in page
    waiting = render_homer_html(None, record, [], "2026-09-13", "studying")
    assert 'http-equiv="refresh"' in waiting

    results_page = render_results_html("2026-09-12", slate, ["2026-09-12"], None, card)
    assert "Every Home Run" in results_page and "Masher" in results_page
    empty = render_results_html("2026-09-01", None, [], None, None)
    assert "Nothing archived" in empty
