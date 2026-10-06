"""The house parlays (the NFL tracker's rules) on the slate, and HOMER's 5-pick parlays.

Run: python -m pytest tests/test_parlays.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlb_hr import homer, parlays  # noqa: E402

SPEC = {s["key"]: s for s in parlays.PARLAYS}
QUALIFIED = {k: {"days": 140, "won": 25, "expected_wins": 29.0, "rate": 0.18, "predicted": 0.2}
             for k in SPEC}


def _hitter(bid, team, prob_hr, p1, side="home", injury=None, pa=4.4):
    return {"batter": f"B{bid}", "batter_id": bid, "team": team, "side": side,
            "facing_hand": "R", "lineup_slot": 3, "prob_hr": prob_hr, "injury_note": injury,
            "hits_proj": {"prob_at_least_one": p1, "projected_hits": 1.0,
                          "projected_hits_raw": 1.0, "expected_pa": pa}}


def _games(n: int, start: str = "2026-09-24T23:05:00Z") -> list[dict]:
    """Game i's best HR bat is 0.12 + i/100 (the away hitter), its best hit bat
    0.66 + i/100, and its home side the favourite at 0.55 + i/100."""
    games = []
    for i in range(n):
        pk = 100 + i
        games.append({"game_pk": pk, "start": start, "home_sp": f"H{i}", "away_sp": f"A{i}",
                      "venue": f"Park {i}", "home": f"Home {i}", "away": f"Away {i}",
                      "projection": {"home_team": f"Home {i}", "away_team": f"Away {i}",
                                     "favorite": f"Home {i}", "underdog": f"Away {i}",
                                     "favorite_prob": 0.55 + i / 100},
                      "hitters": [
                          _hitter(pk * 10 + 1, f"Home {i}", 0.10 + i / 100, 0.66 + i / 100),
                          _hitter(pk * 10 + 2, f"Away {i}", 0.12 + i / 100, 0.60 + i / 100,
                                  side="away")]})
    return games


def test_core_tickets_are_the_best_leg_of_the_five_best_games():
    games = _games(7)
    hr = parlays.build(games, SPEC["hr5"])
    hit = parlays.build(games, SPEC["hit5"])
    win = parlays.build(games, SPEC["win5"])
    for play in (hr, hit, win):
        pks = [leg["game_pk"] for leg in play["leg_list"]]
        assert len(pks) == len(set(pks)) == 5
        assert math.isclose(play["prob"], math.prod(leg["prob"] for leg in play["leg_list"]))
    # Games 6..2 have the best legs; the away bat for home runs, the home bat for hits.
    assert [leg["batter_id"] for leg in hr["leg_list"]] == [1062, 1052, 1042, 1032, 1022]
    assert [leg["batter_id"] for leg in hit["leg_list"]] == [1061, 1051, 1041, 1031, 1021]
    assert all(leg["type"] == "hits" and leg["line"] == 1 for leg in hit["leg_list"])
    assert [leg["team"] for leg in win["leg_list"]] == [f"Home {i}" for i in (6, 5, 4, 3, 2)]
    assert hr["leg_list"][0]["opp_sp"] == "H6"          # an away hitter faces the home starter
    assert hr["fair_odds"].startswith("+") and "," in hr["fair_odds"]
    assert hr["one_in"] == round(1 / hr["prob"])


def test_multi_hit_rungs_agree_with_the_served_one_plus_chance():
    h = _hitter(1, "T", 0.1, 0.72)
    p1, p2, p3 = (parlays.hit_prob(h, k) for k in (1, 2, 3))
    assert p1 == 0.72 and 0 < p3 < p2 < p1
    # The 1+ chance is 1 - (1 - r)^PA; the 2+ rung uses the same r and PA.
    r = 1 - (1 - 0.72) ** (1 / 4.4)
    assert math.isclose(p2, 1 - (1 - r) ** 4.4 - 4.4 * r * (1 - r) ** 3.4, rel_tol=1e-9)


def test_a_core_ticket_needs_its_replay_record():
    games = _games(8)
    assert parlays.choose(games) == [p for p in parlays.choose(games) if p["longshot"]]
    bar = parlays.CORE_MIN_CASHED
    keys = {p["key"] for p in parlays.choose(games, evidence={
        "hit5": {"days": 140, "won": bar}, "hr5": {"days": 140, "won": bar - 1}})}
    assert "hit5" in keys and "hr5" not in keys


def test_chalk_takes_each_games_likeliest_leg_of_any_market():
    play = parlays.build(_games(7), SPEC["chalk5"])
    # A 0.66+ hit bat beats a 0.55+ favourite in every game here.
    assert all(leg["type"] == "hits" for leg in play["leg_list"])
    assert len({leg["game_pk"] for leg in play["leg_list"]}) == 5
    # Leg for leg the Five Hits ticket, so the day posts only one of them.
    keys = [p["key"] for p in parlays.choose(_games(7), evidence=QUALIFIED)]
    assert "hit5" in keys and "chalk5" not in keys


def test_injured_bats_are_passed_over_and_short_slates_post_no_core_ticket():
    games = _games(6)
    games[5]["hitters"][1]["injury_note"] = "Day-To-Day"
    hr = parlays.build(games, SPEC["hr5"])
    assert 1052 not in [leg["batter_id"] for leg in hr["leg_list"]]
    assert 1051 in [leg["batter_id"] for leg in hr["leg_list"]]   # his teammate takes the game
    assert parlays.build(_games(4), SPEC["hr5"]) is None
    assert not [p for p in parlays.choose(_games(4), evidence=QUALIFIED) if not p["longshot"]]


def test_windows_split_at_seven_eastern():
    assert parlays.window_of("2026-09-24T17:05:00Z") == "early"     # 1:05 PM EDT
    assert parlays.window_of("2026-09-24T22:40:00Z") == "early"     # 6:40 PM EDT
    assert parlays.window_of("2026-09-24T23:05:00Z") == "late"      # 7:05 PM EDT
    assert parlays.window_of("2026-09-25T02:10:00Z") == "late"      # 10:10 PM EDT
    assert parlays.window_of("") is None


def _window_slate():
    """Six early games and six late ones, each with three strong hit bats."""
    games = _games(6, "2026-09-24T17:05:00Z") + _games(6, "2026-09-24T23:40:00Z")
    for j, g in enumerate(games):
        g["game_pk"] = 200 + j
        for h in g["hitters"]:
            h["batter_id"] += j * 1000
            h["hits_proj"]["prob_at_least_one"] = 0.74
        g["hitters"].append(_hitter(9000 + j, g["home"], 0.05, 0.73))
    return games


def test_each_ten_stays_in_its_window_and_pays_at_least_plus_1000():
    games = _window_slate()
    early = {g["game_pk"] for g in games[:6]}
    for key, window in (("early10", early), ("late10", {g["game_pk"] for g in games[6:]})):
        play = parlays.build(games, SPEC[key])
        legs = play["leg_list"]
        assert len(legs) == 10 and {leg["game_pk"] for leg in legs} <= window
        assert play["prob"] <= parlays.TEN_TARGET_PROB
        assert sum(leg["type"] == "hr" for leg in legs) == 1          # the one required home run
        assert any(leg["type"] == "hits" for leg in legs)
        per_game = [sum(leg["game_pk"] == pk for leg in legs) for pk in window]
        assert max(per_game) <= parlays.TEN_MAX_PER_GAME
        ids = [leg["batter_id"] or leg["team"] for leg in legs]
        assert len(ids) == len(set(ids))


def test_a_short_window_loosens_the_ten_and_says_so():
    games = _window_slate()[:4]                                      # four early games only
    play = parlays.build(games, SPEC["early10"])
    assert play and len(play["leg_list"]) == 10 and "three legs a game" in play["note"]
    assert parlays.build(games, SPEC["late10"]) is None


def test_a_ticket_locks_once_any_leg_starts_and_backfills_a_finished_day():
    games = _games(8)
    posted = parlays.choose(games, evidence=QUALIFIED)
    hr = next(p for p in posted if p["key"] == "hr5")
    first_leg_game = hr["leg_list"][0]["game_pk"]
    games[7]["hitters"][1]["prob_hr"] = 0.9       # a new favourite after lineups post
    # A leg's game has started: the ticket stands exactly as posted.
    kept = parlays.choose(games, previous=posted, started={first_leg_game}, evidence=QUALIFIED)
    assert next(p for p in kept if p["key"] == "hr5") is hr
    # Only an off-ticket game has started: rebuilt from the games still to come.
    used = {leg["game_pk"] for leg in hr["leg_list"]}
    off_ticket = next(g["game_pk"] for g in games if g["game_pk"] not in used)
    rebuilt = next(p for p in parlays.choose(games, previous=posted, started={off_ticket}, evidence=QUALIFIED)
                   if p["key"] == "hr5")
    assert rebuilt["leg_list"][0]["batter_id"] == 1072
    assert off_ticket not in [leg["game_pk"] for leg in rebuilt["leg_list"]]
    # Every game under way and nothing was posted: built from the morning, flagged.
    late = parlays.choose(games, previous=None, started={g["game_pk"] for g in games}, evidence=QUALIFIED)
    assert late and all(p["backfilled"] for p in late)


def test_tickets_posted_under_the_old_rules_settle_as_retired():
    games = _games(8)
    old = {"key": "hit5", "kind": "hit", "name": "Five-Hit Parlay", "blurb": "", "legs": 5, "prob": 0.2,
           "fair_odds": "+400", "leg_list": [{"type": "hit", "game_pk": 100, "batter_id": 1001,
                                              "batter": "B", "prob": 0.7}]}
    gone = {**old, "key": "old5"}
    out = parlays.choose(games, previous=[old, gone], started={100})
    assert {p["key"] for p in out if p.get("retired")} == {"hit5", "old5"}
    # An old 'hit' leg grades as 1+ hit.
    parlays.grade([old], lambda leg: {"hits": 1, "hr": 0, "final": True, "pa": 4}, lambda leg: None)
    assert old["status"] == "won"


def test_grading_follows_sportsbook_rules():
    play = parlays.build(_games(5), SPEC["hit5"])
    lines = {leg["batter_id"]: {"hits": 1, "hr": 0, "final": True, "pa": 4} for leg in play["leg_list"]}
    ids = [leg["batter_id"] for leg in play["leg_list"]]
    lines.pop(ids[0])                                   # scratched: game final, never batted

    def line_for(leg):
        return lines.get(leg["batter_id"])

    def did_not_play(leg):
        return {"dnp": True, "final": True} if leg["batter_id"] == ids[0] else None

    parlays.grade([play], line_for, did_not_play)
    assert play["leg_list"][0]["status"] == "void"
    assert play["status"] == "won"                      # the rest rode and all hit
    lines[ids[1]] = {"hits": 0, "hr": 0, "final": False, "pa": 2}
    parlays.grade([play], line_for, did_not_play)
    assert play["status"] == "pending"                  # not wrong until his last at-bat
    lines[ids[1]]["final"] = True
    parlays.grade([play], line_for, did_not_play)
    assert play["status"] == "lost"


def test_multi_hit_and_winner_legs_grade():
    games = _games(5)
    multi = parlays.build(games, SPEC["multi5"])
    win = parlays.build(games, SPEC["win5"])
    one_hit = {"hits": 1, "hr": 0, "final": True, "pa": 4}
    parlays.grade([multi], lambda leg: one_hit, lambda leg: None)
    assert multi["status"] == "lost"                    # one hit is not two
    finals = {g["game_pk"]: {"state": "Final", "home_score": 5, "away_score": 2} for g in games}
    parlays.grade([win], lambda leg: None, lambda leg: None, lambda leg: finals[leg["game_pk"]])
    assert win["status"] == "won"
    finals[104]["away_score"] = 9
    parlays.grade([win], lambda leg: None, lambda leg: None, lambda leg: finals[leg["game_pk"]])
    assert win["status"] == "lost"
    parlays.grade([win], lambda leg: None, lambda leg: None, lambda leg: None)
    assert win["status"] == "pending"


def test_slate_tracker_records_the_original_tickets():
    from mlb_hr.snapshot import summary_row

    games = _games(6)
    slate = {"date": "2026-09-27", "games": games, "games_count": 6, "picks_6": [],
             "hit_picks": [], "parlays": parlays.choose(games, evidence=QUALIFIED)}
    next(p for p in slate["parlays"] if p["key"] == "hr5")["status"] = "lost"
    row = summary_row(slate)
    assert row["hr_parlay"] == "lost" and row["hit_parlay"] == "pending"
    assert 0 < row["hr_parlay_prob"] < row["hit_parlay_prob"] < 1


def test_slate_page_shows_the_parlays_and_the_replay():
    from mlb_hr.render import _render_parlays_section

    games = _window_slate()
    replay = {**QUALIFIED, "hr5": {"days": 140, "won": 0, "expected_wins": 0.08, "rate": 0.0,
                                   "predicted": 0.0005, "legs": 5},
              "early10": {"days": 105, "won": 1, "expected_wins": 1.2, "rate": 0.01, "predicted": 0.012,
                          "legs": 10, "leg_rate": 0.66, "leg_projected": 0.67}}
    replay = {k: {"legs": SPEC[k]["legs"], **v} for k, v in replay.items()}
    plays = parlays.choose(games, evidence=replay)
    html = _render_parlays_section({"parlays": plays, "parlay_replay": replay})
    assert "Five Hits" in html and "Early Ten" in html and "Long shots" in html
    assert "Five Homers" in html and f"below {parlays.CORE_MIN_CASHED}" in html   # in the replay table
    assert "cashed <b>25</b> of 140 days" in html
    assert "2+ HITS" in html or "1+ HIT" in html
    assert _render_parlays_section({"parlays": []}) == ""

def _cand(pk, bid, prob, pa=400):
    return {"game_pk": pk, "batter_id": bid, "batter": f"B{bid}", "prob": prob, "season_pa": pa}


def test_homer_five_pick_tickets_take_his_likeliest_legs():
    hr = [_cand(g, 100 + g, 0.10 + g / 100) for g in range(1, 8)] + [_cand(7, 999, 0.30, pa=20)]
    hits = [_cand(g, 200 + g, 0.60 + g / 100) for g in range(1, 8)]
    plays = {p["key"]: p for p in homer.build_parlays(hr, hits, plays=homer.FIVE_PICK_PLAYS)}
    assert set(plays) == {"hr5", "hit5"}
    assert [l["batter_id"] for l in plays["hr5"]["leg_list"]] == [107, 106, 105, 104, 103]
    assert all(l["type"] == "hr" for l in plays["hr5"]["leg_list"])   # 999 is under the PA floor
    assert [l["batter_id"] for l in plays["hit5"]["leg_list"]] == [207, 206, 205, 204, 203]
    # HOMER'S PLAYS stay what they were: no home runs, three or four legs.
    assert all(p["hr"] == 0 and 3 <= p["legs"] <= 4 for p in homer.PARLAY_PLAYS)
    keys = [p["key"] for p in homer.PARLAY_PLAYS + homer.FIVE_PICK_PLAYS]
    assert len(keys) == len(set(keys))


def test_homer_grades_the_five_picks_on_their_own_tally():
    hr = [_cand(g, 100 + g, 0.2) for g in range(1, 6)]
    hits = [_cand(g, 200 + g, 0.7) for g in range(1, 6)]
    five = homer.build_parlays(hr, hits, plays=homer.FIVE_PICK_PLAYS)
    card = {"hr_picks": [], "hit_picks": [], "game_picks": [], "parlays": [], "five_picks": five}
    final = {"state": "Final", "detailed": "Final"}
    results = {"games": {g: final for g in range(1, 6)}, "fetched_at": "t",
               "batters": {**{100 + g: {"hr": 1, "hits": 1, "game_pk": g} for g in range(1, 6)},
                           **{200 + g: {"hr": 0, "hits": 1 if g < 5 else 0, "game_pk": g}
                              for g in range(1, 6)}}}
    homer.grade_card(card, results)
    status = {p["key"]: p["status"] for p in card["five_picks"]}
    assert status == {"hr5": "won", "hit5": "lost"}
    assert card["record"]["five_picks"] == {"won": 1, "lost": 1, "void": 0, "pending": 0}
    assert card["record"]["parlays"]["won"] == 0 and card["record"]["complete"]


def test_homer_page_shows_his_five_picks():
    from mlb_hr.render_review import _five_picks

    hr = [_cand(g, 100 + g, 0.2) for g in range(1, 6)]
    hits = [_cand(g, 200 + g, 0.7) for g in range(1, 6)]
    five = homer.build_parlays(hr, hits, plays=homer.FIVE_PICK_PLAYS)
    five[0]["evidence"] = {"days": 120, "won": 0, "expected_wins": 0.04}
    html = _five_picks({"five_picks": five})
    assert "HOMER's 5-Pick Parlays" in html and "Five-Homer Parlay" in html
    assert "1 in 3,125" in html                          # 0.2 ** 5
    assert "cashed 0 of 120 days" in html
    assert _five_picks({}) == ""
