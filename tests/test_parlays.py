"""The 5-pick parlays: five home runs and five hits, on the slate and on HOMER's page.

Run: python -m pytest tests/test_parlays.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlb_hr import homer, parlays  # noqa: E402


def _hitter(bid, team, prob_hr, p1, side="home", injury=None):
    return {"batter": f"B{bid}", "batter_id": bid, "team": team, "side": side,
            "facing_hand": "R", "lineup_slot": 3, "prob_hr": prob_hr, "injury_note": injury,
            "hits_proj": {"prob_at_least_one": p1, "projected_hits": 1.0,
                          "projected_hits_raw": 1.0}}


def _games(n: int) -> list[dict]:
    """Game i's best HR bat is 0.12 + i/100 (the away hitter), its best hit bat 0.66 + i/100."""
    games = []
    for i in range(n):
        pk = 100 + i
        games.append({"game_pk": pk, "home_sp": f"H{i}", "away_sp": f"A{i}", "venue": f"Park {i}",
                      "home": f"Home {i}", "away": f"Away {i}", "hitters": [
                          _hitter(pk * 10 + 1, f"Home {i}", 0.10 + i / 100, 0.66 + i / 100),
                          _hitter(pk * 10 + 2, f"Away {i}", 0.12 + i / 100, 0.60 + i / 100,
                                  side="away")]})
    return games


def test_each_ticket_is_the_best_leg_of_the_five_best_games():
    games = _games(7)
    hr = parlays.build(games, parlays.PARLAYS[0])
    hit = parlays.build(games, parlays.PARLAYS[1])
    for play, kind in ((hr, "hr"), (hit, "hit")):
        pks = [leg["game_pk"] for leg in play["leg_list"]]
        assert len(pks) == len(set(pks)) == 5
        assert all(leg["type"] == kind for leg in play["leg_list"])
        assert math.isclose(play["prob"], math.prod(leg["prob"] for leg in play["leg_list"]))
    # Games 6..2 have the best legs; the away bat for home runs, the home bat for hits.
    assert [leg["batter_id"] for leg in hr["leg_list"]] == [1062, 1052, 1042, 1032, 1022]
    assert [leg["batter_id"] for leg in hit["leg_list"]] == [1061, 1051, 1041, 1031, 1021]
    assert hr["leg_list"][0]["opp_sp"] == "H6"          # an away hitter faces the home starter
    assert hr["fair_odds"].startswith("+") and "," in hr["fair_odds"]
    assert hr["one_in"] == round(1 / hr["prob"])


def test_injured_bats_are_passed_over_and_short_slates_post_nothing():
    games = _games(6)
    games[5]["hitters"][1]["injury_note"] = "Day-To-Day"
    hr = parlays.build(games, parlays.PARLAYS[0])
    assert 1052 not in [leg["batter_id"] for leg in hr["leg_list"]]
    assert 1051 in [leg["batter_id"] for leg in hr["leg_list"]]   # his teammate takes the game
    assert parlays.build(_games(4), parlays.PARLAYS[0]) is None
    assert parlays.choose(_games(4)) == []


def test_a_ticket_locks_once_any_leg_starts_and_backfills_a_finished_day():
    games = _games(8)
    posted = parlays.choose(games)
    first_leg_game = posted[0]["leg_list"][0]["game_pk"]
    games[7]["hitters"][1]["prob_hr"] = 0.9       # a new favourite after lineups post
    # A leg's game has started: the ticket stands exactly as posted.
    kept = parlays.choose(games, previous=posted, started={first_leg_game})
    assert kept[0] is posted[0]
    # Only an off-ticket game has started: rebuilt from the games still to come.
    off_ticket = next(g["game_pk"] for g in games
                      if g["game_pk"] not in {l["game_pk"] for p in posted for l in p["leg_list"]})
    rebuilt = parlays.choose(games, previous=posted, started={off_ticket})
    assert rebuilt[0]["leg_list"][0]["batter_id"] == 1072
    assert off_ticket not in [leg["game_pk"] for leg in rebuilt[0]["leg_list"]]
    # Every game under way and nothing was posted: built from the morning, flagged.
    late = parlays.choose(games, previous=None, started={g["game_pk"] for g in games},
                          evidence={"hr5": {"days": 140, "won": 0}})
    assert len(late) == 2 and all(p["backfilled"] for p in late)
    assert late[0]["evidence"] == {"days": 140, "won": 0}


def test_grading_follows_sportsbook_rules():
    play = parlays.build(_games(5), parlays.PARLAYS[1])
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


def test_slate_tracker_records_both_tickets():
    from mlb_hr.snapshot import summary_row

    games = _games(6)
    slate = {"date": "2026-09-27", "games": games, "games_count": 6, "picks_6": [],
             "hit_picks": [], "parlays": parlays.choose(games)}
    slate["parlays"][0]["status"] = "lost"
    row = summary_row(slate)
    assert row["hr_parlay"] == "lost" and row["hit_parlay"] == "pending"
    assert 0 < row["hr_parlay_prob"] < row["hit_parlay_prob"] < 1


def test_slate_page_shows_the_parlays():
    from mlb_hr.render import _render_parlays_section

    games = _games(6)
    plays = parlays.choose(games, evidence={"hr5": {"days": 140, "won": 0, "expected_wins": 0.03},
                                            "hit5": {"days": 140, "won": 30, "rate": 0.214,
                                                     "predicted": 0.221}})
    html = _render_parlays_section({"parlays": plays})
    assert "Five-Homer Parlay" in html and "Five-Hit Parlay" in html
    assert "1 in " in html                              # the home run ticket's chance
    assert "cashed 0 of 140 days" in html and "cashed 30 of 140 days" in html
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
