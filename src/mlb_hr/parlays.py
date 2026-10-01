"""The house model's two 5-pick parlays: five home runs, and five hitters with a hit.

Each is the model's own best judgment, built the one way that maximizes the
chance the ticket cashes: every game's single most likely leg, then the five
best of those. One leg per game keeps the legs independent -- players in the
same game rise and fall together, and books reprice same-game parlays to take
that back -- which is also what makes the ticket's chance the product of its
legs. A hitter carrying an injury note is passed over: a scratched leg voids.

The two tickets are not the same kind of bet, and the page says so.
  * Five hits: the model's likeliest hit legs are ~73% each, so the ticket lands
    around one day in five.
  * Five home runs: even the day's five likeliest are ~20% each, so it lands
    about one day in five thousand. It is a lottery ticket and is shown as one,
    with its real odds.
The season replay (mlb_hr.replay, data/model_fit.json "parlays") rebuilds both
tickets with these rules on every day of the season and records how often each
actually cashed.

Like the six picks, a parlay is locked once any of its games is under way and is
rebuilt from the games still to start until then. A day whose games had all
started before a parlay was ever posted (a restart late in the evening, or the
day this feature arrived) is built from that morning's projections, flagged as
backfilled, and graded all the same.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional

LEGS = 5

PARLAYS = [
    {"key": "hr5", "kind": "hr", "name": "Five-Homer Parlay",
     "blurb": "The five likeliest home run bats on the slate, one per game. A true long shot: "
              "every leg has to leave the yard, so this one is priced like the lottery ticket it is."},
    {"key": "hit5", "kind": "hit", "name": "Five-Hit Parlay",
     "blurb": "The five likeliest bats to get a hit, one per game. The model's steadiest five-leg "
              "ticket &mdash; and still one that misses most days."},
]


def fair_odds(prob: float) -> str:
    """American odds at which a ticket of this probability breaks even."""
    if prob <= 0 or prob >= 1:
        return "&mdash;"
    if prob >= 0.5:
        return f"-{round(100 * prob / (1 - prob)):,}"
    return f"+{round(100 * (1 - prob) / prob):,}"


def leg_prob(hitter: dict, kind: str) -> Optional[float]:
    """The model's probability for one leg: a home run, or at least one hit."""
    if kind == "hr":
        return hitter.get("prob_hr")
    return (hitter.get("hits_proj") or {}).get("prob_at_least_one")


def _opposing_starter(game: dict, hitter: dict) -> str:
    return (game.get("away_sp") if hitter.get("side") == "home" else game.get("home_sp")) or ""


def build(games: list[dict], spec: dict, closed: Iterable = (), legs: int = LEGS) -> Optional[dict]:
    """One parlay from the games not in `closed`, or None if too few games.

    Maximizes the product of the legs under one-leg-per-game: each game's best
    leg, then the best `legs` of those.
    """
    closed = set(closed)
    kind = spec["kind"]
    best: dict = {}
    for game in games:
        pk = game.get("game_pk")
        if pk in closed:
            continue
        for h in game.get("hitters", []):
            p = leg_prob(h, kind)
            if p is None or h.get("injury_note"):
                continue
            if pk not in best or p > best[pk][0]:
                best[pk] = (p, h, game)
    chosen = sorted(best.values(), key=lambda t: t[0], reverse=True)[:legs]
    if len(chosen) < legs:
        return None
    leg_list = [{
        "type": kind, "batter": h["batter"], "batter_id": h.get("batter_id"),
        "team": h.get("team", ""), "game_pk": g.get("game_pk"),
        "opp_sp": _opposing_starter(g, h), "facing_hand": h.get("facing_hand"),
        "lineup_slot": h.get("lineup_slot"), "prob": float(p),
    } for p, h, g in chosen]
    prob = math.prod(leg["prob"] for leg in leg_list)
    return {"key": spec["key"], "kind": kind, "name": spec["name"], "blurb": spec["blurb"],
            "legs": legs, "leg_list": leg_list, "prob": prob, "fair_odds": fair_odds(prob),
            "one_in": round(1 / prob) if prob > 0 else None}


def choose(games: list[dict], previous: Optional[list] = None, started: Iterable = (),
           evidence: Optional[dict] = None) -> list[dict]:
    """Both parlays, locked once under way (see the module docstring)."""
    started = set(started)
    prior = {p["key"]: p for p in previous or []}
    all_games = {g.get("game_pk") for g in games}
    out = []
    for spec in PARLAYS:
        old = prior.get(spec["key"])
        if old and any(leg["game_pk"] in started for leg in old["leg_list"]):
            play = old
        elif all_games and all_games <= started:
            # Every game is under way and none was posted in time: built from
            # the morning's projections, and labelled as such.
            play = build(games, spec)
            if play:
                play["backfilled"] = True
        else:
            play = build(games, spec, closed=started)
        if play:
            play["evidence"] = (evidence or {}).get(spec["key"])
            out.append(play)
    return out


def grade(parlays: list[dict], line_for, did_not_play) -> None:
    """Each leg won / lost / void / pending, and the ticket's status, in place.

    `line_for(leg)` returns his line in that game (or None), and
    `did_not_play(leg)` a settled no-result once his game is final. A home run
    leg is won the moment he homers; any leg is lost only once his game is over.
    A leg whose player never bats is void and the ticket rides on the rest, as
    at a sportsbook.
    """
    for play in parlays or []:
        statuses = []
        for leg in play["leg_list"]:
            line = line_for(leg) or did_not_play(leg)
            key = "hr" if leg["type"] == "hr" else "hits"
            if not line:
                status = "pending"
            elif line.get("dnp"):
                status = "void"
            elif (line.get(key) or 0) > 0:
                status = "won"
            elif line.get("final"):
                status = "lost"
            else:
                status = "pending"
            leg["status"] = status
            if line:
                leg["result"] = {k: line.get(k) for k in ("pa", "hits", "hr", "summary", "final")}
            statuses.append(status)
        if "lost" in statuses:
            play["status"] = "lost"
        elif "pending" in statuses:
            play["status"] = "pending"
        elif "won" in statuses:
            play["status"] = "won"
        else:
            play["status"] = "void"
