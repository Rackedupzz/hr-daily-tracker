"""Scoring a slate against what actually happened.

The model publishes six names in the morning; by midnight the answer is a
matter of record. This module pulls the day's box scores and attaches the
outcome to every projection the slate made, so the page shows hits and misses
instead of only forecasts.

It is deliberately separate from the slate build. Building a slate refits the
ensemble over the season and takes minutes; scoring it is two cheap API calls
and can therefore run on every page refresh, which is what makes the numbers
live while games are in progress.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from mlb_hr.fetch import boxscore_batting, live_games
from mlb_hr.parlays import grade as grade_parlays

# A game whose state is not one of these has not started.
_STARTED = ("Live", "Final")


def fetch_results(day: date, game_pks: Optional[list] = None) -> dict:
    """The day's game states and batting lines.

    Returns an empty-but-shaped dict when nothing has started, so callers can
    attach it unconditionally rather than branching on None.
    """
    games = live_games(day)
    if game_pks:
        games = {pk: g for pk, g in games.items() if pk in set(game_pks)}

    started = [pk for pk, g in games.items() if g.get("state") in _STARTED]
    batters = boxscore_batting(started) if started else {}

    return {
        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "games": games,
        "batters": batters,
        "final_games": sum(1 for g in games.values() if g.get("state") == "Final"),
        "live_games": sum(1 for g in games.values() if g.get("state") == "Live"),
        "upcoming_games": sum(
            1 for g in games.values() if g.get("state") not in _STARTED
        ),
    }


def batter_line(batters: dict, batter_id, game_pk=None) -> Optional[dict]:
    """A batter's line in one game.

    In a doubleheader the box-score line is both games combined, with each
    game's own line under `by_game`; a projection is for one game, so it gets
    that game's line, and none at all if he batted only in the other game.
    """
    if batter_id is None:
        return None
    line = batters.get(batter_id)
    if line is None:
        line = batters.get(str(batter_id))
    if not line or game_pk is None:
        return line or None
    by_game = line.get("by_game") or {}
    if by_game:
        game_line = by_game.get(game_pk)
        if game_line is None:
            game_line = by_game.get(str(game_pk))
        if game_line is None:
            return None
        return {**line, **game_line, "game_pk": game_pk}
    if line.get("game_pk") is not None and str(line["game_pk"]) != str(game_pk):
        return None
    return line


def _line_for(results: dict, hitter: dict) -> Optional[dict]:
    """The batting line for one projected hitter, if his game has started."""
    line = batter_line(results.get("batters", {}), hitter.get("batter_id"),
                       hitter.get("game_pk"))
    if not line:
        return None
    game = results.get("games", {}).get(line.get("game_pk"), {})
    return {
        "pa": line.get("pa", 0),
        "ab": line.get("ab", 0),
        "hits": line.get("hits", 0),
        "hr": line.get("hr", 0),
        "rbi": line.get("rbi", 0),
        "summary": line.get("summary", ""),
        "state": game.get("state", ""),
        "detailed": game.get("detailed", ""),
        "final": game.get("state") == "Final",
    }


def _did_not_play(results: dict, hitter: dict) -> Optional[dict]:
    """A settled no-result: his game is over and he never came to the plate.

    Covers rest days, late scratches and postponements. Without this such a
    pick stays "pending" forever and the day never counts as complete.
    """
    game = results.get("games", {}).get(hitter.get("game_pk"))
    if not game or game.get("state") != "Final":
        return None
    detailed = game.get("detailed") or ""
    return {
        "pa": 0, "ab": 0, "hits": 0, "hr": 0, "rbi": 0,
        "summary": "Did not play",
        "state": "Final",
        "detailed": detailed if detailed and detailed != "Final" else "Did not play",
        "final": True,
        "dnp": True,
    }


def attach_results(slate_data: dict, results: dict) -> dict:
    """Annotate a slate in place with the outcomes that are known so far.

    Every hitter on the slate gets a `result` when his game has started, so the
    model page can score each prior's own top six, not just the served one.
    Picks additionally roll up into a hit/miss summary.
    """
    if not results or not results.get("games"):
        slate_data["results"] = None
        return slate_data

    for game in slate_data.get("games", []):
        game["live"] = results["games"].get(game.get("game_pk"))
        for hitter in game.get("hitters", []):
            line = _line_for(results, hitter)
            if line:
                hitter["result"] = line

    # Picks and hit picks are separate dicts from the hitters inside the games,
    # so they are scored directly rather than by reference.
    def score(entries: list, key: str) -> dict:
        hit = scored = dnp = 0
        for entry in entries:
            line = _line_for(results, entry) or _did_not_play(results, entry)
            if not line:
                continue
            entry["result"] = line
            if line.get("dnp"):
                dnp += 1
            elif line["final"]:
                scored += 1
                hit += 1 if line[key] > 0 else 0
        # Anything not yet final is still open, whether the game is underway or
        # has not started; a pick is not wrong until his last at-bat. A pick
        # who never batted is neither a hit nor a miss.
        return {"hit": hit, "scored": scored, "dnp": dnp,
                "pending": len(entries) - scored - dnp}

    picks = score(slate_data.get("picks_6") or [], "hr")
    hits = score(slate_data.get("hit_picks") or [], "hits")
    grade_parlays(slate_data.get("parlays") or [],
                  lambda leg: _line_for(results, leg),
                  lambda leg: _did_not_play(results, leg),
                  lambda leg: results["games"].get(leg.get("game_pk")))

    slate_data["results"] = {
        "fetched_at": results["fetched_at"],
        "final_games": results["final_games"],
        "live_games": results["live_games"],
        "upcoming_games": results["upcoming_games"],
        "picks": picks,
        "hit_picks": hits,
        "parlays": {p["key"]: p.get("status", "pending") for p in slate_data.get("parlays") or []},
        # Anything at all to show? Before first pitch there is not.
        "any": bool(results["batters"]),
    }
    return slate_data


def variant_scoreboard(slate_data: dict, top6_for) -> dict:
    """How each model variant's own top six actually did.

    `top6_for` is a callable taking a variant key and returning that variant's
    six hitters, so this stays independent of how the page ranks them. Only
    finished games count toward hit and miss; a hitter still playing is
    pending, since a pick is not wrong until his last at-bat.
    """
    out: dict[str, dict] = {}
    for key, top in top6_for():
        hit = final = pending = 0
        for hitter in top:
            line = hitter.get("result")
            if not line:
                pending += 1
            elif line["final"]:
                final += 1
                hit += 1 if line["hr"] > 0 else 0
            else:
                pending += 1
        out[key] = {"hit": hit, "final": final, "pending": pending}
    return out
