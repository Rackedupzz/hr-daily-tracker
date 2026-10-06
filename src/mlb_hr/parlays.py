"""The house parlays: the model's own best judgment, built the one way that
maximizes the chance the ticket cashes -- every game's single most likely leg,
then the best few of those. The rules are the NFL tracker's (nfl_tracker.parlays),
carried over market for market.

One leg per game keeps the legs independent (players in the same game rise and
fall together, and books reprice same-game parlays to take that back), which is
also what makes the ticket's chance the product of its legs. A hitter carrying
an injury note is passed over: a scratched leg voids.

The core tickets have five legs each, one per game: Five Hits (1+ hit), Five
Multi-Hit (2+ hits), Five Homers, Five Winners and Chalk Five (the five
likeliest legs of any of those markets, not posted on a day it is another core
ticket leg for leg). A core ticket is offered only if its
rule cashed CORE_MIN_CASHED or more times in the season replay (see qualifies);
the others stay in the replay table with their record. (The NFL board's Best of
the Board takes one leg from each of five markets; baseball has four the replay
can score, so it has no board.)

The long shots, Early Ten and Late Ten, take ten legs from one first-pitch
window only -- games starting before 7 PM Eastern, or at 7 PM and after -- so a
leg in one window can never sink a ticket in the other: one home run, one hits
leg, then the likeliest of the rest (hits at the highest rung the model trusts,
and favourites to win), at most two legs a game, the most probable ten that
pays +1000 or more at fair odds. When that rule cannot be met (a window short
of games or of strong legs) the ticket falls back to a looser shape (see
ten_ticket) and says so on its card.

There is no odds feed for baseball props, so every leg counts at the model's
own number and every ticket at fair odds. The season replay (mlb_hr.replay,
data/model_fit.json "parlays") rebuilds each ticket with these rules on every
replayed day and records how often it actually cashed. Like the picks, a
parlay is locked once any of its games is under way and is rebuilt from the
games still to start until then. A day whose games had all started before a
parlay was ever posted is built from that morning's projections, flagged as
backfilled, and graded all the same.

Stdlib only: the live page (api/live.py) grades tickets with no numpy.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

PARLAYS = [
    {"key": "hit5", "kind": "hit", "legs": 5, "name": "Five Hits",
     "blurb": "Five bats to get a hit, one per game: the model's likeliest hit legs on the slate."},
    {"key": "multi5", "kind": "hit2", "legs": 5, "name": "Five Multi-Hit",
     "blurb": "Five bats to get two or more hits, one per game."},
    {"key": "hr5", "kind": "hr", "legs": 5, "name": "Five Homers",
     "blurb": "The five likeliest home run bats on the slate, one per game. Every leg has to leave "
              "the yard, so this one is priced like the lottery ticket it is."},
    {"key": "win5", "kind": "win", "legs": 5, "name": "Five Winners",
     "blurb": "The five most lopsided games on the slate, straight up."},
    {"key": "chalk5", "kind": "mix", "legs": 5, "name": "Chalk Five",
     "blurb": "The five likeliest legs on the whole board, from any of those markets, one per game."},
    {"key": "early10", "kind": "ladder", "window": "early", "legs": 10, "name": "Early Ten (before 7 PM ET)",
     "longshot": True,
     "blurb": "Ten legs from the games starting before 7 PM Eastern only: a home run, hits at the "
              "highest rung the model trusts, and favourites to win. The most probable ten that pays "
              "+1000 or more at fair odds; no player with an injury note."},
    {"key": "late10", "kind": "ladder", "window": "late", "legs": 10, "name": "Late Ten (7 PM ET & later)",
     "longshot": True,
     "blurb": "Ten legs from the games starting at 7 PM Eastern or later, built the same way. "
              "Nothing on it shares a game with the early ticket."},
]
# A core ticket is offered only with at least CORE_MIN_LEGS legs and a rule that
# cashed at least CORE_MIN_CASHED times in the walk-forward season replay. The
# long shots (the Tens) are shown apart, with their own record, and are exempt.
# The NFL bar is 30 cashes; baseball legs are longer odds (no MLB five-leg rule
# reached 30 in the 140-day 2026 replay, Five Hits 25), so the bar is 20.
CORE_MIN_LEGS = 5
CORE_MIN_CASHED = 20
MIX_KINDS = ("hit", "hit2", "hr", "win")
# Core player markets: (leg type, line).
CORE_MARKETS = {"hit": ("hits", 1), "hit2": ("hits", 2), "hr": ("hr", None)}

# Milestone lines, as books post them. A hitter's leg is the highest rung the
# model gives at least the floor (see pick_rung).
LADDERS = {"hits": (1, 2, 3)}
LEG_FLOOR = 0.75
# The Tens are priced to pay at least +1000 at fair odds: the most probable ticket
# whose chance is at most TEN_TARGET_PROB, found by lowering the rung floor.
TEN_MIN_ODDS = 1000
TEN_TARGET_PROB = 100.0 / (100.0 + TEN_MIN_ODDS)
TEN_FLOORS = tuple(round(0.85 - 0.01 * i, 2) for i in range(36))
TEN_REQUIRED = ("hr", "hits")
TEN_MAX_PER_GAME = 2
# Beyond the one required home run, every leg must be at least this likely.
TEN_FILL_MIN = 0.70
# When the strict ticket cannot be built, ten_ticket loosens it to this.
TEN_RELAXED = {"max_per_game": 3, "fill_min": 0.60, "strict": False}
# The windows split at 7 PM Eastern first pitch.
WINDOW_SPLIT_HOUR_ET = 19


def fair_odds(prob: float) -> str:
    """American odds at which a ticket of this probability breaks even."""
    if prob <= 0 or prob >= 1:
        return "&mdash;"
    if prob >= 0.5:
        return f"-{round(100 * prob / (1 - prob)):,}"
    return f"+{round(100 * (1 - prob) / prob):,}"


def leg_label(leg_type: str, line=None) -> str:
    if leg_type == "hr":
        return "home run"
    if leg_type == "win":
        return "to win"
    n = int(line or 1)
    return f"{n}+ hit" + ("s" if n != 1 else "")


def qualifies(spec: dict, evidence: Optional[dict]) -> bool:
    """Whether a core ticket has earned its place: enough legs, and a rule that
    cashed CORE_MIN_CASHED or more times in the replay."""
    if spec.get("longshot"):
        return True
    return (spec["legs"] >= CORE_MIN_LEGS and bool(evidence)
            and (evidence.get("won") or 0) >= CORE_MIN_CASHED)


# ------------------------------------------------------------------ windows
def _eastern(utc: datetime) -> datetime:
    try:
        from zoneinfo import ZoneInfo
        return utc.astimezone(ZoneInfo("America/New_York"))
    except Exception:  # noqa: BLE001 - no tz database: the season is on daylight time
        return utc.astimezone(timezone(timedelta(hours=-4)))


def window_of(start) -> Optional[str]:
    """'early' for a first pitch before 7 PM Eastern, 'late' for 7 PM and after;
    None when the start time is unknown."""
    try:
        utc = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    if utc.tzinfo is None:
        utc = utc.replace(tzinfo=timezone.utc)
    return "early" if _eastern(utc).hour < WINDOW_SPLIT_HOUR_ET else "late"


def _window(game: dict) -> Optional[str]:
    return game.get("window") or window_of(game.get("start"))


# -------------------------------------------------------------- probabilities
def _binom_tail(n: float, r: float, k: int) -> float:
    """P(at least k successes) in n (possibly fractional) trials at rate r."""
    if k <= 0:
        return 1.0
    below = sum(math.exp(math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1))
                * r ** j * (1 - r) ** (n - j) for j in range(k) if j <= n)
    return min(max(1.0 - below, 0.0), 1.0)


def hit_prob(hitter: dict, k: int = 1) -> Optional[float]:
    """The model's chance of k+ hits. The served 1+ chance comes from a per-PA
    rate over the expected plate appearances (pitching.project_hits); the same
    rate and exposure price the higher rungs, so every rung agrees with it."""
    proj = hitter.get("hits_proj") or {}
    p1 = proj.get("prob_at_least_one")
    if p1 is None:
        return None
    if k <= 1:
        return float(p1)
    exposure = min(max(float(proj.get("expected_pa") or 4.1), 1.5), 5.2)
    rate = 1.0 - (1.0 - min(float(p1), 0.999999)) ** (1.0 / exposure)
    return min(_binom_tail(exposure, rate, k), float(p1))


def leg_prob(hitter: dict, kind: str) -> Optional[float]:
    """The model's probability for a core leg: a home run, or k+ hits."""
    if kind == "hr":
        return hitter.get("prob_hr")
    return hit_prob(hitter, CORE_MARKETS[kind][1])


def _opposing_starter(game: dict, hitter: dict) -> str:
    return (game.get("away_sp") if hitter.get("side") == "home" else game.get("home_sp")) or ""


def _pid(leg: dict):
    return leg.get("batter_id") if leg["type"] != "win" else f"team:{leg['team']}"


def _hitter_leg(game: dict, h: dict, leg_type: str, line=None) -> dict:
    return {"type": leg_type, "line": line, "label": leg_label(leg_type, line),
            "batter": h["batter"], "batter_id": h.get("batter_id"),
            "team": h.get("team", ""), "game_pk": game.get("game_pk"),
            "opp_sp": _opposing_starter(game, h), "facing_hand": h.get("facing_hand"),
            "lineup_slot": h.get("lineup_slot")}


def _win_leg(game: dict) -> Optional[tuple]:
    """(prob, leg) for the game's favourite to win, or None without a projection."""
    proj = game.get("projection") or {}
    fav, p = proj.get("favorite"), proj.get("favorite_prob")
    if not fav or p is None:
        return None
    home = fav == (proj.get("home_team") or game.get("home"))
    opp = proj.get("underdog") or (game.get("away") if home else game.get("home"))
    return float(p), {"type": "win", "line": None, "label": "to win", "batter": fav,
                      "batter_id": None, "team": fav, "side": "home" if home else "away",
                      "opp": opp or "", "game_pk": game.get("game_pk")}


def _eligible(h: dict) -> bool:
    return not h.get("injury_note")


# -------------------------------------------------------------- core tickets
def best_legs(games: list[dict], kind: str, closed: Iterable = ()) -> list:
    """Each open game's single likeliest leg of this kind, as (prob, leg)."""
    closed = set(closed)
    best = []
    for game in games:
        if game.get("game_pk") in closed:
            continue
        if kind == "win":
            w = _win_leg(game)
            if w:
                best.append(w)
            continue
        leg_type, line = CORE_MARKETS[kind]
        top = None
        for h in game.get("hitters", []):
            if not _eligible(h):
                continue
            p = leg_prob(h, kind)
            if p is None or (top is not None and p <= top[0]):
                continue
            top = (float(p), _hitter_leg(game, h, leg_type, line))
        if top:
            best.append(top)
    return best


# ----------------------------------------------------------------- the Tens
def rung_table(hitter: dict) -> list:
    """Every hits line a hitter can be set at, as (prob, line)."""
    out = []
    for line in LADDERS["hits"]:
        p = hit_prob(hitter, line)
        if p is not None:
            out.append((p, line))
    return out


def pick_rung(table: list, floor: float = LEG_FLOOR):
    """(prob, line) at the highest line with prob >= floor, or None."""
    best = None
    for p, line in table:
        if p >= floor and (best is None or line > best[1]):
            best = (p, line)
    return best


def ladder_table(games: list[dict]) -> tuple:
    """Everything a Ten can be built from, priced once: (fixed legs, rung
    tables). Fixed legs are every home run and every favourite, as (prob, leg);
    rung tables hold each hitter's hits lines, so a floor only has to pick from
    them (ladder_pick)."""
    fixed, rungs = [], []
    for game in games:
        w = _win_leg(game)
        if w:
            fixed.append(w)
        for h in game.get("hitters", []):
            if not _eligible(h):
                continue
            if h.get("prob_hr") is not None:
                fixed.append((float(h["prob_hr"]), _hitter_leg(game, h, "hr")))
            table = rung_table(h)
            if table:
                rungs.append((_hitter_leg(game, h, "hits"), table))
    return fixed, rungs


def ladder_pick(tables: tuple, floor: float = LEG_FLOOR) -> list:
    """The candidates at one rung floor: every fixed leg, and each hitter's
    highest hits line the floor allows, as (prob, leg)."""
    fixed, rungs = tables
    out = list(fixed)
    for base, table in rungs:
        rung = pick_rung(table, floor)
        if rung:
            p, line = rung
            out.append((p, {**base, "line": line, "label": leg_label("hits", line)}))
    return out


def ten_legs(cands: list, legs: int = 10, required: tuple = TEN_REQUIRED,
             max_per_game: int = TEN_MAX_PER_GAME, fill_min: float = TEN_FILL_MIN,
             strict: bool = True) -> list:
    """The likeliest required leg of each market, then the likeliest of the
    rest: one leg a player, at most `max_per_game` a game. Not `strict`, a
    required market the candidates do not offer at all is skipped."""
    pool = sorted(cands, key=lambda t: t[0], reverse=True)
    if not strict:
        offered = {leg["type"] for _, leg in pool}
        required = tuple(k for k in required if k in offered)
    out, players, per_game = [], set(), {}

    def take(p, leg):
        out.append((p, leg))
        players.add(_pid(leg))
        per_game[leg["game_pk"]] = per_game.get(leg["game_pk"], 0) + 1

    def open_(leg):
        return _pid(leg) not in players and per_game.get(leg["game_pk"], 0) < max_per_game

    for kind in required:
        for p, leg in pool:
            if leg["type"] == kind and open_(leg):
                take(p, leg)
                break
    if {leg["type"] for _, leg in out} < set(required):
        return []
    for p, leg in pool:
        if len(out) >= legs:
            break
        if p >= fill_min and leg["type"] != "hr" and open_(leg):
            take(p, leg)
    return out if len(out) >= legs else []


def ten_at_odds(candidates, legs: int = 10, target: float = TEN_TARGET_PROB, **shape) -> list:
    """The most probable ten that pays +1000 or more at fair odds:
    `candidates(floor)` gives the legs at that rung floor, and the floor is
    lowered -- every line raised a rung where it costs least -- until the
    ticket reaches the price. `shape` goes to ten_legs."""
    best, best_p = [], 0.0
    for floor in TEN_FLOORS:
        chosen = ten_legs(candidates(floor), legs, **shape)
        p = math.prod(q for q, _ in chosen) if chosen else 0.0
        if chosen and p <= target + 1e-12 and p > best_p:
            best, best_p = chosen, p
    return best


def ten_ticket(games: list[dict], legs: int = 10) -> tuple:
    """(legs, note) for a Ten from these games: the full rule, else the shape
    loosened (TEN_RELAXED: only the required markets the window offers, up to
    three legs a game, fill legs from 60%). The note says when it was loosened."""
    tables = ladder_table(games)
    tries = [({}, ""),
             (TEN_RELAXED, "Built with up to three legs a game: the window is short of legs that "
                           "meet the full rule.")]
    for shape, note in tries:
        chosen = ten_at_odds(lambda f: ladder_pick(tables, f), legs, **shape)
        if chosen:
            return chosen, note
    return [], ""


# ------------------------------------------------------------------ building
def build(games: list[dict], spec: dict, closed: Iterable = ()) -> Optional[dict]:
    """One parlay from the games not in `closed`, or None if it cannot be built."""
    closed = set(closed)
    kind, legs = spec["kind"], spec["legs"]
    if kind == "ladder":
        window = [g for g in games if _window(g) == spec["window"] and g.get("game_pk") not in closed]
        chosen, note = ten_ticket(window, legs)
        if not chosen:
            return None
        chosen.sort(key=lambda t: (str(t[1]["game_pk"]), -t[0]))
        play = _ticket(spec, chosen)
        if note:
            play["note"] = note
        return play
    if kind == "mix":
        per_game: dict = {}
        for k in MIX_KINDS:
            for p, leg in best_legs(games, k, closed):
                if p > per_game.get(leg["game_pk"], (-1.0, None))[0]:
                    per_game[leg["game_pk"]] = (p, leg)
        chosen = sorted(per_game.values(), key=lambda t: t[0], reverse=True)[:legs]
    else:
        chosen = sorted(best_legs(games, kind, closed), key=lambda t: t[0], reverse=True)[:legs]
    if len(chosen) < legs:
        return None
    return _ticket(spec, chosen)


def _ticket(spec: dict, chosen: list) -> dict:
    leg_list = [{**leg, "prob": float(p)} for p, leg in chosen]
    prob = math.prod(leg["prob"] for leg in leg_list)
    return {"key": spec["key"], "kind": spec["kind"], "name": spec["name"], "blurb": spec["blurb"],
            "legs": spec["legs"], "leg_list": leg_list, "prob": prob, "fair_odds": fair_odds(prob),
            "one_in": round(1 / prob) if prob > 0 else None}


def _legs_of(play: dict) -> frozenset:
    return frozenset((leg["type"], leg.get("line"), leg["game_pk"], _pid(leg)) for leg in play["leg_list"])


def choose(games: list[dict], previous: Optional[list] = None, started: Iterable = (),
           evidence: Optional[dict] = None) -> list[dict]:
    """Every parlay, locked once under way (see the module docstring)."""
    started = set(started)
    prior = {p["key"]: p for p in previous or []}
    all_games = {g.get("game_pk") for g in games}
    out, posted = [], set()
    for spec in PARLAYS:
        old = prior.get(spec["key"])
        ok = qualifies(spec, (evidence or {}).get(spec["key"]))
        if old and any(leg["game_pk"] in started for leg in old["leg_list"]):
            if not ok:
                # Posted before its rule fell below the bar: it settles apart.
                out.append({**old, "retired": True})
                continue
            play = old
        elif not ok:
            continue
        elif all_games and all_games <= started:
            # Every game is under way and none was posted in time: built from
            # the morning's projections, and labelled as such.
            play = build(games, spec)
            if play:
                play["backfilled"] = True
        else:
            play = build(games, spec, closed=started)
            window = {g.get("game_pk") for g in games if _window(g) == spec.get("window")}
            if not play and spec.get("longshot") and window and window <= started:
                # Its whole window started before a Ten was posted: built from
                # the morning's projections, so the day still has both.
                play = build(games, spec)
                if play:
                    play["backfilled"] = True
        if play and play is not old and not spec.get("longshot") and _legs_of(play) in posted:
            # Chalk Five is usually Five Hits leg for leg (a 70% hit bat beats
            # a 60% favourite in most games): one ticket is not posted twice.
            continue
        if play:
            if not spec.get("longshot"):
                posted.add(_legs_of(play))
            play["evidence"] = (evidence or {}).get(spec["key"])
            play["longshot"] = bool(spec.get("longshot"))
            out.append(play)
    # A ticket posted under an earlier rule and already under way stays on the
    # board until it settles.
    keys = {spec["key"] for spec in PARLAYS}
    for key, old in prior.items():
        if key not in keys and any(leg["game_pk"] in started for leg in old["leg_list"]):
            out.append({**old, "retired": True})
    return out


# ------------------------------------------------------------------- grading
def leg_need(leg: dict) -> tuple:
    """(stat, at least) a hitter leg needs; tickets posted before the hits
    ladder carry type 'hit', which is 1+ hit."""
    if leg["type"] == "hr":
        return "hr", 1
    return "hits", int(leg.get("line") or 1)


def grade(parlays: list[dict], line_for, did_not_play, game_for=None) -> None:
    """Each leg won / lost / void / pending, and the ticket's status, in place.

    `line_for(leg)` returns his line in that game (or None), `did_not_play(leg)`
    a settled no-result once his game is final, and `game_for(leg)` the live
    game entry (state, home_score, away_score) for a winner leg. A hitter leg is
    won the moment it is satisfied and lost only once his game is over; a leg
    whose player never bats is void and the ticket rides on the rest, as at a
    sportsbook.
    """
    for play in parlays or []:
        statuses = []
        for leg in play["leg_list"]:
            if leg["type"] == "win":
                game = (game_for(leg) if game_for else None) or {}
                if game.get("state") != "Final":
                    status = "pending"
                elif game.get("home_score") == game.get("away_score"):
                    status = "void"
                else:
                    home_won = (game.get("home_score") or 0) > (game.get("away_score") or 0)
                    status = "won" if home_won == (leg.get("side") == "home") else "lost"
                    leg["result"] = {"summary": f"{game.get('away_score')}-{game.get('home_score')} final",
                                     "final": True}
            else:
                line = line_for(leg) or did_not_play(leg)
                key, need = leg_need(leg)
                if not line:
                    status = "pending"
                elif line.get("dnp"):
                    status = "void"
                elif (line.get(key) or 0) >= need:
                    status = "won"
                elif line.get("final"):
                    status = "lost"
                else:
                    status = "pending"
                if line:
                    leg["result"] = {k: line.get(k) for k in ("pa", "hits", "hr", "summary", "final")}
            leg["status"] = status
            statuses.append(status)
        if "lost" in statuses:
            play["status"] = "lost"
        elif "pending" in statuses:
            play["status"] = "pending"
        elif "won" in statuses:
            play["status"] = "won"
        else:
            play["status"] = "void"
