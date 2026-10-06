"""DraftKings' lines for the day's games, as ESPN carries them.

ESPN's core odds API lists every line DraftKings (ESPN provider 100) posts for a
game, each with its American price: the moneyline, and every hitter's Home Run
and Hits milestones (1+, 2+, 3+). The house parlays and HOMER's take every
hitter leg from these lines only (mlb_hr.parlays), the NFL tracker's rule.

Props are keyed by ESPN athlete id. They are matched to the slate's hitters by
folded name (espn.normalize_name) within the same game, using the two clubs'
ESPN rosters, so a name shared across clubs never crosses games.

Standard library only. Any failure leaves a game without a book, and the
parlays then fall back to the model's numbers, unpriced, as the replay does.
"""
from __future__ import annotations

import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Optional

from mlb_hr.espn import normalize_name

SITE = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb"
CORE = "https://sports.core.api.espn.com/v2/sports/baseball/leagues/mlb"
PROVIDER = "100"   # DraftKings, the book ESPN carries
# ESPN prop-bet type ids -> the parlay leg types they price.
PROP_TYPES = {"240": "hr", "234": "hits"}


def _get(url: str, timeout: int = 20, attempts: int = 2) -> Optional[dict]:
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except Exception:  # noqa: BLE001 - a missing book is not an error
            if i + 1 < attempts:
                time.sleep(1.0 + i)
    return None


def american(value) -> Optional[int]:
    """'-115', '+140', 'EVEN', -115.0 -> American odds as an int (None if absent)."""
    if value is None or value == "":
        return None
    text = str(value).strip().upper()
    if text in ("EVEN", "EV", "PK"):
        return 100
    try:
        v = int(round(float(text.replace("+", ""))))
    except ValueError:
        return None
    return v if abs(v) >= 100 else None


def _price(item: dict) -> Optional[int]:
    odds = item.get("odds") or {}
    got = american((odds.get("american") or {}).get("value"))
    if got is None:
        got = american(((item.get("current") or {}).get("american") or {}).get("value"))
    return got


def events(day: date) -> list[dict]:
    """ESPN's games on a date: id, first pitch, home and away club names."""
    data = _get(f"{SITE}/scoreboard?dates={day:%Y%m%d}&limit=50") or {}
    out = []
    for e in data.get("events", []):
        comp = (e.get("competitions") or [{}])[0]
        sides = {c.get("homeAway"): c.get("team") or {} for c in comp.get("competitors", [])}
        out.append({"id": str(e.get("id")), "start": e.get("date", ""),
                    "home": (sides.get("home") or {}).get("displayName", ""),
                    "away": (sides.get("away") or {}).get("displayName", ""),
                    "team_ids": [str(t.get("id")) for t in sides.values() if t.get("id")]})
    return out


def _roster_names(team_id: str) -> dict:
    """{ESPN athlete id: folded name} for one club's ESPN roster."""
    data = _get(f"{SITE}/teams/{team_id}/roster", timeout=25) or {}
    out = {}
    for group in data.get("athletes") or []:
        for ath in group.get("items") or []:
            if ath.get("id"):
                out[str(ath["id"])] = normalize_name(ath.get("displayName") or ath.get("fullName") or "")
    return out


def event_lines(event_id: str) -> Optional[dict]:
    """DraftKings' moneyline and every hitter's home run and hits lines for
    one game: {"home_ml", "away_ml", "props": {ESPN id: {leg type: [[line,
    price], ...]}}}. None when ESPN carries no DraftKings book for it."""
    base = f"{CORE}/events/{event_id}/competitions/{event_id}/odds"
    index = _get(f"{base}?lang=en&region=us") or {}
    game = next((it for it in index.get("items", [])
                 if str((it.get("provider") or {}).get("id")) == PROVIDER), None)
    out = {"provider": "DraftKings", "fetched": time.strftime("%Y-%m-%d %H:%M:%S"),
           "home_ml": american(((game or {}).get("homeTeamOdds") or {}).get("moneyLine")),
           "away_ml": american(((game or {}).get("awayTeamOdds") or {}).get("moneyLine")),
           "props": {}}
    page, pages = 1, 1
    while page <= pages:
        data = _get(f"{base}/{PROVIDER}/propBets?lang=en&region=us&limit=1000&page={page}",
                    timeout=30) or {}
        pages = int(data.get("pageCount") or 1)
        for item in data.get("items", []):
            kind = PROP_TYPES.get(str((item.get("type") or {}).get("id")))
            ref = (item.get("athlete") or {}).get("$ref", "")
            if not kind or "/athletes/" not in ref:
                continue
            athlete = ref.split("/athletes/")[1].split("?")[0]
            target = ((item.get("current") or {}).get("target") or {}).get("value")
            if target is None:
                continue
            lines = out["props"].setdefault(athlete, {}).setdefault(kind, [])
            if all(existing[0] != float(target) for existing in lines):
                lines.append([float(target), _price(item)])
        page += 1
    if game is None and not out["props"]:
        return None
    for kinds in out["props"].values():
        for lines in kinds.values():
            lines.sort(key=lambda x: x[0])
    return out


def attach(games: list[dict], day: date, workers: int = 6) -> int:
    """Put DraftKings' book on each slate game as game["book"], props keyed
    by the slate's MLB batter id. Returns how many games got one."""
    evs = events(day)
    by_teams: dict = {}
    for e in evs:
        by_teams.setdefault((e["home"], e["away"]), []).append(e)
    pairs = []
    for g in sorted(games, key=lambda g: (g.get("start") or "", g.get("game_pk") or 0)):
        cands = by_teams.get((g.get("home"), g.get("away"))) or []
        if cands:
            # A doubleheader lists the pair twice: take them in first-pitch order.
            pairs.append((g, cands.pop(0)))
    if not pairs:
        return 0
    team_ids = sorted({t for _, e in pairs for t in e["team_ids"]})
    with ThreadPoolExecutor(max_workers=min(workers, len(pairs))) as ex:
        books = list(ex.map(lambda p: event_lines(p[1]["id"]), pairs))
        rosters = dict(zip(team_ids, ex.map(_roster_names, team_ids)))
    n = 0
    for (g, e), book in zip(pairs, books):
        if not book:
            continue
        roster = {k: v for t in e["team_ids"] for k, v in rosters.get(t, {}).items()}
        ids = {normalize_name(h.get("batter") or ""): h.get("batter_id") for h in g.get("hitters", [])}
        props = {}
        for espn_id, kinds in book["props"].items():
            bid = ids.get(roster.get(espn_id, ""))
            if bid is not None:
                props[str(bid)] = kinds
        g["book"] = {**book, "props": props, "espn_event": e["id"],
                     "priced": sum(1 for kinds in props.values() for lines in kinds.values()
                                   for _, price in lines if price is not None)}
        n += 1
    return n
