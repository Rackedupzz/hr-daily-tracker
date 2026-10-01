"""MLB Stats API client.

The unit of extraction is the plate appearance, not the home run. Fetching only HRs
tells you the numerator but leaves the denominator to be guessed — the previous model
divided by team games played, which silently assumed every hitter started every game.
Pulling every PA gives measured exposure, real per-batter platoon splits, and the
zero-HR outcomes a calibration backtest needs.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import date, timedelta
from typing import Iterable, Iterator

BASE = "https://statsapi.mlb.com"
UA = {"User-Agent": "mlb-hr-model/1.0 (research)"}


def _get(url: str, retries: int = 4, timeout: int = 30) -> dict | None:
    delay = 1.0
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError,
                json.JSONDecodeError) as exc:
            if attempt == retries - 1:
                print(f"  FAILED {url}: {exc}", file=sys.stderr)
                return None
            time.sleep(delay)
            delay *= 2
    return None


# ── schedule ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class GameRef:
    game_pk: int
    date: str
    venue: str
    home: str
    away: str
    home_id: int
    away_id: int
    state: str


def schedule(start: date, end: date, game_type: str = "R") -> list[GameRef]:
    """Every scheduled game in [start, end]. Includes non-final games; filter on .state."""
    out: list[GameRef] = []
    # The API caps a single schedule request, so walk it in month-sized chunks.
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=30), end)
        url = (f"{BASE}/api/v1/schedule?sportId=1&gameType={game_type}"
               f"&startDate={cur:%Y-%m-%d}&endDate={chunk_end:%Y-%m-%d}")
        data = _get(url)
        for day in (data or {}).get("dates", []):
            for g in day.get("games", []):
                out.append(GameRef(
                    game_pk=g["gamePk"],
                    date=g.get("officialDate") or day["date"],
                    venue=g.get("venue", {}).get("name", ""),
                    home=g["teams"]["home"]["team"]["name"],
                    away=g["teams"]["away"]["team"]["name"],
                    home_id=g["teams"]["home"]["team"]["id"],
                    away_id=g["teams"]["away"]["team"]["id"],
                    state=g.get("status", {}).get("abstractGameState", ""),
                ))
        cur = chunk_end + timedelta(days=1)
    # A doubleheader can repeat a game_pk across chunk boundaries; de-dup.
    seen, uniq = set(), []
    for g in out:
        if g.game_pk not in seen:
            seen.add(g.game_pk)
            uniq.append(g)
    return sorted(uniq, key=lambda g: (g.date, g.game_pk))


# ── plate appearances ─────────────────────────────────────────────────────
PA_FIELDS = (
    "game_pk", "date", "venue", "batting_team", "pitching_team", "is_home",
    "batter_id", "batter", "bat_side", "pitcher_id", "pitcher", "pitch_hand",
    "inning", "pa_index", "event", "event_type", "is_hr",
    "launch_speed", "launch_angle", "distance", "pitch_type", "pitch_speed",
    # Added for the park / spray / weather / umpire / lineup models.
    "coord_x", "coord_y", "spray_angle", "pull_angle", "trajectory",
    "lineup_slot", "tto", "temp_f", "wind_mph", "wind_dir", "condition",
    "ump_hp",
)

# Home plate in the MLB hit-coordinate frame. Batted-ball location is published
# as a 2-D screen coordinate, so spray angle has to be derived from it.
_SPRAY_ORIGIN_X = 125.42
_SPRAY_ORIGIN_Y = 198.27


def _spray_angle(coord_x, coord_y) -> float | None:
    """Horizontal angle of a batted ball, in degrees.

    Negative is toward left field, positive toward right, 0 straight up the
    middle; the foul lines sit near -45 and +45.
    """
    if coord_x is None or coord_y is None:
        return None
    dy = _SPRAY_ORIGIN_Y - coord_y
    if dy <= 0:
        return None
    return math.degrees(math.atan2(coord_x - _SPRAY_ORIGIN_X, dy))


def _parse_wind(wind: str) -> tuple[float | None, str | None]:
    """Split '7 mph, L To R' into speed and direction."""
    if not wind:
        return None, None
    speed, _, direction = wind.partition(",")
    try:
        mph = float(speed.strip().split()[0])
    except (ValueError, IndexError):
        mph = None
    return mph, direction.strip() or None

# Events that are true plate appearances. Anything else in allPlays (pickoffs,
# stolen bases, substitutions) is not an opportunity to homer and would inflate
# the denominator if counted.
_NON_PA = {
    "pickoff_1b", "pickoff_2b", "pickoff_3b", "caught_stealing_2b", "caught_stealing_3b",
    "caught_stealing_home", "stolen_base_2b", "stolen_base_3b", "stolen_base_home",
    "wild_pitch", "passed_ball", "balk", "defensive_switch", "pitching_substitution",
    "offensive_substitution", "defensive_substitution", "runner_double_play",
    "pickoff_error_1b", "pickoff_error_2b", "pickoff_error_3b", "other_advance",
    "game_advisory", "ejection", "injury", "stolen_base_home", "error",
    "cs_double_play", "pickoff_caught_stealing_2b", "pickoff_caught_stealing_3b",
    "pickoff_caught_stealing_home", "defensive_indiff",
}


def plate_appearances(game: GameRef) -> list[dict]:
    """Every plate appearance in one game, flattened."""
    data = _get(f"{BASE}/api/v1.1/game/{game.game_pk}/feed/live")
    if not data:
        return []
    live = data.get("liveData", {})
    plays = live.get("plays", {}).get("allPlays", [])
    if not plays:
        return []

    # Game-level context: weather and the plate umpire apply to every PA.
    game_data = data.get("gameData", {})
    weather = game_data.get("weather") or {}
    try:
        temp_f = float(weather.get("temp"))
    except (TypeError, ValueError):
        temp_f = None
    wind_mph, wind_dir = _parse_wind(weather.get("wind", ""))
    condition = weather.get("condition")

    boxscore = live.get("boxscore", {})
    ump_hp = next(
        (o.get("official", {}).get("fullName")
         for o in boxscore.get("officials", [])
         if o.get("officialType") == "Home Plate"),
        None,
    )

    # Lineup slot per batter. battingOrder is a string like "601": the first
    # digit is the slot, the rest a substitution index.
    slot_by_batter: dict[int, int] = {}
    for team_side in ("home", "away"):
        for player in (boxscore.get("teams", {}).get(team_side, {})
                       .get("players", {}) or {}).values():
            order = player.get("battingOrder")
            pid = (player.get("person") or {}).get("id")
            if order and pid:
                try:
                    slot_by_batter[int(pid)] = int(str(order)[0])
                except (ValueError, IndexError):
                    continue

    # Times through the order, counted per pitcher as the game runs.
    faced: dict[int, int] = {}

    rows: list[dict] = []
    for idx, play in enumerate(plays):
        result = play.get("result", {})
        etype = result.get("eventType", "")
        if not etype or etype in _NON_PA:
            continue
        about = play.get("about", {})
        matchup = play.get("matchup", {})
        # halfInning 'top' -> away team bats
        top = about.get("halfInning") == "top"
        events = play.get("playEvents", [])

        hit = next((e["hitData"] for e in events if e.get("hitData")), None)
        pitch = next((e for e in reversed(events) if e.get("isPitch")), None)
        ptype = pspeed = None
        if pitch:
            det = pitch.get("details", {}) or {}
            ptype = (det.get("type") or {}).get("description")
            pspeed = (pitch.get("pitchData") or {}).get("startSpeed")

        batter_id = (matchup.get("batter") or {}).get("id")
        pitcher_id = (matchup.get("pitcher") or {}).get("id")
        bat_side = (matchup.get("batSide") or {}).get("code", "")

        # Third time through the order is measurably worse for the pitcher, so
        # the trip has to be recorded at the plate appearance that it happened.
        seen = faced.get(pitcher_id, 0)
        faced[pitcher_id] = seen + 1
        tto = seen // 9 + 1

        coords = (hit or {}).get("coordinates") or {}
        coord_x, coord_y = coords.get("coordX"), coords.get("coordY")
        spray = _spray_angle(coord_x, coord_y)
        # Positive pull_angle means the ball was pulled, for either handedness.
        pull = None if spray is None else (-spray if bat_side == "R" else spray)

        rows.append({
            "game_pk": game.game_pk,
            "date": game.date,
            "venue": game.venue,
            "batting_team": game.away if top else game.home,
            "pitching_team": game.home if top else game.away,
            "is_home": 0 if top else 1,
            "batter_id": batter_id,
            "batter": (matchup.get("batter") or {}).get("fullName", ""),
            "bat_side": bat_side,
            "pitcher_id": pitcher_id,
            "pitcher": (matchup.get("pitcher") or {}).get("fullName", ""),
            "pitch_hand": (matchup.get("pitchHand") or {}).get("code", ""),
            "inning": about.get("inning"),
            "pa_index": idx,
            "event": result.get("event", ""),
            "event_type": etype,
            "is_hr": 1 if etype == "home_run" else 0,
            "launch_speed": (hit or {}).get("launchSpeed"),
            "launch_angle": (hit or {}).get("launchAngle"),
            "distance": (hit or {}).get("totalDistance"),
            "pitch_type": ptype,
            "pitch_speed": pspeed,
            "coord_x": coord_x,
            "coord_y": coord_y,
            "spray_angle": None if spray is None else round(spray, 2),
            "pull_angle": None if pull is None else round(pull, 2),
            "trajectory": (hit or {}).get("trajectory"),
            "lineup_slot": slot_by_batter.get(batter_id),
            "tto": tto,
            "temp_f": temp_f,
            "wind_mph": wind_mph,
            "wind_dir": wind_dir,
            "condition": condition,
            "ump_hp": ump_hp,
        })
    return rows


def fetch_season(start: date, end: date, out_path: str, workers: int = 12,
                 resume: bool = True, progress_every: int = 100) -> int:
    """Fetch every final game in the range into a JSONL of plate appearances.

    Resumable: game_pks already present in out_path are skipped, so an interrupted
    run (or a later date range) tops up the same file instead of starting over.
    Raw feeds are never cached — they are large and only a small slice is kept.
    """
    games = [g for g in schedule(start, end) if g.state == "Final"]
    done: set[int] = set()
    if resume and os.path.exists(out_path):
        with open(out_path) as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["game_pk"])
                except (json.JSONDecodeError, KeyError):
                    continue
    todo = [g for g in games if g.game_pk not in done]
    print(f"{len(games)} final games in range; {len(done)} already cached; "
          f"fetching {len(todo)}", flush=True)

    written = 0
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "a") as fh, ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(plate_appearances, g): g for g in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                rows = fut.result()
            except Exception as exc:                      # noqa: BLE001
                print(f"  error on {futs[fut].game_pk}: {exc}", file=sys.stderr)
                continue
            for r in rows:
                fh.write(json.dumps(r) + "\n")
            written += len(rows)
            if i % progress_every == 0:
                print(f"  {i}/{len(todo)} games · {written:,} PAs", flush=True)
    print(f"done: {written:,} new plate appearances -> {out_path}", flush=True)
    return written


# ── supporting reference data ─────────────────────────────────────────────
def probable_pitchers(day: date) -> list[dict]:
    """Today's card with probable starters (used for forward-looking slates)."""
    url = (f"{BASE}/api/v1/schedule?sportId=1&date={day:%m/%d/%Y}"
           f"&hydrate=probablePitcher,venue")
    data = _get(url)
    out = []
    for d in (data or {}).get("dates", []):
        for g in d.get("games", []):
            row = {"game_pk": g["gamePk"], "date": g.get("officialDate", ""),
                   "venue": g.get("venue", {}).get("name", ""),
                   "state": g.get("status", {}).get("abstractGameState", "")}
            for side in ("away", "home"):
                t = g["teams"][side]
                pp = t.get("probablePitcher") or {}
                row[f"{side}_team"] = t["team"]["name"]
                row[f"{side}_team_id"] = t["team"]["id"]
                row[f"{side}_sp_id"] = pp.get("id")
                row[f"{side}_sp"] = pp.get("fullName")
            out.append(row)
    return out


def pitcher_hands(pitcher_ids: Iterable[int]) -> dict[int, str]:
    """Throwing hand for a batch of pitchers (one call per 100 ids)."""
    ids = [int(p) for p in pitcher_ids if p]
    hands: dict[int, str] = {}
    for i in range(0, len(ids), 100):
        batch = ",".join(str(x) for x in ids[i:i + 100])
        data = _get(f"{BASE}/api/v1/people?personIds={batch}")
        for p in (data or {}).get("people", []):
            h = (p.get("pitchHand") or {}).get("code")
            if h:
                hands[p["id"]] = h
    return hands


def active_rosters(team_ids: Iterable[int], season: int) -> dict[int, list[dict]]:
    """Active position players per club — a board for today should not list
    hitters who have since been traded, optioned or placed on the IL."""
    def one(tid: int):
        d = _get(f"{BASE}/api/v1/teams/{tid}/roster/active?season={season}")
        return tid, [{"id": p["person"]["id"], "name": p["person"]["fullName"],
                      "pos": p["position"]["abbreviation"]}
                     for p in (d or {}).get("roster", [])
                     if p["position"]["abbreviation"] != "P"]

    out: dict[int, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=10) as ex:
        for tid, roster in ex.map(one, [int(t) for t in team_ids]):
            out[tid] = roster
    return out


def iter_jsonl(path: str) -> Iterator[dict]:
    with open(path) as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def team_standings(season: int) -> dict[int, dict]:
    """Season record and run totals per team id.

    The PA feed carries no score, so team strength for the game projection has
    to come from here. Runs scored/allowed matter more than raw W-L: a team's
    Pythagorean expectation from run differential is a steadier estimate of
    true talent than its actual record, which absorbs bullpen luck and
    one-run-game variance.
    """
    url = (f"{BASE}/api/v1/standings?leagueId=103,104&season={season}"
           f"&standingsTypes=regularSeason")
    data = _get(url)
    out: dict[int, dict] = {}
    for record in (data or {}).get("records", []):
        for team in record.get("teamRecords", []):
            tid = (team.get("team") or {}).get("id")
            if not tid:
                continue
            wins, losses = team.get("wins", 0), team.get("losses", 0)
            out[int(tid)] = {
                "team_id": int(tid),
                "name": (team.get("team") or {}).get("name", ""),
                "wins": wins,
                "losses": losses,
                "games_played": team.get("gamesPlayed") or (wins + losses),
                "runs_scored": team.get("runsScored", 0),
                "runs_allowed": team.get("runsAllowed", 0),
                "run_differential": team.get("runDifferential", 0),
                "division_rank": team.get("divisionRank", ""),
                "streak": (team.get("streak") or {}).get("streakCode", ""),
            }
    return out


def lineups(day: date) -> dict[int, dict]:
    """Confirmed batting orders per game, keyed by game_pk.

    Exposure is the multiplier on every per-game projection, and lineup slot
    sets it: a leadoff hitter averages about 4.65 plate appearances, a ninth
    hitter about 3.75. Just as important, this is what separates the nine men
    who will actually bat from the rest of the active roster.

    Returns {game_pk: {"home": [ids in order], "away": [...],
                       "home_names": [...], "away_names": [...]}}.
    Empty when lineups have not been posted yet, which is typical until an hour
    or two before first pitch.
    """
    url = (f"{BASE}/api/v1/schedule?sportId=1&date={day:%m/%d/%Y}"
           f"&hydrate=lineups")
    data = _get(url)
    out: dict[int, dict] = {}
    for d in (data or {}).get("dates", []):
        for g in d.get("games", []):
            lu = g.get("lineups") or {}
            home = lu.get("homePlayers") or []
            away = lu.get("awayPlayers") or []
            if not home and not away:
                continue
            out[g["gamePk"]] = {
                "home": [p["id"] for p in home],
                "away": [p["id"] for p in away],
                "home_names": [p.get("fullName", "") for p in home],
                "away_names": [p.get("fullName", "") for p in away],
            }
    return out


def game_context(game_pks: Iterable[int]) -> dict[int, dict]:
    """Weather and plate umpire for specific games.

    Read from the live feed so a slate built for today gets the conditions of
    today, not a season average. Failures degrade to an empty entry and the
    context model then applies a neutral 1.0.
    """
    def one(pk: int):
        data = _get(f"{BASE}/api/v1.1/game/{pk}/feed/live")
        if not data:
            return pk, {}
        weather = (data.get("gameData") or {}).get("weather") or {}
        try:
            temp_f = float(weather.get("temp"))
        except (TypeError, ValueError):
            temp_f = None
        wind_mph, wind_dir = _parse_wind(weather.get("wind", ""))
        officials = (data.get("liveData") or {}).get("boxscore", {}).get("officials", [])
        ump = next(
            (o.get("official", {}).get("fullName") for o in officials
             if o.get("officialType") == "Home Plate"),
            None,
        )
        return pk, {
            "temp_f": temp_f,
            "wind_mph": wind_mph,
            "wind_dir": wind_dir,
            "condition": weather.get("condition"),
            "ump_hp": ump,
        }

    out: dict[int, dict] = {}
    pks = [int(p) for p in game_pks if p]
    if not pks:
        return out
    with ThreadPoolExecutor(max_workers=8) as ex:
        for pk, ctx in ex.map(one, pks):
            out[pk] = ctx
    return out


def live_games(day: date) -> dict[int, dict]:
    """State and score for every game on a date, keyed by game_pk.

    One request for the whole slate, so it is cheap enough to call on every
    page refresh. `state` is the coarse Preview/Live/Final; `detailed` carries
    the readable version ("Warmup", "Delayed: Rain", "Final"), which is what a
    reader actually wants when a game is not simply underway.
    """
    url = (f"{BASE}/api/v1/schedule?sportId=1"
           f"&startDate={day:%Y-%m-%d}&endDate={day:%Y-%m-%d}"
           f"&hydrate=linescore")
    data = _get(url)
    out: dict[int, dict] = {}
    for d in (data or {}).get("dates", []):
        for g in d.get("games", []):
            status = g.get("status", {})
            line = g.get("linescore") or {}
            teams = g.get("teams", {})
            out[g["gamePk"]] = {
                "state": status.get("abstractGameState", ""),
                "detailed": status.get("detailedState", ""),
                "inning": line.get("currentInning"),
                "inning_ordinal": line.get("currentInningOrdinal"),
                "inning_state": line.get("inningState"),
                "home_score": teams.get("home", {}).get("score"),
                "away_score": teams.get("away", {}).get("score"),
            }
    return out


def boxscore_batting(game_pks: Iterable[int], workers: int = 8) -> dict[int, dict]:
    """Every batter's line in the given games, keyed by batter id.

    The box score is a tenth the size of the live feed and already carries the
    aggregated batting line, so scoring a slate against what happened does not
    need the play-by-play the season fetch pulls. A batter appears once; the
    same person cannot bat in two games on one slate except in a doubleheader,
    where the top-level line is the two games combined and `by_game` keeps each
    game's own line. A projection is for one game, so it must be scored with
    that game's line (results.batter_line) -- scoring the combined line let a
    pick in game one collect the home run he hit in game two.
    """
    pks = list(game_pks)
    out: dict[int, dict] = {}

    def one(pk: int) -> tuple[int, list[dict]]:
        data = _get(f"{BASE}/api/v1/game/{pk}/boxscore")
        rows: list[dict] = []
        for side in ("home", "away"):
            players = (data or {}).get("teams", {}).get(side, {}).get("players", {})
            for player in (players or {}).values():
                batting = (player.get("stats") or {}).get("batting") or {}
                # Pitchers and unused bench players carry an empty batting dict;
                # a batter who came up has at least a plate appearance.
                if not batting.get("plateAppearances"):
                    continue
                person = player.get("person") or {}
                rows.append({
                    "batter_id": person.get("id"),
                    "batter": person.get("fullName", ""),
                    "pa": batting.get("plateAppearances", 0),
                    "ab": batting.get("atBats", 0),
                    "hits": batting.get("hits", 0),
                    "hr": batting.get("homeRuns", 0),
                    "rbi": batting.get("rbi", 0),
                    "summary": batting.get("summary", ""),
                })
        return pk, rows

    if not pks:
        return out
    with ThreadPoolExecutor(max_workers=min(workers, len(pks))) as ex:
        for pk, rows in ex.map(one, pks):
            for row in rows:
                bid = row["batter_id"]
                if bid is None:
                    continue
                game_line = {k: row[k] for k in ("pa", "ab", "hits", "hr", "rbi", "summary")}
                prior = out.get(bid)
                if prior:  # doubleheader: add the second line to the first
                    row["pa"] += prior["pa"]
                    row["ab"] += prior["ab"]
                    row["hits"] += prior["hits"]
                    row["hr"] += prior["hr"]
                    row["rbi"] += prior["rbi"]
                    row["summary"] = f"{prior['summary']}; {row['summary']}"
                row["by_game"] = {**(prior or {}).get("by_game", {}), pk: game_line}
                row["game_pk"] = pk
                out[bid] = row
    return out
