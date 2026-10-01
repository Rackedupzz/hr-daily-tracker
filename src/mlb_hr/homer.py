"""HOMER: the tracker's resident MLB expert, with his own book and his own picks.

The house model is a fitted ensemble. HOMER is deliberately something else: a
handicapper who reads the raw plate-appearance feed and builds his scouting
report the way an analyst with a spreadsheet would. None of the ensemble's
probabilities feed him. From the slate he takes only facts -- who is in the
lineup, who is starting, the ballpark and the weather -- and from the season
feed he measures everything else himself:

  * **Expected home runs (xHR).** Every ball in play is binned by exit velocity
    and launch angle; the league's home-run rate per bin turns a hitter's
    contact into the home runs it deserved.
  * **Form.** The last 21 days of xHR, for hitters and for starters.
  * **Splits.** The hitter's own record against today's pitcher hand, and the
    league-wide platoon effect.
  * **Pitchers.** HR and xHR allowed per batter faced, fly-ball rate, and the
    bullpen that takes over after the starter.
  * **Park** from home/road HR rates, **weather** from temperature and wind.

**He learns what matters.** Those scouting factors are the inputs, but how much
each is worth is not hand-set: `homer_fit` replays the season week by week --
HOMER's book frozen at each week's start, graded on games he had not seen --
fits the weights, drops any factor that does not improve predictions on
unseen weeks, and records how his picks actually did. HOMER serves the learned
weights only if they beat his original rule-of-thumb formula in that replay,
and every grade and "take" on his card is backed by those measured results.

For games he runs a wOBA runs model (lineup against the starter and bullpen it
faces, park-adjusted) and season run differential, weighted the same way.

His card is saved per day under ``<snapshots>/homer/``. A rebuild during the day
keeps every pick in a game that has already started, exactly like the house
slate, so his record is never rewritten after first pitch.
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

from mlb_hr.results import batter_line

NAME = "HOMER"
DATA_DIR = Path(__file__).parent / "data"
FIT_PATH = DATA_DIR / "homer_fit.json"

# Linear weights for wOBA. Outs, errors and sacrifices are worth nothing.
_WOBA = {
    "walk": 0.69, "hit_by_pitch": 0.72, "single": 0.88,
    "double": 1.25, "triple": 1.58, "home_run": 2.03,
}
# Not a real trip to the plate for rate purposes.
_NOT_PA = {"intent_walk", "sac_bunt", "catcher_interf"}

# Exit velocity / launch angle bins for expected home runs.
_EV_LO, _EV_STEP, _EV_BINS = 60.0, 3.0, 22     # 60..126 mph
_LA_LO, _LA_STEP, _LA_BINS = -20.0, 4.0, 20    # -20..60 degrees

RECENT_DAYS = 21

# Shrinkage strengths, in plate appearances / batters faced / balls in play.
K_BATTER_XHR = 120
K_BATTER_HR = 250
K_RECENT = 60
K_SPLIT = 150
K_BARREL = 50
K_PITCHER = 350
K_PITCHER_RECENT = 80
K_PITCHER_FLY = 150
K_BULLPEN = 900
K_PARK = 4000
K_WOBA_SP = 250
K_WOBA_PEN = 1000
K_WOBA_TEAM = 1500

# Expected plate appearances by lineup slot (league-typical).
_PA_BY_SLOT = [4.65, 4.55, 4.45, 4.35, 4.25, 4.13, 4.01, 3.90, 3.78]
_PA_UNCONFIRMED = 3.9
_TEAM_PA_PER_GAME = 38.3

# Calibration dial for the original rule-of-thumb formula, used when no
# learned fit is available: p = SCALE * league * (raw / league) ** POWER.
CAL_SCALE = 0.9
CAL_POWER = 0.8

MIN_SEASON_PA = 80
HR_PICKS = 6

# The scouting factors HOMER can weigh, in display order.
HR_FEATURES = [
    "xhr", "hr", "form", "barrel", "split", "platoon", "sp_hr", "sp_fly",
    "sp_form", "pen", "park", "weather", "exp_pa", "sample",
]
GAME_FEATURES = ["offense", "starter", "bullpen", "record", "park"]
HIT_FEATURES = [
    "hit_rate", "xhit", "hit_form", "k_rate", "bb_rate", "hit_split", "hit_platoon",
    "sp_hit", "sp_k", "pen_hit", "hit_park", "exp_pa", "sample",
]
HIT_PICKS = 6

K_HIT = 200
K_XHIT = 150
K_K = 150
K_SP_HIT = 300
K_SP_K = 200
K_PEN_HIT = 900

FEATURE_LABELS = {
    "xhr": "Expected HR rate (contact quality)",
    "hr": "Actual HR rate",
    "form": "Last 21 days vs season",
    "barrel": "Barrel rate",
    "split": "His own split vs today's hand",
    "platoon": "League platoon edge",
    "sp_hr": "Starter's HR allowed",
    "sp_fly": "Starter's fly-ball rate",
    "sp_form": "Starter's last 21 days",
    "pen": "Opposing bullpen HR allowed",
    "park": "Ballpark",
    "weather": "Temperature & wind",
    "exp_pa": "Lineup spot (trips to the plate)",
    "sample": "Size of track record",
    "offense": "Lineup quality (wOBA)",
    "starter": "Starting pitcher matchup",
    "bullpen": "Bullpen matchup",
    "record": "Season run differential",
    "hit_rate": "Hits per plate appearance",
    "xhit": "Expected hits (contact quality)",
    "hit_form": "Last 21 days vs season (hits)",
    "k_rate": "Strikeout rate",
    "bb_rate": "Walk rate",
    "hit_split": "His own hit split vs today's hand",
    "hit_platoon": "League platoon edge (hits)",
    "sp_hit": "Starter's hits allowed",
    "sp_k": "Starter's strikeout rate",
    "pen_hit": "Opposing bullpen hits allowed",
    "hit_park": "Ballpark (hits)",
}


# ────────────────────────────────────────────────────────────── the feed

_ROW_FIELDS = (
    "game_pk", "date", "venue", "batting_team", "pitching_team", "is_home",
    "batter_id", "batter", "bat_side", "pitcher_id", "pitcher", "pitch_hand",
    "inning", "pa_index", "event_type", "is_hr", "launch_speed", "launch_angle",
    "trajectory", "lineup_slot", "temp_f", "wind_mph", "wind_dir", "condition",
)


class Row:
    """One plate appearance with only the fields HOMER reads.

    Slotted and with interned strings, so the whole season fits in memory for
    the week-by-week replay instead of re-parsing the file for every week.
    """
    __slots__ = _ROW_FIELDS

    def __init__(self, d: dict):
        for f in _ROW_FIELDS:
            v = d.get(f)
            setattr(self, f, sys.intern(v) if isinstance(v, str) else v)


def iter_rows(pa_path: str) -> Iterable[Row]:
    with open(pa_path, encoding="utf-8") as fh:
        for raw in fh:
            try:
                yield Row(json.loads(raw))
            except json.JSONDecodeError:
                continue


def load_rows(pa_path: str) -> list[Row]:
    return list(iter_rows(pa_path))


# ─────────────────────────────────────────────────────────────── the book


def _bin(ev, la) -> Optional[int]:
    if ev is None or la is None:
        return None
    e = min(max(int((ev - _EV_LO) // _EV_STEP), 0), _EV_BINS - 1)
    a = min(max(int((la - _LA_LO) // _LA_STEP), 0), _LA_BINS - 1)
    return e * _LA_BINS + a


_HIT_EVENTS = {"single", "double", "triple", "home_run"}
_K_EVENTS = {"strikeout", "strikeout_double_play"}
_BB_EVENTS = {"walk", "hit_by_pitch"}


def _new_line() -> dict:
    return {"pa": 0, "hr": 0, "hit": 0, "k": 0, "bb": 0, "woba": 0.0, "bins": Counter()}


def _add(line: dict, hr: int, woba: float, b: Optional[int], hit: int = 0,
         k: int = 0, bb: int = 0) -> None:
    line["pa"] += 1
    line["hr"] += hr
    line["hit"] += hit
    line["k"] += k
    line["bb"] += bb
    line["woba"] += woba
    if b is not None:
        line["bins"][b] += 1


def build_book(source, before: date) -> dict:
    """Everything HOMER knows, from plate appearances strictly before `before`.

    `source` is a path to the season feed (streamed) or already-loaded rows.
    Nothing from the day being handicapped is read, so his picks cannot see
    their own outcomes.
    """
    rows = iter_rows(source) if isinstance(source, (str, Path)) else source
    cutoff = before.isoformat()
    recent_from = (before - timedelta(days=RECENT_DAYS)).isoformat()

    league = {"pa": 0, "hr": 0, "hit": 0, "k": 0, "bb": 0, "woba": 0.0, "bip": 0,
              "fly": 0, "barrels": 0, "games": set()}
    bin_n: Counter = Counter()
    bin_hr: Counter = Counter()
    bin_hit: Counter = Counter()
    platoon = defaultdict(lambda: [0, 0, 0])        # (bat, pitch) -> [pa, hr, hit]

    batters: dict = {}
    pitchers: dict = {}
    teams_off = defaultdict(lambda: {"pa": 0, "woba": 0.0, "hr": 0})
    venue = defaultdict(lambda: {"pa": 0, "hr": 0, "hit": 0, "woba": 0.0})
    road = defaultdict(lambda: {"pa": 0, "hr": 0, "hit": 0, "woba": 0.0})
    home_venue: dict = defaultdict(Counter)
    # (game_pk, pitching_team) -> {pitcher_id: [first_pa_index, bf, hr, woba, hit]}
    staff = defaultdict(dict)

    for r in rows:
        day = r.date or ""
        if day >= cutoff:
            continue
        event = r.event_type or ""
        if event in _NOT_PA:
            continue
        hr = 1 if r.is_hr else 0
        hit = 1 if event in _HIT_EVENTS else 0
        k = 1 if event in _K_EVENTS else 0
        bb = 1 if event in _BB_EVENTS else 0
        woba = _WOBA.get(event, 0.0)
        ev, la = r.launch_speed, r.launch_angle
        b = _bin(ev, la)
        recent = day >= recent_from
        fly = r.trajectory == "fly_ball"
        barrel = b is not None and ev >= 98 and 24 <= la <= 36
        counts = (hit, k, bb)

        league["pa"] += 1
        league["hr"] += hr
        league["hit"] += hit
        league["k"] += k
        league["bb"] += bb
        league["woba"] += woba
        league["games"].add(r.game_pk)
        if b is not None:
            league["bip"] += 1
            league["fly"] += fly
            league["barrels"] += barrel
            bin_n[b] += 1
            bin_hr[b] += hr
            bin_hit[b] += hit

        bat_side, pitch_hand = r.bat_side, r.pitch_hand
        if bat_side in ("L", "R") and pitch_hand in ("L", "R"):
            cell = platoon[(bat_side, pitch_hand)]
            cell[0] += 1
            cell[1] += hr
            cell[2] += hit

        if r.batter_id is not None:
            bat = batters.get(r.batter_id)
            if bat is None:
                bat = batters[r.batter_id] = {
                    "name": r.batter or "", "team": r.batting_team,
                    "season": _new_line(), "recent": _new_line(),
                    "hands": {"L": _new_line(), "R": _new_line()},
                    "barrels": 0, "hard": 0, "bip": 0, "fly": 0,
                    "slots": {},    # game_pk -> lineup slot, recent games only
                }
            bat["team"] = r.batting_team or bat["team"]
            _add(bat["season"], hr, woba, b, *counts)
            if recent:
                _add(bat["recent"], hr, woba, b, *counts)
                if r.lineup_slot:
                    bat["slots"].setdefault(r.game_pk, r.lineup_slot)
            if pitch_hand in ("L", "R"):
                _add(bat["hands"][pitch_hand], hr, woba, b, *counts)
            if b is not None:
                bat["bip"] += 1
                bat["hard"] += ev >= 95
                bat["barrels"] += barrel
                bat["fly"] += fly

        pid = r.pitcher_id
        if pid is not None:
            pit = pitchers.get(pid)
            if pit is None:
                pit = pitchers[pid] = {
                    "name": r.pitcher or "", "team": r.pitching_team,
                    "hand": pitch_hand, "season": _new_line(), "recent": _new_line(),
                    "bip": 0, "fly": 0,
                }
            pit["team"] = r.pitching_team or pit["team"]
            _add(pit["season"], hr, woba, b, *counts)
            if recent:
                _add(pit["recent"], hr, woba, b, *counts)
            if b is not None:
                pit["bip"] += 1
                pit["fly"] += fly
            arms = staff[(r.game_pk, r.pitching_team)]
            idx = r.pa_index or 0
            entry = arms.get(pid)
            if entry is None:
                arms[pid] = [idx, 1, hr, woba, hit]
            else:
                entry[0] = min(entry[0], idx)
                entry[1] += 1
                entry[2] += hr
                entry[3] += woba
                entry[4] += hit

        off = teams_off[r.batting_team]
        off["pa"] += 1
        off["woba"] += woba
        off["hr"] += hr

        v = r.venue or ""
        home_team = r.batting_team if r.is_home else r.pitching_team
        away_team = r.pitching_team if r.is_home else r.batting_team
        home_venue[home_team][v] += 1
        for bucket in (venue[v], road[away_team]):
            bucket["pa"] += 1
            bucket["hr"] += hr
            bucket["hit"] += hit
            bucket["woba"] += woba

    lg_pa = max(league["pa"], 1)
    lg_hr = league["hr"] / lg_pa
    lg_hit = league["hit"] / lg_pa
    lg_woba = league["woba"] / lg_pa
    lg_bip = max(league["bip"], 1)
    lg_bip_hr = sum(bin_hr.values()) / lg_bip
    lg_bip_hit = sum(bin_hit.values()) / lg_bip
    # League HR and hit rates per contact bin, lightly smoothed toward overall.
    xhr_table = {k: (bin_hr[k] + 2 * lg_bip_hr) / (n + 2) for k, n in bin_n.items()}
    xhit_table = {k: (bin_hit[k] + 2 * lg_bip_hit) / (n + 2) for k, n in bin_n.items()}

    def settle(line: dict) -> None:
        bins = line.pop("bins")
        line["xhr"] = sum(n * xhr_table.get(k, lg_bip_hr) for k, n in bins.items())
        line["xhit"] = sum(n * xhit_table.get(k, lg_bip_hit) for k, n in bins.items())

    starts = defaultdict(lambda: {"starts": 0, "bf": 0})
    pens = defaultdict(lambda: {"bf": 0, "hr": 0, "hit": 0, "woba": 0.0})
    for (_, team), arms in staff.items():
        starter = min(arms, key=lambda p: arms[p][0])
        for pid, (_, bf, hrs, wob, hits) in arms.items():
            if pid == starter:
                starts[pid]["starts"] += 1
                starts[pid]["bf"] += bf
            else:
                pen = pens[team]
                pen["bf"] += bf
                pen["hr"] += hrs
                pen["hit"] += hits
                pen["woba"] += wob

    # Home/road park factors. A venue is only rated for the team that plays
    # most of its home games there, so neutral-site series do not pollute it.
    parks = {}
    for team, counts in home_venue.items():
        v, _ = counts.most_common(1)[0]
        at, away = venue[v], road.get(team)
        if not away or not away["pa"] or not at["pa"]:
            continue
        road_hr = away["hr"] / away["pa"]
        road_woba = away["woba"] / away["pa"]
        road_hit = away["hit"] / away["pa"]
        hr_ratio = (at["hr"] / at["pa"]) / road_hr if road_hr else 1.0
        hit_ratio = (at["hit"] / at["pa"]) / road_hit if road_hit else 1.0
        woba_ratio = (at["woba"] / at["pa"]) / road_woba if road_woba else 1.0
        w = at["pa"] / (at["pa"] + K_PARK)
        parks[v] = {
            "hr": round(1 + w * (hr_ratio - 1), 3),
            "hit": round(1 + w * (hit_ratio - 1), 3),
            "runs": round(1 + w * (woba_ratio ** 2 - 1), 3),
            "pa": at["pa"],
        }

    platoon_mult, platoon_hit = {}, {}
    for side in ("L", "R"):
        tot_pa = sum(platoon[(side, h)][0] for h in ("L", "R"))
        tot_hr = sum(platoon[(side, h)][1] for h in ("L", "R"))
        tot_hit = sum(platoon[(side, h)][2] for h in ("L", "R"))
        base = tot_hr / tot_pa if tot_pa else lg_hr
        base_hit = tot_hit / tot_pa if tot_pa else lg_hit
        for hand in ("L", "R"):
            pa, hrs, hits = platoon[(side, hand)]
            platoon_mult[f"{side}{hand}"] = (hrs / pa) / base if pa and base else 1.0
            platoon_hit[f"{side}{hand}"] = (hits / pa) / base_hit if pa and base_hit else 1.0

    for bat in batters.values():
        # Where he usually hits lately: the projected lineup spot for a game
        # whose lineup has not been posted yet.
        slots = Counter(bat.pop("slots").values())
        bat["usual_slot"] = slots.most_common(1)[0][0] if slots else None
        settle(bat["season"])
        settle(bat["recent"])
        for line in bat["hands"].values():
            settle(line)
    for pid, pit in pitchers.items():
        settle(pit["season"])
        settle(pit["recent"])
        pit["starts"] = starts[pid]["starts"]
        pit["start_bf"] = starts[pid]["bf"]

    return {
        "before": cutoff,
        "built_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "league": {
            "pa": league["pa"], "hr_rate": lg_hr, "woba": lg_woba,
            "hit_rate": lg_hit, "k_rate": league["k"] / lg_pa, "bb_rate": league["bb"] / lg_pa,
            "bip_hr_rate": lg_bip_hr, "games": len(league["games"]),
            "fly_rate": league["fly"] / lg_bip, "barrel_rate": league["barrels"] / lg_bip,
            "bf_per_start": (
                sum(s["bf"] for s in starts.values())
                / max(sum(s["starts"] for s in starts.values()), 1)
            ),
        },
        "platoon": platoon_mult,
        "platoon_hit": platoon_hit,
        "batters": batters,
        "pitchers": pitchers,
        "pens": dict(pens),
        "offense": dict(teams_off),
        "parks": parks,
    }


# ─────────────────────────────────────────────────────────── calculations


def _shrink(num: float, den: float, prior: float, k: float) -> float:
    return (num + k * prior) / (den + k)


def _log(x: float) -> float:
    return math.log(max(x, 1e-6))


def _log_ratio(a: float, b: float) -> float:
    """log(a / b), neutral when the reference rate is still empty."""
    return _log(a / b) if b > 0 else 0.0


def _ratio(a: float, b: float) -> float:
    return a / b if b > 0 else 1.0


def log5(batter: float, pitcher: float, league: float) -> float:
    """Odds-ratio matchup: a hitter's rate against a pitcher's, via league."""
    if not 0 < league < 1:
        return batter
    b = min(max(batter, 1e-6), 1 - 1e-6)
    p = min(max(pitcher, 1e-6), 1 - 1e-6)
    odds = (b / (1 - b)) * (p / (1 - p)) / (league / (1 - league))
    return odds / (1 + odds)


def calibrate(raw: float, league: float) -> float:
    """Pull a raw per-PA rate toward league by the rule-of-thumb dial."""
    if raw <= 0 or league <= 0:
        return raw
    return CAL_SCALE * league * (raw / league) ** CAL_POWER


def weather_factor(weather: Optional[dict], bat_side: Optional[str]) -> tuple[float, str]:
    """Carry from temperature and wind. Returns (factor, plain-English reason)."""
    w = weather or {}
    condition = (w.get("condition") or "").lower()
    if "dome" in condition or "roof closed" in condition:
        return 1.0, "roof closed"
    notes = []
    factor = 1.0
    temp = w.get("temp_f")
    if temp is not None:
        # About 1% more carry per degree over 70F, capped either way.
        factor *= 1 + max(min((temp - 70) * 0.009, 0.15), -0.15)
        if temp >= 82:
            notes.append(f"{temp:.0f}&deg;F air carries")
        elif temp <= 55:
            notes.append(f"cold {temp:.0f}&deg;F knocks balls down")
    mph = w.get("wind_mph") or 0
    direction = (w.get("wind_dir") or "")
    if mph and direction:
        pull_field = "LF" if bat_side == "R" else "RF" if bat_side == "L" else None
        field = direction[-2:]
        if field == "CF":
            per = 0.012
        elif pull_field and field == pull_field:
            per = 0.015
        else:
            per = 0.006
        if direction.startswith("Out"):
            factor *= 1 + min(per * mph, 0.22)
            if mph >= 8:
                notes.append(f"wind {mph:.0f} mph out to {field}")
        elif direction.startswith("In"):
            factor *= 1 - min(per * mph, 0.22)
            if mph >= 8:
                notes.append(f"wind {mph:.0f} mph blowing in from {field}")
    return round(factor, 3), ", ".join(notes)


def _pitcher_hr_rate(book: dict, pit: Optional[dict]) -> float:
    lg = book["league"]["hr_rate"]
    if not pit:
        return lg
    s = pit["season"]
    return (0.6 * _shrink(s["xhr"], s["pa"], lg, K_PITCHER)
            + 0.4 * _shrink(s["hr"], s["pa"], lg, K_PITCHER))


def _pen_rates(book: dict, team: str) -> tuple[float, float]:
    lg = book["league"]
    pen = book["pens"].get(team) or {"bf": 0, "hr": 0, "woba": 0.0}
    return (_shrink(pen["hr"], pen["bf"], lg["hr_rate"], K_BULLPEN),
            _shrink(pen["woba"], pen["bf"], lg["woba"], K_WOBA_PEN))


def _starter_share(book: dict, pit: Optional[dict]) -> float:
    lg_bf = book["league"]["bf_per_start"] or 21.0
    if pit and pit.get("starts"):
        bf = _shrink(pit["start_bf"], pit["starts"], lg_bf, 3)
    else:
        bf = lg_bf
    return min(max(bf / _TEAM_PA_PER_GAME, 0.35), 0.75)


def _find_pitcher(book: dict, name: Optional[str], team: Optional[str]):
    if not name or name == "TBD":
        return None, None
    best = None
    for pid, pit in book["pitchers"].items():
        if pit["name"] != name:
            continue
        if pit["team"] == team:
            return pid, pit
        best = best or (pid, pit)
    return best or (None, None)


def hr_features(book: dict, batter_id, bat_side: Optional[str], sp_id,
                sp_hand: Optional[str], opp_team: Optional[str], venue: Optional[str],
                weather: Optional[dict], slot: Optional[int]) -> Optional[dict]:
    """One hitter's scouting report for one game.

    Shared by the live card and the season replay, so the numbers HOMER was
    graded on are exactly the numbers he picks with. Returns the readable
    components, the factor vector `x` the learned weights apply to, and the
    rule-of-thumb probability.
    """
    bat = book["batters"].get(batter_id)
    if not bat or not bat["season"]["pa"]:
        return None
    lg_row = book["league"]
    lg = lg_row["hr_rate"]
    s, rec = bat["season"], bat["recent"]
    hand = sp_hand if sp_hand in ("L", "R") else "R"
    side = bat_side if bat_side in ("L", "R") else ("L" if hand == "R" else "R")

    xhr_rate = _shrink(s["xhr"], s["pa"], lg, K_BATTER_XHR)
    hr_rate = _shrink(s["hr"], s["pa"], lg, K_BATTER_HR)
    recent_rate = _shrink(rec["xhr"], rec["pa"], xhr_rate, K_RECENT)
    power = 0.45 * xhr_rate + 0.25 * hr_rate + 0.30 * recent_rate
    platoon = book["platoon"].get(f"{side}{hand}", 1.0)
    split_line = bat["hands"][hand]
    split_prior = xhr_rate * platoon
    split_rate = _shrink(split_line["xhr"], split_line["pa"], split_prior, K_SPLIT)
    barrel = _shrink(bat["barrels"], bat["bip"], lg_row["barrel_rate"], K_BARREL)

    sp = book["pitchers"].get(sp_id) if sp_id is not None else None
    sp_rate = _pitcher_hr_rate(book, sp)
    if sp:
        sp_fly = _shrink(sp["fly"], sp["bip"], lg_row["fly_rate"], K_PITCHER_FLY)
        sp_recent = _shrink(sp["recent"]["xhr"], sp["recent"]["pa"], sp_rate, K_PITCHER_RECENT)
    else:
        sp_fly, sp_recent = lg_row["fly_rate"], sp_rate
    pen_rate, _ = _pen_rates(book, opp_team)
    share = _starter_share(book, sp)
    park = book["parks"].get(venue, {}).get("hr", 1.0)
    wx, wx_note = weather_factor(weather, side)

    use_slot = slot or bat.get("usual_slot")
    exp_pa = _PA_BY_SLOT[use_slot - 1] if use_slot and 1 <= use_slot <= 9 else _PA_UNCONFIRMED
    pa_sp = exp_pa * share
    pa_pen = exp_pa - pa_sp
    env = park * wx
    p_sp = min(calibrate(log5(power * platoon, sp_rate, lg), lg) * env, 0.5)
    p_pen = min(calibrate(log5(power, pen_rate, lg), lg) * env, 0.5)
    rules_prob = 1 - (1 - p_sp) ** pa_sp * (1 - p_pen) ** pa_pen

    x = {
        "xhr": _log_ratio(xhr_rate, lg),
        "hr": _log_ratio(hr_rate, lg),
        "form": _log_ratio(recent_rate, xhr_rate),
        "barrel": barrel,
        "split": _log_ratio(split_rate, split_prior),
        "platoon": _log(platoon),
        "sp_hr": _log_ratio(sp_rate, lg),
        "sp_fly": _log_ratio(sp_fly, lg_row["fly_rate"]),
        "sp_form": _log_ratio(sp_recent, sp_rate),
        "pen": _log_ratio(pen_rate, lg),
        "park": _log(park),
        "weather": _log(wx),
        "exp_pa": exp_pa,
        "sample": _log(s["pa"]),
    }
    return {
        "x": x, "rules_prob": rules_prob,
        "xhr_rate": xhr_rate, "hr_rate": hr_rate, "recent_rate": recent_rate,
        "recent_pa": rec["pa"], "recent_hr": rec["hr"],
        "season_pa": s["pa"], "season_hr": s["hr"], "season_xhr": s["xhr"],
        "barrel_rate": barrel, "hard_rate": bat["hard"] / max(bat["bip"], 1),
        "split_index": _ratio(split_rate, split_prior), "split_pa": split_line["pa"],
        "split_hr": split_line["hr"],
        "sp_index": _ratio(sp_rate, lg), "sp_fly": sp_fly,
        "sp_form_index": _ratio(sp_recent, sp_rate),
        "pen_index": _ratio(pen_rate, lg), "platoon": platoon, "park": park,
        "weather": wx, "weather_note": wx_note, "exp_pa": exp_pa, "slot": slot,
        "usual_slot": bat.get("usual_slot"),
        "pa_vs_sp": round(pa_sp, 2), "pa_vs_pen": round(pa_pen, 2),
        "sp_hand": hand, "bat_side": side,
    }


def hit_features(book: dict, batter_id, bat_side: Optional[str], sp_id,
                 sp_hand: Optional[str], opp_team: Optional[str], venue: Optional[str],
                 slot: Optional[int]) -> Optional[dict]:
    """One hitter's chance of at least one hit, built like `hr_features`.

    Contact (hit rate, expected hits from exit velocity and launch angle) and
    approach (strikeouts, walks -- a walk is a trip without an at-bat) against
    a starter's hits allowed and strikeout rate, then the bullpen.
    """
    bat = book["batters"].get(batter_id)
    if not bat or not bat["season"]["pa"]:
        return None
    lg_row = book["league"]
    lg = lg_row.get("hit_rate") or 0.22
    lg_k, lg_bb = lg_row.get("k_rate") or 0.22, lg_row.get("bb_rate") or 0.09
    s, rec = bat["season"], bat["recent"]
    hand = sp_hand if sp_hand in ("L", "R") else "R"
    side = bat_side if bat_side in ("L", "R") else ("L" if hand == "R" else "R")

    hit_rate = _shrink(s["hit"], s["pa"], lg, K_HIT)
    xhit_rate = _shrink(s["xhit"], s["pa"], lg, K_XHIT)
    contact = 0.5 * hit_rate + 0.5 * xhit_rate
    recent = _shrink(rec["hit"], rec["pa"], hit_rate, K_RECENT)
    k_rate = _shrink(s["k"], s["pa"], lg_k, K_K)
    bb_rate = _shrink(s["bb"], s["pa"], lg_bb, K_K)
    platoon = (book.get("platoon_hit") or {}).get(f"{side}{hand}", 1.0)
    split_line = bat["hands"][hand]
    split_prior = hit_rate * platoon
    split_rate = _shrink(split_line["hit"], split_line["pa"], split_prior, K_SPLIT)

    sp = book["pitchers"].get(sp_id) if sp_id is not None else None
    if sp:
        sl = sp["season"]
        sp_hit = (0.5 * _shrink(sl["hit"], sl["pa"], lg, K_SP_HIT)
                  + 0.5 * _shrink(sl["xhit"], sl["pa"], lg, K_SP_HIT))
        sp_k = _shrink(sl["k"], sl["pa"], lg_k, K_SP_K)
    else:
        sp_hit, sp_k = lg, lg_k
    pen = book["pens"].get(opp_team) or {"bf": 0, "hit": 0}
    pen_hit = _shrink(pen.get("hit", 0), pen["bf"], lg, K_PEN_HIT)
    share = _starter_share(book, sp)
    park = book["parks"].get(venue, {}).get("hit", 1.0)

    use_slot = slot or bat.get("usual_slot")
    exp_pa = _PA_BY_SLOT[use_slot - 1] if use_slot and 1 <= use_slot <= 9 else _PA_UNCONFIRMED
    pa_sp = exp_pa * share
    pa_pen = exp_pa - pa_sp
    p_sp = min(log5(contact * platoon, sp_hit, lg) * park, 0.6)
    p_pen = min(log5(contact, pen_hit, lg) * park, 0.6)
    rules_prob = 1 - (1 - p_sp) ** pa_sp * (1 - p_pen) ** pa_pen

    x = {
        "hit_rate": _log_ratio(hit_rate, lg),
        "xhit": _log_ratio(xhit_rate, lg),
        "hit_form": _log_ratio(recent, hit_rate),
        "k_rate": _log_ratio(k_rate, lg_k),
        "bb_rate": _log_ratio(bb_rate, lg_bb),
        "hit_split": _log_ratio(split_rate, split_prior),
        "hit_platoon": _log(platoon),
        "sp_hit": _log_ratio(sp_hit, lg),
        "sp_k": _log_ratio(sp_k, lg_k),
        "pen_hit": _log_ratio(pen_hit, lg),
        "hit_park": _log(park),
        "exp_pa": exp_pa,
        "sample": _log(s["pa"]),
    }
    return {
        "x": x, "rules_prob": rules_prob, "season_pa": s["pa"],
        "season_hits": s["hit"], "avg_line": s["hit"] / max(s["pa"], 1),
        "hit_rate": hit_rate, "xhit_rate": xhit_rate, "recent_hit_rate": recent,
        "recent_pa": rec["pa"], "recent_hits": rec["hit"], "k_rate": k_rate, "bb_rate": bb_rate,
        "split_index": _ratio(split_rate, split_prior), "split_pa": split_line["pa"],
        "split_hits": split_line["hit"], "platoon": platoon,
        "sp_hit_index": _ratio(sp_hit, lg), "sp_k_rate": sp_k, "pen_hit_index": _ratio(pen_hit, lg),
        "park": park, "exp_pa": exp_pa, "slot": slot, "usual_slot": bat.get("usual_slot"),
        "sp_hand": hand, "lg_hit": lg, "lg_k": lg_k,
    }


def standings_features(rs: float, ra: float, games: int) -> float:
    """Pythagorean win% from runs to date, regressed with ten .500 games."""
    lg_runs = 4.5 * 10
    return _pythag_exp(rs + lg_runs, ra + lg_runs, 1.83)


def _pythag_exp(rs: float, ra: float, x: float) -> float:
    if rs <= 0 or ra <= 0:
        return 0.5
    return rs ** x / (rs ** x + ra ** x)


def _pythag(rs: float, ra: float) -> float:
    if rs <= 0 or ra <= 0:
        return 0.5
    return _pythag_exp(rs, ra, (rs + ra) ** 0.287)


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def game_features(book: dict, home: str, away: str, home_sp_id, away_sp_id,
                  venue: Optional[str], home_record: Optional[dict],
                  away_record: Optional[dict], lg_rpg: float) -> dict:
    """Both sides of one game, shared by the live card and the replay.

    Records are {"rs", "ra", "games"} to date. Returns runs, the rule-of-thumb
    home win probability and the factor vector for the learned weights.
    """
    lg = book["league"]
    park_runs = book["parks"].get(venue, {}).get("runs", 1.0)
    sides = {}
    for side, team, opp, opp_sp_id in (("home", home, away, away_sp_id),
                                       ("away", away, home, home_sp_id)):
        off = book["offense"].get(team) or {"pa": 0, "woba": 0.0}
        off_woba = _shrink(off["woba"], off["pa"], lg["woba"], K_WOBA_TEAM)
        sp = book["pitchers"].get(opp_sp_id) if opp_sp_id is not None else None
        sp_line = sp["season"] if sp else {"pa": 0, "woba": 0.0}
        sp_woba = _shrink(sp_line["woba"], sp_line["pa"], lg["woba"], K_WOBA_SP)
        _, pen_woba = _pen_rates(book, opp)
        share = _starter_share(book, sp)
        off_idx = (off_woba / lg["woba"]) ** 2
        prevent = share * (sp_woba / lg["woba"]) ** 2 + (1 - share) * (pen_woba / lg["woba"]) ** 2
        runs = lg_rpg * off_idx * prevent * park_runs * (1.02 if side == "home" else 0.98)
        sides[side] = {"team": team, "runs": runs, "off_woba": off_woba,
                       "opp_sp": sp["name"] if sp else "TBD", "opp_sp_woba": sp_woba,
                       "opp_pen_woba": pen_woba}

    h, a = sides["home"], sides["away"]
    matchup_home = _pythag(h["runs"], a["runs"])
    if home_record and away_record:
        ph = standings_features(home_record["rs"], home_record["ra"], home_record["games"])
        pa = standings_features(away_record["rs"], away_record["ra"], away_record["games"])
        den = ph + pa - 2 * ph * pa
        base = (ph - ph * pa) / den if den else 0.5
        odds = base / (1 - base) * 1.15      # home field, about .535
        record_home = odds / (1 + odds)
        rules_home = 0.65 * matchup_home + 0.35 * record_home
        record_x = _logit(ph) - _logit(pa)
    else:
        ph = pa = None
        rules_home = matchup_home
        record_x = 0.0

    x = {
        "offense": _log(h["off_woba"] / a["off_woba"]),
        # Home benefits when the starter it faces (the away SP) allows more.
        "starter": _log(h["opp_sp_woba"] / a["opp_sp_woba"]),
        "bullpen": _log(h["opp_pen_woba"] / a["opp_pen_woba"]),
        "record": record_x,
        "park": _log(park_runs),
    }
    return {"x": x, "rules_home": rules_home, "home": h, "away": a,
            "park_runs": park_runs, "home_pythag": ph, "away_pythag": pa}


# ────────────────────────────────────────────────────── learned weights


def load_fit(path: Path = FIT_PATH) -> Optional[dict]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def apply_model(model: Optional[dict], x: dict) -> Optional[tuple[float, dict]]:
    """(probability, per-factor log-odds contributions) under learned weights."""
    if not model or not model.get("features"):
        return None
    logit = model["intercept"]
    contrib = {}
    for f, w, mu, sd in zip(model["features"], model["coef"], model["mean"], model["scale"]):
        c = w * (x.get(f, mu) - mu) / (sd or 1.0)
        contrib[f] = c
        logit += c
    # Monotone map fitted on the replay's held-out predictions.
    return interp_calibration(model.get("calibration"), 1 / (1 + math.exp(-logit))), contrib


def interp_calibration(cal: Optional[dict], prob: float) -> float:
    """Read a probability through a fitted {x, y} calibration curve."""
    if not cal or not cal.get("x"):
        return prob
    xs, ys = cal["x"], cal["y"]
    if prob <= xs[0]:
        return ys[0]
    if prob >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if prob <= xs[i]:
            t = (prob - xs[i - 1]) / ((xs[i] - xs[i - 1]) or 1)
            return ys[i - 1] + t * (ys[i] - ys[i - 1])
    return ys[-1]


def _odds_pct(c: float) -> str:
    return f"{(math.exp(c) - 1) * 100:+.0f}% odds"


def grade_for(prob: float, fit: Optional[dict], kind: str = "hr") -> tuple[str, Optional[dict]]:
    """Letter grade plus the measured record of picks in that band."""
    bands = ((fit or {}).get(kind) or {}).get("grades") or []
    if kind != "hr":
        for band in bands:
            if prob >= band["min_prob"]:
                return band["grade"], band
        return ("A" if prob >= 0.72 else "B" if prob >= 0.66 else "C"), None
    for band in bands:
        if prob >= band["min_prob"]:
            return band["grade"], band
    if prob >= 0.20:
        return "A", None
    if prob >= 0.17:
        return "B+", None
    if prob >= 0.14:
        return "B", None
    return "C", None


def _hr_reason(f: str, c: dict, sp_name: str) -> Optional[str]:
    lg_note = lambda v: f"{v:.2f}&times; league"  # noqa: E731
    return {
        "xhr": f"Contact quality: {c['season_xhr']:.1f} xHR in {c['season_pa']} PA ({lg_note(c['xhr_rate'] / c['lg'])})",
        "hr": f"Proven power: {c['season_hr']} HR in {c['season_pa']} PA",
        "form": (f"Last {RECENT_DAYS} days: contact running {c['recent_rate'] / c['xhr_rate']:.2f}&times; "
                 f"his season pace ({c['recent_hr']} HR in {c['recent_pa']} PA)"),
        "barrel": f"Barrels {c['barrel_rate']:.1%} of balls in play",
        "split": (f"His own line vs {c['sp_hand']}HP: {c['split_hr']} HR in {c['split_pa']} PA "
                  f"({c['split_index']:.2f}&times; his norm)"),
        "platoon": f"Platoon matchup vs a {c['sp_hand']}HP ({c['platoon']:.2f}&times;)",
        "sp_hr": f"{sp_name} allows HRs at {lg_note(c['sp_index'])}",
        "sp_fly": f"{sp_name} lives in the air: {c['sp_fly']:.0%} fly balls",
        "sp_form": f"{sp_name}'s last {RECENT_DAYS} days: {c['sp_form_index']:.2f}&times; his season HR contact",
        "pen": f"Opposing bullpen allows HRs at {lg_note(c['pen_index'])}",
        "park": f"{c.get('venue') or 'Park'} plays {c['park']:.2f}&times; for home runs",
        "weather": f"Weather {c['weather']:.2f}&times;{': ' + c['weather_note'] if c['weather_note'] else ''}",
        "exp_pa": (f"Batting {c['slot']}: {c['exp_pa']:.1f} projected trips" if c.get("slot")
                   else f"Lineup not posted; usually bats {c['usual_slot']} ({c['exp_pa']:.1f} trips)"
                   if c.get("usual_slot")
                   else f"Lineup not posted: {c['exp_pa']:.1f} trips assumed"),
        "sample": f"Established: {c['season_pa']} PA of track record",
    }.get(f)


# The direction each factor should push a home run, in baseball terms. A
# learned weight pointing the other way (barrel rate, which expected HR already
# counts) is the fit removing double-counting, not a real reason, so it is
# never offered as one.
EXPECTED_SIGN = {f: 1 for f in HR_FEATURES + HIT_FEATURES}
# Strikeouts end at-bats hitless, and a walk is a trip with no chance of a hit.
EXPECTED_SIGN.update({"k_rate": -1, "sp_k": -1, "bb_rate": -1})


def overlap_features(model: Optional[dict]) -> set:
    if not model:
        return set()
    return {f for f, w in zip(model["features"], model["coef"])
            if EXPECTED_SIGN.get(f, 1) * w < 0}


def _hit_reason(f: str, c: dict, sp_name: str) -> Optional[str]:
    return {
        "hit_rate": f"Pure hitter: {c['season_hits']} hits in {c['season_pa']} PA ({c['avg_line']:.3f} per trip)",
        "xhit": f"Contact quality says {c['xhit_rate'] / c['lg_hit']:.2f}&times; league hits",
        "hit_form": (f"Locked in: {c['recent_hits']} hits in his last {c['recent_pa']} PA "
                     f"({c['recent_hit_rate'] / c['hit_rate']:.2f}&times; his season pace)"),
        "k_rate": f"Strikeout rate {c['k_rate']:.0%} (league {c['lg_k']:.0%})",
        "bb_rate": f"Walk rate {c['bb_rate']:.0%} &mdash; walks are trips without a hit chance",
        "hit_split": (f"Owns {c['sp_hand']}HP: {c['split_hits']} hits in {c['split_pa']} PA "
                      f"({c['split_index']:.2f}&times; his norm)"),
        "hit_platoon": f"Platoon matchup vs a {c['sp_hand']}HP ({c['platoon']:.2f}&times;)",
        "sp_hit": f"{sp_name} allows hits at {c['sp_hit_index']:.2f}&times; league",
        "sp_k": f"{sp_name} strikes out {c['sp_k_rate']:.0%} (league {c['lg_k']:.0%})",
        "pen_hit": f"Opposing bullpen allows hits at {c['pen_hit_index']:.2f}&times; league",
        "hit_park": f"{c.get('venue') or 'Park'} plays {c['park']:.2f}&times; for hits",
        "exp_pa": (f"Batting {c['slot']}: {c['exp_pa']:.1f} trips to the plate" if c.get("slot")
                   else f"Usually bats {c['usual_slot']}: {c['exp_pa']:.1f} trips" if c.get("usual_slot")
                   else f"{c['exp_pa']:.1f} trips assumed"),
        "sample": f"Established: {c['season_pa']} PA of track record",
    }.get(f)


# HOMER's voice. The tag follows the grade, and the grade follows the replay,
# so the swagger is never louder than the evidence behind it.
HR_TAGS = {"A": "HAMMER IT &#128296;", "B": "LOVE IT", "C": "LIKE IT", "D": "SPRINKLE",
           "E": "LOTTO TICKET"}
HIT_TAGS = {"A": "BANK IT &#128176;", "B": "LOVE IT", "C": "LIKE IT", "D": "SOLID", "E": "FILLER"}
GAME_TAGS = {"Strong": "HAMMER", "Lean": "LEAN", "Coin flip": "SMALL PLAY"}


def _learned_take(c: dict, contrib: dict, sp_name: str, skip: set = frozenset(),
                  reason=None) -> list[str]:
    reason = reason or _hr_reason
    ranked = sorted(((f, v) for f, v in contrib.items() if f not in skip and reason(f, c, sp_name)),
                    key=lambda kv: kv[1], reverse=True)
    take = [f"{reason(f, c, sp_name)} <span class='muted'>({_odds_pct(v)})</span>"
            for f, v in ranked[:3] if v > 0.02]
    worst = ranked[-1] if ranked else None
    if worst and worst[1] < -0.05:
        take.append(f"Knock: {reason(worst[0], c, sp_name)} "
                    f"<span class='muted'>({_odds_pct(worst[1])})</span>")
    return take or ["No single standout &mdash; he rates well across the board"]


def _rules_take(c: dict, sp_name: str) -> list[str]:
    """Reasoning for the rule-of-thumb formula, strongest point first."""
    points = []
    if c["barrel_rate"] >= 0.12:
        points.append((c["barrel_rate"], _hr_reason("barrel", c, sp_name)))
    if c["xhr_rate"] / c["lg"] >= 1.4:
        points.append((0.2, _hr_reason("xhr", c, sp_name)))
    if c["recent_pa"] >= 35 and c["recent_rate"] / c["xhr_rate"] >= 1.2:
        points.append((0.2, _hr_reason("form", c, sp_name)))
    if c["sp_index"] >= 1.12:
        points.append((0.25, _hr_reason("sp_hr", c, sp_name)))
    if c["park"] >= 1.08:
        points.append((0.15, _hr_reason("park", c, sp_name)))
    if c["weather_note"] and c["weather"] > 1:
        points.append((0.1, _hr_reason("weather", c, sp_name)))
    points.sort(key=lambda p: p[0], reverse=True)
    return [t for _, t in points[:4]] or ["Steady all-around profile"]


def hr_calculations(slate: dict, book: dict, fit: Optional[dict] = None) -> list[dict]:
    """HOMER's own per-game HR probability for every hitter on the slate."""
    lg = book["league"]["hr_rate"]
    model = ((fit or {}).get("hr") or {}).get("model")
    use_learned = ((fit or {}).get("hr") or {}).get("serve") == "learned" and model
    overlap = overlap_features(model)
    out = []
    for game in slate.get("games", []):
        for side, opp_side in (("home", "away"), ("away", "home")):
            team, opp_team = game.get(side), game.get(opp_side)
            sp_name = game.get(f"{opp_side}_sp") or "TBD"
            sp_id, _ = _find_pitcher(book, sp_name, opp_team)
            hitters = [h for h in game.get("hitters", []) if h.get("side") == side]
            posted = any(h.get("lineup_slot") for h in hitters)
            for h in hitters:
                c = hr_features(book, h.get("batter_id"), h.get("bat_side"), sp_id,
                                game.get(f"{opp_side}_sp_hand"), opp_team, game.get("venue"),
                                game.get("weather"), h.get("lineup_slot"))
                if not c:
                    continue
                c.update({
                    "lg": lg, "batter": h.get("batter"), "batter_id": h.get("batter_id"),
                    "team": team, "game_pk": game.get("game_pk"),
                    "matchup": f"{game.get('away')} @ {game.get('home')}",
                    "venue": game.get("venue"), "sp_name": sp_name,
                    "model_prob": h.get("prob_hr"), "injury_note": h.get("injury_note"),
                    "lineup_posted": posted, "lineup_confirmed": h.get("lineup_slot") is not None,
                })
                learned = apply_model(model, c["x"]) if use_learned else None
                if learned:
                    c["prob"], contrib = learned
                    c["method"] = "learned"
                    c["contrib"] = {k: round(v, 4) for k, v in contrib.items()}
                    c["take"] = _learned_take(c, contrib, sp_name, overlap)
                else:
                    c["prob"] = c["rules_prob"]
                    c["method"] = "rules"
                    c["take"] = _rules_take(c, sp_name)
                del c["x"]
                price_leg(c, "hr", 1 - (1 - c["hr_rate"]) ** c["exp_pa"], fit)
                out.append(c)
    out.sort(key=lambda c: c["prob"], reverse=True)
    return out


def hit_calculations(slate: dict, book: dict, fit: Optional[dict] = None) -> list[dict]:
    """HOMER's own chance of at least one hit for every hitter on the slate."""
    hfit = (fit or {}).get("hits") or {}
    model = hfit.get("model") if hfit.get("serve") == "learned" else None
    overlap = overlap_features(model)
    out = []
    for game in slate.get("games", []):
        for side, opp_side in (("home", "away"), ("away", "home")):
            team, opp_team = game.get(side), game.get(opp_side)
            sp_name = game.get(f"{opp_side}_sp") or "TBD"
            sp_id, _ = _find_pitcher(book, sp_name, opp_team)
            hitters = [h for h in game.get("hitters", []) if h.get("side") == side]
            posted = any(h.get("lineup_slot") for h in hitters)
            for h in hitters:
                c = hit_features(book, h.get("batter_id"), h.get("bat_side"), sp_id,
                                 game.get(f"{opp_side}_sp_hand"), opp_team, game.get("venue"),
                                 h.get("lineup_slot"))
                if not c:
                    continue
                c.update({
                    "batter": h.get("batter"), "batter_id": h.get("batter_id"),
                    "team": team, "game_pk": game.get("game_pk"),
                    "matchup": f"{game.get('away')} @ {game.get('home')}",
                    "venue": game.get("venue"), "sp_name": sp_name,
                    "model_prob": (h.get("hits_proj") or {}).get("prob_at_least_one"),
                    "injury_note": h.get("injury_note"),
                    "lineup_posted": posted, "lineup_confirmed": h.get("lineup_slot") is not None,
                })
                learned = apply_model(model, c["x"]) if model else None
                if learned:
                    c["prob"], contrib = learned
                    c["method"] = "learned"
                    c["take"] = _learned_take(c, contrib, sp_name, overlap, _hit_reason)
                else:
                    c["prob"] = c["rules_prob"]
                    c["method"] = "rules"
                    c["take"] = [t for t in (_hit_reason("hit_rate", c, sp_name),
                                             _hit_reason("sp_hit", c, sp_name),
                                             _hit_reason("exp_pa", c, sp_name)) if t]
                del c["x"]
                price_leg(c, "hits", 1 - (1 - c["hit_rate"]) ** c["exp_pa"], fit)
                out.append(c)
    out.sort(key=lambda c: c["prob"], reverse=True)
    return out


def eligible(c: dict, min_pa: Optional[int] = None) -> bool:
    """An established hitter, no injury note, and in the lineup once it posts."""
    min_pa = MIN_SEASON_PA if min_pa is None else min_pa
    if c["season_pa"] < min_pa or c.get("injury_note"):
        return False
    return not (c.get("lineup_posted") and not c.get("lineup_confirmed", True))


def select_hr_picks(candidates: list[dict], n: int = HR_PICKS, min_pa: Optional[int] = None,
                    skip_games: Iterable = ()) -> list[dict]:
    """HOMER's selection rules, used live and in the replay alike.

    Best probability first, one hitter per game, and only eligible hitters.
    Hit picks go through the same rules.
    """
    used = set(skip_games)
    picks = []
    for c in sorted(candidates, key=lambda c: c["prob"], reverse=True):
        if len(picks) >= n:
            break
        if c["game_pk"] in used or not eligible(c, min_pa):
            continue
        used.add(c["game_pk"])
        picks.append(c)
    return picks


# ───────────────────────────────────────────────────────────── HOMER'S PLAYS

# How the book prices a prop. A sportsbook starts from what the market thinks
# (for player props that is mostly the season line: his rate times his trips)
# and adds its margin to the side you bet. A parlay multiplies the legs' prices,
# so the margin compounds: five hit legs carry about 1.065^5 = 1.37x the true
# odds against you. HOMER's counter is to only pay that toll on legs where his
# matchup read (pitcher, park, weather, contact quality, lineup slot) rates the
# hitter clearly above the season line the book leans on.
#
# The margins are typical US-book holds on the "yes" side: 1+ hit props are
# two-way markets with a modest hold, and HR props are yes-heavy markets with a
# big one. Until a real odds feed is wired in, every book price is an estimate.
BOOK_MARGIN = {"hr": 1.22, "hits": 1.065}
MAX_IMPLIED = 0.97

# A value leg still has to be a real contender. Below these HOMER probabilities
# the "edge" is a long shot the book mispriced, not a leg worth carrying. The hit
# floor sits at .68 because every play now has to cash on at least three days in
# ten: at .66 the cheapest value legs dragged the all-value ticket under that.
VALUE_FLOOR = {"hr": 0.17, "hits": 0.68}


def american(decimal: float) -> str:
    """Decimal odds as a US price."""
    if decimal <= 1:
        return "&mdash;"
    if decimal >= 2:
        return f"+{round(100 * (decimal - 1))}"
    return f"-{round(100 / (decimal - 1))}"


def price_leg(c: dict, kind: str, season_prob: float, fit: Optional[dict] = None) -> dict:
    """Put the book's estimated line on a leg, in place.

    `season_prob` is the season-line chance (his rate over his expected trips);
    the fit's market curve turns it into what that line actually cashes at, and
    the margin turns that into the price the book would hang.
    """
    market = interp_calibration(((fit or {}).get(kind) or {}).get("market"), season_prob)
    implied = min(market * BOOK_MARGIN[kind], MAX_IMPLIED)
    c["market_prob"] = market
    c["book_implied"] = implied
    c["book_price"] = american(1 / implied)
    c["book_edge"] = c["prob"] / implied - 1  # >0: HOMER's number beats the price
    return c


# Each play: total legs, how many are home-run legs, and how HOMER picks them.
# "prob" takes his most likely legs (built to cash). "value" takes his biggest
# edges over the book's price among real contenders (built to pay). Every leg
# comes from a different game, so legs don't rise and fall together -- which is
# what lets the chance be the product of the legs, and what keeps the book from
# repricing it as a correlated same-game parlay.
# Every play has to cash on at least three days in ten, which decides the whole
# board. A parlay can never beat its weakest leg, and HOMER's best home run leg
# cashed 21% in the replay, so no ticket carrying one can clear that bar: these
# are all hit legs. Two legs is too short to fall under 46% and five is too long
# to reach 30%, so every play is three or four legs, and what separates them is
# how many of those legs are taken for price instead of for likelihood.
#
# `key` is what the replay record is filed under, so a key means one exact shape:
# leg count, home-run legs and value legs. Give a play a different shape and it
# needs a new key, or the page would quote the old play's record under new legs.
# The keys spell out the shape for that reason.
PARLAY_PLAYS = [
    {"key": "hit3", "name": "Parlay of the Day", "legs": 3, "hr": 0, "value": 0,
     "featured": True,
     "blurb": "Three legs, three different games, straight off the top of HOMER's hit board. "
              "Nothing here is taken for the price &mdash; it is the shortest ticket on the "
              "board and the one that cashes most often."},
    {"key": "hit3e1", "name": "The Lock", "legs": 3, "hr": 0, "value": 1,
     "blurb": "Two of HOMER's most likely hits, plus one bat the book has underpriced. A "
              "longer number than the Parlay of the Day for about the same cash rate."},
    {"key": "hit3e2", "name": "The Edge", "legs": 3, "hr": 0, "value": 2,
     "blurb": "Flips the mix: one anchor and two legs where HOMER's matchup read beats the "
              "season line the book prices off. More price, a little less certainty."},
    {"key": "hit4e1", "name": "The Stretch", "legs": 4, "hr": 0, "value": 1,
     "blurb": "A fourth leg for a bigger number &mdash; three likely hits and one value bat. "
              "The longest price that still clears the cash bar."},
    {"key": "hit3e3", "name": "Full Value", "legs": 3, "hr": 0, "value": 3,
     "blurb": "Every leg chosen for what the book is paying rather than for how likely it is. "
              "The purest read on whether HOMER beats the price."},
]

LEG_KEYS = ("batter", "batter_id", "team", "game_pk", "matchup", "sp_name", "prob",
            "grade", "y", "market_prob", "book_implied", "book_price", "book_edge")

# HOMER's two 5-pick parlays, posted beside the slate's (mlb_hr.parlays) at the
# user's request on 2026-09-30: five home runs, and five hitters with a hit, each
# his five likeliest legs from five different games. They sit outside HOMER'S
# PLAYS on purpose. That board is held to cashing three days in ten, which no
# five-leg ticket can do -- the five-hit ticket lands about one day in five,
# the five-homer ticket about one day in a few thousand -- so these carry their
# own section, their own record, and their real odds. Same builder, locking and
# grading as the plays; "hr_build": "prob" takes the likeliest home run bats
# rather than the value-first ones the plays would use.
FIVE_PICK_PLAYS = [
    {"key": "hr5", "name": "Five-Homer Parlay", "legs": 5, "hr": 5, "value": 0,
     "hr_build": "prob",
     "blurb": "HOMER's five likeliest home run bats, one per game. Every one of them has to go deep, "
              "so this is a moonshot priced like one &mdash; a ticket for the story, not the bankroll."},
    {"key": "hit5", "name": "Five-Hit Parlay", "legs": 5, "hr": 0, "value": 0,
     "blurb": "HOMER's five likeliest bats to get a hit, one per game. His steadiest five-leg ticket, "
              "and still a long way from a sure thing."},
]


def fair_odds(prob: float) -> str:
    """American odds at which a play of this probability breaks even."""
    if prob <= 0 or prob >= 1:
        return "&mdash;"
    if prob >= 0.5:
        return f"-{round(100 * prob / (1 - prob))}"
    return f"+{round(100 * (1 - prob) / prob)}"


def _leg_rank(build: str, kind: str):
    if build != "value":
        return lambda c: c["prob"]
    floor = VALUE_FLOOR[kind]
    # Contenders first, then the biggest edge over the book's price.
    return lambda c: (c["prob"] >= floor, c.get("book_edge", 0.0), c["prob"])


def build_parlays(hr_cands: list[dict], hit_cands: list[dict],
                  plays: list[dict] = PARLAY_PLAYS, skip_games: Iterable = ()) -> list[dict]:
    """HOMER's plays, no two legs from the same game. Used live and in the replay.

    Home-run legs are taken first, then hit legs fill in from other games, in
    the order the play's build calls for. Each play carries its chance, the
    fair odds, and the book's estimated price with HOMER's expected return at it.
    """
    hr_ok = [c for c in hr_cands if eligible(c)]
    hit_ok = [c for c in hit_cands if eligible(c)]
    out = []
    for play in plays:
        used = set(skip_games)
        legs = []
        n_value = play.get("value", 0)
        n_likely = play["legs"] - play["hr"] - n_value
        # Home run legs are the scarcest, so they pick first when a play wants
        # one. The likely hit legs then claim their games, and the value legs
        # fill what is left -- which is what the replay measured.
        for kind, fit_kind, pool, build, want in (
                ("hr", "hr", hr_ok, play.get("hr_build", "value"), play["hr"]),
                ("hit", "hits", hit_ok, "prob", n_likely),
                ("hit", "hits", hit_ok, "value", n_value)):
            taken = 0
            for c in sorted(pool, key=_leg_rank(build, fit_kind), reverse=True):
                if taken >= want:
                    break
                if c["game_pk"] in used:
                    continue
                used.add(c["game_pk"])
                legs.append(dict({k: c[k] for k in LEG_KEYS if k in c}, type=kind))
                taken += 1
        if len(legs) < play["legs"]:
            continue
        prob = math.prod(leg["prob"] for leg in legs)
        entry = {k: play[k] for k in ("key", "name", "legs", "hr", "blurb")} | {
            "value": n_value, "featured": play.get("featured", False), "leg_list": legs,
            "prob": prob, "fair_odds": fair_odds(prob)}
        if all(leg.get("book_implied") for leg in legs):
            decimal = math.prod(1 / leg["book_implied"] for leg in legs)
            entry |= {"book_decimal": decimal, "book_odds": american(decimal),
                      "ev": prob * decimal - 1}
        out.append(entry)
    return out


def play_evidence(records: dict, play: dict) -> Optional[dict]:
    """This play's replay record, but only if the replay built it the same way.

    Records are filed by `key`, so a key rebuilt with different legs would quote
    the previous play's track record. A record written before the shape was
    stored carries none and is taken at its word.
    """
    rec = records.get(play["key"])
    if not rec:
        return None
    shape = {k: rec[k] for k in ("legs", "hr", "value") if k in rec}
    if any((play.get(k) or 0) != v for k, v in shape.items()):
        return None
    return rec


def _records_from_projection(proj: dict) -> tuple[Optional[dict], Optional[dict]]:
    recs = proj.get("records") or {}
    out = []
    for side in ("home", "away"):
        r = recs.get(side) or {}
        if r.get("runs_scored") is None:
            return None, None
        out.append({"rs": r["runs_scored"], "ra": r["runs_allowed"],
                    "games": (r.get("wins") or 0) + (r.get("losses") or 0)})
    return out[0], out[1]


def _game_reason(f: str, g: dict, pick_side: str) -> str:
    me, them = g[pick_side], g["away" if pick_side == "home" else "home"]
    return {
        "offense": f"{me['team']} lineup {me['off_woba']:.3f} wOBA vs {them['off_woba']:.3f}",
        "starter": (f"Pitching edge: {them['opp_sp']} allows {them['opp_sp_woba']:.3f} wOBA, "
                    f"{me['opp_sp']} {me['opp_sp_woba']:.3f}"),
        "bullpen": (f"Bullpen edge: {me['team']} faces a {me['opp_pen_woba']:.3f} pen, "
                    f"{them['team']} a {them['opp_pen_woba']:.3f}"),
        "record": "Season run differential favors " + me["team"],
        "park": f"Park plays {g['park_runs']:.2f}&times; for runs",
    }.get(f, f)


def game_calculations(slate: dict, book: dict, fit: Optional[dict] = None) -> list[dict]:
    """HOMER's runs and win probability for every game."""
    gfit = (fit or {}).get("games") or {}
    model = gfit.get("model") if gfit.get("serve") == "learned" else None
    out = []
    for game in slate.get("games", []):
        proj = game.get("projection") or {}
        home_rec, away_rec = _records_from_projection(proj)
        home_sp_id, _ = _find_pitcher(book, game.get("home_sp"), game.get("home"))
        away_sp_id, _ = _find_pitcher(book, game.get("away_sp"), game.get("away"))
        g = game_features(book, game.get("home"), game.get("away"), home_sp_id, away_sp_id,
                          game.get("venue"), home_rec, away_rec, proj.get("league_rpg") or 4.5)
        learned = apply_model(model, g["x"]) if model else None
        home_win = learned[0] if learned else g["rules_home"]

        pick_side = "home" if home_win >= 0.5 else "away"
        win_prob = home_win if pick_side == "home" else 1 - home_win
        sign = 1 if pick_side == "home" else -1
        take = []
        if learned:
            ranked = sorted(learned[1].items(), key=lambda kv: kv[1] * sign, reverse=True)
            take = [f"{_game_reason(f, g, pick_side)} <span class='muted'>({_odds_pct(v * sign)})</span>"
                    for f, v in ranked[:3] if v * sign > 0.02]
            if pick_side == "home" and model["intercept"] > 0:
                take.append(f"Home field <span class='muted'>({_odds_pct(model['intercept'])})</span>")
        else:
            if g["away"]["opp_sp_woba"] != g["home"]["opp_sp_woba"]:
                take.append(_game_reason("starter", g, pick_side))
            take.append(_game_reason("offense", g, pick_side))
        if not take:
            take.append("Dead even on paper &mdash; taking the small edge")

        bucket = None
        for b in gfit.get("confidence") or []:
            if win_prob >= b["min_prob"]:
                bucket = b
                break
        model_home = proj.get("home_win_prob")
        out.append({
            "game_pk": game.get("game_pk"),
            "matchup": f"{game.get('away')} @ {game.get('home')}",
            "away": game.get("away"), "home": game.get("home"),
            "venue": game.get("venue"),
            "pick": g[pick_side]["team"], "pick_side": pick_side,
            "win_prob": win_prob,
            "confidence": (bucket or {}).get("label") or (
                "Strong" if win_prob >= 0.60 else "Lean" if win_prob >= 0.54 else "Coin flip"),
            "evidence": bucket,
            "method": "learned" if learned else "rules",
            "away_runs": round(g["away"]["runs"], 2), "home_runs": round(g["home"]["runs"], 2),
            "total": round(g["away"]["runs"] + g["home"]["runs"], 1),
            "park_runs": g["park_runs"],
            "take": take[:4],
            "model_pick_prob": (
                None if model_home is None
                else (model_home if pick_side == "home" else 1 - model_home)
            ),
        })
    return out


# ────────────────────────────────────────────────────────────── the card


def _started(results: Optional[dict]) -> set:
    games = (results or {}).get("games") or {}
    return {int(pk) for pk, g in games.items() if g.get("state") in ("Live", "Final")}


def evidence_summary(fit: Optional[dict]) -> Optional[dict]:
    """The slice of the replay that belongs on a card."""
    if not fit:
        return None
    hr, games, hits = fit.get("hr") or {}, fit.get("games") or {}, fit.get("hits") or {}
    return {
        "fitted_at": fit.get("fitted_at"), "through": fit.get("through"),
        "hits_serve": hits.get("serve"), "hit_scores": hits.get("scores"),
        "hit_picks": hits.get("picks"), "hit_grades": hits.get("grades"),
        "hit_weights": hits.get("weights"), "hit_dropped": hits.get("dropped"),
        "hit_board_rate": hits.get("board_rate"), "parlays": fit.get("parlays"),
        "book": fit.get("book"),
        "hr_serve": hr.get("serve"), "games_serve": games.get("serve"),
        "hr_scores": hr.get("scores"), "hr_picks": hr.get("picks"),
        "hr_weights": hr.get("weights"), "hr_dropped": hr.get("dropped"),
        "grades": hr.get("grades"), "board_rate": hr.get("board_rate"),
        "game_scores": games.get("scores"),
        "game_weights": games.get("weights"), "confidence": games.get("confidence"),
        "house_winners": games.get("house_walk_forward"),
    }


def build_card(slate: dict, book: dict, previous: Optional[dict] = None,
               results: Optional[dict] = None, fit: Optional[dict] = None) -> dict:
    """HOMER's picks for the day.

    Picks in games that have already started are carried over from `previous`
    untouched; only the rest of the board is re-handicapped.
    """
    started = _started(results)
    hr_all = hr_calculations(slate, book, fit)
    hit_all = hit_calculations(slate, book, fit)
    games = game_calculations(slate, book, fit)
    for c in hr_all:
        c["grade"], c["evidence"] = grade_for(c["prob"], fit, "hr")
        c["tag"] = HR_TAGS.get(c["grade"][0], "")
    for c in hit_all:
        c["grade"], c["evidence"] = grade_for(c["prob"], fit, "hits")
        c["tag"] = HIT_TAGS.get(c["grade"][0], "")

    def locked_and_fresh(key: str, pool: list, n: int) -> list:
        locked = [p for p in (previous or {}).get(key, []) if p["game_pk"] in started]
        fresh = select_hr_picks([c for c in pool if c["game_pk"] not in started],
                                n=n - len(locked), skip_games={p["game_pk"] for p in locked})
        return sorted(locked + fresh, key=lambda c: c["prob"], reverse=True)

    hr_picks = locked_and_fresh("hr_picks", hr_all, HR_PICKS)
    hit_picks = locked_and_fresh("hit_picks", hit_all, HIT_PICKS)

    # A play is locked as soon as any of its legs is under way; otherwise it is
    # rebuilt from whatever has not started.
    evidence = (fit or {}).get("parlays") or {}

    def locked_plays(key: str, specs: list, backfill: bool = False) -> list:
        prev_plays = {p["key"]: p for p in (previous or {}).get(key, [])}
        open_hr = [c for c in hr_all if c["game_pk"] not in started]
        open_hit = [c for c in hit_all if c["game_pk"] not in started]
        fresh = {p["key"]: p for p in build_parlays(open_hr, open_hit, plays=specs)}
        slate_games = {g.get("game_pk") for g in slate.get("games", [])}
        # A card first built after every game started (the five-pick plays
        # arriving at season's end) gets them from the morning's numbers,
        # flagged as built after the fact.
        late = {}
        if backfill and slate_games and slate_games <= started:
            late = {p["key"]: dict(p, backfilled=True)
                    for p in build_parlays(hr_all, hit_all, plays=specs)}
        out = []
        for play in specs:
            old = prev_plays.get(play["key"])
            if old and any(leg["game_pk"] in started for leg in old["leg_list"]):
                out.append(old)
            elif play["key"] in fresh:
                out.append(fresh[play["key"]])
            elif play["key"] in late:
                out.append(late[play["key"]])
        for p in out:
            p["evidence"] = play_evidence(evidence, p)
        return out

    parlays = locked_plays("parlays", PARLAY_PLAYS)
    five_picks = locked_plays("five_picks", FIVE_PICK_PLAYS, backfill=True)

    prev_games = {g["game_pk"]: g for g in (previous or {}).get("game_picks", [])}
    game_picks = [
        prev_games[g["game_pk"]] if g["game_pk"] in started and g["game_pk"] in prev_games else g
        for g in games
    ]
    game_picks.sort(key=lambda g: g["win_prob"], reverse=True)
    for g in game_picks:
        g["tag"] = GAME_TAGS.get(g["confidence"], "")

    # Where HOMER and the house model part ways the most.
    compared = [c for c in hr_all if c.get("model_prob") and c["season_pa"] >= MIN_SEASON_PA]
    for c in compared:
        c["edge"] = c["prob"] - c["model_prob"]
    likes = sorted(compared, key=lambda c: c["edge"], reverse=True)[:5]
    fades = sorted(compared, key=lambda c: c["edge"])[:5]

    def slim(c):
        return {k: c[k] for k in ("batter", "batter_id", "team", "game_pk", "matchup",
                                  "prob", "model_prob", "edge", "barrel_rate",
                                  "sp_name", "park", "weather")}

    return {
        "bot": NAME,
        "date": slate.get("date"),
        "built_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "book_through": book["before"],
        "league": {k: book["league"][k] for k in ("pa", "hr_rate", "woba", "games")},
        "method": hr_all[0]["method"] if hr_all else "rules",
        "evidence": evidence_summary(fit),
        "hr_picks": hr_picks,
        "hit_picks": hit_picks,
        "parlays": parlays,
        "five_picks": five_picks,
        "game_picks": game_picks,
        "likes": [slim(c) for c in likes],
        "fades": [slim(c) for c in fades],
        "locked": sorted(started),
    }


# ─────────────────────────────────────────────────────────────── grading


def grade_card(card: dict, results: Optional[dict]) -> dict:
    """Mark each pick hit, miss, pending or no-result, in place."""
    if not results or not results.get("games"):
        return card
    games = {int(pk): g for pk, g in results["games"].items()}
    batters = {int(b): line for b, line in (results.get("batters") or {}).items()}

    hr_tally = {"hit": 0, "scored": 0, "dnp": 0, "pending": 0}
    for p in card.get("hr_picks", []):
        game = games.get(p["game_pk"]) or {}
        line = batter_line(batters, p.get("batter_id"), p["game_pk"])
        final = game.get("state") == "Final"
        if line:
            p["result"] = {
                "hr": line.get("hr", 0), "summary": line.get("summary", ""),
                "final": final, "state": game.get("detailed", ""),
            }
            if final:
                hr_tally["scored"] += 1
                hr_tally["hit"] += 1 if line.get("hr") else 0
            elif line.get("hr"):
                # A home run in a game still going is already a winner.
                hr_tally["scored"] += 1
                hr_tally["hit"] += 1
                p["result"]["final"] = True
            else:
                hr_tally["pending"] += 1
        elif final:
            p["result"] = {"hr": 0, "summary": "Did not play", "final": True,
                           "dnp": True, "state": game.get("detailed", "")}
            hr_tally["dnp"] += 1
        else:
            if game.get("state") == "Live":
                p["result"] = {"hr": 0, "summary": "", "final": False,
                               "state": game.get("detailed", "")}
            hr_tally["pending"] += 1

    game_tally = {"won": 0, "lost": 0, "pending": 0}
    for g in card.get("game_picks", []):
        game = games.get(g["game_pk"]) or {}
        if game.get("state") != "Final":
            game_tally["pending"] += 1
            if game.get("state") == "Live":
                g["result"] = {"final": False, "home_score": game.get("home_score"),
                               "away_score": game.get("away_score"),
                               "state": game.get("detailed", "")}
            continue
        hs, as_ = game.get("home_score"), game.get("away_score")
        if hs is None or as_ is None or hs == as_:
            # Postponed or suspended: no decision.
            g["result"] = {"final": True, "no_decision": True,
                           "state": game.get("detailed", "")}
            continue
        winner = "home" if hs > as_ else "away"
        won = winner == g["pick_side"]
        g["result"] = {"final": True, "won": won, "home_score": hs, "away_score": as_,
                       "total": hs + as_, "state": game.get("detailed", "")}
        game_tally["won" if won else "lost"] += 1

    def leg_status(kind: str, pk: int, bid) -> str:
        """won / lost / void / pending for one leg (a hit or a home run)."""
        game = games.get(pk) or {}
        line = batter_line(batters, bid, pk)
        got = (line or {}).get("hr" if kind == "hr" else "hits", 0)
        if got:
            return "won"  # already in the book, even mid-game
        final = game.get("state") == "Final"
        if final:
            return "lost" if line else "void"
        return "pending"

    hit_tally = {"hit": 0, "scored": 0, "dnp": 0, "pending": 0}
    for p in card.get("hit_picks", []):
        status = leg_status("hit", p["game_pk"], p.get("batter_id"))
        line = batter_line(batters, p.get("batter_id"), p["game_pk"]) or {}
        game = games.get(p["game_pk"]) or {}
        if status == "pending" and game.get("state") != "Live":
            p.pop("result", None)
        else:
            p["result"] = {"hits": line.get("hits", 0), "summary": line.get("summary", ""),
                           "final": status in ("won", "lost", "void"), "dnp": status == "void",
                           "state": game.get("detailed", "")}
        if status in ("won", "lost"):
            hit_tally["scored"] += 1
            hit_tally["hit"] += status == "won"
        else:
            hit_tally["dnp" if status == "void" else "pending"] += 1

    def grade_plays(plays: list) -> dict:
        tally = {"won": 0, "lost": 0, "void": 0, "pending": 0}
        for play in plays:
            statuses = []
            for leg in play["leg_list"]:
                leg["status"] = leg_status(leg["type"], leg["game_pk"], leg.get("batter_id"))
                statuses.append(leg["status"])
            # Sportsbook rules: a leg whose player never batted is voided and
            # the parlay rides on the rest.
            if "lost" in statuses:
                outcome = "lost"
            elif "pending" in statuses:
                outcome = "pending"
            elif "won" in statuses:
                outcome = "won"
            else:
                outcome = "void"
            play["status"] = outcome
            tally[outcome] += 1
        return tally

    parlay_tally = grade_plays(card.get("parlays", []))
    # The 5-pick parlays keep their own tally, apart from HOMER'S PLAYS.
    five_tally = grade_plays(card.get("five_picks", []))

    card["record"] = {
        "hr": hr_tally, "games": game_tally, "hits": hit_tally, "parlays": parlay_tally,
        "five_picks": five_tally,
        "graded_at": results.get("fetched_at"),
        "complete": (hr_tally["pending"] == 0 and game_tally["pending"] == 0
                     and hit_tally["pending"] == 0 and parlay_tally["pending"] == 0
                     and five_tally["pending"] == 0),
    }
    return card


def results_from_slate(slate: dict) -> dict:
    """The results shape `grade_card` wants, rebuilt from a scored slate.

    A saved slate already carries each game's final state and every slate
    hitter's line, and HOMER only ever picks slate hitters, so a past day can
    be graded without another API call.
    """
    games, batters = {}, {}
    for g in slate.get("games", []):
        if g.get("live"):
            games[int(g["game_pk"])] = g["live"]
        for h in g.get("hitters", []):
            line = h.get("result")
            if line and h.get("batter_id") is not None and not line.get("dnp"):
                # A doubleheader lists him in both games: keep each game's line.
                entry = batters.setdefault(int(h["batter_id"]), {"by_game": {}})
                entry["by_game"][int(g["game_pk"])] = dict(line)
                entry.update(dict(line, game_pk=g["game_pk"]))
    return {"games": games, "batters": batters,
            "fetched_at": (slate.get("results") or {}).get("fetched_at")}


# ──────────────────────────────────────────────────────────── persistence


def card_dir(snapshot_dir: Path) -> Path:
    path = Path(snapshot_dir) / "homer"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_card(card: dict, snapshot_dir: Path) -> Path:
    path = card_dir(snapshot_dir) / f"{card['date']}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(card, indent=1, default=str), encoding="utf-8")
    tmp.replace(path)
    return path


def load_card(day: str, snapshot_dir: Path) -> Optional[dict]:
    path = card_dir(snapshot_dir) / f"{day}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[homer.py] could not read {path.name} ({exc})")
        return None


def card_days(snapshot_dir: Path) -> list[str]:
    return sorted(p.stem for p in card_dir(snapshot_dir).glob("*.json"))


def season_record(snapshot_dir: Path, through: Optional[str] = None) -> dict:
    """HOMER's running totals across every saved card, plus a per-day log."""
    total = {"hr_hit": 0, "hr_scored": 0, "hr_expected": 0.0,
             "games_won": 0, "games_lost": 0, "hits_hit": 0, "hits_scored": 0,
             "parlays_won": 0, "parlays_lost": 0, "five_won": 0, "five_lost": 0, "days": []}
    for day in card_days(snapshot_dir):
        if through and day > through:
            continue
        card = load_card(day, snapshot_dir)
        rec = (card or {}).get("record")
        if not rec:
            continue
        hr, gm = rec["hr"], rec["games"]
        expected = sum(
            p["prob"] for p in card["hr_picks"]
            if (p.get("result") or {}).get("final") and not (p.get("result") or {}).get("dnp")
        )
        total["hr_hit"] += hr["hit"]
        total["hr_scored"] += hr["scored"]
        total["hr_expected"] += expected
        total["games_won"] += gm["won"]
        total["games_lost"] += gm["lost"]
        hits = rec.get("hits") or {"hit": 0, "scored": 0}
        plays = rec.get("parlays") or {"won": 0, "lost": 0}
        total["hits_hit"] += hits["hit"]
        total["hits_scored"] += hits["scored"]
        total["parlays_won"] += plays["won"]
        total["parlays_lost"] += plays["lost"]
        five = rec.get("five_picks") or {"won": 0, "lost": 0}
        total["five_won"] += five["won"]
        total["five_lost"] += five["lost"]
        total["days"].append({
            "date": day, "hr_hit": hr["hit"], "hr_scored": hr["scored"],
            "hr_expected": round(expected, 2), "games_won": gm["won"],
            "games_lost": gm["lost"], "hits_hit": hits["hit"], "hits_scored": hits["scored"],
            "parlays_won": plays["won"], "parlays_lost": plays["lost"],
            "complete": rec.get("complete", False),
            "backfilled": bool(card.get("backfilled")),
        })
    total["hr_expected"] = round(total["hr_expected"], 2)
    total["days"].sort(key=lambda d: d["date"], reverse=True)
    return total


if __name__ == "__main__":  # python -m mlb_hr.homer 2026-09-13
    from mlb_hr.results import fetch_results
    from mlb_hr.snapshot import DEFAULT_DIR, load_slate

    target = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()
    out = DEFAULT_DIR if DEFAULT_DIR.exists() else Path.cwd() / "snapshots"
    slate = load_slate(target, out)
    if slate is None:
        raise SystemExit(f"no saved slate for {target}")
    pa = str(DATA_DIR / "season_pa_v2.jsonl")
    book = build_book(pa, date.fromisoformat(target))
    card = build_card(slate, book, fit=load_fit())
    # Built after the fact from the published slate. The book still sees only
    # games before the date, but the card was not public before first pitch,
    # so it is labelled as a backfill wherever it is shown.
    card["backfilled"] = True
    grade_card(card, fetch_results(date.fromisoformat(target),
                                   [g["game_pk"] for g in slate["games"]]))
    save_card(card, out)
    for p in card["hr_picks"]:
        print(f"{p['grade']:>2} {p['prob']:.1%} (model {p['model_prob'] or 0:.1%}) "
              f"{p['batter']} {p.get('result', {}).get('summary', '')}")
    print(json.dumps(card["record"], indent=1))
