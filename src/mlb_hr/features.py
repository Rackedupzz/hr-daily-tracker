"""Batter profile + feature matrix construction.

The unit of observation here is **one batter**, not one plate appearance. That
is the central fix: the previous ensemble replicated a batter-level aggregate
across every one of that batter's PA rows, producing thousands of duplicate
points with zero within-batter variance (see ensemble.py docstring).

Features describe *how hard and at what angle a batter hits the ball* --
quality-of-contact metrics that cause home runs -- plus physical attributes
from ESPN. Home-run counts are deliberately excluded from the feature vector:
they are the prediction target, and feeding them back in is what let the old
model "predict" a rate it had already been told.

Profiles accept a date window so the model can be fit on one slice of the
season and scored on a later, unseen slice.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Quality-of-contact thresholds (standard Statcast conventions).
HARD_HIT_EV = 95.0          # mph
BARREL_MIN_EV = 98.0        # mph
BARREL_LA_RANGE = (24.0, 34.0)   # degrees
FLY_BALL_LA_RANGE = (25.0, 50.0)
# Degrees toward the batter's pull side that count as a pulled ball. Home runs
# average +19.8 degrees in this feed, with 70% beyond this threshold.
PULL_ANGLE_MIN = 15.0

# The feed records the side a hitter batted from in each plate appearance, so a
# switch hitter shows up as a left-handed hitter against right-handers and a
# right-handed one against left-handers. Keeping whichever side he used last
# (as the profile did) gave him the same-hand platoon penalty about half the
# time -- the one matchup he never actually has. 57 hitters in 2026 batted 20+
# times from each side; a hitter is treated as a switch hitter once his
# minority side reaches both floors below.
SWITCH_MIN_PA = 10
SWITCH_MIN_SHARE = 0.10

# Expected home runs (xHR): every ball in play with a tracked exit velocity and
# launch angle is scored at the league's home-run rate for its bin, the same
# 3 mph x 4 degree grid HOMER uses. It measures power without the luck of which
# fly balls cleared the wall, and on the 2026 replay it carried more weight
# than a hitter's actual home runs in the stacked model (ensemble.HR_STACK).
XHR_EV_LO, XHR_EV_STEP, XHR_EV_BINS = 60.0, 3.0, 22
XHR_LA_LO, XHR_LA_STEP, XHR_LA_BINS = -20.0, 4.0, 20
HIT_EVENT_TYPES = frozenset({"single", "double", "triple", "home_run"})


def contact_bin(ev: Optional[float], la: Optional[float]) -> Optional[int]:
    """Exit velocity x launch angle cell for one batted ball, or None."""
    if ev is None or la is None:
        return None
    e = min(max(int((float(ev) - XHR_EV_LO) // XHR_EV_STEP), 0), XHR_EV_BINS - 1)
    a = min(max(int((float(la) - XHR_LA_LO) // XHR_LA_STEP), 0), XHR_LA_BINS - 1)
    return e * XHR_LA_BINS + a

FEATURE_NAMES = [
    "log_pa",
    "avg_ev",
    "ev90",
    "hard_hit_rate",
    "barrel_rate",
    "fly_ball_rate",
    "avg_la",
    "bip_rate",
    "k_rate",
    "bb_rate",
    "xbh_rate",
    "pull_rate",
    "height_in",
    "weight_lb",
    "age",
]

# Fallbacks when a batter has no balls in play or no ESPN bio match.
#
# Tried and rejected 2026-09-12: shrinking thin-sample rate features toward
# these means (40 BIP / 60 PA of prior weight). It was meant to stop call-ups
# like Yohandy Morales (19 PA) reading as elite comparables, but on the rolling
# backtest it lowered core AUC 0.5882 -> 0.5863 and thin-sample AUC
# 0.5642 -> 0.5565, both intervals excluding zero. Early contact quality is
# noisy but still informative; the raw rates stay.
_LEAGUE_DEFAULTS = {
    "avg_ev": 88.0, "ev90": 103.0, "hard_hit_rate": 0.35, "barrel_rate": 0.06,
    "fly_ball_rate": 0.22, "avg_la": 12.0, "pull_rate": 0.40, "height_in": 73.0,
    "weight_lb": 205.0, "age": 28.0,
}


@dataclass
class BatterProfile:
    """Accumulated batting profile for one batter over a date window."""
    batter_id: int
    name: str = ""
    bat_side: str = "R"

    pa: int = 0
    hr: int = 0
    pa_vs_l: int = 0
    hr_vs_l: int = 0
    pa_vs_r: int = 0
    hr_vs_r: int = 0

    strikeouts: int = 0
    walks: int = 0
    xbh: int = 0            # doubles + triples (home runs excluded)
    hits: int = 0           # singles + doubles + triples + home runs

    # Recency-weighted counts. Equal weighting across a whole season hides real
    # changes -- a swing change, lost velocity, a return from injury. These are
    # exponentially decayed toward a reference date and used as the observed
    # term in the empirical-Bayes posterior. They are *weights*, not events, so
    # they are floats.
    w_pa: float = 0.0
    w_hr: float = 0.0
    w_pa_vs_l: float = 0.0
    w_hr_vs_l: float = 0.0
    w_pa_vs_r: float = 0.0
    w_hr_vs_r: float = 0.0

    bip: int = 0            # balls in play with tracked exit velocity
    hard_hit: int = 0
    barrels: int = 0
    fly_balls: int = 0
    pulled: int = 0        # batted balls hit to the batter's pull side
    spray_tracked: int = 0

    # Plate appearances by the side he actually batted from (see SWITCH_MIN_PA).
    pa_as_l: int = 0
    pa_as_r: int = 0

    # Expected home runs (raw and recency-weighted) and expected hits, from the
    # league's rates per contact bin; set by build_profiles once those are known.
    xhr: float = 0.0
    xhr_w: float = 0.0
    xhit: float = 0.0
    _bins: dict = field(default_factory=dict, repr=False)   # bin -> [count, weight]
    _ev_values: list = field(default_factory=list, repr=False)
    _la_sum: float = 0.0
    _games: set = field(default_factory=set, repr=False)

    @property
    def games_played(self) -> int:
        return len(self._games)

    @property
    def bats(self) -> str:
        """'L', 'R', or 'S' for a switch hitter -- his batting hand, not the
        side of whichever plate appearance happened to come last."""
        low, high = sorted((self.pa_as_l, self.pa_as_r))
        if low >= SWITCH_MIN_PA and low / (low + high) >= SWITCH_MIN_SHARE:
            return "S"
        if self.pa_as_l == self.pa_as_r:
            return self.bat_side if self.bat_side in ("L", "R") else "R"
        return "L" if self.pa_as_l > self.pa_as_r else "R"

    def side_vs(self, pitcher_hand: str) -> str:
        """The side he bats from against a pitcher of this hand."""
        bats = self.bats
        if bats == "S":
            return "R" if pitcher_hand == "L" else "L"
        return bats

    @property
    def pa_per_game(self) -> float:
        """Average plate appearances per game started, used to convert a
        per-PA rate into a per-game probability."""
        if not self._games:
            return 0.0
        return self.pa / len(self._games)

    @property
    def hr_rate(self) -> float:
        return self.hr / self.pa if self.pa else 0.0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.pa if self.pa else 0.0

    @property
    def pull_rate(self) -> float:
        """Share of tracked batted balls hit to the pull side.

        Home runs are directional: a pull-heavy left-handed hitter and a spray
        hitter with identical exit velocity are different propositions in a park
        with a short right field, and without this the comparables model cannot
        tell them apart.
        """
        if self.spray_tracked < 20:
            return _LEAGUE_DEFAULTS["pull_rate"]
        return self.pulled / self.spray_tracked

    def observed(self, pitcher_hand: str) -> tuple[int, int]:
        """(home runs, plate appearances) against the given pitcher hand."""
        if pitcher_hand == "L":
            return self.hr_vs_l, self.pa_vs_l
        return self.hr_vs_r, self.pa_vs_r

    def observed_weighted(self, pitcher_hand: str) -> tuple[float, float]:
        """Recency-weighted (home runs, plate appearances) vs a pitcher hand.

        Falls back to raw counts when no decay was applied, so callers do not
        need to know which mode the profiles were built in.
        """
        if self.w_pa <= 0:
            if pitcher_hand == "ALL":
                return float(self.hr), float(self.pa)
            return tuple(float(x) for x in self.observed(pitcher_hand))
        if pitcher_hand == "ALL":
            return self.w_hr, self.w_pa
        if pitcher_hand == "L":
            return self.w_hr_vs_l, self.w_pa_vs_l
        return self.w_hr_vs_r, self.w_pa_vs_r

    def add_pa(self, record: dict, weight: float = 1.0) -> None:
        """Fold one plate-appearance record into the profile."""
        self.pa += 1
        self.name = record.get("batter", self.name)
        side = record.get("bat_side")
        if side in ("L", "R", "S"):
            self.bat_side = side
        if side == "L":
            self.pa_as_l += 1
        elif side == "R":
            self.pa_as_r += 1
        self._games.add(record.get("game_pk"))

        is_hr = int(record.get("is_hr", 0) or 0)
        self.hr += is_hr
        self.w_pa += weight
        self.w_hr += is_hr * weight
        if record.get("pitch_hand") == "L":
            self.pa_vs_l += 1
            self.hr_vs_l += is_hr
            self.w_pa_vs_l += weight
            self.w_hr_vs_l += is_hr * weight
        else:
            self.pa_vs_r += 1
            self.hr_vs_r += is_hr
            self.w_pa_vs_r += weight
            self.w_hr_vs_r += is_hr * weight

        event = record.get("event_type") or ""
        if event in ("single", "double", "triple", "home_run"):
            self.hits += 1
        if event == "strikeout":
            self.strikeouts += 1
        elif event == "walk":
            self.walks += 1
        elif event in ("double", "triple"):
            self.xbh += 1

        ev = record.get("launch_speed")
        la = record.get("launch_angle")
        if ev is None:
            return
        cell = contact_bin(ev, la)
        if cell is not None:
            tally = self._bins.get(cell)
            if tally is None:
                self._bins[cell] = [1, weight]
            else:
                tally[0] += 1
                tally[1] += weight
        self.bip += 1
        self._ev_values.append(float(ev))
        pull_angle = record.get("pull_angle")
        if pull_angle is not None:
            self.spray_tracked += 1
            if pull_angle > PULL_ANGLE_MIN:
                self.pulled += 1
        if ev >= HARD_HIT_EV:
            self.hard_hit += 1
        if la is not None:
            self._la_sum += float(la)
            if ev >= BARREL_MIN_EV and BARREL_LA_RANGE[0] <= la <= BARREL_LA_RANGE[1]:
                self.barrels += 1
            if FLY_BALL_LA_RANGE[0] <= la <= FLY_BALL_LA_RANGE[1]:
                self.fly_balls += 1

    def feature_vector(self, bio=None) -> np.ndarray:
        """Build the KNN/SVR feature vector. Never includes home-run counts."""
        d = _LEAGUE_DEFAULTS
        if self.bip > 0:
            avg_ev = float(np.mean(self._ev_values))
            ev90 = float(np.percentile(self._ev_values, 90))
            hard_hit_rate = self.hard_hit / self.bip
            barrel_rate = self.barrels / self.bip
            fly_ball_rate = self.fly_balls / self.bip
            avg_la = self._la_sum / self.bip
        else:
            avg_ev, ev90 = d["avg_ev"], d["ev90"]
            hard_hit_rate, barrel_rate = d["hard_hit_rate"], d["barrel_rate"]
            fly_ball_rate, avg_la = d["fly_ball_rate"], d["avg_la"]

        pa = max(self.pa, 1)
        height = _or_default(getattr(bio, "height_in", None), d["height_in"])
        weight = _or_default(getattr(bio, "weight_lb", None), d["weight_lb"])
        age = _or_default(getattr(bio, "age", None), d["age"])

        return np.array([
            math.log1p(self.pa),
            avg_ev,
            ev90,
            hard_hit_rate,
            barrel_rate,
            fly_ball_rate,
            avg_la,
            self.bip / pa,
            self.strikeouts / pa,
            self.walks / pa,
            self.xbh / pa,
            self.pull_rate,
            height,
            weight,
            age,
        ], dtype=np.float64)


def _or_default(value: Optional[float], default: float) -> float:
    return float(value) if value is not None else default


def build_profiles(
    pa_jsonl_path: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    half_life_days: Optional[float] = None,
    ref_date: Optional[str] = None,
) -> dict[int, BatterProfile]:
    """Aggregate PA records into per-batter profiles within a date window.

    Args:
        pa_jsonl_path: season PA JSONL.
        start_date: inclusive ISO date lower bound, or None for no bound.
        end_date: exclusive ISO date upper bound, or None for no bound.
        half_life_days: if set, weight each PA by 0.5 ** (age / half_life) so
            recent form counts for more. The reference point is ref_date, which
            during validation must be the cutoff rather than today -- otherwise
            the weighting would peek at the scoring window.
        ref_date: ISO date that decay is measured back from. Defaults to
            end_date when a window is given, else the latest date seen.
    """
    profiles: dict[int, BatterProfile] = {}
    anchor = _parse_date(ref_date) or _parse_date(end_date)
    pending: list = [] if (half_life_days and anchor is None) else None
    # League home runs and hits per contact bin, for expected HR and hits.
    bin_n: dict[int, int] = {}
    bin_hr: dict[int, int] = {}
    bin_hit: dict[int, int] = {}
    with open(pa_jsonl_path) as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            batter_id = record.get("batter_id")
            if not batter_id:
                continue

            game_date = record.get("date") or ""
            if start_date and game_date < start_date:
                continue
            if end_date and game_date >= end_date:
                continue

            cell = contact_bin(record.get("launch_speed"), record.get("launch_angle"))
            if cell is not None:
                bin_n[cell] = bin_n.get(cell, 0) + 1
                bin_hr[cell] = bin_hr.get(cell, 0) + int(record.get("is_hr", 0) or 0)
                bin_hit[cell] = bin_hit.get(cell, 0) + (
                    (record.get("event_type") or "") in HIT_EVENT_TYPES
                )

            if pending is not None:
                # No anchor supplied: hold records so decay can be measured
                # from the latest date actually present in the window.
                pending.append((game_date, record))
                continue

            profile = profiles.get(batter_id)
            if profile is None:
                profile = BatterProfile(batter_id=batter_id)
                profiles[batter_id] = profile
            profile.add_pa(record, _decay_weight(game_date, anchor, half_life_days))

    if pending is not None:
        anchor = max((_parse_date(d) for d, _ in pending if d), default=None)
        for game_date, record in pending:
            batter_id = record["batter_id"]
            profile = profiles.get(batter_id)
            if profile is None:
                profile = BatterProfile(batter_id=batter_id)
                profiles[batter_id] = profile
            profile.add_pa(record, _decay_weight(game_date, anchor, half_life_days))

    settle_expected_contact(profiles, bin_n, bin_hr, bin_hit)
    return profiles


def settle_expected_contact(
    profiles: dict[int, BatterProfile], bin_n: dict[int, int],
    bin_hr: dict[int, int], bin_hit: dict[int, int],
) -> None:
    """Score every profile's batted balls at the league's HR and hit rates per bin.

    Each bin's rate is pulled lightly (two balls' worth) toward the league's
    overall rate on contact, so a sparse corner of the grid cannot hand a
    hitter a full home run for one ball.
    """
    total = sum(bin_n.values())
    if not total:
        return
    lg_hr = sum(bin_hr.values()) / total
    lg_hit = sum(bin_hit.values()) / total
    hr_table = {k: (bin_hr.get(k, 0) + 2 * lg_hr) / (n + 2) for k, n in bin_n.items()}
    hit_table = {k: (bin_hit.get(k, 0) + 2 * lg_hit) / (n + 2) for k, n in bin_n.items()}
    for profile in profiles.values():
        xhr = xhr_w = xhit = 0.0
        for cell, (count, weight) in profile._bins.items():
            rate = hr_table.get(cell, lg_hr)
            xhr += count * rate
            xhr_w += weight * rate
            xhit += count * hit_table.get(cell, lg_hit)
        profile.xhr, profile.xhr_w, profile.xhit = xhr, xhr_w, xhit


def _parse_date(value: Optional[str]):
    from datetime import date as _d
    if not value:
        return None
    try:
        return _d.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def _decay_weight(game_date: str, anchor, half_life_days: Optional[float]) -> float:
    """Exponential recency weight for one plate appearance."""
    if not half_life_days or anchor is None:
        return 1.0
    played = _parse_date(game_date)
    if played is None:
        return 1.0
    age = (anchor - played).days
    if age <= 0:
        return 1.0
    return float(0.5 ** (age / half_life_days))


def season_bounds(pa_jsonl_path: str, sample_every: int = 25) -> tuple[str, str]:
    """Min and max game date in the file, sampled for speed."""
    lo, hi = "9999", "0000"
    with open(pa_jsonl_path) as fh:
        for i, line in enumerate(fh):
            if i % sample_every or not line.strip():
                continue
            try:
                d = json.loads(line).get("date") or ""
            except json.JSONDecodeError:
                continue
            if d:
                lo, hi = min(lo, d), max(hi, d)
    return lo, hi


def build_matrix(
    profiles: dict[int, BatterProfile],
    espn=None,
    min_pa: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Assemble the batter-level design matrix.

    Returns:
        X: (n_batters, n_features) feature matrix
        hr: (n_batters,) home-run counts
        pa: (n_batters,) plate-appearance counts
        ids: batter ids, row-aligned with X
    """
    rows, hrs, pas, ids = [], [], [], []
    for batter_id, profile in profiles.items():
        if profile.pa < min_pa:
            continue
        bio = espn.bio_for(profile.name) if espn else None
        rows.append(profile.feature_vector(bio))
        hrs.append(profile.hr)
        pas.append(profile.pa)
        ids.append(batter_id)

    if not rows:
        n = len(FEATURE_NAMES)
        return np.empty((0, n)), np.empty(0), np.empty(0), []

    return (
        np.vstack(rows),
        np.array(hrs, dtype=np.float64),
        np.array(pas, dtype=np.float64),
        ids,
    )


def platoon_factors(profiles: dict[int, BatterProfile]) -> dict[tuple[str, str], float]:
    """League platoon multipliers, measured rather than assumed.

    Returns a multiplier per (bat_side, pitcher_hand) giving that matchup's HR
    rate relative to the overall rate for batters of that side. Replaces the
    previous hardcoded 1.10 / 0.93 constants.

    Hitters are grouped by batting hand (BatterProfile.bats), not by the side of
    their last plate appearance: that used to drop each switch hitter into the
    left- or right-handed bucket more or less at random, carrying his
    opposite-hand plate appearances into the same-hand cell (LHB vs LHP read
    0.81 instead of 0.76). Switch hitters themselves get 1.0 both ways -- they
    always have the platoon advantage, and their own measured split (x1.15 vs
    LHP, x0.93 vs RHP) rests on ~60 hitters and did not hold up forward: on the
    2026 replay it overstated what they actually did (10.2% vs 9.5% HR games).
    """
    totals: dict[tuple[str, str], list[int]] = {}
    by_side: dict[str, list[int]] = {}

    for profile in profiles.values():
        side = profile.bats
        if side == "S":
            continue
        for hand, (hr, pa) in (("L", profile.observed("L")), ("R", profile.observed("R"))):
            bucket = totals.setdefault((side, hand), [0, 0])
            bucket[0] += hr
            bucket[1] += pa
            side_bucket = by_side.setdefault(side, [0, 0])
            side_bucket[0] += hr
            side_bucket[1] += pa

    factors: dict[tuple[str, str], float] = {}
    for key, (hr, pa) in totals.items():
        side = key[0]
        side_hr, side_pa = by_side.get(side, [0, 0])
        if pa < 500 or side_pa == 0 or side_hr == 0:
            factors[key] = 1.0
            continue
        matchup_rate = hr / pa
        side_rate = side_hr / side_pa
        factors[key] = float(np.clip(matchup_rate / side_rate, 0.75, 1.35))
    factors[("S", "L")] = factors[("S", "R")] = 1.0
    return factors


# Plate appearances per game for the hitter who *starts* in each lineup slot,
# measured across the 2026 season's 4,858 team-games. Exposure is the
# multiplier on every per-game projection, and the spread from leadoff to ninth
# is 30% -- larger than most of the rate effects the model works hard to
# estimate.
#
# This used to count every plate appearance taken in the slot (4.637 ... 3.753),
# which includes the pinch hitters and defensive replacements who bat in it
# once the starter is gone. The starter himself gets 3% fewer at the top of the
# order and 8% fewer at the bottom, where the late-game substitutions happen,
# so every projection was high and the bottom of the order most of all: before
# the fix, 9th hitters homered 18% less often than projected and the hit
# projection ran 3.4% over the season total.
SLOT_PA = {
    1: 4.491, 2: 4.402, 3: 4.303, 4: 4.197, 5: 4.059,
    6: 3.901, 7: 3.759, 8: 3.590, 9: 3.444,
}


def recent_starts(
    pa_jsonl_path: str, before_date: str, team_games: int = 7
) -> dict[int, int]:
    """Lineup starts per batter across his team's last `team_games` games.

    Before lineups are posted the whole active roster competes for a pick, and
    a bench bat with a good rate can win a slot he will never play in. How
    often a hitter has actually started lately is the best morning evidence of
    whether he will start today.
    """
    return recent_lineups(pa_jsonl_path, before_date, team_games)[0]


def recent_lineups(
    pa_jsonl_path: str, before_date: str, team_games: int = 7, slot_starts: int = 10
) -> tuple[dict[int, int], dict[int, int]]:
    """(starts in his team's last `team_games` games, usual lineup slot).

    A start is the first hitter to bat in a lineup slot. The feed gives a
    pinch hitter or defensive replacement the slot he entered, so counting
    every hitter with a slot credited bench players with starts.

    The usual slot is the most common one across his last `slot_starts` starts
    (the most recent wins a tie). It stands in for the batting order until
    lineups post, so a regular's exposure is his slot's, not a per-game average
    dragged down by the odd pinch-hitting appearance.
    """
    team_dates: dict[str, set] = {}
    # (game, team, slot) -> (pa_index, batter, date) of its first hitter
    first: dict[tuple, tuple] = {}
    with open(pa_jsonl_path) as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            day, team = r.get("date") or "", r.get("batting_team") or ""
            if not day or day >= before_date or not team:
                continue
            team_dates.setdefault(team, set()).add(day)
            slot, batter = r.get("lineup_slot"), r.get("batter_id")
            if slot and batter:
                key = (r.get("game_pk"), team, slot)
                index = r.get("pa_index") or 0
                if key not in first or index < first[key][0]:
                    first[key] = (index, batter, day)

    starts: dict[tuple, set] = {}   # (team, batter) -> game dates started
    history: dict[int, list] = {}   # batter -> [(date, game, slot)]
    for (game_pk, team, slot), (_index, batter, day) in first.items():
        starts.setdefault((team, batter), set()).add(day)
        history.setdefault(batter, []).append((day, game_pk, slot))

    recent = {t: set(sorted(d)[-team_games:]) for t, d in team_dates.items()}
    counts: dict[int, int] = {}
    for (team, batter), days in starts.items():
        n = len(days & recent[team])
        counts[batter] = max(counts.get(batter, 0), n)

    usual: dict[int, int] = {}
    for batter, games in history.items():
        games.sort()
        last = [slot for _day, _game, slot in games[-slot_starts:]]
        tally: dict[int, int] = {}
        for slot in last:
            tally[slot] = tally.get(slot, 0) + 1
        top = max(tally.values())
        usual[batter] = next(s for s in reversed(last) if tally[s] == top)
    return counts, usual


def expected_pa_for_slot(slot: Optional[int], fallback: float) -> float:
    """Expected plate appearances for the starter in a lineup slot.

    Falls back to `fallback` when there is no slot at all -- no posted lineup
    and no recent start to take a usual slot from.
    """
    if slot and slot in SLOT_PA:
        return SLOT_PA[slot]
    return fallback
