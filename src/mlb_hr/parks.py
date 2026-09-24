"""Measured park factors, split by batter handedness.

Replaces a hand-entered table that covered 15 of the 34 venues in the feed and
silently treated the other 22 as neutral -- including Wrigley Field, which is
the most home-run-friendly park in the data.

Method is the standard home-versus-road comparison: a park's factor is the home
run rate in games played there divided by the home run rate in that club's road
games. Using the same team on both sides is what removes the confound -- Yankee
Stadium looks homer-friendly partly because the Yankees bat there 81 times, and
a raw per-park rate cannot tell the ballpark apart from the lineup.

Handedness matters as much as the park: Yankee Stadium's short right field and
Fenway's left field wall push in opposite directions depending on who is
batting, and a single scalar per venue averages that away.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Regression strength, in plate appearances. One season of one park is a small
# sample for an effect this size, so factors are pulled toward neutral.
PARK_PRIOR_PA = 6000.0
PARK_PRIOR_PA_SPLIT = 3500.0   # per-handedness samples are roughly half as big

# Clamps. Real park factors live inside these; anything outside is noise.
PARK_CLAMP = (0.72, 1.35)


@dataclass
class ParkFactor:
    """Measured home-run factor for one venue."""
    venue: str
    factor: float = 1.0
    factor_vs_lhb: float = 1.0
    factor_vs_rhb: float = 1.0
    pa: int = 0
    hr: int = 0
    raw_factor: float = 1.0
    home_team: str = ""
    measured: bool = False

    def for_batter(self, bat_side: str) -> float:
        """Factor for a batter of the given side. Switch hitters get the blend."""
        if bat_side == "L":
            return self.factor_vs_lhb
        if bat_side == "R":
            return self.factor_vs_rhb
        return self.factor


def _regress(rate_home: float, rate_road: float, pa: float, prior_pa: float) -> float:
    """Ratio of two rates, pulled toward 1.0 by sample size."""
    if rate_road <= 0 or rate_home <= 0 or pa <= 0:
        return 1.0
    raw = rate_home / rate_road
    weight = pa / (pa + prior_pa)
    return float(np.clip(1.0 + (raw - 1.0) * weight, *PARK_CLAMP))


def build_park_factors(pa_jsonl_path: str) -> dict[str, ParkFactor]:
    """Measure per-venue home-run factors from the season's plate appearances."""
    # game_pk -> home team, and the venue each game was played in.
    game_home: dict[int, str] = {}
    game_venue: dict[int, str] = {}

    # venue -> [hr, pa] for all batters, and per batter side.
    venue_totals: dict[str, list] = defaultdict(lambda: [0, 0])
    venue_by_side: dict[tuple, list] = defaultdict(lambda: [0, 0])
    # team -> [hr, pa] in games away from their own park.
    road_totals: dict[str, list] = defaultdict(lambda: [0, 0])
    road_by_side: dict[tuple, list] = defaultdict(lambda: [0, 0])

    rows: list[tuple] = []

    with open(pa_jsonl_path) as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            game_pk = record.get("game_pk")
            venue = record.get("venue") or ""
            if not game_pk or not venue:
                continue

            game_venue[game_pk] = venue
            if record.get("is_home"):
                game_home[game_pk] = record.get("batting_team") or ""

            rows.append((
                game_pk,
                venue,
                record.get("batting_team") or "",
                record.get("bat_side") or "R",
                int(record.get("is_hr", 0) or 0),
            ))

    # Each club's home park is where it bats as the home team.
    home_park: dict[str, str] = {}
    park_counts: dict[tuple, int] = defaultdict(int)
    for game_pk, home_team in game_home.items():
        if home_team:
            park_counts[(home_team, game_venue.get(game_pk, ""))] += 1
    for (team, venue), count in sorted(park_counts.items(), key=lambda kv: -kv[1]):
        home_park.setdefault(team, venue)

    for game_pk, venue, batting_team, bat_side, is_hr in rows:
        side = bat_side if bat_side in ("L", "R") else "R"
        venue_totals[venue][0] += is_hr
        venue_totals[venue][1] += 1
        venue_by_side[(venue, side)][0] += is_hr
        venue_by_side[(venue, side)][1] += 1

        # Road exposure is judged from the perspective of the club that owns the
        # park being measured, so accumulate every team's away production.
        home_team = game_home.get(game_pk, "")
        for team in (batting_team, home_team):
            if team and home_park.get(team) and home_park[team] != venue:
                road_totals[team][0] += is_hr
                road_totals[team][1] += 1
                road_by_side[(team, side)][0] += is_hr
                road_by_side[(team, side)][1] += 1
                break

    factors: dict[str, ParkFactor] = {}
    venue_owner = {v: t for t, v in home_park.items()}

    for venue, (hr, pa) in venue_totals.items():
        owner = venue_owner.get(venue, "")
        entry = ParkFactor(venue=venue, pa=pa, hr=hr, home_team=owner)

        road = road_totals.get(owner)
        if owner and road and road[1] > 1000 and pa > 500:
            rate_home = hr / pa
            rate_road = road[0] / road[1]
            entry.raw_factor = (
                rate_home / rate_road if rate_road > 0 else 1.0
            )
            entry.factor = _regress(rate_home, rate_road, pa, PARK_PRIOR_PA)
            entry.measured = True

            for side, attr in (("L", "factor_vs_lhb"), ("R", "factor_vs_rhb")):
                v_hr, v_pa = venue_by_side.get((venue, side), [0, 0])
                r_hr, r_pa = road_by_side.get((owner, side), [0, 0])
                if v_pa > 300 and r_pa > 500 and r_hr > 0:
                    setattr(entry, attr, _regress(
                        v_hr / v_pa, r_hr / r_pa, v_pa, PARK_PRIOR_PA_SPLIT
                    ))
                else:
                    setattr(entry, attr, entry.factor)
        else:
            # Neutral-site or too-thin venues stay at 1.0 rather than inventing
            # a factor from a handful of games.
            entry.factor_vs_lhb = entry.factor_vs_rhb = 1.0

        factors[venue] = entry

    return factors


def summarize(factors: dict[str, ParkFactor], top: int = 5) -> str:
    """One-line-per-park summary, most homer-friendly first."""
    measured = [f for f in factors.values() if f.measured]
    measured.sort(key=lambda f: -f.factor)
    lines = [
        f"[parks.py] measured {len(measured)} of {len(factors)} venues"
    ]
    for f in measured[:top] + measured[-top:]:
        lines.append(
            f"  {f.venue:34} {f.factor:.3f}  "
            f"(L {f.factor_vs_lhb:.3f} / R {f.factor_vs_rhb:.3f})  "
            f"raw {f.raw_factor:.3f}  {f.pa:,} PA"
        )
    return "\n".join(lines)
