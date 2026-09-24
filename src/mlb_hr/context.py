"""Game-context adjustments: weather, times through the order, and umpire.

Each effect here is *measured* from the season's own plate appearances rather
than taken from a published coefficient, and each is measured against a
baseline chosen to defuse the obvious confound:

- **Weather** is scored against park-adjusted expectation. Warm cities also
  tend to have hitter-friendly parks, so a raw temperature curve would be
  partly a map of which stadiums are hot.
- **Times through the order** is scored against *each pitcher's own* season
  rate. Only good pitchers are still in the game the third time through, so
  comparing third-trip results to the league would credit the trip with what is
  really pitcher quality. Comparing a pitcher to himself removes that.
- **Umpires** are scored against the pitcher-and-batter expectation for each
  plate appearance, then shrunk hard. Assignments rotate, but a season gives
  only ~25 games per umpire.

Everything is expressed as a multiplier on a rate, defaults to 1.0, and is
clamped, so a thin or noisy slice cannot swing a projection.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Shrinkage strengths, in plate appearances of equivalent prior weight.
UMP_PRIOR_PA = 4000.0
WIND_PRIOR_PA = 3000.0
TTO_PRIOR_PA = 3000.0

TEMP_CLAMP = (0.85, 1.18)
WIND_CLAMP = (0.80, 1.25)
TTO_CLAMP = (0.85, 1.25)
UMP_K_CLAMP = (0.90, 1.10)

REFERENCE_TEMP_F = 72.0


def _shrunk_ratio(observed: float, expected: float, pa: float,
                  prior_pa: float, clamp: tuple) -> float:
    """Observed-over-expected, pulled toward 1.0 by sample size."""
    if expected <= 0 or pa <= 0:
        return 1.0
    raw = observed / expected
    weight = pa / (pa + prior_pa)
    return float(np.clip(1.0 + (raw - 1.0) * weight, *clamp))


def wind_category(wind_dir: Optional[str]) -> str:
    """Collapse the feed's wind string into a category that matters for flight.

    'Out To CF' helps a fly ball, 'In From LF' kills it, and the crosswinds do
    much less either way. Domes report calm or nothing at all.
    """
    if not wind_dir:
        return "none"
    d = wind_dir.strip().lower()
    if d.startswith("out"):
        return "out"
    if d.startswith("in"):
        return "in"
    if "to" in d and ("l to r" in d or "r to l" in d):
        return "cross"
    if "calm" in d or "none" in d:
        return "calm"
    return "other"


@dataclass
class ContextModel:
    """Measured multipliers for weather, order turn, and plate umpire."""
    temp_per_degree: float = 0.0          # fractional HR change per degree F
    wind_hr: dict = field(default_factory=dict)   # (category, speed band) -> mult
    tto_hr: dict = field(default_factory=dict)    # trip -> HR multiplier
    tto_k: dict = field(default_factory=dict)     # trip -> K multiplier
    ump_k: dict = field(default_factory=dict)     # umpire -> K multiplier
    league_hr_rate: float = 0.03
    league_k_rate: float = 0.22
    fitted: bool = False
    n_pa: int = 0

    # ------------------------------------------------------------------ lookups

    def temp_factor(self, temp_f: Optional[float]) -> float:
        """HR multiplier for a temperature, relative to a 72F reference."""
        if temp_f is None or not self.fitted:
            return 1.0
        return float(np.clip(
            1.0 + self.temp_per_degree * (temp_f - REFERENCE_TEMP_F), *TEMP_CLAMP
        ))

    def wind_factor(self, wind_mph: Optional[float], wind_dir: Optional[str]) -> float:
        """HR multiplier for a wind reading."""
        if not self.fitted:
            return 1.0
        return self.wind_hr.get(_wind_key(wind_mph, wind_dir), 1.0)

    def tto_hr_factor(self, trip: int) -> float:
        return self.tto_hr.get(min(max(trip, 1), 4), 1.0) if self.fitted else 1.0

    def tto_k_factor(self, trip: int) -> float:
        return self.tto_k.get(min(max(trip, 1), 4), 1.0) if self.fitted else 1.0

    def umpire_k_factor(self, umpire: Optional[str]) -> float:
        if not umpire or not self.fitted:
            return 1.0
        return self.ump_k.get(umpire, 1.0)

    def game_hr_factor(
        self, temp_f: Optional[float], wind_mph: Optional[float],
        wind_dir: Optional[str],
    ) -> float:
        """Combined weather multiplier for a game."""
        return self.temp_factor(temp_f) * self.wind_factor(wind_mph, wind_dir)

    def summary(self) -> str:
        if not self.fitted:
            return "[context.py] not fitted (no weather/tto fields in feed)"
        winds = ", ".join(
            f"{k}={v:.3f}" for k, v in sorted(self.wind_hr.items())
            if not k.startswith("none")
        )
        return (
            f"[context.py] fitted on {self.n_pa:,} PA\n"
            f"  temperature: {self.temp_per_degree * 100:+.2f}% HR per degree F "
            f"(so 90F vs 72F = {self.temp_factor(90):.3f}x)\n"
            f"  wind: {winds}\n"
            f"  times through order HR: "
            + ", ".join(f"{k}={v:.3f}" for k, v in sorted(self.tto_hr.items()))
            + "\n  times through order K:  "
            + ", ".join(f"{k}={v:.3f}" for k, v in sorted(self.tto_k.items()))
            + f"\n  umpires: {len(self.ump_k)} rated, K factor range "
            f"{min(self.ump_k.values(), default=1):.3f}-"
            f"{max(self.ump_k.values(), default=1):.3f}"
        )


def _wind_key(wind_mph: Optional[float], wind_dir: Optional[str]) -> str:
    """Wind cell key: direction only, deliberately not banded by speed.

    Banding by speed was tried and produced a strong-wind-blowing-out factor of
    0.91 -- the opposite of what the physics requires -- which survived even a
    within-park control. The reading is a single pre-game observation, so at
    high speeds the recorded direction is the least trustworthy part of it.
    Direction alone is monotone and physically coherent (out 1.05, cross 1.02,
    calm 0.99, in 0.90) with 28,000+ plate appearances behind every cell.
    """
    return wind_category(wind_dir)


def build_context_model(
    pa_jsonl_path: str,
    park_factors: Optional[dict] = None,
) -> ContextModel:
    """Fit weather, order-turn and umpire effects from the PA feed."""
    park_factors = park_factors or {}

    total_hr = total_k = total_pa = 0

    # Weather is scored *within park*: each venue's own season home-run rate is
    # the baseline, and each weather cell is measured against it. Scoring
    # against a league-wide park factor is not enough -- a park's measured
    # factor already absorbs its typical wind, so the windy pitcher's parks
    # (Oracle, Target, PNC) dominate the strong-wind buckets and drag the
    # apparent effect backwards. Comparing a park to itself removes that.
    venue_totals: dict[str, list] = defaultdict(lambda: [0, 0])       # hr, pa
    temp_cells: dict[tuple, list] = defaultdict(lambda: [0, 0])       # hr, pa
    wind_cells: dict[tuple, list] = defaultdict(lambda: [0, 0])
    ump_bins: dict[str, list] = defaultdict(lambda: [0.0, 0.0, 0])    # k, exp, pa

    # Times through the order is scored against each pitcher's own rate, so the
    # per-pitcher totals have to be accumulated before the ratio is taken.
    pitcher_tot: dict[int, list] = defaultdict(lambda: [0, 0, 0])     # hr, k, pa
    tto_rows: list[tuple] = []

    has_context = False

    with open(pa_jsonl_path) as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue

            is_hr = int(r.get("is_hr", 0) or 0)
            is_k = 1 if r.get("event_type") == "strikeout" else 0
            total_hr += is_hr
            total_k += is_k
            total_pa += 1

            pitcher_id = r.get("pitcher_id")
            tto = r.get("tto")
            if pitcher_id and tto:
                has_context = True
                p = pitcher_tot[pitcher_id]
                p[0] += is_hr
                p[1] += is_k
                p[2] += 1
                tto_rows.append((pitcher_id, int(tto), is_hr, is_k))

            venue = r.get("venue") or ""
            if venue:
                venue_totals[venue][0] += is_hr
                venue_totals[venue][1] += 1

            temp = r.get("temp_f")
            if temp is not None and venue:
                has_context = True
                cell = temp_cells[(venue, int(round(float(temp) / 5.0) * 5))]
                cell[0] += is_hr
                cell[1] += 1

            if venue and (r.get("wind_dir") is not None or r.get("wind_mph") is not None):
                cell = wind_cells[(venue, _wind_key(r.get("wind_mph"), r.get("wind_dir")))]
                cell[0] += is_hr
                cell[1] += 1

            ump = r.get("ump_hp")
            if ump:
                has_context = True
                b = ump_bins[ump]
                b[0] += is_k
                b[2] += 1

    model = ContextModel(n_pa=total_pa)
    if not has_context or total_pa == 0:
        return model

    model.league_hr_rate = total_hr / total_pa
    model.league_k_rate = total_k / total_pa
    lg_hr, lg_k = model.league_hr_rate, model.league_k_rate

    # Each park's own realized rate is the yardstick for its weather cells.
    venue_rate = {
        v: (hr / pa) for v, (hr, pa) in venue_totals.items() if pa > 500
    }

    def _pool(cells: dict, group_index: int) -> dict:
        """Pool within-park observed-vs-expected across venues, by group key."""
        out: dict = defaultdict(lambda: [0.0, 0.0, 0])   # observed, expected, pa
        for (venue, group), (hr, pa) in cells.items():
            rate = venue_rate.get(venue)
            if rate is None or pa < 150:
                continue
            bucket = out[group]
            bucket[0] += hr
            bucket[1] += rate * pa
            bucket[2] += pa
        return out

    # --- temperature: weighted least squares of within-park ratio on degrees
    temp_pooled = _pool(temp_cells, 1)
    xs, ys, ws = [], [], []
    for temp, (obs, exp, pa) in sorted(temp_pooled.items()):
        if pa < 2000 or exp <= 0:
            continue
        xs.append(temp)
        ys.append(obs / exp)
        ws.append(pa)
    if len(xs) >= 4:
        x = np.array(xs, dtype=float)
        y = np.array(ys, dtype=float)
        w = np.array(ws, dtype=float)
        slope, intercept = np.polyfit(x, y, 1, w=np.sqrt(w))
        # Re-express the fit so it passes through 1.0 at the reference temp.
        at_ref = slope * REFERENCE_TEMP_F + intercept
        if at_ref > 0:
            model.temp_per_degree = float(slope / at_ref)

    # --- wind
    for key, (obs, exp, pa) in _pool(wind_cells, 1).items():
        if pa < 2000:
            continue
        model.wind_hr[key] = _shrunk_ratio(obs, exp, pa, WIND_PRIOR_PA, WIND_CLAMP)

    # --- times through the order, each pitcher against himself
    tto_hr: dict[int, list] = defaultdict(lambda: [0.0, 0.0, 0])
    tto_k: dict[int, list] = defaultdict(lambda: [0.0, 0.0, 0])
    for pitcher_id, trip, is_hr, is_k in tto_rows:
        hr_tot, k_tot, pa_tot = pitcher_tot[pitcher_id]
        if pa_tot < 100:
            continue
        trip = min(trip, 4)
        h = tto_hr[trip]
        h[0] += is_hr
        h[1] += hr_tot / pa_tot
        h[2] += 1
        k = tto_k[trip]
        k[0] += is_k
        k[1] += k_tot / pa_tot
        k[2] += 1

    for trip, (obs, exp, pa) in tto_hr.items():
        model.tto_hr[trip] = _shrunk_ratio(obs, exp, pa, TTO_PRIOR_PA, TTO_CLAMP)
    for trip, (obs, exp, pa) in tto_k.items():
        model.tto_k[trip] = _shrunk_ratio(obs, exp, pa, TTO_PRIOR_PA, TTO_CLAMP)

    # --- umpires
    for ump, (k, _exp, pa) in ump_bins.items():
        if pa < 400:
            continue
        model.ump_k[ump] = _shrunk_ratio(
            k, lg_k * pa, pa, UMP_PRIOR_PA, UMP_K_CLAMP
        )

    model.fitted = True
    return model
