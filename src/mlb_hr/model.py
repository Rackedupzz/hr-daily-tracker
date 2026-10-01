"""HR probability model with per-batter PA exposure and handedness splits.

Fixes the exposure-bias bug in the previous team-games denominator:
- Uses real PA count per batter (not team games × 9 assumptions)
- Applies per-batter L/R splits (not flat 1.10/0.93)
- Shrinks via empirical-Bayes with per-batter K (not one global K)
- Validates via Brier score / log loss on held-out test windows
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import date, timedelta
from typing import Iterator

import numpy as np
import pandas as pd


# Fallback park factors for venues the measured pass cannot resolve.
DEFAULT_PARK_FACTORS = {
    "Coors Field": 1.22,
    "Comerica Park": 0.94,
    "Globe Life Field": 1.15,
    "Citizens Bank Park": 0.99,
    "Great American Ball Park": 1.05,
    "Oriole Park at Camden Yards": 1.03,
    "Oracle Park": 0.78,
    "Petco Park": 0.82,
    "Sutter Health Park": 1.05,
    "Guaranteed Rate Field": 0.99,
    "Minute Maid Park": 1.02,
    "Dodger Stadium": 0.95,
    "Fenway Park": 1.08,
    "Yankee Stadium": 1.06,
    "Tropicana Field": 0.88,
}


@dataclass
class BatterStats:
    """Per-batter, per-handedness PA-based HR stats."""
    batter_id: int
    batter: str
    hand: str  # 'L' or 'R' (batting hand)
    pa_total: int  # plate appearances
    hr_total: int  # home runs
    pa_vs_l: int  # PAs vs LHP
    pa_vs_r: int  # PAs vs RHP
    hr_vs_l: int  # HRs vs LHP
    hr_vs_r: int  # HRs vs RHP
    games_played: int

    def hr_rate_vs(self, pitcher_hand: str) -> float:
        """HR rate vs left or right pitcher. Shrunk toward league average."""
        if pitcher_hand == "L":
            pa, hr = self.pa_vs_l, self.hr_vs_l
        else:
            pa, hr = self.pa_vs_r, self.hr_vs_r

        if pa == 0:
            # Never faced this handedness; use overall rate
            return self.overall_hr_rate()

        # Empirical Bayes shrinkage. K (regularization) scales with PA.
        # Hitters with many PAs (>200) get little shrinkage; low-PA (<20) gets heavy.
        K = max(10, 50 - pa // 5)  # K from 10 to 50, tuned for 0-200 PA range
        lg_rate = 0.034  # ~3.4% of all PAs are HRs across MLB 2026
        shrunk_rate = (hr + lg_rate * K) / (pa + K)
        return np.clip(shrunk_rate, 0, 0.35)  # Cap at 35% to avoid overconfidence

    def overall_hr_rate(self) -> float:
        """Shrunk overall HR rate (all handedness)."""
        if self.pa_total == 0:
            return 0.0
        K = max(10, 50 - self.pa_total // 10)
        lg_rate = 0.034
        return (self.hr_total + lg_rate * K) / (self.pa_total + K)


class HRModel:
    """Builds and queries PA-based HR probabilities."""

    def __init__(self, pa_jsonl_path: str):
        """Load season PA data and compute per-batter splits."""
        self.pa_path = pa_jsonl_path
        self.batters: dict[int, BatterStats] = {}
        self.pitcher_hands: dict[int, str] = {}
        self.park_factors = self._default_park_factors()
        self.park_model: dict = {}
        self._build_from_pa_data()

    def set_park_model(self, park_model: dict) -> None:
        """Install measured, handedness-split park factors.

        The hardcoded table below stays as the fallback for venues the measured
        pass could not resolve (neutral sites, parks with too few games).
        """
        self.park_model = park_model or {}
        for venue, entry in self.park_model.items():
            if getattr(entry, "measured", False):
                self.park_factors[venue] = entry.factor

    def park_factor_for(self, venue: str, bat_side: str = "R") -> float:
        """Park factor for this venue and batter side.

        Handedness is not a refinement here, it is most of the effect: Yankee
        Stadium plays at 1.10 for left-handed hitters and 0.99 for right-handed
        ones, and a single scalar averages those into a number true of neither.
        """
        entry = getattr(self, "park_model", {}).get(venue)
        if entry is not None and getattr(entry, "measured", False):
            return entry.for_batter(bat_side)
        return self.park_factors.get(venue, 1.0)

    def hit_park_factor_for(self, venue: str, bat_side: str = "R") -> float:
        """Hits-per-PA park factor for this venue and batter side (1.0 unmeasured)."""
        entry = getattr(self, "park_model", {}).get(venue)
        if entry is not None and getattr(entry, "measured", False):
            return entry.hits_for_batter(bat_side)
        return 1.0

    def _default_park_factors(self) -> dict[str, float]:
        """Fallback park factors for venues the measured pass cannot resolve."""
        return dict(DEFAULT_PARK_FACTORS)

    def _build_from_pa_data(self) -> None:
        """Read PA JSONL, aggregate to per-batter splits."""
        data = defaultdict(lambda: {
            "name": "", "pa_total": 0, "hr_total": 0,
            "pa_vs_l": 0, "pa_vs_r": 0,
            "hr_vs_l": 0, "hr_vs_r": 0,
            "games": set(), "bat_side": "R", "as_l": 0, "as_r": 0,
        })

        with open(self.pa_path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    pa = json.loads(line)
                except json.JSONDecodeError:
                    continue

                bid = pa.get("batter_id")
                if not bid:
                    continue

                pitch_hand = pa.get("pitch_hand", "R")
                is_hr = pa.get("is_hr", 0)

                batter_data = data[bid]
                batter_data["name"] = pa.get("batter", "")
                if pa.get("bat_side") in ("L", "R", "S"):
                    batter_data["bat_side"] = pa["bat_side"]
                if pa.get("bat_side") == "L":
                    batter_data["as_l"] += 1
                elif pa.get("bat_side") == "R":
                    batter_data["as_r"] += 1
                batter_data["pa_total"] += 1
                batter_data["hr_total"] += is_hr
                batter_data["games"].add((pa.get("date"), pa.get("game_pk")))

                if pitch_hand == "L":
                    batter_data["pa_vs_l"] += 1
                    batter_data["hr_vs_l"] += is_hr
                else:
                    batter_data["pa_vs_r"] += 1
                    batter_data["hr_vs_r"] += is_hr

        # Convert to BatterStats objects
        for bid, d in data.items():
            self.batters[bid] = BatterStats(
                batter_id=bid,
                batter=d["name"],
                hand=_batting_hand(d["as_l"], d["as_r"], d["bat_side"]),
                pa_total=d["pa_total"],
                hr_total=d["hr_total"],
                pa_vs_l=d["pa_vs_l"],
                pa_vs_r=d["pa_vs_r"],
                hr_vs_l=d["hr_vs_l"],
                hr_vs_r=d["hr_vs_r"],
                games_played=len(d["games"]),
            )

    def get_batter(self, batter_id: int) -> BatterStats | None:
        """Retrieve batter stats by ID."""
        return self.batters.get(batter_id)

    def get_batter_by_name(self, name: str) -> BatterStats | None:
        """Retrieve batter stats by name (case-insensitive)."""
        name_lower = name.lower()
        for b in self.batters.values():
            if b.batter.lower() == name_lower:
                return b
        return None

    def pr_hr_today(
        self,
        batter: BatterStats,
        pitcher_id: int,
        venue: str,
        pitcher_hand: str = "R",
    ) -> float:
        """Probability of home run in today's game.

        Factors:
        - Per-batter HR rate vs pitcher handedness (shrunk)
        - Park factor adjustment
        - Pitcher HR-per-9 (simplified: not included here)
        """
        base_rate = batter.hr_rate_vs(pitcher_hand)
        park_adj = self.park_factors.get(venue, 1.0)
        prob = base_rate * park_adj
        return np.clip(prob, 0.001, 0.35)  # Reasonable bounds

    def to_json(self, out_path: str) -> None:
        """Export model stats as JSON for web frontend."""
        records = [asdict(b) for b in self.batters.values()]
        with open(out_path, "w") as fh:
            json.dump(records, fh, indent=1)

    @staticmethod
    def from_json(json_path: str) -> dict[int, BatterStats]:
        """Load pre-computed batter stats from JSON."""
        with open(json_path) as fh:
            records = json.load(fh)
        return {
            r["batter_id"]: BatterStats(**r)
            for r in records
        }


def _batting_hand(as_l: int, as_r: int, last_side: str) -> str:
    """'L', 'R' or 'S' from the sides a hitter has actually batted from.

    The feed records the side of each plate appearance, so a switch hitter's
    last one is just whichever hand the last pitcher threw with.
    """
    from mlb_hr.features import SWITCH_MIN_PA, SWITCH_MIN_SHARE

    low, high = sorted((as_l, as_r))
    if low >= SWITCH_MIN_PA and low / (low + high) >= SWITCH_MIN_SHARE:
        return "S"
    if as_l == as_r:
        return last_side if last_side in ("L", "R") else "R"
    return "L" if as_l > as_r else "R"


def compute_calibration_metrics(
    observed_events: np.ndarray,  # 1D array of 0/1
    predicted_probs: np.ndarray,  # 1D array of probabilities
) -> dict[str, float]:
    """Compute Brier score, log loss, and reliability stats."""
    n = len(observed_events)
    if n == 0:
        return {"brier": 0, "log_loss": 0, "calibration_rmse": 0}

    # Brier score: mean squared error of probability predictions
    brier = np.mean((predicted_probs - observed_events) ** 2)

    # Log loss: cross-entropy
    probs_clipped = np.clip(predicted_probs, 1e-7, 1 - 1e-7)
    log_loss = -np.mean(
        observed_events * np.log(probs_clipped)
        + (1 - observed_events) * np.log(1 - probs_clipped)
    )

    # Calibration: reliability over equal-count bins of the predictions.
    #
    # This used fixed bins across [0, 1], which was the wrong tool for a 3%
    # event. Every prediction lives below 0.15, so nine of the ten bins were
    # always empty and the score collapsed to whatever happened in the handful
    # of predictions above 0.10. A model that never crossed 0.10 scored ~0.002
    # and one that occasionally did scored ~0.07 -- a thirty-fold gap that
    # measured the bin layout rather than the calibration, and it made two
    # perfectly reasonable priors look broken.
    #
    # Quantile bins put every model on the same ten buckets of its own
    # distribution, and weighting by bin population stops a sparse tail bin
    # from dominating the average.
    n_bins = min(10, max(3, n // 50))
    edges = np.unique(np.quantile(predicted_probs, np.linspace(0, 1, n_bins + 1)))
    calibration_errors: list[float] = []
    bin_counts: list[int] = []
    if len(edges) > 1:
        idx = np.clip(
            np.searchsorted(edges, predicted_probs, side="right") - 1,
            0, len(edges) - 2,
        )
        for b in range(len(edges) - 1):
            mask = idx == b
            if mask.sum() > 0:
                calibration_errors.append(
                    abs(observed_events[mask].mean() - predicted_probs[mask].mean())
                )
                bin_counts.append(int(mask.sum()))
    calibration_rmse = (
        float(np.sqrt(np.average(
            np.square(calibration_errors), weights=bin_counts
        )))
        if calibration_errors else 0.0
    )

    # Discrimination. The slate ranks hitters and takes the top six, so its
    # usefulness rests on ordering, not on absolute calibration -- and per-PA
    # Brier/log loss are so dominated by the ~3% base rate that they barely
    # move between models. AUC asks whether home runs were ranked above
    # non-home-runs; top-decile lift asks how much hotter the model's most
    # confident 10% actually ran.
    auc = float("nan")
    lift = float("nan")
    positives = observed_events.sum()
    if 0 < positives < n:
        order = np.argsort(predicted_probs)
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(1, n + 1)
        # Ties must share a rank or AUC is inflated by arbitrary ordering.
        _, inverse, counts = np.unique(
            predicted_probs, return_inverse=True, return_counts=True
        )
        rank_sums = np.zeros(len(counts))
        np.add.at(rank_sums, inverse, ranks)
        ranks = (rank_sums / counts)[inverse]

        n_pos = float(positives)
        n_neg = float(n - positives)
        auc = float((ranks[observed_events == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

        top_n = max(1, int(round(n * 0.10)))
        top_idx = np.argsort(-predicted_probs)[:top_n]
        base_rate = float(observed_events.mean())
        if base_rate > 0:
            lift = float(observed_events[top_idx].mean() / base_rate)

    return {
        "brier": float(brier),
        "log_loss": float(log_loss),
        "calibration_rmse": float(calibration_rmse),
        "auc": auc,
        "top_decile_lift": lift,
    }
