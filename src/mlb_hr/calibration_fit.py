"""Refit the per-game HR probability calibration (ensemble.CAL_*).

Served per-game probabilities are compared with outcomes for every starting
hitter's game in the rolling backtest, and a logistic recalibration
logit(p') = a + b * logit(p) is fitted -- once on everything, and once per
month on the months before it, to show the constants hold up forward.

Run after the backtest (it reads the backtest cache):
    python -m mlb_hr.calibration_fit
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from mlb_hr.backtest import weekly_windows
from mlb_hr.pick_fit import DATA_DIR, _read_pa, _served_hr_rates, build_rows


def _fit(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    def loss(w):
        t = w[0] + w[1] * z
        return np.sum(np.logaddexp(0, t) - y * t)
    return minimize(loss, [0.0, 1.0], method="L-BFGS-B").x


def main(pa_path: str = str(DATA_DIR / "season_pa_v2.jsonl")) -> None:
    records = _read_pa(pa_path)
    rows = [r for r in build_rows(records, _served_hr_rates(records, weekly_windows(pa_path)))
            if r["current_hr"] is not None]
    y = np.array([r["hr"] for r in rows], dtype=float)
    p = np.clip(np.array([r["current_hr"] for r in rows]), 1e-4, 1 - 1e-4)
    z = np.log(p / (1 - p))
    months = np.array([r["date"][:7] for r in rows])

    a, b = _fit(z, y)
    print(f"all {len(y):,} hitter-games: CAL_INTERCEPT={a:.3f} CAL_SLOPE={b:.3f}")

    raw = cal = 0.0
    for month in sorted(set(months))[2:]:
        before, now = months < month, months == month
        a_m, b_m = _fit(z[before], y[before])
        q = expit(a_m + b_m * z[now])

        def ll(pp):
            return -np.sum(y[now] * np.log(pp) + (1 - y[now]) * np.log(1 - pp))

        raw, cal = raw + ll(p[now]), cal + ll(q)
        print(f"  {month}: fitted on prior months a={a_m:.3f} b={b_m:.3f}")
    print(f"forward log loss: uncalibrated {raw:.1f}, calibrated {cal:.1f}")

    print("deciles (predicted -> actual, uncalibrated):")
    for chunk in np.array_split(np.argsort(p), 10):
        print(f"  {p[chunk].mean():.3f} -> {y[chunk].mean():.3f}")


if __name__ == "__main__":
    main()
