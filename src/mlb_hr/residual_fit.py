"""Backtest residual correction: learn where the served model was wrong, forward.

The rolling backtest leaves behind, for every starting hitter's game from
mid-April on, what the served model predicted and what actually happened. The
gap between them is the model's residual. Most of it is noise; the question
is whether any of it is *systematic* -- the model underrating hitters facing
home-run-prone starters, say, or overrating them in cold weather -- because a
systematic miss can be corrected.

The algorithm reverse-engineers those misses:

    logit(p_corrected) = logit(p_served) + b0 + sum_j  w_j * x_j

The served prediction enters as a fixed offset, so the model it produces can
only *adjust* the served model, never replace it. The x_j are context the
served per-PA rate does not already carry (opposing starter, park, weather,
batting-order exposure, platoon, sample size) plus the served log-odds itself,
which lets it fix over- or under-confidence. The weights are L2-penalised
toward zero -- no correction -- and the penalty is chosen on the most recent
four weeks of the training window, so a pattern has to have held up forward
once already before it is allowed to move a projection.

Every correction is fitted only on backtest weeks before the one it is applied
to, and scored on the slate's own metric: of each day's top six, how many
connected. The gain over the served model gets a 90% interval from resampling
whole weeks, so a lucky week cannot pass for an improvement.

Run: python -m mlb_hr.residual_fit   (needs the backtest cache from mlb_hr.backtest)
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import date as _date, timedelta
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from mlb_hr.backtest import weekly_windows
from mlb_hr.pick_fit import (
    DATA_DIR, TOP_N, _read_pa, _scores, _served_hr_rates, _top_n_rate, build_rows,
)

LAMBDAS = (1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0)
VALIDATION_DAYS = 28


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def features(rows: list[dict], target: str) -> tuple[np.ndarray, list[str]]:
    """Context the served rate does not already account for, as log ratios."""
    rate = "hr" if target == "hr" else "hit"
    lg = np.array([r[f"lg_{rate}"] for r in rows])
    cols = {
        "opposing starter": np.log(np.array([r[f"sp_{rate}_rate"] for r in rows]) / lg),
        "park to date": np.log(np.array([r[f"park_{rate}"] for r in rows]) / lg),
        "temperature": np.array([r["temp"] for r in rows]) - 72.0,
        "lineup slot": np.array([r["lineup_slot"] for r in rows], dtype=float),
        "same-hand matchup": np.array([r["same_hand"] for r in rows], dtype=float),
        "home team": np.array([r["is_home"] for r in rows], dtype=float),
        "sample size (log PA)": np.log1p([r["bat_pa"] for r in rows]),
        "served confidence": _logit(np.array([r[f"current_{target}"] for r in rows])),
    }
    return np.column_stack(list(cols.values())), list(cols)


class ResidualCorrection:
    """Penalised logistic regression with the served log-odds as an offset."""

    def __init__(self, lam: float):
        self.lam = lam

    def fit(self, X: np.ndarray, y: np.ndarray, served: np.ndarray) -> "ResidualCorrection":
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = np.column_stack([np.ones(len(X)), (X - self.mu) / self.sd])
        offset = _logit(served)

        def loss(w):
            z = offset + Z @ w
            value = np.sum(np.logaddexp(0.0, z) - y * z) + 0.5 * self.lam * w[1:] @ w[1:]
            grad = Z.T @ (expit(z) - y)
            grad[1:] += self.lam * w[1:]
            return value, grad

        self.w = minimize(loss, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B").x
        return self

    def predict(self, X: np.ndarray, served: np.ndarray) -> np.ndarray:
        Z = np.column_stack([np.ones(len(X)), (X - self.mu) / self.sd])
        return expit(_logit(served) + Z @ self.w)


def _loglik(y, p) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_forward(rows: list[dict], target: str) -> ResidualCorrection | None:
    """Choose the penalty on the last four weeks, then refit on everything."""
    dates = sorted({r["date"] for r in rows})
    if len(dates) < VALIDATION_DAYS + 14:
        return None
    split = str(_date.fromisoformat(dates[-1]) - timedelta(days=VALIDATION_DAYS))
    early = [r for r in rows if r["date"] <= split]
    late = [r for r in rows if r["date"] > split]
    y_e = np.array([r[target] for r in early], dtype=float)
    y_l = np.array([r[target] for r in late], dtype=float)
    X_e, _ = features(early, target)
    X_l, _ = features(late, target)
    s_e = np.array([r[f"current_{target}"] for r in early])
    s_l = np.array([r[f"current_{target}"] for r in late])

    # The served model itself is the bar: a penalty only wins if it beats
    # making no correction at all on the held-back weeks.
    best_lam, best = None, _loglik(y_l, s_l)
    for lam in LAMBDAS:
        ll = _loglik(y_l, ResidualCorrection(lam).fit(X_e, y_e, s_e).predict(X_l, s_l))
        if ll > best:
            best_lam, best = lam, ll
    if best_lam is None:
        return None

    X, _ = features(rows, target)
    y = np.array([r[target] for r in rows], dtype=float)
    s = np.array([r[f"current_{target}"] for r in rows])
    return ResidualCorrection(best_lam).fit(X, y, s)


def walk_forward(rows: list[dict], windows, target: str) -> dict:
    scored = [r for r in rows if r[f"current_{target}"] is not None]
    test_rows, served, corrected, week_of = [], [], [], []
    weights, lams = [], []
    for k, (cutoff, end) in enumerate(windows):
        train = [r for r in scored if r["date"] < cutoff]
        test = [r for r in scored if cutoff <= r["date"] < end]
        if not test:
            continue
        s = np.array([r[f"current_{target}"] for r in test])
        model = fit_forward(train, target) if train else None
        if model is None:
            p = s  # nothing learned yet, or nothing that held up: no correction
        else:
            p = model.predict(features(test, target)[0], s)
            weights.append(model.w[1:])
            lams.append(model.lam)
        test_rows.extend(test)
        served.extend(s)
        corrected.extend(p)
        week_of.extend([k] * len(test))
    return {
        "rows": test_rows,
        "served": np.array(served),
        "corrected": np.array(corrected),
        "week": np.array(week_of),
        "weights": np.array(weights),
        "lams": lams,
    }


def bootstrap_gain(wf: dict, target: str, n_boot: int = 300, seed: int = 0) -> dict:
    """Corrected minus served, top-6 rate and AUC, resampling whole weeks."""
    rng = np.random.default_rng(seed)
    rows, week = wf["rows"], wf["week"]
    y = np.array([r[target] for r in rows], dtype=float)
    weeks = np.unique(week)
    members = {w: np.flatnonzero(week == w) for w in weeks}
    top, auc = [], []
    for _ in range(n_boot):
        draw = rng.choice(weeks, len(weeks))
        idx = np.concatenate([members[w] for w in draw])
        copy = np.concatenate([[c] * len(members[w]) for c, w in enumerate(draw)])
        # A week drawn twice repeats its days; tag each copy so the top six is
        # taken per day per copy, not from a doubled pool.
        sub = [dict(rows[i], date=f"{rows[i]['date']}#{c}") for i, c in zip(idx, copy)]
        t_c, _ = _top_n_rate(sub, wf["corrected"][idx], target)
        t_s, _ = _top_n_rate(sub, wf["served"][idx], target)
        top.append(t_c - t_s)
        auc.append(roc_auc_score(y[idx], wf["corrected"][idx])
                   - roc_auc_score(y[idx], wf["served"][idx]))
    return {
        "top6_gain_ci": [float(np.quantile(top, 0.05)), float(np.quantile(top, 0.95))],
        "auc_gain_ci": [float(np.quantile(auc, 0.05)), float(np.quantile(auc, 0.95))],
        "share_weeks_resamples_positive_top6": float(np.mean(np.array(top) > 0)),
    }


def main(pa_path: str = str(DATA_DIR / "season_pa_v2.jsonl")) -> None:
    records = _read_pa(pa_path)
    windows = weekly_windows(pa_path)
    rows = build_rows(records, _served_hr_rates(records, windows))
    report = {}
    for target, label in (("hr", "HOME RUNS"), ("hit", "HITS")):
        wf = walk_forward(rows, windows, target)
        s_served = _scores(wf["rows"], wf["served"], target)
        s_corr = _scores(wf["rows"], wf["corrected"], target)
        gain = bootstrap_gain(wf, target)
        _, names = features(wf["rows"][:1], target)

        # What the backtest says the served model gets wrong: the correction
        # fitted on every backtest week, as odds multipliers per 1 SD.
        final = fit_forward([r for r in rows if r[f"current_{target}"] is not None], target)
        stable = (
            np.mean(np.sign(wf["weights"]) == np.sign(final.w[1:]), axis=0)
            if final is not None and len(wf["weights"]) else None
        )
        report[target] = {
            "served": s_served, "corrected": s_corr, **gain,
            "weeks_corrected": len(wf["lams"]), "weeks_total": int(len(np.unique(wf["week"]))),
            "final_penalty": final.lam if final else None,
            "final_effects": (
                {n: {"odds_per_sd": float(np.exp(w)), "sign_agreement": float(a)}
                 for n, w, a in zip(names, final.w[1:], stable)}
                if final is not None else {}
            ),
        }

        print("\n" + "=" * 80)
        print(f"{label}: served model vs backtest residual correction "
              f"({s_served['n']:,} hitter-games, {s_served['top6_n']:,} top-{TOP_N} picks)")
        print(f"  {'':<24}{'Top-6 hit%':>11}{'AUC':>9}{'Brier':>9}{'LogLoss':>9}")
        for name, s in (("Served model", s_served), ("Corrected", s_corr)):
            print(f"  {name:<24}{s['top6_rate']:>11.1%}{s['auc']:>9.4f}"
                  f"{s['brier']:>9.4f}{s['log_loss']:>9.4f}")
        lo, hi = gain["top6_gain_ci"]
        alo, ahi = gain["auc_gain_ci"]
        print(f"  top-6 gain {s_corr['top6_rate'] - s_served['top6_rate']:+.1%} "
              f"(90% CI {lo:+.1%} to {hi:+.1%}); "
              f"AUC gain {s_corr['auc'] - s_served['auc']:+.4f} (90% CI {alo:+.4f} to {ahi:+.4f})")
        print(f"  a correction passed the forward check in {len(wf['lams'])} of "
              f"{report[target]['weeks_total']} weeks")
        if final is not None:
            print(f"  what the full backtest says the served model misses "
                  f"(odds x per 1 SD; share of weekly fits with the same sign):")
            for n, e in sorted(report[target]["final_effects"].items(),
                               key=lambda kv: -abs(np.log(kv[1]["odds_per_sd"]))):
                print(f"    {n:<24} x{e['odds_per_sd']:.3f}   {e['sign_agreement']:.0%}")
        else:
            print("  on the full backtest no correction beat the served model forward")
    print("=" * 80)
    (DATA_DIR / "residual_fit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[residual_fit.py] wrote {DATA_DIR / 'residual_fit.json'}")


if __name__ == "__main__":
    main()
