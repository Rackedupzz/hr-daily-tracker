"""Does the model actually learn? Four tests that can each fail loudly.

A held-out AUC of 0.58 on a 3% base rate is easy to mistake for a broken
model. It is also exactly what a *working* model looks like when the event is
nearly all noise, so the number alone cannot distinguish the two. These tests
separate them:

1. RECOVERY -- fit the same estimators on synthetic data with a known, strong
   relationship. If they cannot recover a signal that is definitely there, the
   fitting code is broken. This is the one test that indicts the machinery.

2. PERMUTATION -- shuffle the target across batters and refit. Any score above
   chance now is leakage or an artefact of the scoring path, because there is
   no longer anything to learn. Real skill must collapse to ~0.500.

3. LEARNING CURVE -- refit on 25/50/75/100% of the batters. Error that falls
   as data grows is the signature of learning; a flat curve means the model
   saturated, and a rising one means it is fitting noise.

4. CEILING -- score an oracle that knows each batter's true talent, on the same
   held-out plate appearances. Per-PA prediction is capped far below AUC 1.0 by
   binomial noise, and this measures where that cap actually is. Model quality
   is only interpretable as a fraction of it.

Run: python -m mlb_hr.diagnostics [path-to-season-pa.jsonl]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from mlb_hr.ensemble import (
    ComparablesKNN,
    RELIABLE_PA,
    TRAIN_FRACTION,
    _TORCH,
    _XGB,
    _fit_logistic_rate,
    _new_linear,
    _new_nn,
    _new_rf,
    _new_svr,
    _new_xgb,
)
from mlb_hr.features import build_matrix, build_profiles, season_bounds


def _cutoff(pa_path: str) -> str:
    """The same temporal split the served model uses."""
    from datetime import date as _date, timedelta

    lo, hi = season_bounds(pa_path)
    start, end = _date.fromisoformat(lo), _date.fromisoformat(hi)
    span = (end - start).days
    return str(start + timedelta(days=int(span * TRAIN_FRACTION)))


def _fit_all(X: np.ndarray, hr: np.ndarray, pa: np.ndarray, ids: list) -> dict:
    """Every prior's out-of-fold prediction of each batter's HR rate."""
    rates = np.divide(hr, np.maximum(pa, 1))
    weights = np.sqrt(pa)
    out = {}

    knn = ComparablesKNN(n_neighbors=15).fit(X, hr, pa, ids)
    out["knn"] = np.array([knn.prior(X[i], ids[i]) for i in range(len(X))])

    preds = {k: np.full(len(X), rates.mean()) for k in
             ("linear", "logistic", "rf", "svr", "nn", "xgb")}
    n_splits = min(5, max(2, len(X) // 50))
    for tr, te in KFold(n_splits=n_splits, shuffle=True, random_state=0).split(X):
        w = weights[tr]
        preds["linear"][te] = _new_linear().fit(
            X[tr], rates[tr], sample_weight=w).predict(X[te])
        preds["rf"][te] = _new_rf().fit(
            X[tr], rates[tr], sample_weight=w).predict(X[te])
        preds["svr"][te] = _new_svr().fit(
            X[tr], rates[tr], sample_weight=w).predict(X[te])
        logit = _fit_logistic_rate(X[tr], hr[tr], pa[tr])
        if logit is not None:
            preds["logistic"][te] = logit.predict_proba(X[te])[:, 1]
        if _XGB:
            preds["xgb"][te] = _new_xgb().fit(
                X[tr], rates[tr], sample_weight=w).predict(X[te])
        if _TORCH:
            p = _new_nn().fit(X[tr], rates[tr], w).predict(X[te])
            if not np.isnan(p).any():
                preds["nn"][te] = p

    out.update(preds)
    return {k: np.clip(v, 0.0005, 0.15) for k, v in out.items()}


# ---------------------------------------------------------------- test 1


def test_recovery(n: int = 600, d: int = 11, seed: int = 0) -> dict:
    """Can each estimator recover a signal that is definitely present?

    Rates are a known function of two features plus noise, and the counts are
    real binomial draws at realistic plate-appearance volumes, so this is the
    actual problem with the guesswork removed.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    true = np.clip(0.030 + 0.014 * X[:, 0] + 0.009 * X[:, 1], 0.003, 0.13)
    pa = rng.integers(60, 680, size=n).astype(float)
    hr = rng.binomial(pa.astype(int), true).astype(float)
    ids = list(range(n))

    scores = {}
    for name, pred in _fit_all(X, hr, pa, ids).items():
        # Correlation with the TRUE rate, which only synthetic data exposes.
        scores[name] = float(np.corrcoef(pred, true)[0, 1])
    scores["observed_rate"] = float(
        np.corrcoef(np.divide(hr, np.maximum(pa, 1)), true)[0, 1]
    )
    return scores


# ---------------------------------------------------------------- test 2


def test_permutation(X, hr, pa, ids, seed: int = 0) -> dict:
    """Shuffle the counts across batters; skill must vanish.

    The features stay put and the targets move, so any remaining correlation
    is the fitting or scoring path inventing signal rather than finding it.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(hr))
    hr_s, pa_s = hr[order], pa[order]
    rates_s = np.divide(hr_s, np.maximum(pa_s, 1))
    heavy = pa_s >= RELIABLE_PA

    out = {}
    for name, pred in _fit_all(X, hr_s, pa_s, ids).items():
        out[name] = float(np.corrcoef(pred[heavy], rates_s[heavy])[0, 1])
    return out


# ---------------------------------------------------------------- test 3


def test_learning_curve(X, hr, pa, ids, future: dict, seed: int = 0) -> list:
    """Does more data help? Scored against what hitters did *next*.

    The evaluation batters are fixed and held out of every fit, and only the
    training pool grows. Scoring whoever happens to be in the subsample instead
    would confound sample size with who is being scored -- adding thin-record
    hitters to the evaluation set raises the error no matter what the model
    learned, which reads as the model getting worse.

    Scoring against the same window's observed rate would reward memorising
    binomial noise, so the target is the later window.
    """
    rng = np.random.default_rng(seed)
    rows = []

    scorable = np.array([i for i, b in enumerate(ids) if b in future])
    if len(scorable) < 120:
        return rows
    shuffled = rng.permutation(scorable)
    eval_idx = shuffled[: len(shuffled) // 3]
    pool = np.array([i for i in range(len(ids)) if i not in set(eval_idx.tolist())])

    y_next = np.array([future[ids[i]][0] / max(future[ids[i]][1], 1) for i in eval_idx])
    w_next = np.array([future[ids[i]][1] for i in eval_idx])
    rates = np.divide(hr, np.maximum(pa, 1))

    for frac in (0.25, 0.50, 0.75, 1.00):
        take = pool[rng.permutation(len(pool))[: max(60, int(len(pool) * frac))]]
        w = np.sqrt(pa[take])
        row = {"fraction": frac, "batters": len(take), "scored": len(eval_idx)}

        # Every prior is fit on the training pool only and asked about batters
        # it has never seen, which is what a new hitter's prior really is.
        fitted = {
            "knn": ComparablesKNN(n_neighbors=15).fit(
                X[take], hr[take], pa[take], [ids[i] for i in take]
            ),
            "linear": _new_linear().fit(X[take], rates[take], sample_weight=w),
            "rf": _new_rf().fit(X[take], rates[take], sample_weight=w),
            "svr": _new_svr().fit(X[take], rates[take], sample_weight=w),
        }
        preds = {
            "knn": np.array([fitted["knn"].prior(X[i]) for i in eval_idx]),
            "linear": fitted["linear"].predict(X[eval_idx]),
            "rf": fitted["rf"].predict(X[eval_idx]),
            "svr": fitted["svr"].predict(X[eval_idx]),
        }
        logit = _fit_logistic_rate(X[take], hr[take], pa[take])
        if logit is not None:
            preds["logistic"] = logit.predict_proba(X[eval_idx])[:, 1]
        if _XGB:
            preds["xgb"] = _new_xgb().fit(
                X[take], rates[take], sample_weight=w).predict(X[eval_idx])
        if _TORCH:
            p = _new_nn().fit(X[take], rates[take], w).predict(X[eval_idx])
            if not np.isnan(p).any():
                preds["nn"] = p

        for name, pred in preds.items():
            err = (np.clip(pred, 0.0005, 0.15) - y_next) ** 2
            row[name] = float(np.average(err, weights=w_next))
        rows.append(row)
    return rows


# ---------------------------------------------------------------- test 4


def test_ceiling(pa_path: str, cutoff: str, prior_rates: dict) -> dict:
    """How much per-PA signal is attainable at all?

    Two oracles, and the difference between them matters.

    The *cheating* oracle is handed each batter's rate measured in the very
    plate appearances being scored. It is an upper bound but an inflated one:
    a batter's home runs are inside the rate used to rank them, so part of its
    score is self-fulfilling rather than knowledge.

    The *honest* oracle sees the first half of the held-out window and is
    scored on the second. That is genuine foreknowledge of talent with none of
    the outcome being scored leaking into it, and it is the number a model
    could actually aspire to.
    """
    from mlb_hr.ensemble import _load_events

    events = _load_events(pa_path, start_date=cutoff)
    if not events:
        return {}

    half = len(events) // 2
    first, second = events[:half], events[half:]

    def tally(rows):
        hr_by, pa_by = {}, {}
        for bid, _hand, is_hr in rows:
            pa_by[bid] = pa_by.get(bid, 0) + 1
            hr_by[bid] = hr_by.get(bid, 0) + int(is_hr)
        return hr_by, pa_by

    hr_all, pa_all = tally(events)
    hr_1st, pa_1st = tally(first)

    y, cheat, model = [], [], []
    for bid, _hand, is_hr in events:
        if bid not in prior_rates or pa_all[bid] < 20:
            continue
        y.append(int(is_hr))
        cheat.append(hr_all[bid] / pa_all[bid])
        model.append(prior_rates[bid])

    y2, honest, model2 = [], [], []
    for bid, _hand, is_hr in second:
        if bid not in prior_rates or pa_1st.get(bid, 0) < 20:
            continue
        y2.append(int(is_hr))
        honest.append(hr_1st[bid] / pa_1st[bid])
        model2.append(prior_rates[bid])

    if len(set(y)) < 2:
        return {}
    out = {
        "events": int(len(y)),
        "base_rate": float(np.mean(y)),
        "cheating_oracle_auc": float(roc_auc_score(y, cheat)),
        "model_auc": float(roc_auc_score(y, model)),
    }
    if len(set(y2)) > 1 and len(y2) > 1000:
        out.update({
            "split_half_events": int(len(y2)),
            "empirical_rate_auc": float(roc_auc_score(y2, honest)),
            "model_auc_same_events": float(roc_auc_score(y2, model2)),
        })

    # The noise ceiling. Take each batter's full-season posterior as if it were
    # his true talent, simulate home runs from it, and rank by that same truth.
    # No predictor can beat this, because it *is* the data-generating process --
    # so the gap between it and 1.0 is pure binomial noise, and the gap between
    # it and the model is the only part that is anyone's fault.
    talent = np.array([prior_rates[bid] for bid, _h, _o in events
                       if bid in prior_rates and pa_all[bid] >= 20])
    if len(talent) > 1000:
        rng = np.random.default_rng(0)
        aucs = []
        for _ in range(5):
            simulated = (rng.random(len(talent)) < talent).astype(int)
            if len(set(simulated.tolist())) > 1:
                aucs.append(roc_auc_score(simulated, talent))
        if aucs:
            out["noise_ceiling_auc"] = float(np.mean(aucs))
    return out


# ---------------------------------------------------------------------- CLI


def main(pa_path: str) -> None:
    print("=" * 72)
    print("LEARNING DIAGNOSTICS")
    print("=" * 72)

    print("\n[1] RECOVERY -- known signal, synthetic data")
    print("    correlation between each prior and the TRUE rate")
    rec = test_recovery()
    for name, score in sorted(rec.items(), key=lambda kv: -kv[1]):
        flag = "" if score > 0.5 else "   <-- FAILS to recover a real signal"
        print(f"    {name:<16} {score:+.3f}{flag}")

    cutoff = _cutoff(pa_path)
    print(f"\n    loading real data (temporal split at {cutoff})")
    train = build_profiles(pa_path, end_date=cutoff)
    later = build_profiles(pa_path, start_date=cutoff)
    X_raw, hr, pa, ids = build_matrix(train, min_pa=1)
    X = StandardScaler().fit_transform(X_raw)
    future = {b: (p.hr, p.pa) for b, p in later.items() if p.pa > 0}
    print(f"    {len(ids)} batters fitted, {len(future)} scored in the later window")

    print("\n[2] PERMUTATION -- targets shuffled, skill must vanish")
    print("    correlation with the (shuffled) observed rate; ~0.00 is correct")
    for name, score in sorted(test_permutation(X, hr, pa, ids).items()):
        flag = "   <-- LEAK: skill on shuffled labels" if abs(score) > 0.15 else ""
        print(f"    {name:<16} {score:+.3f}{flag}")

    print("\n[3] LEARNING CURVE -- error vs sample size, scored on the future")
    rows = test_learning_curve(X, hr, pa, ids, future)
    if rows:
        names = [k for k in rows[0] if k not in ("fraction", "batters", "scored")]
        print("    " + "frac  batters  " + "  ".join(f"{n:>9}" for n in names))
        for r in rows:
            cells = "  ".join(f"{r[n]:9.3e}" for n in names)
            print(f"    {r['fraction']:.2f}  {r['batters']:7d}  {cells}")
        first, last = rows[0], rows[-1]
        print("\n    change from smallest to largest sample:")
        for n in names:
            delta = (last[n] - first[n]) / first[n] * 100
            verdict = "learns" if delta < -1 else ("flat" if delta < 1 else "WORSE")
            print(f"    {n:<16} {delta:+6.1f}%   {verdict}")

    print("\n[4] CEILING -- what per-PA AUC is even attainable")
    knn = ComparablesKNN(n_neighbors=15).fit(X, hr, pa, ids)
    prior_rates = {}
    for i, bid in enumerate(ids):
        prior = knn.prior(X[i], bid)
        k = 248.0
        prior_rates[bid] = (hr[i] + k * prior) / (pa[i] + k)
    ceil = test_ceiling(pa_path, cutoff, prior_rates)
    if ceil:
        print(f"    scored plate appearances : {ceil['events']:,}")
        print(f"    base rate                : {ceil['base_rate']:.4f}")
        print(f"    cheating oracle auc      : {ceil['cheating_oracle_auc']:.4f}"
              "   (sees the outcomes it ranks; inflated)")
        print(f"    model auc                : {ceil['model_auc']:.4f}")
        span = ceil["cheating_oracle_auc"] - 0.5
        print(f"    captured                 : "
              f"{(ceil['model_auc'] - 0.5) / span:.1%} of that inflated bound")
        if "empirical_rate_auc" in ceil:
            print(f"\n    split-half, {ceil['split_half_events']:,} events -- the model"
                  " vs a naive rival")
            print("    rival: rank by the batter's own rate over the first half only")
            print(f"    naive empirical rate auc : {ceil['empirical_rate_auc']:.4f}")
            print(f"    model auc                : {ceil['model_auc_same_events']:.4f}"
                  f"   ({'model wins' if ceil['model_auc_same_events'] > ceil['empirical_rate_auc'] else 'model loses'})")
        if "noise_ceiling_auc" in ceil:
            span = ceil["noise_ceiling_auc"] - 0.5
            got = ceil["model_auc"] - 0.5
            print(f"\n    NOISE CEILING            : {ceil['noise_ceiling_auc']:.4f}"
                  "   <-- best score any predictor can get")
            print(f"    model auc                : {ceil['model_auc']:.4f}")
            if span > 0:
                print(f"    captured                 : {got / span:.1%} of what is "
                      "achievable at all")
    print("\n" + "=" * 72)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(__file__).parent / "data" / "season_pa_v2.jsonl"
    )
    main(path)
