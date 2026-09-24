"""Rolling-origin backtest: every prior scored across the whole season.

The page's held-out scores come from a single split -- fit on the first 80% of
the season, score the last 20%. That is honest but thin: a few thousand home
runs, one slice of the calendar, and AUC gaps between priors that sit inside
the noise. This walks the split forward instead:

    fit before week 4, score week 4
    fit before week 5, score week 5
    ...

Every prediction is made before its outcome, exactly as on a live slate, and
together the windows cover the season from the end of the warm-up onward. The
windows are pooled into one score per prior (not averaged, so a light week
does not count as much as a full one), broken out by month, and the gap to the
served prior is given a confidence interval by resampling whole weeks.

Early windows are fitted on very little data and will score worse than the
model does today. That is what April looked like, not what September looks
like -- read the monthly table for the current picture.

Each window's predictions are cached, so an interrupted run resumes where it
stopped. Pass --fresh after changing the model to refit everything.

Run: python -m mlb_hr.backtest [--workers N] [--step-days 7] [--warmup-days 21]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date as _date, timedelta
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from mlb_hr.ensemble import RELIABLE_PA, EnsembleHRModel
from mlb_hr.features import season_bounds
from mlb_hr.model import compute_calibration_metrics

DATA_DIR = Path(__file__).parent / "data"

# Same order and names as the Model Lab page.
VARIANTS = [
    ("empirical_bayes", "Flat league prior"),
    ("knn", "KNN comparables"),
    ("svm", "SVR"),
    ("rf", "Random forest"),
    ("core", "Core model"),
    ("forest", "Forest model"),
    ("linear", "Linear"),
    ("logistic", "Logistic"),
    ("neural", "Neural net"),
    ("xgboost", "XGBoost"),
]
SERVED = "core"

_espn = None


def weekly_windows(
    pa_path: str, warmup_days: int = 21, step_days: int = 7
) -> list[tuple[str, str]]:
    """[cutoff, end) scoring windows from the end of the warm-up to the last game."""
    lo, hi = season_bounds(pa_path, sample_every=1)
    start, last = _date.fromisoformat(lo), _date.fromisoformat(hi)
    stop = last + timedelta(days=1)
    windows = []
    cutoff = start + timedelta(days=warmup_days)
    while cutoff < stop:
        end = cutoff + timedelta(days=step_days)
        # A stub of a few days at the season's edge joins the week before it
        # rather than being scored on its own handful of games.
        if (stop - end).days < step_days / 2:
            end = stop
        windows.append((str(cutoff), str(end)))
        cutoff = end
    return windows


def _init_worker(espn_cache: str) -> None:
    """Load ESPN bios once per process, and keep each fit to its own cores."""
    global _espn
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    from mlb_hr.espn import ESPNData
    # Backtests read the cache as-is rather than refetching mid-run.
    _espn = ESPNData(espn_cache, ttl_hours=1e9).load()


def _fit_window(pa_path: str, cutoff: str, end: str, half_life_days: float,
                cache_path: str) -> dict:
    """Fit before `cutoff`, score [cutoff, end), cache the raw predictions."""
    t0 = time.perf_counter()
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        ens = EnsembleHRModel(
            None, pa_path, espn=_espn, half_life_days=half_life_days,
            cutoff=cutoff, holdout_end=end,
        )
    Path(cache_path).with_suffix(".log").write_text(log.getvalue(), encoding="utf-8")

    held = ens.holdout
    seconds = time.perf_counter() - t0
    if not held:
        return {"cutoff": cutoff, "end": end, "seconds": seconds, "empty": True}

    np.savez_compressed(
        cache_path,
        y=held["y"],
        low_pa=held["low_pa"],
        unmatched=ens.calibration.get("unmatched_events", 0),
        seconds=seconds,
        **{f"pred_{k}": v for k, v in held["preds"].items()},
    )
    return {"cutoff": cutoff, "end": end, "seconds": seconds, "empty": False}


def _load_window(cache_path: Path) -> dict:
    with np.load(cache_path) as z:
        return {
            "y": z["y"],
            "low_pa": z["low_pa"],
            "unmatched": int(z["unmatched"]),
            "seconds": float(z["seconds"]),
            "preds": {k: z[f"pred_{k}"] for k, _ in VARIANTS if f"pred_{k}" in z},
        }


def _metrics(y: np.ndarray, preds: dict) -> dict:
    return {k: compute_calibration_metrics(y, p) for k, p in preds.items()}


def _concat(windows: list[dict], mask_key: str | None = None) -> tuple[np.ndarray, dict]:
    ys, preds = [], {k: [] for k, _ in VARIANTS}
    for w in windows:
        m = w[mask_key] if mask_key else slice(None)
        ys.append(w["y"][m])
        for k in preds:
            if k in w["preds"]:
                preds[k].append(w["preds"][k][m])
    y = np.concatenate(ys) if ys else np.empty(0)
    return y, {k: np.concatenate(v) for k, v in preds.items() if v}


def bootstrap_auc_gap(
    windows: list[dict], served: str = SERVED, n_boot: int = 300, seed: int = 0
) -> dict:
    """AUC minus the served prior's, with a 90% interval from resampling weeks.

    Whole weeks are resampled rather than single plate appearances, because
    PAs inside a week share the same fitted model and the same slate of games;
    treating them as independent would make the interval too narrow.
    """
    rng = np.random.default_rng(seed)
    point_y, point = _concat(windows)
    base = roc_auc_score(point_y, point[served])
    gaps = {k: [] for k in point if k != served}
    for _ in range(n_boot):
        sample = [windows[i] for i in rng.integers(0, len(windows), len(windows))]
        y, preds = _concat(sample)
        if y.sum() == 0 or y.sum() == len(y):
            continue
        ref = roc_auc_score(y, preds[served])
        for k in gaps:
            gaps[k].append(roc_auc_score(y, preds[k]) - ref)
    out = {}
    for k, vals in gaps.items():
        vals = np.array(vals)
        out[k] = {
            "gap": float(roc_auc_score(point_y, point[k]) - base),
            "lo": float(np.quantile(vals, 0.05)),
            "hi": float(np.quantile(vals, 0.95)),
            "share_above_served": float((vals > 0).mean()),
        }
    return out


def summarize(windows: list[dict], spans: list[tuple[str, str]]) -> dict:
    for w, (cutoff, end) in zip(windows, spans):
        w["cutoff"], w["end"] = cutoff, end

    y, preds = _concat(windows)
    report = {
        "n_windows": len(windows),
        "n_test_samples": int(len(y)),
        "n_home_runs": int(y.sum()),
        "holdout_hr_rate": float(y.mean()),
        "first_cutoff": spans[0][0],
        "last_end": spans[-1][1],
        "served": SERVED,
        "pooled": _metrics(y, preds),
    }

    y_low, preds_low = _concat(windows, "low_pa")
    if len(y_low) >= 200 and 0 < y_low.sum() < len(y_low):
        report["low_pa"] = {
            "threshold_pa": RELIABLE_PA,
            "n_test_samples": int(len(y_low)),
            "holdout_hr_rate": float(y_low.mean()),
            **_metrics(y_low, preds_low),
        }

    # Month of the scoring window's first day.
    months: dict[str, list[dict]] = {}
    for w in windows:
        months.setdefault(w["cutoff"][:7], []).append(w)
    report["monthly"] = {}
    for month, ws in months.items():
        my, mp = _concat(ws)
        report["monthly"][month] = {
            "n_test_samples": int(len(my)),
            "holdout_hr_rate": float(my.mean()),
            "auc": {k: compute_calibration_metrics(my, p)["auc"] for k, p in mp.items()},
        }

    report["weekly"] = []
    wins = {k: 0 for k in preds}
    for w in windows:
        aucs = {
            k: compute_calibration_metrics(w["y"], p)["auc"]
            for k, p in w["preds"].items()
        }
        finite = {k: v for k, v in aucs.items() if np.isfinite(v)}
        best = max(finite, key=finite.get) if finite else None
        if best:
            wins[best] += 1
        report["weekly"].append({
            "cutoff": w["cutoff"], "end": w["end"],
            "n_test_samples": int(len(w["y"])), "home_runs": int(w["y"].sum()),
            "unmatched_events": w["unmatched"], "fit_seconds": w["seconds"],
            "auc": aucs, "best": best,
        })
    report["weekly_wins"] = wins
    report["auc_gap_vs_served"] = bootstrap_auc_gap(windows)
    return report


def print_report(r: dict) -> None:
    name = dict(VARIANTS)
    gaps = r["auc_gap_vs_served"]
    print("\n" + "=" * 96)
    print(
        f"ROLLING BACKTEST  {r['n_windows']} windows, {r['first_cutoff']} .. "
        f"{r['last_end']}   {r['n_test_samples']:,} PA, {r['n_home_runs']:,} HR "
        f"({r['holdout_hr_rate']:.2%})"
    )
    print("=" * 96)
    print(
        f"{'':<20}{'AUC':>8}  {'vs served (90% CI)':<26}{'Lift':>7}{'Brier':>10}"
        f"{'LogLoss':>10}{'CalRMSE':>10}{'Wks won':>9}"
    )
    order = sorted(r["pooled"], key=lambda k: -np.nan_to_num(r["pooled"][k]["auc"]))
    for k in order:
        m = r["pooled"][k]
        if k == r["served"]:
            gap = "(served)"
        else:
            g = gaps[k]
            gap = f"{g['gap']:+.4f} [{g['lo']:+.4f}, {g['hi']:+.4f}]"
        print(
            f"{name.get(k, k):<20}{m['auc']:>8.4f}  {gap:<26}"
            f"{m['top_decile_lift']:>6.2f}x{m['brier']:>10.5f}{m['log_loss']:>10.5f}"
            f"{m['calibration_rmse']:>10.5f}{r['weekly_wins'].get(k, 0):>9}"
        )
    print(
        "\nA gap whose interval includes zero is not a real difference: the two"
        " priors rank\nhitters equally well as far as this season can tell."
    )

    months = list(r["monthly"])
    print("\nAUC BY MONTH (month the scoring window starts)")
    print(f"{'':<20}" + "".join(f"{m:>10}" for m in months))
    print(f"{'PA scored':<20}" + "".join(
        f"{r['monthly'][m]['n_test_samples']:>10,}" for m in months))
    for k in order:
        print(f"{name.get(k, k):<20}" + "".join(
            f"{r['monthly'][m]['auc'].get(k, float('nan')):>10.4f}" for m in months))

    low = r.get("low_pa")
    if low:
        print(
            f"\nTHIN SAMPLE (<{low['threshold_pa']} PA before each cutoff): "
            f"{low['n_test_samples']:,} PA, HR rate {low['holdout_hr_rate']:.2%}"
        )
        for k in order:
            if k in low:
                print(f"  {name.get(k, k):<18}AUC {low[k]['auc']:.4f}  "
                      f"lift {low[k]['top_decile_lift']:.2f}x")
    print("=" * 96)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("pa_path", nargs="?", default=str(DATA_DIR / "season_pa_v2.jsonl"))
    # One fit at a time by default: each already spreads the forest across
    # every core, and two at once can exhaust a small Docker VM's memory.
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--step-days", type=int, default=7)
    parser.add_argument("--warmup-days", type=int, default=21)
    parser.add_argument("--half-life-days", type=float, default=45)
    parser.add_argument("--limit", type=int, default=0,
                        help="score only the last N windows (for a quick trial)")
    parser.add_argument("--fresh", action="store_true", help="ignore cached windows")
    parser.add_argument("--out", default=str(DATA_DIR / "backtest.json"))
    args = parser.parse_args()

    spans = weekly_windows(args.pa_path, args.warmup_days, args.step_days)
    if args.limit:
        spans = spans[-args.limit:]
    cache_dir = DATA_DIR / "backtest_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_of = {s: cache_dir / f"{s[0]}_{s[1]}.npz" for s in spans}
    if args.fresh:
        for p in cache_of.values():
            p.unlink(missing_ok=True)

    todo = [s for s in spans if not cache_of[s].exists()]
    print(
        f"[backtest.py] {len(spans)} windows of {args.step_days} days after a "
        f"{args.warmup_days}-day warm-up; {len(spans) - len(todo)} cached, "
        f"{len(todo)} to fit on {args.workers} workers"
    )

    started = time.perf_counter()
    if todo:
        # A fresh process per window, so each fit starts with clean memory and
        # a crash or kill loses only the window in flight.
        with ProcessPoolExecutor(
            max_workers=args.workers,
            max_tasks_per_child=1,
            initializer=_init_worker,
            initargs=(str(DATA_DIR / "espn_cache.json"),),
        ) as pool:
            futures = {
                pool.submit(
                    _fit_window, args.pa_path, c, e, args.half_life_days,
                    str(cache_of[(c, e)]),
                ): (c, e)
                for c, e in todo
            }
            for done, fut in enumerate(as_completed(futures), 1):
                c, e = futures[fut]
                res = fut.result()
                elapsed = time.perf_counter() - started
                eta = elapsed / done * (len(todo) - done)
                print(
                    f"[backtest.py] {done}/{len(todo)} {c}..{e} fit in "
                    f"{res['seconds'] / 60:.1f} min"
                    f"{' (no scorable PA)' if res['empty'] else ''}; "
                    f"elapsed {elapsed / 60:.0f} min, ~{eta / 60:.0f} min left"
                )

    kept = [s for s in spans if cache_of[s].exists()]
    if not kept:
        print("[backtest.py] nothing to score")
        return
    windows = [_load_window(cache_of[s]) for s in kept]
    report = summarize(windows, kept)
    report["settings"] = {
        "step_days": args.step_days, "warmup_days": args.warmup_days,
        "half_life_days": args.half_life_days, "pa_path": Path(args.pa_path).name,
    }
    Path(args.out).write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print_report(report)
    print(f"[backtest.py] wrote {args.out}")


if __name__ == "__main__":
    main()
