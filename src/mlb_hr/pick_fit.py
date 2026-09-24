"""Reverse-engineering who homered and who got a hit, and whether it carries forward.

The same question as winloss_fit.py, asked of the outcomes the picks are
actually made on: for every starting hitter in every game, did he hit a home
run, and did he get a hit? Three predictors per outcome, all from what was
known the morning of the game:

  current   The served models. For home runs, the core prior's per-PA rate
            exactly as the rolling backtest produced it (backtest.py must have
            been run); for hits, project_hits() on to-date batter and starter
            lines.
  perfect   An unpruned decision tree on the to-date stats plus batter,
            starter, game and date ids. It reproduces its fitting window
            exactly, by construction.
  fitted    Regularised logistic regression on the to-date stats.

Scored in the same weekly walk-forward windows as the backtest, on the metric
the slate lives by: of each day's top six, how many connected.

Park and weather are left out of every predictor alike, as in the backtest's
scoring, so the comparison isolates the model.

Run: python -m mlb_hr.pick_fit
"""
from __future__ import annotations

import json
import pickle
from collections import defaultdict
from datetime import date as _date
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from mlb_hr.backtest import weekly_windows
from mlb_hr.features import expected_pa_for_slot
from mlb_hr.pitching import HIT_EVENTS, LeagueRates, PitcherProfile, _shrink, project_hits

DATA_DIR = Path(__file__).parent / "data"
CACHE_DIR = DATA_DIR / "backtest_cache"
TOP_N = 6

NUMERIC = [
    "bat_pa", "bat_hr_rate", "bat_hit_rate", "bat_pa_per_game",
    "sp_bf", "sp_hr_rate", "sp_hit_rate",
    "lineup_slot", "is_home", "same_hand",
]
IDS = ["batter_id", "sp_id", "game_pk", "day"]


def _read_pa(pa_path: str) -> list[tuple]:
    """Plate appearances in file order -- the order the backtest scored them in."""
    out = []
    with open(pa_path) as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out.append((
                r.get("date") or "", r.get("game_pk"), r.get("batter_id"),
                r.get("pitcher_id"), r.get("pitching_team"), r.get("pa_index", 0),
                r.get("event_type") or "", int(r.get("is_hr", 0) or 0),
                r.get("lineup_slot"), int(r.get("is_home", 0) or 0),
                r.get("bat_side") or "R",
                "L" if r.get("pitch_hand") == "L" else "R",
                r.get("venue") or "", r.get("temp_f"),
            ))
    return out


def _served_hr_rates(records: list[tuple], windows: list[tuple[str, str]]) -> dict[int, float]:
    """Per-PA core-model rate from the backtest cache, keyed by record index.

    The cache holds predictions in the order backtest scoring visited them:
    file order within the window, keeping batters seen before the cutoff.
    That order is rebuilt here and checked against the cached outcomes, so a
    misalignment fails loudly rather than scoring the wrong hitter.
    """
    rates: dict[int, float] = {}
    for cutoff, end in windows:
        path = CACHE_DIR / f"{cutoff}_{end}.npz"
        if not path.exists():
            raise SystemExit(f"missing {path.name}; run python -m mlb_hr.backtest first")
        seen = {r[2] for r in records if r[2] and r[0] < cutoff}
        idx = [i for i, r in enumerate(records)
               if r[2] and cutoff <= r[0] < end and r[2] in seen]
        with np.load(path) as z:
            y, core = z["y"], z["pred_core"]
        hr = np.array([records[i][7] for i in idx])
        if len(idx) != len(y) or not np.array_equal(hr, y.astype(int)):
            raise SystemExit(f"backtest cache {path.name} does not line up with the PA file")
        rates.update(zip(idx, core.tolist()))
    return rates


def build_rows(records: list[tuple], hr_rates: dict[int, float]) -> list[dict]:
    """One row per starting hitter per game, described as of that morning."""
    openers: dict[tuple, tuple[int, int]] = {}
    games: dict[tuple, list[int]] = defaultdict(list)
    by_date: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        by_date[r[0]].append(i)
        key = (r[1], r[4])
        if key not in openers or r[5] < openers[key][0]:
            openers[key] = (r[5], r[3])
        if r[2]:
            games[(r[1], r[2])].append(i)

    # First PA of each batter-game carries his slot, side and opponent.
    firsts_by_date: dict[str, list[tuple]] = defaultdict(list)
    for key, idx in games.items():
        first = min(idx, key=lambda i: records[i][5])
        firsts_by_date[records[first][0]].append((key, idx, first))

    bat = defaultdict(lambda: [0, 0, 0, set()])  # pa, hits, hr, games
    pit = defaultdict(lambda: [0, 0, 0])         # bf, hits, hr
    park = defaultdict(lambda: [0, 0, 0])        # pa, hits, hr, per venue
    lg = [0, 0, 0]                               # pa, hits, hr
    rows = []
    for day in sorted(by_date):
        league = LeagueRates(
            hit_rate=lg[1] / lg[0] if lg[0] else 0.225,
            hr_rate=lg[2] / lg[0] if lg[0] else 0.030,
        )
        for (game_pk, batter_id), idx, first in firsts_by_date.get(day, []):
            r = records[first]
            slot = r[8]
            if not slot:
                continue  # a pinch hitter or substitute, not a lineup pick
            sp_id = openers.get((game_pk, r[4]), (0, 0))[1]
            b_pa, b_h, b_hr, b_games = bat[batter_id]
            s_bf, s_h, s_hr = pit[sp_id]
            pa_per_game = b_pa / len(b_games) if b_games else 4.1
            exp_pa = float(np.clip(expected_pa_for_slot(slot, pa_per_game), 1.5, 5.2))
            hits = project_hits(
                batter_hit_rate=b_h / b_pa if b_pa else league.hit_rate,
                batter_pa=b_pa,
                pitcher=PitcherProfile(pitcher_id=sp_id, bf=s_bf, hits_allowed=s_h),
                bullpen=None, league=league, expected_pa=exp_pa,
            )
            hr_rate = hr_rates.get(first)
            rows.append({
                "date": day, "day": _date.fromisoformat(day).toordinal(),
                "game_pk": game_pk, "batter_id": batter_id, "sp_id": sp_id,
                "hr": int(any(records[i][7] for i in idx)),
                "hit": int(any(records[i][6] in HIT_EVENTS for i in idx)),
                "current_hr": (
                    1.0 - (1.0 - hr_rate) ** exp_pa if hr_rate is not None else None
                ),
                "current_hit": hits["prob_at_least_one"],
                "bat_pa": b_pa,
                "bat_hr_rate": _shrink(b_hr, b_pa, league.hr_rate, 300.0),
                "bat_hit_rate": _shrink(b_h, b_pa, league.hit_rate, 200.0),
                "bat_pa_per_game": pa_per_game,
                "sp_bf": s_bf,
                "sp_hr_rate": _shrink(s_hr, s_bf, league.hr_rate, 900.0),
                "sp_hit_rate": _shrink(s_h, s_bf, league.hit_rate, 400.0),
                "lineup_slot": slot, "is_home": r[9],
                "same_hand": int(r[10] == r[11]),
                # Context the served per-PA rate does not carry, for the
                # residual correction in residual_fit.py.
                "lg_hr": league.hr_rate, "lg_hit": league.hit_rate,
                "park_hr": _shrink(park[r[12]][2], park[r[12]][0], league.hr_rate, 3000.0),
                "park_hit": _shrink(park[r[12]][1], park[r[12]][0], league.hit_rate, 3000.0),
                "temp": float(r[13]) if r[13] is not None else 72.0,
            })

        for i in by_date[day]:
            r = records[i]
            is_hit = r[6] in HIT_EVENTS
            if r[2]:
                b = bat[r[2]]
                b[0] += 1; b[1] += is_hit; b[2] += r[7]; b[3].add(r[1])
            if r[3]:
                p = pit[r[3]]
                p[0] += 1; p[1] += is_hit; p[2] += r[7]
            v = park[r[12]]
            v[0] += 1; v[1] += is_hit; v[2] += r[7]
            lg[0] += 1; lg[1] += is_hit; lg[2] += r[7]
    return rows


def _X(rows, with_ids=False) -> np.ndarray:
    cols = NUMERIC + (IDS if with_ids else [])
    return np.array([[float(r[c]) for c in cols] for r in rows])


def _top_n_rate(rows: list[dict], p: np.ndarray, target: str) -> tuple[float, int]:
    """Share of each day's top-N that connected, pooled over days.

    Ties are broken at random. The perfect-fit tree scores thousands of hitters
    at exactly 1.0, and falling back on row order would quietly pick leadoff
    hitters -- who get the most plate appearances -- and credit the tree with
    the batting order's skill.
    """
    by_day: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_day[r["date"]].append(i)
    tiebreak = np.random.default_rng(0).random(len(rows))
    hits = n = 0
    for idx in by_day.values():
        top = sorted(idx, key=lambda i: (-p[i], tiebreak[i]))[:TOP_N]
        hits += sum(rows[i][target] for i in top)
        n += len(top)
    return (hits / n if n else float("nan")), n


def _scores(rows, p, target) -> dict:
    y = np.array([r[target] for r in rows], dtype=float)
    pc = np.clip(p, 1e-3, 1 - 1e-3)
    top, n_top = _top_n_rate(rows, p, target)
    return {
        "n": int(len(y)),
        "base_rate": float(y.mean()),
        "top6_rate": top,
        "top6_n": n_top,
        "top6_se": float(np.sqrt(top * (1 - top) / max(n_top, 1))),
        "auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan"),
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
    }


def evaluate(rows: list[dict], windows: list[tuple[str, str]], target: str) -> dict:
    test_rows, preds, train_acc = [], {"current": [], "perfect": [], "fitted": []}, []
    for cutoff, end in windows:
        train = [r for r in rows if r["date"] < cutoff]
        test = [r for r in rows if cutoff <= r["date"] < end
                and r[f"current_{target}"] is not None]
        if not test:
            continue
        y = np.array([r[target] for r in train])
        tree = DecisionTreeClassifier(random_state=0).fit(_X(train, True), y)
        train_acc.append(float(np.mean(tree.predict(_X(train, True)) == y)))
        logit = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))
        logit.fit(_X(train), y)
        test_rows.extend(test)
        preds["current"].extend(r[f"current_{target}"] for r in test)
        preds["perfect"].extend(tree.predict_proba(_X(test, True))[:, 1])
        preds["fitted"].extend(logit.predict_proba(_X(test))[:, 1])
    return {
        "scores": {k: _scores(test_rows, np.array(v, dtype=float), target)
                   for k, v in preds.items()},
        "perfect_train_accuracy_min": min(train_acc),
    }


def main(pa_path: str = str(DATA_DIR / "season_pa_v2.jsonl")) -> None:
    records = _read_pa(pa_path)
    windows = weekly_windows(pa_path)
    hr_rates = _served_hr_rates(records, windows)
    rows = build_rows(records, hr_rates)
    print(f"[pick_fit.py] {len(rows):,} starting-hitter games, "
          f"{len(hr_rates):,} backtest PA predictions matched")

    names = {"perfect": "Perfect fit (tree)", "current": "Current model",
             "fitted": "Regularised fit"}
    report = {"batter_games": len(rows)}
    for target, label in (("hr", "HOME RUNS"), ("hit", "HITS")):
        y = np.array([r[target] for r in rows])
        tree = DecisionTreeClassifier(random_state=0).fit(_X(rows, True), y)
        in_sample_acc = float(np.mean(tree.predict(_X(rows, True)) == y))
        with open(DATA_DIR / f"pick_perfect_fit_{target}.pkl", "wb") as fh:
            pickle.dump(tree, fh)
        wf = evaluate(rows, windows, target)
        report[target] = {
            "in_sample_perfect_accuracy": in_sample_acc,
            "perfect_leaves": int(tree.get_n_leaves()),
            **wf,
        }

        s0 = wf["scores"]["current"]
        print("\n" + "=" * 80)
        print(f"{label}  -- a starting hitter's game, did he get one? "
              f"(season rate {y.mean():.1%})")
        print(f"  perfect fit on the whole season: {in_sample_acc:.1%} reproduced, "
              f"{tree.get_n_leaves():,} leaves for {len(rows):,} hitter-games")
        print(f"  walk-forward, {s0['n']:,} hitter-games, {s0['top6_n']:,} top-{TOP_N} picks:")
        print(f"  {'':<22}{'Top-6 hit%':>11}{'± 1 SE':>8}{'AUC':>9}{'Brier':>9}{'LogLoss':>9}")
        print(f"  {'Random pick':<22}{s0['base_rate']:>11.1%}")
        for k in ("current", "fitted", "perfect"):
            s = wf["scores"][k]
            print(f"  {names[k]:<22}{s['top6_rate']:>11.1%}{s['top6_se']:>7.1%} "
                  f"{s['auc']:>9.4f}{s['brier']:>9.4f}{s['log_loss']:>9.4f}")
        print(f"  perfect fit reproduced its own fitting window at no less than "
              f"{wf['perfect_train_accuracy_min']:.0%} in every week")
    print("=" * 80)
    (DATA_DIR / "pick_fit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[pick_fit.py] wrote {DATA_DIR / 'pick_fit.json'}")


if __name__ == "__main__":
    main()
