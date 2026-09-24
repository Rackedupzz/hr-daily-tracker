"""Reverse-engineering the game win/loss results, and whether it means anything.

The question: can an algorithm be fitted that reproduces the season's actual
wins and losses exactly, and would it predict the next game any better than
the runs model in game.py?

Three predictors are built from identical pre-game inputs -- for each game,
only what was known the morning of it (team runs scored/allowed to date, each
starter's run-value line to date):

  current   game.py's Pythagorean runs model, unchanged. Nothing is fitted to
            outcomes; its constants were set by baseball convention.
  perfect   An unpruned decision tree on the same inputs plus team and starter
            ids. Every game has a distinct input row, so the tree can carve out
            a leaf per game and reproduce the fitted record exactly -- 100% by
            construction. This is the reverse-engineered algorithm.
  fitted    Regularised logistic regression on the model's numeric inputs: a
            fit to results that is only allowed to learn smooth, general
            relationships.

All three are scored walk-forward, week by week: fit on every game before the
week, predict the week. A reproduction of the past is only worth something if
it carries forward to games it has not seen.

Run: python -m mlb_hr.winloss_fit [--refresh]
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from datetime import date as _date, timedelta
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from mlb_hr.features import season_bounds
from mlb_hr.fetch import BASE, _get
from mlb_hr.game import project_game
from mlb_hr.pitching import (
    DEFAULT_RUN_VALUE,
    RUN_VALUES,
    LeagueRates,
    PitcherProfile,
)

DATA_DIR = Path(__file__).parent / "data"
RESULTS_PATH = DATA_DIR / "game_results.json"

NUMERIC = [
    "home_offense_index", "away_offense_index",
    "home_bullpen_index", "away_bullpen_index",
    "home_sp_index", "away_sp_index",
]


# ── outcomes ─────────────────────────────────────────────────────────────────
def fetch_game_results(start: _date, end: _date) -> list[dict]:
    """Final score of every regular-season game in [start, end]."""
    games: dict[int, dict] = {}
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=30), end)
        url = (f"{BASE}/api/v1/schedule?sportId=1&gameType=R"
               f"&startDate={cur:%Y-%m-%d}&endDate={chunk_end:%Y-%m-%d}"
               f"&hydrate=linescore")
        for day in (_get(url) or {}).get("dates", []):
            for g in day.get("games", []):
                if g.get("status", {}).get("abstractGameState") != "Final":
                    continue
                home, away = g["teams"]["home"], g["teams"]["away"]
                hs, as_ = home.get("score"), away.get("score")
                # Postponed and suspended games can report Final with no score;
                # a tie has no winner to reproduce.
                if hs is None or as_ is None or hs == as_:
                    continue
                games[g["gamePk"]] = {
                    "game_pk": g["gamePk"],
                    "date": g.get("officialDate") or day["date"],
                    "home": home["team"]["name"],
                    "away": away["team"]["name"],
                    "home_id": home["team"]["id"],
                    "away_id": away["team"]["id"],
                    "home_score": int(hs),
                    "away_score": int(as_),
                }
        cur = chunk_end + timedelta(days=1)
    return sorted(games.values(), key=lambda g: (g["date"], g["game_pk"]))


def load_results(pa_path: str, refresh: bool = False) -> list[dict]:
    """Game results, cached; refetched when asked or when the cache is behind."""
    lo, hi = season_bounds(pa_path, sample_every=1)
    if RESULTS_PATH.exists() and not refresh:
        cached = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        if cached and cached[-1]["date"] >= hi:
            return cached
    results = fetch_game_results(_date.fromisoformat(lo), _date.fromisoformat(hi))
    RESULTS_PATH.write_text(json.dumps(results), encoding="utf-8")
    return results


# ── pre-game inputs ──────────────────────────────────────────────────────────
def _starters_and_pitching(pa_path: str):
    """Each game's two starters, and every pitcher's run-value line per date.

    The starter is the pitcher on the first recorded plate appearance of each
    half, the same rule pitching.py uses.
    """
    openers: dict[tuple, tuple[int, int]] = {}
    daily: dict[str, dict[int, list]] = defaultdict(dict)
    with open(pa_path) as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = r.get("pitcher_id")
            if not pid:
                continue
            key = (r.get("game_pk"), r.get("pitching_team"))
            idx = r.get("pa_index", 0)
            if key not in openers or idx < openers[key][0]:
                openers[key] = (idx, pid)
            rv = RUN_VALUES.get(r.get("event_type") or "", DEFAULT_RUN_VALUE)
            line_ = daily[r.get("date") or ""].setdefault(pid, [0, 0.0])
            line_[0] += 1
            line_[1] += rv
    starters = {key: pid for key, (_i, pid) in openers.items()}
    return starters, daily


def build_games(pa_path: str, results: list[dict]) -> list[dict]:
    """One row per game: current model's inputs and prediction, as of that morning.

    Walks the season a date at a time. Every game on a date is described with
    the totals from earlier dates only, then that date's outcomes are added.
    """
    starters, daily = _starters_and_pitching(pa_path)
    by_date: dict[str, list[dict]] = defaultdict(list)
    for g in results:
        by_date[g["date"]].append(g)

    team = defaultdict(lambda: {"runs_scored": 0, "runs_allowed": 0, "games_played": 0})
    pitcher: dict[int, list] = defaultdict(lambda: [0, 0.0])
    league_bf, league_rv = 0, 0.0

    rows = []
    for day in sorted(set(by_date) | set(daily)):
        league = LeagueRates(
            run_value_per_bf=league_rv / league_bf if league_bf else 0.0
        )
        standings = {tid: dict(v) for tid, v in team.items()}
        for g in by_date.get(day, []):
            for tid in (g["home_id"], g["away_id"]):
                standings.setdefault(
                    tid, {"runs_scored": 0, "runs_allowed": 0, "games_played": 0}
                )
            # PA records name the pitching team: the home starter pitches to
            # the away lineup, so he is the opener of the home team's half.
            sp = {}
            for side in ("home", "away"):
                pid = starters.get((g["game_pk"], g[side]))
                bf, rv = pitcher[pid] if pid else (0, 0.0)
                sp[side] = (
                    PitcherProfile(pitcher_id=pid, bf=bf, run_value_sum=rv)
                    if pid else None
                )
            proj = project_game(
                g["home_id"], g["away_id"], g["home"], g["away"], standings,
                sp["home"], sp["away"], league, park_factor=1.0,
            )
            if proj is None:
                continue
            rows.append({
                **g,
                "home_sp_id": sp["home"].pitcher_id if sp["home"] else 0,
                "away_sp_id": sp["away"].pitcher_id if sp["away"] else 0,
                "home_games_before": standings[g["home_id"]]["games_played"],
                "home_win": int(g["home_score"] > g["away_score"]),
                "current_prob": proj["home_win_prob"],
                **{k: proj["components"][k] for k in NUMERIC},
            })

        # Now that the date is described, fold in what happened on it.
        for g in by_date.get(day, []):
            for tid, rs, ra in (
                (g["home_id"], g["home_score"], g["away_score"]),
                (g["away_id"], g["away_score"], g["home_score"]),
            ):
                team[tid]["runs_scored"] += rs
                team[tid]["runs_allowed"] += ra
                team[tid]["games_played"] += 1
        for pid, (bf, rv) in daily.get(day, {}).items():
            pitcher[pid][0] += bf
            pitcher[pid][1] += rv
            league_bf += bf
            league_rv += rv
    return rows


# ── the three predictors ─────────────────────────────────────────────────────
def _numeric(rows: list[dict]) -> np.ndarray:
    X = np.array([[r[k] for k in NUMERIC] for r in rows], dtype=float)
    p = np.clip([r["current_prob"] for r in rows], 1e-3, 1 - 1e-3)
    return np.column_stack([X, np.log(p / (1 - p))])


def _perfect_features(rows: list[dict]) -> np.ndarray:
    """Numeric inputs plus identities. Ids let the tree isolate single games."""
    ids = np.array(
        [[r["home_id"], r["away_id"], r["home_sp_id"], r["away_sp_id"],
          _date.fromisoformat(r["date"]).toordinal()] for r in rows],
        dtype=float,
    )
    return np.column_stack([_numeric(rows), ids])


def fit_perfect(rows: list[dict]) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(random_state=0).fit(
        _perfect_features(rows), [r["home_win"] for r in rows]
    )


def fit_regularised(rows: list[dict]):
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1)).fit(
        _numeric(rows), [r["home_win"] for r in rows]
    )


def _scores(y: np.ndarray, p: np.ndarray) -> dict:
    pc = np.clip(p, 1e-3, 1 - 1e-3)
    acc = float(np.mean((p >= 0.5) == (y == 1)))
    return {
        "n": int(len(y)),
        "accuracy": acc,
        "accuracy_se": float(np.sqrt(acc * (1 - acc) / max(len(y), 1))),
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
    }


def walk_forward(rows: list[dict], warmup_days: int = 21, step_days: int = 7) -> dict:
    """Fit on everything before each week, predict the week, pool the weeks."""
    first = _date.fromisoformat(rows[0]["date"])
    last = _date.fromisoformat(rows[-1]["date"])
    cutoff = first + timedelta(days=warmup_days)

    y_all, preds = [], {"current": [], "perfect": [], "fitted": [], "home": []}
    train_acc = []
    while cutoff <= last:
        end = cutoff + timedelta(days=step_days)
        train = [r for r in rows if r["date"] < str(cutoff)]
        test = [r for r in rows if str(cutoff) <= r["date"] < str(end)]
        cutoff = end
        if not test or len(train) < 50:
            continue
        y_train = np.array([r["home_win"] for r in train])
        tree = fit_perfect(train)
        train_acc.append(float(np.mean(tree.predict(_perfect_features(train)) == y_train)))
        logit = fit_regularised(train)

        y_all.extend(r["home_win"] for r in test)
        preds["current"].extend(r["current_prob"] for r in test)
        preds["perfect"].extend(tree.predict_proba(_perfect_features(test))[:, 1])
        preds["fitted"].extend(logit.predict_proba(_numeric(test))[:, 1])
        # Home-field alone: the home team's share of wins in the fitting window.
        preds["home"].extend([float(y_train.mean())] * len(test))

    y = np.array(y_all, dtype=float)
    return {
        "scores": {k: _scores(y, np.array(v)) for k, v in preds.items()},
        "perfect_train_accuracy_min": min(train_acc) if train_acc else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("pa_path", nargs="?", default=str(DATA_DIR / "season_pa_v2.jsonl"))
    parser.add_argument("--refresh", action="store_true", help="refetch game results")
    args = parser.parse_args()

    results = load_results(args.pa_path, args.refresh)
    rows = build_games(args.pa_path, results)
    print(f"[winloss_fit.py] {len(results):,} final games, {len(rows):,} with inputs, "
          f"{rows[0]['date']} .. {rows[-1]['date']}")

    # The reverse-engineered algorithm, fitted on the whole season.
    y = np.array([r["home_win"] for r in rows])
    tree = fit_perfect(rows)
    in_sample = {
        "perfect": _scores(y, tree.predict_proba(_perfect_features(rows))[:, 1]),
        "current": _scores(y, np.array([r["current_prob"] for r in rows])),
        "fitted": _scores(y, fit_regularised(rows).predict_proba(_numeric(rows))[:, 1]),
    }
    with open(DATA_DIR / "winloss_perfect_fit.pkl", "wb") as fh:
        pickle.dump(tree, fh)

    wf = walk_forward(rows)
    report = {
        "games": len(rows),
        "home_win_rate": float(y.mean()),
        "perfect_tree": {"depth": int(tree.get_depth()), "leaves": int(tree.get_n_leaves())},
        "in_sample": in_sample,
        "walk_forward": wf,
    }
    (DATA_DIR / "winloss_fit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    names = {
        "perfect": "Perfect fit (tree)", "current": "Current runs model",
        "fitted": "Regularised fit", "home": "Always pick home",
    }
    print("\n" + "=" * 78)
    print(f"FIT TO THE SEASON THAT PRODUCED IT  ({len(rows):,} games, "
          f"home won {y.mean():.1%})")
    print(f"{'':<22}{'Accuracy':>10}{'Brier':>10}{'LogLoss':>10}")
    for k, s in in_sample.items():
        print(f"{names[k]:<22}{s['accuracy']:>10.1%}{s['brier']:>10.4f}{s['log_loss']:>10.4f}")
    print(f"  perfect-fit tree: {tree.get_n_leaves():,} leaves for {len(rows):,} games, "
          f"depth {tree.get_depth()}")

    s0 = wf["scores"]["current"]
    print(f"\nPREDICTING GAMES NOT YET SEEN  (walk-forward, {s0['n']:,} games)")
    print(f"{'':<22}{'Accuracy':>10}{'± 1 SE':>9}{'Brier':>10}{'LogLoss':>10}")
    for k in ("current", "fitted", "home", "perfect"):
        s = wf["scores"][k]
        print(f"{names[k]:<22}{s['accuracy']:>10.1%}{s['accuracy_se']:>8.1%} "
              f"{s['brier']:>10.4f}{s['log_loss']:>10.4f}")
    print(f"  perfect fit reproduced its own fitting window at no less than "
          f"{wf['perfect_train_accuracy_min']:.0%} in every week")
    print("=" * 78)
    print(f"[winloss_fit.py] wrote {DATA_DIR / 'winloss_fit.json'}")


if __name__ == "__main__":
    main()
