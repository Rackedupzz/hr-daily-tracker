"""HOMER's homework: replay the season, learn what matters, prove it.

HOMER's scouting factors (`homer.hr_features`, `homer.game_features`) say what
he looks at. This decides how much each is worth, and whether the result is
any good, using only games he had not seen:

1. **Replay.** The season is cut into weeks. For each week HOMER's book is
   frozen at the week's first day and every hitter in every starting lineup
   that week gets a scouting report -- the same function the live card uses --
   paired with whether he homered. Every game gets a report paired with who won.
   Lineups, starters and weather are the pre-game facts the live card also has.
2. **Walk forward.** A logistic regression is fitted on every week before week
   k and predicts week k, so no prediction has seen its own outcome.
3. **Keep what works.** Factors are removed one at a time while removing one
   improves held-out log loss. What survives is what carries forward.
4. **Decide.** The learned weights are served only if they beat HOMER's
   original rule-of-thumb formula on the same held-out games.
5. **Evidence.** HOMER's daily six are re-run through his live selection rules
   on every held-out day, and grades and confidence labels are set from
   measured hit rates, so a card can say what picks like each one did.

Run: python -m mlb_hr.homer_fit            (writes data/homer_fit.json, ~3 min)
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from mlb_hr import homer

DATA_DIR = Path(__file__).parent / "data"
PA_PATH = DATA_DIR / "season_pa_v2.jsonl"
RESULTS_PATH = DATA_DIR / "game_results.json"
WINLOSS_PATH = DATA_DIR / "winloss_fit.json"

FIRST_WEEK = date(2026, 4, 22)   # about four weeks of book before the first replay
MIN_TRAIN_WEEKS = 3
C = 0.5


# ─────────────────────────────────────────────────────────────── outcomes


def game_results(through: date) -> list[dict]:
    """Final scores, topped up from the API when the cache is behind."""
    cached = []
    if RESULTS_PATH.exists():
        cached = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    last = cached[-1]["date"] if cached else "2026-03-01"
    if last < through.isoformat():
        try:
            from mlb_hr.winloss_fit import fetch_game_results
            fresh = fetch_game_results(date.fromisoformat(last), through)
            merged = {g["game_pk"]: g for g in cached}
            merged.update({g["game_pk"]: g for g in fresh})
            cached = sorted(merged.values(), key=lambda g: (g["date"], g["game_pk"]))
            RESULTS_PATH.write_text(json.dumps(cached), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - replay what is cached
            print(f"[homer_fit] could not top up game results ({exc})")
    return cached


def standings_by_date(results: list[dict]) -> dict:
    """date -> ({team: {rs, ra, games}}, league runs per team-game) before that date."""
    teams = defaultdict(lambda: {"rs": 0, "ra": 0, "games": 0})
    runs = games = 0
    out = {}
    by_date = defaultdict(list)
    for g in results:
        by_date[g["date"]].append(g)
    for day in sorted(by_date):
        out[day] = ({t: dict(v) for t, v in teams.items()},
                    runs / (2 * games) if games else 4.5)
        for g in by_date[day]:
            for team, rs, ra in ((g["home"], g["home_score"], g["away_score"]),
                                 (g["away"], g["away_score"], g["home_score"])):
                teams[team]["rs"] += rs
                teams[team]["ra"] += ra
                teams[team]["games"] += 1
            runs += g["home_score"] + g["away_score"]
            games += 1
    return out


# ───────────────────────────────────────────────────────────────── replay


def _setup(game_rows: list) -> dict:
    """Pre-game facts for one game, read back from its plate appearances."""
    game_rows.sort(key=lambda r: r.pa_index or 0)
    first = game_rows[0]
    starters, lineups, homered = {}, defaultdict(dict), defaultdict(int)
    hits = defaultdict(int)
    home = away = None
    for r in game_rows:
        if r.is_home:
            home = r.batting_team
        else:
            away = r.batting_team
        starters.setdefault(r.pitching_team, (r.pitcher_id, r.pitch_hand))
        # The first hitter to bat in each slot is the one who started there.
        if r.lineup_slot and r.lineup_slot not in lineups[r.batting_team]:
            lineups[r.batting_team][r.lineup_slot] = (r.batter_id, r.bat_side)
        if r.is_hr:
            homered[r.batter_id] += 1
        if r.event_type in homer._HIT_EVENTS:
            hits[r.batter_id] += 1
    return {
        "game_pk": first.game_pk, "date": first.date, "venue": first.venue,
        "home": home, "away": away, "starters": starters, "lineups": lineups,
        "homered": homered, "hits": hits,
        "weather": {"temp_f": first.temp_f, "wind_mph": first.wind_mph,
                    "wind_dir": first.wind_dir, "condition": first.condition},
    }


def replay(rows: list, results: list[dict], through: date) -> tuple[list, list]:
    """(hitter-game samples, game samples) for every week in the replay."""
    by_game = defaultdict(list)
    for r in rows:
        if r.date and FIRST_WEEK.isoformat() <= r.date < through.isoformat():
            by_game[r.game_pk].append(r)
    setups = [_setup(g) for g in by_game.values()]
    finals = {g["game_pk"]: g for g in results}
    standings = standings_by_date(results)

    weeks = []
    cur = FIRST_WEEK
    while cur < through:
        weeks.append(cur)
        cur += timedelta(days=7)

    hr_samples, game_samples = [], []
    for week in weeks:
        end = min(week + timedelta(days=7), through)
        todo = [s for s in setups if week.isoformat() <= s["date"] < end.isoformat()]
        if not todo:
            continue
        book = homer.build_book(rows, week)
        wk = week.isoformat()
        for s in todo:
            if not s["home"] or not s["away"]:
                continue
            for team, opp in ((s["home"], s["away"]), (s["away"], s["home"])):
                sp_id, sp_hand = s["starters"].get(opp, (None, None))
                for slot, (bid, bat_side) in s["lineups"][team].items():
                    c = homer.hr_features(book, bid, bat_side, sp_id, sp_hand, opp,
                                          s["venue"], s["weather"], slot)
                    h = homer.hit_features(book, bid, bat_side, sp_id, sp_hand, opp,
                                           s["venue"], slot)
                    if not c or not h:
                        continue
                    base = 1 - (1 - c["hr_rate"]) ** c["exp_pa"]
                    hr_samples.append({
                        "week": wk, "date": s["date"], "game_pk": s["game_pk"],
                        "batter_id": bid,
                        "x": [c["x"][f] for f in homer.HR_FEATURES],
                        "rules": c["rules_prob"], "base": base,
                        "season_pa": c["season_pa"], "y": int(s["homered"][bid] > 0),
                        "xh": [h["x"][f] for f in homer.HIT_FEATURES],
                        "hrules": h["rules_prob"],
                        "hbase": 1 - (1 - h["hit_rate"]) ** h["exp_pa"],
                        "yh": int(s["hits"][bid] > 0),
                    })

            final = finals.get(s["game_pk"])
            if not final:
                continue
            table, lg_rpg = standings.get(s["date"], ({}, 4.5))
            home_rec, away_rec = table.get(s["home"]), table.get(s["away"])
            g = homer.game_features(
                book, s["home"], s["away"], s["starters"].get(s["home"], (None,))[0],
                s["starters"].get(s["away"], (None,))[0], s["venue"],
                home_rec if home_rec and home_rec["games"] else None,
                away_rec if away_rec and away_rec["games"] else None, lg_rpg,
            )
            game_samples.append({
                "week": wk, "date": s["date"], "game_pk": s["game_pk"],
                "x": [g["x"][f] for f in homer.GAME_FEATURES], "rules": g["rules_home"],
                "y": int(final["home_score"] > final["away_score"]),
            })
        print(f"[homer_fit] week {wk}: {len(hr_samples)} hitter-games, "
              f"{len(game_samples)} games so far")
    return hr_samples, game_samples


# ──────────────────────────────────────────────────────────── walk forward


def _fit(X: np.ndarray, y: np.ndarray):
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=2000).fit(scaler.transform(X), y)
    return scaler, clf


def walk_forward(X: np.ndarray, y: np.ndarray, weeks: np.ndarray, cols: list[int]) -> np.ndarray:
    preds = np.full(len(y), np.nan)
    uniq = sorted(set(weeks))
    for i, wk in enumerate(uniq):
        if i < MIN_TRAIN_WEEKS:
            continue
        train, test = weeks < wk, weeks == wk
        if len(set(y[train])) < 2:
            continue
        scaler, clf = _fit(X[train][:, cols], y[train])
        preds[test] = clf.predict_proba(scaler.transform(X[test][:, cols]))[:, 1]
    return preds


def _log_loss(y, p) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def scores(y: np.ndarray, p: np.ndarray) -> dict:
    out = {"n": int(len(y)), "log_loss": round(_log_loss(y, p), 5),
           "brier": round(float(np.mean((p - y) ** 2)), 5)}
    if len(set(y)) > 1:
        out["auc"] = round(float(roc_auc_score(y, p)), 4)
    acc = float(np.mean((p >= 0.5) == (y == 1)))
    out["accuracy"] = round(acc, 4)
    out["accuracy_se"] = round(math.sqrt(acc * (1 - acc) / max(len(y), 1)), 4)
    return out


def select_features(X, y, weeks, names: list[str]) -> tuple[list[str], list[dict]]:
    """Backward elimination on held-out log loss. Returns (kept, drop log)."""
    cols = list(range(len(names)))
    mask = None

    def loss(c):
        nonlocal mask
        p = walk_forward(X, y, weeks, c)
        if mask is None:
            mask = ~np.isnan(p)
        return _log_loss(y[mask], p[mask])

    current = loss(cols)
    log = []
    while len(cols) > 1:
        trials = [(loss([c for c in cols if c != drop]), drop) for drop in cols]
        best, drop = min(trials)
        if best >= current - 1e-6:
            break
        log.append({"dropped": names[drop], "log_loss_before": round(current, 6),
                    "log_loss_after": round(best, 6)})
        print(f"[homer_fit] dropping {names[drop]}: {current:.6f} -> {best:.6f}")
        cols = [c for c in cols if c != drop]
        current = best
    return [names[c] for c in cols], log


def final_model(X, y, names: list[str], kept: list[str]) -> dict:
    cols = [names.index(f) for f in kept]
    scaler, clf = _fit(X[:, cols], y)
    return {
        "features": kept,
        "coef": [round(float(v), 6) for v in clf.coef_[0]],
        "intercept": round(float(clf.intercept_[0]), 6),
        "mean": [round(float(v), 6) for v in scaler.mean_],
        "scale": [round(float(v), 6) for v in scaler.scale_],
    }


def weights_table(model: dict) -> list[dict]:
    """Plain-English weights: odds change for a one-standard-deviation edge."""
    player = set(homer.HR_FEATURES) | set(homer.HIT_FEATURES)
    overlap = homer.overlap_features(model) if set(model["features"]) <= player else set()
    rows = [{"feature": f, "label": homer.FEATURE_LABELS.get(f, f),
             "odds_per_sd": round(math.exp(w) - 1, 4), "overlap": f in overlap}
            for f, w in zip(model["features"], model["coef"])]
    return sorted(rows, key=lambda r: abs(r["odds_per_sd"]), reverse=True)


# ─────────────────────────────────────────────────────────────── evidence


def daily_picks(samples: list[dict], key: str, ykey: str = "y") -> dict:
    """HOMER's live selection rules re-run on every held-out day."""
    by_day = defaultdict(list)
    for s in samples:
        by_day[s["date"]].append({"game_pk": s["game_pk"], "prob": s[key],
                                  "season_pa": s["season_pa"], "y": s[ykey]})
    hit = n = 0
    expected = 0.0
    for cands in by_day.values():
        for p in homer.select_hr_picks(cands):
            n += 1
            hit += p["y"]
            expected += p["prob"]
    rate = hit / n if n else 0.0
    return {"days": len(by_day), "picks": n, "hit": hit, "rate": round(rate, 4),
            "rate_se": round(math.sqrt(rate * (1 - rate) / max(n, 1)), 4),
            "expected": round(expected, 1)}


def pav_calibration(p: np.ndarray, y: np.ndarray, blocks: int = 40) -> dict:
    """Monotone map from predicted to observed rate (pool adjacent violators).

    Fitted on walk-forward predictions, so it corrects the overconfidence the
    replay actually measured without changing anyone's rank.
    """
    order = np.argsort(p)
    ps, ys = p[order], y[order]
    size = max(len(ps) // blocks, 1)
    pools = [[float(ps[i:i + size].sum()), float(ys[i:i + size].sum()), len(ps[i:i + size])]
             for i in range(0, len(ps), size)]
    merged: list = []
    for block in pools:
        merged.append(block)
        while len(merged) > 1 and merged[-2][1] / merged[-2][2] >= merged[-1][1] / merged[-1][2]:
            b = merged.pop()
            merged[-1] = [merged[-1][0] + b[0], merged[-1][1] + b[1], merged[-1][2] + b[2]]
    return {"x": [round(sp / n, 5) for sp, _, n in merged],
            "y": [round(sy / n, 5) for _, sy, n in merged]}


def price_samples(samples: list[dict]) -> dict:
    """Model the book's line on every held-out leg.

    The market curve maps the season line (rate over expected trips) to what it
    actually cashed at, so the modeled book is an honest season-rate book, not a
    straw man. Returns the curves for the fit; samples gain `hr_leg`/`hit_leg`.
    """
    curves = {}
    for kind, base_key, ykey, serve_key, leg_key in (
            ("hr", "base", "y", "hr_serve", "hr_leg"),
            ("hits", "hbase", "yh", "hits_serve", "hit_leg")):
        base = np.array([s[base_key] for s in samples])
        curves[kind] = pav_calibration(base, np.array([s[ykey] for s in samples]))
        fit = {kind: {"market": curves[kind]}}
        for s in samples:
            s[leg_key] = homer.price_leg(
                {"game_pk": s["game_pk"], "batter_id": s["batter_id"], "prob": s[serve_key],
                 "season_pa": s["season_pa"], "y": s[ykey]}, kind, s[base_key], fit)
    return curves


def book_check(samples: list[dict]) -> dict:
    """Leg by leg: does HOMER's edge clear the book's margin?

    HOMER's daily five legs of each kind, picked by likelihood and by value,
    each bet straight at the modeled book price.
    """
    by_day = defaultdict(list)
    for s in samples:
        by_day[s["date"]].append(s)
    out = {}
    for kind, fit_kind, leg_key in (("hr", "hr", "hr_leg"), ("hit", "hits", "hit_leg")):
        for build in ("prob", "value"):
            legs = []
            for rows in by_day.values():
                pool = [s[leg_key] for s in rows if homer.eligible(s[leg_key])]
                used = set()
                for c in sorted(pool, key=homer._leg_rank(build, fit_kind), reverse=True):
                    if c["game_pk"] in used:
                        continue
                    used.add(c["game_pk"])
                    legs.append(c)
                    if len(used) == 5:
                        break
            n = max(len(legs), 1)
            out[f"{kind}_{build}"] = {
                "legs": len(legs), "actual": round(sum(c["y"] for c in legs) / n, 4),
                "homer": round(sum(c["prob"] for c in legs) / n, 4),
                "market": round(sum(c["market_prob"] for c in legs) / n, 4),
                "break_even": round(sum(c["book_implied"] for c in legs) / n, 4),
                "roi": round(sum((1 / c["book_implied"] - 1) if c["y"] else -1
                                 for c in legs) / n, 4),
            }
    return {"margins": homer.BOOK_MARGIN, "legs": out}


def parlay_record(samples: list[dict]) -> dict:
    """Every play type, built with the live rules on every held-out day.

    A play cashes when every leg does. Its predicted rate is the average of the
    legs' product, so the page can say whether the combined probabilities HOMER
    quotes are honest. Returns at the modeled book price are an estimate: the
    record says how often it cashed, the price says what that would have paid.
    """
    by_day = defaultdict(list)
    for s in samples:
        by_day[s["date"]].append(s)
    out = {p["key"]: {"name": p["name"], "legs": p["legs"], "hr": p["hr"],
                      "value": p.get("value", 0), "days": 0, "won": 0,
                      "predicted": 0.0, "prices": [], "profit": 0.0, "wins_on": []}
           for p in homer.PARLAY_PLAYS}
    for day, rows in sorted(by_day.items()):
        for play in homer.build_parlays([s["hr_leg"] for s in rows], [s["hit_leg"] for s in rows]):
            rec = out[play["key"]]
            won = all(leg["y"] for leg in play["leg_list"])
            rec["days"] += 1
            rec["won"] += won
            rec["predicted"] += play["prob"]
            rec["prices"].append(play["book_decimal"])
            rec["profit"] += play["book_decimal"] - 1 if won else -1
            if won:
                rec["wins_on"].append(day)
    for rec in out.values():
        n = max(rec["days"], 1)
        prices = sorted(rec.pop("prices"))
        rec["rate"] = round(rec["won"] / n, 4)
        rec["rate_se"] = round(math.sqrt(rec["rate"] * (1 - rec["rate"]) / n), 4)
        rec["predicted"] = round(rec["predicted"] / n, 4)
        rec["typical_odds"] = homer.american(prices[len(prices) // 2]) if prices else None
        rec["est_roi"] = round(rec.pop("profit") / n, 4)
        rec["days_per_win"] = round(n / rec["won"], 1) if rec["won"] else None
    return out


def fit_target(samples: list[dict], xkey: str, ykey: str, rules_key: str, base_key: str,
               names: list[str], label: str) -> tuple[dict, list[dict]]:
    """Feature selection, walk-forward scoring, calibration and grades for one target.

    Returns the fit block and the held-out samples, each carrying `<label>_cal`
    (calibrated learned probability) and `<label>_serve` (whatever is served).
    """
    X = np.array([s[xkey] for s in samples], dtype=float)
    y = np.array([s[ykey] for s in samples], dtype=int)
    weeks = np.array([s["week"] for s in samples])
    kept, dropped = select_features(X, y, weeks, names)
    preds = walk_forward(X, y, weeks, [names.index(f) for f in kept])
    held = ~np.isnan(preds)
    yt = y[held]
    calibration = pav_calibration(preds[held], yt)
    calibrated = np.interp(preds[held], calibration["x"], calibration["y"])
    test = [s for s, h in zip(samples, held) if h]
    rules = np.array([s[rules_key] for s in test])
    block_scores = {
        "learned": scores(yt, preds[held]),
        "rules": scores(yt, rules),
        "baseline": scores(yt, np.array([s[base_key] for s in test])),
    }
    serve = "learned" if block_scores["learned"]["log_loss"] < block_scores["rules"]["log_loss"] else "rules"
    for s, c in zip(test, calibrated):
        s[f"{label}_cal"] = float(c)
        s[f"{label}_serve"] = float(c) if serve == "learned" else s[rules_key]
    model = final_model(X, y, names, kept)
    model["calibration"] = calibration
    eligible = [s for s in test if s["season_pa"] >= homer.MIN_SEASON_PA]
    graded = [dict(s, calibrated=s[f"{label}_serve"], y=s[ykey]) for s in test]
    block = {
        "serve": serve, "scores": block_scores, "dropped": dropped, "model": model,
        "weights": weights_table(model),
        "picks": {"calibrated": daily_picks(test, f"{label}_serve", ykey),
                  "rules": daily_picks(test, rules_key, ykey),
                  "base": daily_picks(test, base_key, ykey)},
        "board_rate": round(sum(s[ykey] for s in eligible) / max(len(eligible), 1), 4),
        "grades": grade_bands(graded),
        "samples": len(samples), "held_out": int(held.sum()),
        "held_out_from": min(s["date"] for s in test) if test else None,
        "actual_rate": round(float(yt.mean()), 4) if len(yt) else None,
    }
    print(f"[homer_fit] {label}: serving {serve}; {json.dumps(block_scores)}")
    return block, test


def grade_bands(samples: list[dict], key: str = "calibrated") -> list[dict]:
    """Letter grades from the held-out board.

    Candidate bands are cut by board percentile, then merged until every
    higher band actually homered more often than the one below it -- a grade
    only exists if the results say it means something.
    """
    eligible = sorted((s for s in samples if s["season_pa"] >= homer.MIN_SEASON_PA),
                      key=lambda s: s[key], reverse=True)
    if not eligible:
        return []
    cuts = [0.02, 0.05, 0.13, 0.30, 1.0]
    bands, start = [], 0
    for share in cuts:
        end = len(eligible) if share == 1.0 else max(start + 1, int(len(eligible) * share))
        chunk = eligible[start:end]
        if chunk:
            bands.append({"top_share": share, "rows": chunk})
        start = end

    def rate(b):
        return sum(s["y"] for s in b["rows"]) / len(b["rows"])

    merged = True
    while merged and len(bands) > 1:
        merged = False
        for i in range(len(bands) - 1):
            if rate(bands[i]) <= rate(bands[i + 1]):
                bands[i] = {"top_share": bands[i + 1]["top_share"],
                            "rows": bands[i]["rows"] + bands[i + 1]["rows"]}
                del bands[i + 1]
                merged = True
                break

    out = []
    for letter, b in zip("ABCDE", bands):
        rows = b["rows"]
        last = b is bands[-1]
        out.append({
            "grade": letter, "min_prob": 0.0 if last else round(rows[-1][key], 4),
            "top_share": b["top_share"], "n": len(rows), "hit": sum(s["y"] for s in rows),
            "rate": round(rate(b), 4),
            "avg_prob": round(sum(s[key] for s in rows) / len(rows), 4),
        })
    return out


def confidence_bands(samples: list[dict]) -> list[dict]:
    out = []
    edges = (("Strong", 0.60, 1.01), ("Lean", 0.55, 0.60), ("Coin flip", 0.0, 0.55))
    for label, lo, hi in edges:
        chunk = [s for s in samples if lo <= s["pick_prob"] < hi]
        wins = sum(s["pick_won"] for s in chunk)
        out.append({"label": label, "min_prob": lo, "n": len(chunk), "won": wins,
                    "rate": round(wins / len(chunk), 4) if chunk else None})
    return out


def run(pa_path: Path = PA_PATH, out_path: Path = homer.FIT_PATH,
        through: Optional[date] = None) -> dict:
    through = through or date.today()
    print("[homer_fit] loading the season feed")
    rows = homer.load_rows(str(pa_path))
    results = game_results(through - timedelta(days=1))
    hr_s, game_s = replay(rows, results, through)
    del rows

    fit: dict = {"fitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 "through": (through - timedelta(days=1)).isoformat(),
                 "first_week": FIRST_WEEK.isoformat(), "min_train_weeks": MIN_TRAIN_WEEKS}

    # ── home runs and hits (same hitter-games, same held-out weeks)
    fit["hr"], _ = fit_target(hr_s, "x", "y", "rules", "base", homer.HR_FEATURES, "hr")
    fit["hits"], test = fit_target(hr_s, "xh", "yh", "hrules", "hbase",
                                   homer.HIT_FEATURES, "hits")

    # ── the book's line on every leg, then HOMER's plays built with the live rules
    curves = price_samples(test)
    fit["hr"]["market"], fit["hits"]["market"] = curves["hr"], curves["hits"]
    fit["book"] = book_check(test)
    print(f"[homer_fit] book: {json.dumps(fit['book'])}")
    fit["parlays"] = parlay_record(test)
    print(f"[homer_fit] parlays: {json.dumps(fit['parlays'])}")

    # ── winners
    gnames = homer.GAME_FEATURES
    GX = np.array([s["x"] for s in game_s], dtype=float)
    gy = np.array([s["y"] for s in game_s], dtype=int)
    gweeks = np.array([s["week"] for s in game_s])
    gkept, gdropped = select_features(GX, gy, gweeks, gnames)
    gpreds = walk_forward(GX, gy, gweeks, [gnames.index(f) for f in gkept])
    gheld = ~np.isnan(gpreds)
    gtest = [dict(s, learned=float(p)) for s, p, h in zip(game_s, gpreds, gheld) if h]
    gyt = gy[gheld]
    game_scores = {
        "learned": scores(gyt, gpreds[gheld]),
        "rules": scores(gyt, np.array([s["rules"] for s in gtest])),
        "home_team": scores(gyt, np.full(len(gyt), 0.53)),
    }
    gserve = ("learned" if game_scores["learned"]["log_loss"] < game_scores["rules"]["log_loss"]
              else "rules")
    key = "learned" if gserve == "learned" else "rules"
    for s in gtest:
        home_p = s[key]
        s["pick_prob"] = max(home_p, 1 - home_p)
        s["pick_won"] = int((home_p >= 0.5) == (s["y"] == 1))
    gmodel = final_model(GX, gy, gnames, gkept)
    house = None
    if WINLOSS_PATH.exists():
        try:
            house = json.loads(WINLOSS_PATH.read_text(encoding="utf-8"))["walk_forward"]["scores"]["current"]
        except (KeyError, ValueError):
            house = None
    fit["games"] = {
        "serve": gserve, "scores": game_scores, "dropped": gdropped, "model": gmodel,
        "weights": weights_table(gmodel), "confidence": confidence_bands(gtest),
        "samples": len(game_s), "held_out": int(gheld.sum()),
        "house_walk_forward": house,
    }
    print(f"[homer_fit] games: serving {gserve}; {json.dumps(game_scores)}")

    out_path.write_text(json.dumps(fit, indent=1), encoding="utf-8")
    print(f"[homer_fit] wrote {out_path}")
    return fit


if __name__ == "__main__":
    run()
