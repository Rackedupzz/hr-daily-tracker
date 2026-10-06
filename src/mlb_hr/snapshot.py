"""Daily archive: a PDF to read, a spreadsheet to analyse.

Two different jobs, so two artefacts.

The **PDF** is the page exactly as it stood -- projections, verdicts, game
cards, model scores -- frozen for a date. It answers "what did we say that
morning, and what happened", months later, without needing the model or the
API to still work.

The **tracker** is the same day reduced to rows. A PDF cannot answer "are the
25% picks actually hitting 25% over sixty days"; a CSV can, and that question
is the only way a six-pick-a-day slate ever accumulates into evidence. The CSV
is the source of truth (append-only by date, rewritten if a day is re-scored)
and the .xlsx is regenerated from it, so a corrupt spreadsheet is never a lost
record.

Nothing here is destructive: re-running for a date replaces that date's rows
and overwrites that date's PDF, and leaves every other day alone.

The day's full slate is also saved (`<date>-slate.json.gz`), so a day can be
settled after the fact: the archive is first written while games are still
being played, and `finalize_pending` rescores any recent day that still has
open picks once its box scores are final, without refitting -- the picks stay
exactly as published.

Run: python -m mlb_hr.snapshot             # snapshot the current slate
     python -m mlb_hr.snapshot 2026-09-09  # a specific date
     python -m mlb_hr.snapshot --finalize  # settle recent days with open picks
"""
from __future__ import annotations

import csv
import gzip
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

DEFAULT_DIR = Path("/app/snapshots")

# One row per projection, in the order a reader would want them.
PICK_COLUMNS = [
    "date", "kind", "rank", "batter", "batter_id", "team", "game_pk",
    "opponent_sp", "facing_hand", "lineup_slot", "projected", "projected_prob",
    "projected_per_pa", "expected_pa", "park_factor", "weather_factor",
    "bullpen_index", "sp_hr_index", "season_hr", "season_pa", "result_pa",
    "result_ab", "result_hits", "result_hr", "result_rbi", "result_line",
    "game_state", "final", "won",
]

SUMMARY_COLUMNS = [
    "date", "games", "hr_picks", "hr_picks_final", "hr_picks_hit",
    "hr_expected", "hit_picks", "hit_picks_final", "hit_picks_hit",
    "hit_expected", "avg_projected_hr", "picks_dnp", "settled", "built_at",
    # Every starter on the slate, not just the picks (see slate_calibration).
    "slate_starters", "slate_hr_expected", "slate_hr_base", "slate_hr_actual",
    "slate_hits_expected", "slate_hits_actual", "hr_level", "model_version",
    # The two 5-pick parlays (mlb_hr.parlays): won / lost / void / pending.
    "hr_parlay", "hr_parlay_prob", "hit_parlay", "hit_parlay_prob",
]

# The tracker's level (hr_level): recent actual-over-projected home runs across
# every starter, as an odds multiplier on tomorrow's projections. On the 2026
# replay a 7-day half-life with 300 home runs of pseudo-count toward 1.0 was
# worth +7.7 log-likelihood (90% CI +3.9 to +12.6) on top of the stacked model
# and cut the monthly miss from 0.89-1.15 to 0.92-1.03. The home-run
# environment runs hot and cold for stretches that weather alone does not
# explain.
LEVEL_HALF_LIFE_DAYS = 7.0
LEVEL_PSEUDO_HR = 300.0
LEVEL_LOOKBACK_DAYS = 45
LEVEL_CLAMP = (0.8, 1.25)

MODEL_COLUMNS = [
    "date", "prior", "auc", "top_decile_lift", "brier", "log_loss",
    "calibration_rmse", "n_test_samples", "cutoff_date",
]


def _out_dir(out_dir: Optional[Path] = None) -> Path:
    path = Path(out_dir) if out_dir else DEFAULT_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _opponent_sp(slate_data: dict, pick: dict) -> str:
    """The starter this hitter faces, dug out of his game record."""
    for game in slate_data.get("games", []):
        if game.get("game_pk") != pick.get("game_pk"):
            continue
        is_home = pick.get("side") == "home"
        return game.get("away_sp" if is_home else "home_sp") or ""
    return ""


def pick_rows(slate_data: dict) -> list[dict]:
    """Every projection on the slate, flattened, with its outcome attached."""
    day = slate_data.get("date", "")
    rows: list[dict] = []

    for kind, entries in (
        ("hr", slate_data.get("picks_6") or []),
        ("hit", slate_data.get("hit_picks") or []),
    ):
        for rank, pick in enumerate(entries, 1):
            line = pick.get("result") or {}
            hits_proj = pick.get("hits_proj") or {}
            proj = (
                hits_proj.get("projected_hits")
                if kind == "hit" and hits_proj
                else pick.get("prob_hr")
            )
            # The probability of the thing "won" records -- a home run, or at
            # least one hit -- so the two can be compared over many days.
            prob = (
                hits_proj.get("prob_at_least_one") if kind == "hit"
                else pick.get("prob_hr")
            )
            # "Won" is only meaningful once the game is over; an unfinished
            # game leaves it blank rather than counting as a loss, and so does
            # a pick who never came to the plate.
            if line.get("final") and not line.get("dnp"):
                won = bool(line.get("hr" if kind == "hr" else "hits", 0))
            else:
                won = None
            rows.append({
                "date": day,
                "kind": kind,
                "rank": rank,
                "batter": pick.get("batter", ""),
                "batter_id": pick.get("batter_id"),
                "team": pick.get("team", ""),
                "game_pk": pick.get("game_pk"),
                "opponent_sp": _opponent_sp(slate_data, pick),
                "facing_hand": pick.get("facing_hand", ""),
                "lineup_slot": pick.get("lineup_slot"),
                "projected": round(proj, 4) if proj is not None else None,
                "projected_prob": round(prob, 4) if prob is not None else None,
                "projected_per_pa": pick.get("prob_ensemble_per_pa"),
                "expected_pa": pick.get("prob_expected_pa"),
                "park_factor": pick.get("prob_park_effective"),
                "weather_factor": pick.get("prob_weather_factor"),
                "bullpen_index": pick.get("prob_pen_hr_index"),
                "sp_hr_index": pick.get("prob_sp_hr_index"),
                "season_hr": pick.get("season_hrs"),
                "season_pa": pick.get("season_pas"),
                "result_pa": line.get("pa"),
                "result_ab": line.get("ab"),
                "result_hits": line.get("hits"),
                "result_hr": line.get("hr"),
                "result_rbi": line.get("rbi"),
                "result_line": line.get("summary", ""),
                "game_state": line.get("detailed", ""),
                "final": line.get("final"),
                "won": won,
            })
    return rows


def summary_row(slate_data: dict) -> dict:
    """One row per day: what was promised against what was delivered."""
    row = summarize_rows(
        pick_rows(slate_data),
        day=slate_data.get("date", ""),
        games=slate_data.get("games_count", 0),
        built_at=slate_data.get("built_at", ""),
    )
    row.update(slate_calibration(slate_data))
    for play in slate_data.get("parlays") or []:
        prefix = "hr_parlay" if play.get("kind") == "hr" else "hit_parlay"
        row[prefix] = play.get("status", "pending") + (" (backfilled)" if play.get("backfilled") else "")
        row[f"{prefix}_prob"] = round(play["prob"], 6)
    return row


def slate_calibration(slate_data: dict) -> dict:
    """Projected against actual for every starter on the slate, not just the picks.

    Six picks a day is too few to judge calibration: about 36 home runs are
    expected over 30 days, so even a perfectly calibrated model misses by 10%
    or more in roughly half of all 30-day windows. The whole slate -- ~250
    starters, ~28 expected home runs a day -- settles that question in days.

    Counts every hitter with a posted lineup slot whose game is final and who
    came to the plate. Unposted-lineup hitters and substitutes are left out:
    they were projected as starters and would read as misses.
    """
    n = 0
    hr_exp = hr_base = hr_act = hits_exp = hits_act = 0.0
    for game in slate_data.get("games", []):
        for h in game.get("hitters", []):
            line = h.get("result") or {}
            if not h.get("lineup_slot") or not line.get("final") or not line.get("pa"):
                continue
            n += 1
            p = float(h.get("prob_hr") or 0.0)
            hr_exp += p
            hr_base += float(h.get("prob_ensemble_base", p) or p)
            hr_act += 1.0 if (line.get("hr") or 0) > 0 else 0.0
            hits_exp += float((h.get("hits_proj") or {}).get("projected_hits_raw") or 0.0)
            hits_act += float(line.get("hits") or 0)
    if not n:
        return {"hr_level": slate_data.get("hr_level"),
                "model_version": slate_data.get("model_version")}
    return {
        "slate_starters": n,
        "slate_hr_expected": round(hr_exp, 3),
        "slate_hr_base": round(hr_base, 3),
        "slate_hr_actual": int(hr_act),
        "slate_hits_expected": round(hits_exp, 2),
        "slate_hits_actual": int(hits_act),
        "hr_level": slate_data.get("hr_level"),
        "model_version": slate_data.get("model_version"),
    }


def hr_level(out_dir: Optional[Path] = None, before: Optional[date] = None,
             model_version: Optional[str] = None) -> float:
    """Odds multiplier from recent actual-over-projected home runs (see LEVEL_*).

    Reads the daily summary's whole-slate columns for settled days before
    `before`, from the current model version only -- a different model's
    projections say nothing about this one's level. It compares actual home
    runs with the projections *before* any level was applied, so it measures
    the model rather than its own last correction. 1.0 with no record.
    """
    before = before or date.today()
    earliest = str(before - timedelta(days=LEVEL_LOOKBACK_DAYS))
    actual = expected = 0.0
    for r in _read_csv(_out_dir(out_dir) / "daily_summary.csv"):
        day = r.get("date") or ""
        if not (earliest <= day < str(before)) or not _truthy(r.get("settled")):
            continue
        if model_version and r.get("model_version") != model_version:
            continue
        base, act = _num(r.get("slate_hr_base")), _num(r.get("slate_hr_actual"))
        if base is None or act is None or base <= 0:
            continue
        age = (before - date.fromisoformat(day)).days
        w = 0.5 ** (age / LEVEL_HALF_LIFE_DAYS)
        actual += w * act
        expected += w * base
    level = (actual + LEVEL_PSEUDO_HR) / (expected + LEVEL_PSEUDO_HR)
    return float(min(max(level, LEVEL_CLAMP[0]), LEVEL_CLAMP[1]))


def _truthy(value) -> bool:
    return value is True or str(value).strip().lower() == "true"


def _num(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_rows(rows: list[dict], day: str, games, built_at: str) -> dict:
    """The daily summary from pick rows, whether fresh or read back from CSV."""
    hr = [r for r in rows if r["kind"] == "hr"]
    hits = [r for r in rows if r["kind"] == "hit"]

    def settled(r) -> bool:
        return r.get("won") not in (None, "")

    def tally(group):
        # Only picks who batted in a finished game count either way.
        final = [r for r in group if settled(r)]
        won = [r for r in final if _truthy(r["won"])]
        # Expected wins is the sum of the probabilities over the same picks,
        # which is what makes a 2-for-6 day readable: it is only a miss
        # against 1.6, not against 6.
        probs = [prob_of(r) for r in final]
        expected = None if None in probs else round(sum(probs), 3)
        return len(group), len(final), len(won), expected

    def prob_of(r) -> Optional[float]:
        p = _num(r.get("projected_prob"))
        if p is None and r["kind"] == "hr":
            p = _num(r.get("projected"))  # rows written before projected_prob
        return p

    n_hr, hr_final, hr_won, hr_expected = tally(hr)
    n_hit, hit_final, hit_won, hit_expected = tally(hits)
    projected = [_num(r["projected"]) for r in hr if _num(r["projected"]) is not None]
    dnp = sum(1 for r in rows if _truthy(r.get("final")) and not settled(r))

    return {
        "date": day,
        "games": games,
        "hr_picks": n_hr,
        "hr_picks_final": hr_final,
        "hr_picks_hit": hr_won,
        "hr_expected": hr_expected,
        "hit_picks": n_hit,
        "hit_picks_final": hit_final,
        "hit_picks_hit": hit_won,
        "hit_expected": hit_expected if hits else None,
        "avg_projected_hr": (
            round(sum(projected) / len(projected), 4) if projected else None
        ),
        "picks_dnp": dnp,
        # Every pick has either a result or a confirmed no-result.
        "settled": bool(rows) and all(_truthy(r.get("final")) for r in rows),
        "built_at": built_at,
    }


def model_rows(slate_data: dict) -> list[dict]:
    """Each prior's held-out score on this date's fit, for tracking drift."""
    metrics = slate_data.get("ensemble_metrics") or {}
    day = slate_data.get("date", "")
    rows = []
    for prior in ("empirical_bayes", "knn", "svm", "rf", "core", "forest",
                  "linear", "logistic", "neural", "xgboost"):
        m = metrics.get(prior)
        if not isinstance(m, dict) or "auc" not in m:
            continue
        rows.append({
            "date": day,
            "prior": prior,
            "auc": round(m.get("auc", 0), 5),
            "top_decile_lift": round(m.get("top_decile_lift", 0), 4),
            "brier": round(m.get("brier", 0), 6),
            "log_loss": round(m.get("log_loss", 0), 6),
            "calibration_rmse": round(m.get("calibration_rmse", 0), 6),
            "n_test_samples": metrics.get("n_test_samples"),
            "cutoff_date": metrics.get("cutoff_date", ""),
        })
    return rows


# ------------------------------------------------------------------ tracking


def _merge_csv(path: Path, rows: list[dict], columns: list[str], day: str):
    """Replace this date's rows, keep every other date, write back sorted.

    Re-running a day is normal -- the first snapshot of an evening has games in
    progress and the last one has them final -- so the merge is by date rather
    than append-only.
    """
    import csv

    existing: list[dict] = []
    if path.exists():
        with open(path, newline="", encoding="utf-8") as fh:
            existing = [r for r in csv.DictReader(fh) if r.get("date") != day]

    merged = existing + [{k: r.get(k) for k in columns} for r in rows]
    merged.sort(key=lambda r: (str(r.get("date", "")), str(r.get("kind", "")),
                               str(r.get("rank", "")), str(r.get("prior", ""))))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(merged)
    return merged


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_tracker(slate_data: dict, out_dir: Optional[Path] = None) -> dict:
    """Update the CSVs and rebuild the workbook from them."""
    return _write_tracker_rows(
        _out_dir(out_dir), slate_data.get("date", ""), pick_rows(slate_data),
        summary_row(slate_data), model_rows(slate_data),
    )


def _write_tracker_rows(out: Path, day: str, picks_rows: list[dict],
                        summary: dict, models_rows: Optional[list[dict]]) -> dict:
    """Merge one day's rows into the CSVs; `models_rows=None` leaves scores alone."""
    picks = _merge_csv(out / "picks.csv", picks_rows, PICK_COLUMNS, day)
    summary = _merge_csv(
        out / "daily_summary.csv", [summary], SUMMARY_COLUMNS, day
    )
    if models_rows is None:
        models = _read_csv(out / "model_scores.csv")
    else:
        models = _merge_csv(
            out / "model_scores.csv", models_rows, MODEL_COLUMNS, day
        )

    written = {"picks.csv": len(picks), "daily_summary.csv": len(summary),
               "model_scores.csv": len(models)}

    # The workbook is a convenience view, always rebuilt from the CSVs. If
    # openpyxl is missing the CSVs still stand on their own.
    try:
        import pandas as pd

        with pd.ExcelWriter(out / "tracker.xlsx", engine="openpyxl") as writer:
            frames = {
                "Daily Summary": pd.DataFrame(summary, columns=SUMMARY_COLUMNS),
                "Picks": pd.DataFrame(picks, columns=PICK_COLUMNS),
                "Model Scores": pd.DataFrame(models, columns=MODEL_COLUMNS),
            }
            for sheet, frame in frames.items():
                frame.to_excel(writer, sheet_name=sheet, index=False)
                worksheet = writer.sheets[sheet]
                for i, column in enumerate(frame.columns, 1):
                    width = max(len(str(column)) + 2, 12)
                    worksheet.column_dimensions[
                        worksheet.cell(row=1, column=i).column_letter
                    ].width = min(width, 40)
                worksheet.freeze_panes = "A2"
        written["tracker.xlsx"] = sum(len(f) for f in frames.values())
    except Exception as exc:  # noqa: BLE001 - CSVs are the source of truth
        print(f"[snapshot.py] workbook not written ({exc}); CSVs are current")

    return written


# ---------------------------------------------------------------------- PDF


def write_pdf(slate_data: dict, out_dir: Optional[Path] = None,
              include_models: bool = True) -> list[Path]:
    """Render the day's pages to PDF. Returns the files written."""
    from mlb_hr.render import render_html, render_models_html
    from mlb_hr.ui import printing

    out = _out_dir(out_dir)
    day = slate_data.get("date", "unknown")
    written: list[Path] = []

    try:
        from weasyprint import HTML
    except ImportError as exc:
        print(f"[snapshot.py] weasyprint unavailable ({exc}); saving HTML instead")
        path = out / f"{day}-slate.html"
        with printing():
            html = render_html(slate_data)
        path.write_text(html, encoding="utf-8")
        return [path]

    # Print mode: light theme, no script, no remote images to wait on.
    with printing():
        pages = [(f"{day}-slate.pdf", render_html(slate_data))]
        if include_models and slate_data.get("ensemble_metrics"):
            pages.append((f"{day}-models.pdf", render_models_html(slate_data)))

    for name, html in pages:
        path = out / name
        HTML(string=html).write_pdf(str(path))
        written.append(path)
    return written


def write_snapshot(slate_data: dict, out_dir: Optional[Path] = None) -> dict:
    """Everything for one day: PDFs, CSV rows, workbook and a JSON sidecar."""
    out = _out_dir(out_dir)
    day = slate_data.get("date", "unknown")

    pdfs = write_pdf(slate_data, out)
    tracker = write_tracker(slate_data, out)

    # The raw rows, so a later question can be answered without re-deriving
    # anything from a PDF.
    sidecar = out / f"{day}-picks.json"
    sidecar.write_text(
        json.dumps(
            {"date": day, "built_at": slate_data.get("built_at"),
             "results": slate_data.get("results"),
             "summary": summary_row(slate_data),
             "picks": pick_rows(slate_data),
             "parlays": slate_data.get("parlays") or []},
            indent=2, default=str,
        ),
        encoding="utf-8",
    )
    save_slate(slate_data, out)

    return {
        "date": day,
        "pdfs": [str(p) for p in pdfs],
        "json": str(sidecar),
        "tracker": tracker,
    }


# ------------------------------------------------------------ settling days


def _jsonable(value):
    """json.dumps fallback for the numpy scalars and sets a slate carries."""
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if hasattr(value, "tolist"):  # numpy scalars and arrays
        return value.tolist()
    return str(value)


def _slate_path(out: Path, day: str) -> Path:
    return out / f"{day}-slate.json.gz"


def save_slate(slate_data: dict, out_dir: Optional[Path] = None) -> Path:
    """The whole slate as published, so the day can be re-scored later."""
    out = _out_dir(out_dir)
    path = _slate_path(out, slate_data.get("date", "unknown"))
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(slate_data, fh, default=_jsonable)
    tmp.replace(path)
    return path


def load_slate(day: str, out_dir: Optional[Path] = None) -> Optional[dict]:
    path = _slate_path(_out_dir(out_dir), day)
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[snapshot.py] could not read {path.name} ({exc})")
        return None


def _batter_ids(pa_path: Optional[str], names: set) -> dict:
    """Batter name -> id from the season feed, for rows written without ids."""
    found: dict = {}
    if not pa_path or not names or not Path(pa_path).exists():
        return found
    with open(pa_path, encoding="utf-8") as fh:
        for line in fh:
            if not any(n in line for n in names - found.keys()):
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("batter") in names and r.get("batter_id"):
                found[r["batter"]] = r["batter_id"]
            if len(found) == len(names):
                break
    return found


def _placeholder_pick(r: dict, games: list[dict], batter_id) -> Optional[dict]:
    """A published pick who is not in the refit (usually not in the lineup).

    Kept so the day records him -- as did-not-play, or with his line if he
    came off the bench -- instead of silently dropping a published pick.
    """
    game = next((g for g in games if r["team"] in (g.get("home"), g.get("away"))), None)
    projected = _num(r.get("projected"))
    if game is None or projected is None:
        return None
    pick = {
        "batter": r["batter"], "batter_id": batter_id, "team": r["team"],
        "game_pk": game["game_pk"], "facing_hand": r.get("facing_hand", ""),
        "side": "home" if r["team"] == game.get("home") else "away",
        "season_hrs": int(_num(r.get("season_hr")) or 0),
        "season_pas": int(_num(r.get("season_pa")) or 0),
        "prob_hr": projected if r["kind"] == "hr" else 0.0,
    }
    if r["kind"] == "hit":
        pick["hits_proj"] = {
            "projected_hits": projected, "projected_hits_raw": projected,
            "prob_at_least_one": 0.0, "hitter_rate": 0.0, "sp_hit_rate": 0.0,
            "rate_vs_sp": 0.0, "pa_vs_sp": 0.0, "rate_vs_pen": 0.0, "pa_vs_pen": 0.0,
        }
    return pick


def published_picks(day: str, fresh: dict, out_dir: Optional[Path] = None,
                    pa_path: Optional[str] = None) -> Optional[dict]:
    """The day's published picks, rebuilt from the tracker CSV.

    Fallback for locking picks when no saved slate exists (days archived before
    slates were saved). Each row is matched to the same hitter in a freshly
    built slate for his game and team, and the published projection is put
    back in place of the refit's number. A pick missing from the refit is kept
    as a placeholder rather than dropped.
    """
    rows = [r for r in _read_csv(_out_dir(out_dir) / "picks.csv") if r["date"] == day]
    if not rows:
        return None
    games = fresh.get("games", [])
    hitters = [h for g in games for h in g.get("hitters", [])]
    by_id = {str(h.get("batter_id")): h for h in hitters}
    by_name = {h.get("batter"): h for h in hitters}
    unmatched = {
        r["batter"] for r in rows
        if not by_id.get(str(r.get("batter_id") or "")) and r["batter"] not in by_name
        and not r.get("batter_id")
    }
    ids = _batter_ids(pa_path, unmatched)

    previous: dict = {"picks_6": [], "hit_picks": []}
    for r in sorted(rows, key=lambda r: int(r["rank"])):
        h = by_id.get(str(r.get("batter_id") or "")) or by_name.get(r["batter"])
        projected = _num(r.get("projected"))
        if projected is None:
            continue
        if h is None:
            batter_id = int(r["batter_id"]) if r.get("batter_id") else ids.get(r["batter"])
            placeholder = _placeholder_pick(r, games, batter_id)
            if placeholder:
                previous["picks_6" if r["kind"] == "hr" else "hit_picks"].append(placeholder)
            continue
        pick = dict(h)
        if r["kind"] == "hr":
            pick["prob_hr"] = projected
            previous["picks_6"].append(pick)
        elif pick.get("hits_proj"):
            pick["hits_proj"] = dict(pick["hits_proj"], projected_hits=projected,
                                     projected_hits_raw=projected)
            previous["hit_picks"].append(pick)
    return previous


def unsettled_days(out_dir: Optional[Path] = None, before: Optional[date] = None,
                   lookback_days: int = 7) -> list[str]:
    """Recent dates in the tracker with a pick that has no final result."""
    out = _out_dir(out_dir)
    before = before or date.today()
    earliest = str(before - timedelta(days=lookback_days))
    open_days = {
        r["date"] for r in _read_csv(out / "picks.csv")
        if earliest <= r["date"] < str(before) and not _truthy(r.get("final"))
    }
    return sorted(open_days)


def finalize_day(day: str, out_dir: Optional[Path] = None) -> dict:
    """Re-score a past day against its box scores, picks unchanged.

    Uses the saved slate when there is one (refreshing PDFs too). Days archived
    before slates were saved are settled from the CSV rows, matched by batter
    id or name; their PDFs keep the partial results they were written with.
    """
    from mlb_hr.results import attach_results, fetch_results

    out = _out_dir(out_dir)
    slate = load_slate(day, out)
    if slate is not None:
        results = fetch_results(
            date.fromisoformat(day), [g.get("game_pk") for g in slate.get("games", [])]
        )
        attach_results(slate, results)
        written = write_snapshot(slate, out)
        return {"date": day, "source": "slate", "settled": summary_row(slate)["settled"],
                "pdfs": written["pdfs"]}
    return _finalize_from_csv(day, out)


def _finalize_from_csv(day: str, out: Path) -> dict:
    import unicodedata

    from mlb_hr.fetch import boxscore_batting, live_games
    from mlb_hr.results import batter_line

    def norm(name: str) -> str:
        return unicodedata.normalize("NFKD", name or "").encode(
            "ascii", "ignore").decode().lower().strip()

    rows = [r for r in _read_csv(out / "picks.csv") if r["date"] == day]
    if not rows:
        return {"date": day, "source": "csv", "settled": False, "error": "no rows"}

    games = live_games(date.fromisoformat(day))
    lines = boxscore_batting([pk for pk, g in games.items()
                              if g.get("state") in ("Live", "Final")])
    by_id = {str(bid): line for bid, line in lines.items()}
    by_name: dict[str, list] = {}
    for line in lines.values():
        by_name.setdefault(norm(line.get("batter", "")), []).append(line)
    all_final = bool(games) and all(g.get("state") == "Final" for g in games.values())

    for r in rows:
        line = by_id.get(str(r.get("batter_id") or ""))
        if line is None:
            matches = by_name.get(norm(r["batter"]), [])
            line = matches[0] if len(matches) == 1 else None
        if line is not None and r.get("game_pk"):
            # A doubleheader's other game is not this pick's game.
            line = batter_line({0: line}, 0, int(float(r["game_pk"])))
        game = games.get(line["game_pk"]) if line else None
        if line and game:
            final = game.get("state") == "Final"
            r.update({
                "game_pk": r.get("game_pk") or line["game_pk"],
                "batter_id": r.get("batter_id") or line.get("batter_id"),
                "result_pa": line.get("pa"), "result_ab": line.get("ab"),
                "result_hits": line.get("hits"), "result_hr": line.get("hr"),
                "result_rbi": line.get("rbi"), "result_line": line.get("summary", ""),
                "game_state": game.get("detailed", ""), "final": final,
                "won": (bool(line.get("hr" if r["kind"] == "hr" else "hits", 0))
                        if final else None),
            })
        elif all_final:
            # Every game that day is over and he has no line: did not play.
            r.update({"result_pa": 0, "result_ab": 0, "result_hits": 0,
                      "result_hr": 0, "result_rbi": 0,
                      "result_line": "Did not play", "game_state": "Did not play",
                      "final": True, "won": None})

    prior = next((s for s in _read_csv(out / "daily_summary.csv") if s["date"] == day), {})
    summary = summarize_rows(rows, day=day, games=prior.get("games", ""),
                             built_at=prior.get("built_at", ""))
    # No saved slate here, so the whole-slate columns stay as last written.
    summary.update({k: prior.get(k) for k in SUMMARY_COLUMNS if k not in summary})
    _write_tracker_rows(out, day, rows, summary, None)
    return {"date": day, "source": "csv", "settled": summary["settled"]}


def finalize_pending(out_dir: Optional[Path] = None, before: Optional[date] = None,
                     lookback_days: int = 7) -> list[dict]:
    """Settle every recent day that still has open picks. Safe to call often."""
    done = []
    for day in unsettled_days(out_dir, before, lookback_days):
        try:
            done.append(finalize_day(day, out_dir))
        except Exception as exc:  # noqa: BLE001 - one bad day must not stop the rest
            print(f"[snapshot.py] could not settle {day}: {exc}")
    return done


def is_complete(slate_data: dict) -> bool:
    """True when every game on the slate is final."""
    results = slate_data.get("results") or {}
    return bool(results) and results.get("live_games", 1) == 0 and \
        results.get("upcoming_games", 1) == 0 and results.get("final_games", 0) > 0


def main(day: Optional[str] = None) -> None:
    from mlb_hr.results import attach_results, fetch_results
    from mlb_hr.slate import build_slate_for_date

    from mlb_hr.ensemble import MODEL_VERSION

    target = date.fromisoformat(day) if day else date.today()
    pa_path = str(Path(__file__).parent / "data" / "season_pa_v2.jsonl")
    out = Path.cwd() / "snapshots" if not DEFAULT_DIR.exists() else DEFAULT_DIR
    print(f"[snapshot.py] building slate for {target}")
    slate = build_slate_for_date(target, pa_path, use_ensemble=True,
                                 hr_level=hr_level(out, target, MODEL_VERSION))
    slate["built_at"] = ""
    attach_results(slate, fetch_results(target,
                                        [g.get("game_pk") for g in slate["games"]]))
    written = write_snapshot(slate, out)
    print(json.dumps(written, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--finalize":
        target_dir = DEFAULT_DIR if DEFAULT_DIR.exists() else Path.cwd() / "snapshots"
        days = sys.argv[2:]
        report = (
            [finalize_day(d, target_dir) for d in days] if days
            else finalize_pending(target_dir)
        )
        print(json.dumps(report, indent=2))
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else None)
