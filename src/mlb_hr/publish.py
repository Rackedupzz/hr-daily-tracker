"""One pass of the server's warm-up loop, written out as a static site.

Vercel serves files, not this Flask app: the fit needs torch and minutes of
CPU, and the snapshots must outlive the request. So a scheduled GitHub Action
runs this module instead -- it does what `app.warm_slates` does in one go
(top up the feed, refit when stale, lock started picks, attach live scores,
archive, handicap and grade HOMER) and then renders every page into `public/`,
which the workflow commits and Vercel deploys.

    python -m mlb_hr.publish [--out public] [--force-fit]

`?date=` links become paths (`/results/2026-10-04/`) because a static host
cannot route on a query string.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

from mlb_hr import homer
from mlb_hr.app import SNAPSHOT_DIR, baseball_day
from mlb_hr.fetch import fetch_season, live_games, schedule
from mlb_hr.publish_links import staticize
from mlb_hr.render import remaining_projections, render_html, render_models_html
from mlb_hr.render_review import render_homer_html, render_results_html
from mlb_hr.results import attach_results, fetch_results
from mlb_hr.slate import build_slate_for_date, lock_started_picks
from mlb_hr.snapshot import (
    finalize_pending, hr_level, is_complete, load_slate, published_picks, write_snapshot,
)

PA_PATH = Path(__file__).parent / "data" / "season_pa_v2.jsonl"
# Scheduled runs are hours apart, so anything older than this is refit.
SLATE_TTL = timedelta(minutes=int(os.environ.get("PUBLISH_SLATE_TTL_MINUTES", "150")))


def log(msg: str) -> None:
    print(f"[publish] {msg}", flush=True)


def slate_target() -> date | None:
    """Same lookup as the server: today, else the next or latest day with games."""
    today = baseball_day()
    for offset in [0, 1, 2, 3] + [-d for d in range(1, 14)]:
        day = today + timedelta(days=offset)
        if any(g.state in ("Preview", "Live", "Final") for g in schedule(day, day)):
            return day
    return None


def results_for(day: date, slate: dict) -> dict:
    try:
        return fetch_results(day, [g.get("game_pk") for g in slate.get("games", [])])
    except Exception as exc:  # noqa: BLE001 - outcomes are a bonus layer
        log(f"could not fetch results for {day}: {exc}")
        return {}


def needs_fit(saved: dict | None, force: bool) -> bool:
    if force or saved is None:
        return True
    if is_complete(saved):
        return False
    try:
        built = datetime.strptime(saved.get("built_at") or "", "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return True
    return datetime.now() - built > SLATE_TTL


def fit(day: date, saved: dict | None) -> dict:
    """Refit the day, keeping picks in games that have started as published."""
    from mlb_hr.ensemble import MODEL_VERSION

    level = hr_level(SNAPSHOT_DIR, day, MODEL_VERSION)
    log(f"fitting {day} (HR level x{level:.3f})")
    fresh = build_slate_for_date(day, str(PA_PATH), use_ensemble=True, hr_level=level)
    fresh["built_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    previous = saved or published_picks(str(day), fresh, SNAPSHOT_DIR, str(PA_PATH))
    if previous is not None:
        try:
            started = {pk for pk, g in live_games(day).items()
                       if g.get("state") in ("Live", "Final")}
            lock_started_picks(previous, fresh, started)
        except Exception as exc:  # noqa: BLE001
            log(f"could not lock started picks: {exc}")
    return fresh


def update_homer(day: date, slate: dict, results: dict) -> None:
    card = homer.load_card(str(day), SNAPSHOT_DIR)
    fitted = (homer.load_fit() or {}).get("fitted_at")
    stale = (card is None or "five_picks" not in card
             or (not card.get("record", {}).get("complete")
                 and ((fitted and fitted != (card.get("evidence") or {}).get("fitted_at"))
                      or str(slate.get("built_at") or "") > str(card.get("built_at") or ""))))
    if stale:
        log(f"HOMER handicapping {day}")
        book = homer.build_book(str(PA_PATH), day)
        card = homer.build_card(slate, book, card, results, homer.load_fit())
    homer.grade_card(card, results)
    homer.save_card(card, SNAPSHOT_DIR)


def settle_past(target: date) -> None:
    for done in finalize_pending(SNAPSHOT_DIR, before=target):
        log(f"settled {done['date']} from {done['source']}: {done['settled']}")
    for day in homer.card_days(SNAPSHOT_DIR):
        if day >= str(target):
            continue
        card = homer.load_card(day, SNAPSHOT_DIR)
        if not card or card.get("record", {}).get("complete"):
            continue
        slate = load_slate(day, SNAPSHOT_DIR)
        if slate is None:
            continue
        if not is_complete(slate):
            attach_results(slate, results_for(date.fromisoformat(day), slate))
        if slate.get("results"):
            homer.grade_card(card, homer.results_from_slate(slate))
            homer.save_card(card, SNAPSHOT_DIR)


def maybe_refit_homer() -> None:
    fitted = (homer.load_fit() or {}).get("fitted_at", "")
    if fitted and datetime.now() - datetime.strptime(fitted, "%Y-%m-%d %H:%M:%S") < timedelta(hours=20):
        return
    from mlb_hr import homer_fit
    log("HOMER replaying the season")
    try:
        homer_fit.run(PA_PATH)
    except Exception as exc:  # noqa: BLE001
        log(f"HOMER refit failed: {exc}")


# ── static output ───────────────────────────────────────────────────────────

def write_page(out: Path, route: str, html: str) -> None:
    path = out / route.strip("/") / "index.html" if route.strip("/") else out / "index.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(staticize(html), encoding="utf-8")


def write_fallback(out: Path, page: str, html: str) -> None:
    """`/`, `/models` and `/homer` are served live by api/live.py. A file at
    those paths would shadow the rewrite, so the prerendered copy lives under
    fallback/ for the function to serve when the MLB API is unreachable."""
    path = out / "fallback" / f"{page}.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(staticize(html), encoding="utf-8")


def write_json(out: Path, route: str, data) -> None:
    path = out / route.strip("/")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def archived_days(before: date) -> list[str]:
    found = set()
    for pattern in ("*-slate.json.gz", "*-picks.json"):
        for path in SNAPSHOT_DIR.glob(pattern):
            day = path.name[:10]
            try:
                date.fromisoformat(day)
            except ValueError:
                continue
            if day < str(before):
                found.add(day)
    return sorted(found)


def render_site(out: Path, slate: dict | None, target: date | None) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    if slate is not None:
        write_fallback(out, "slate", render_html(slate))
        write_fallback(out, "models", render_models_html(slate))
        write_json(out, "/api/remaining.json", {
            "date": slate["date"], "built_at": slate.get("built_at"),
            "fetched_at": (slate.get("results") or {}).get("fetched_at"),
            "games": remaining_projections(slate),
        })

    # Results: one page per archived day; /results is the newest.
    cutoff = target or baseball_day()
    days = archived_days(before=cutoff)
    for day in days:
        saved = load_slate(day, SNAPSHOT_DIR)
        sidecar_path = SNAPSHOT_DIR / f"{day}-picks.json"
        sidecar = None
        if saved is None and sidecar_path.exists():
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        card = homer.load_card(day, SNAPSHOT_DIR)
        html = render_results_html(day, saved, days, sidecar, card)
        write_page(out, f"/results/{day}", html)
        if day == days[-1]:
            write_page(out, "/results", html)

    # HOMER: every card, with the current one at /homer.
    record = homer.season_record(SNAPSHOT_DIR)
    card_days = homer.card_days(SNAPSHOT_DIR)
    for day in card_days:
        write_page(out, f"/homer/{day}",
                   render_homer_html(homer.load_card(day, SNAPSHOT_DIR), record, card_days, day, ""))
    current = str(slate["date"]) if slate else (card_days[-1] if card_days else str(cutoff))
    card = homer.load_card(current, SNAPSHOT_DIR)
    write_fallback(out, "homer", render_homer_html(card, record, card_days, current, ""))
    write_json(out, "/api/homer.json",
               {"date": current, "status": "", "card": card, "record": record})
    write_json(out, "/health.json", {"status": "ok",
                                     "published_at": datetime.now().isoformat(timespec="seconds")})

    if slate is None:
        from mlb_hr import ui
        body = ui.hero("Off day", "No games on the schedule",
                       'Nothing is scheduled in the next few days. Past slates are under '
                       '<a href="/results">Results</a>.')
        write_fallback(out, "slate", ui.page("HR Daily Tracker", "slate", body))
    log(f"site written to {out}")


def main(argv=None) -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="public")
    ap.add_argument("--force-fit", action="store_true")
    ap.add_argument("--render-only", action="store_true",
                    help="skip fetching and fitting; render what is saved")
    args = ap.parse_args(argv)

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    target = slate_target()
    log(f"target day: {target}")
    slate = load_slate(str(target), SNAPSHOT_DIR) if target else None

    if not args.render_only and target is not None:
        if needs_fit(slate, args.force_fit):
            log("topping up the season feed")
            fetch_season(date(2026, 1, 1), baseball_day(), str(PA_PATH),
                         workers=8, resume=True, progress_every=500)
            slate = fit(target, slate)
        results = results_for(target, slate)
        attach_results(slate, results)
        write_snapshot(slate, SNAPSHOT_DIR)
        try:
            update_homer(target, slate, results)
        except Exception as exc:  # noqa: BLE001 - the slate still publishes
            log(f"HOMER failed: {exc}")
        settle_past(target)
        maybe_refit_homer()

    # The live function (api/live.py) re-scores this day on every request.
    (SNAPSHOT_DIR / "current.json").write_text(
        json.dumps({"date": str(slate["date"]) if slate else None}), encoding="utf-8")
    render_site(Path(args.out), slate, target)


if __name__ == "__main__":
    main()
