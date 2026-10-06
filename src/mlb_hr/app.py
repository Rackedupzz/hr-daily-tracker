"""Flask server: auto-fetch data on startup, render HTML, serve on localhost:5000."""
from __future__ import annotations

import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from threading import Event, Lock, Thread

from html import escape

from flask import Flask, render_template_string, request

from mlb_hr import homer, ui
from mlb_hr.fetch import fetch_season, probable_pitchers, schedule
from mlb_hr.model import HRModel
from mlb_hr.render import remaining_projections, render_html, render_models_html
from mlb_hr.render_review import render_homer_html, render_results_html
from mlb_hr.results import attach_results, fetch_results
from mlb_hr.fetch import live_games
from mlb_hr.slate import build_slate_for_date, lock_started_picks
from mlb_hr.snapshot import (
    finalize_pending, hr_level, is_complete, load_slate, published_picks, write_snapshot,
)


SNAPSHOT_DIR = Path(os.environ.get("SNAPSHOT_DIR", "/app/snapshots"))


def baseball_day(now: datetime = None) -> date:
    """The date of the slate currently being played.

    A server clock in UTC and MLB's schedule day are not the same thing. At
    03:00 UTC it is still the previous evening in the United States and the
    night games are in the middle of the seventh inning, but `date.today()`
    has already rolled over. Taking the UTC date there swaps a slate of live
    games for tomorrow's unplayed one -- the games in progress vanish, and with
    them every hit and miss the page had been showing.

    So the day is measured in US Eastern time, and rolls at 06:00 rather than
    midnight, because a west-coast game that starts at 22:10 ET is still part
    of the previous day's slate when it ends after 01:00.
    """
    now = now or datetime.now(timezone.utc)
    try:
        eastern = now.astimezone(ZoneInfo("America/New_York"))
    except Exception:  # noqa: BLE001 - no tz database available
        # UTC-4/5 is the right offset for the entire baseball season.
        eastern = now.astimezone(timezone(timedelta(hours=-4)))
    return (eastern - timedelta(hours=6)).date()


def create_app(pa_data_path: str = None) -> Flask:
    """Create and initialize Flask app with fresh data."""
    app = Flask(__name__)

    # Default data path
    if pa_data_path is None:
        pa_data_path = str(
            Path(__file__).parent / "data" / "season_pa_v2.jsonl"
        )

    # Ensure data directory exists
    Path(pa_data_path).parent.mkdir(parents=True, exist_ok=True)

    last_refresh = [datetime.min]

    def refresh_season_data() -> None:
        """Top up the PA feed with every newly final game.

        This used to run only at startup, so a container left up for days kept
        refitting on the same frozen data -- 9/11 and 9/12 produced identical
        model scores. It now also runs before each new day's first build. The
        slate build ignores anything from its own date, so topping up with
        today's finished games cannot leak outcomes into today's picks.
        """
        last_refresh[0] = datetime.now()
        print("[app.py] Fetching latest 2026 season data...")
        try:
            fetch_season(
                date(2026, 1, 1),
                baseball_day(),
                pa_data_path,
                workers=8,
                resume=True,
                progress_every=100,
            )
            print(f"[app.py] Data updated: {pa_data_path}")
        except Exception as e:
            print(f"[app.py] Warning: could not fetch season data: {e}")
            print("[app.py] Proceeding with existing data if available.")

    refresh_season_data()

    # Building a slate refits the whole ensemble over the season and takes
    # minutes. The slate page and the model page render from the same slate, so
    # it is built once per date and reused -- switching tabs must not retrain
    # the model. The lock means a second request waits for the first build
    # rather than starting a duplicate one.
    slate_cache: dict = {}
    slate_lock = Lock()          # guards the cache dict only, never a fit
    building: set = set()        # dates with a fit in flight
    build_lock = Lock()
    target_cache: dict = {}
    target_lock = Lock()
    archived: set = set()        # dates whose final archive is already written

    # How stale the fitted slate may get before it is refit. Lineups get posted,
    # starters get scratched and injuries land during the day, so the model is
    # rebuilt on a timer -- in the background, since a foreground refit would
    # hang the page for minutes.
    slate_ttl = timedelta(minutes=int(os.environ.get("SLATE_TTL_MINUTES", "45")))
    # Live scores and box scores are two cheap calls, so they are pulled fresh
    # on every page load, with just enough caching to survive a reload storm.
    results_ttl = timedelta(seconds=int(os.environ.get("RESULTS_TTL_SECONDS", "45")))
    # How often the season PA feed is topped up with newly final games.
    feed_ttl = timedelta(hours=float(os.environ.get("FEED_REFRESH_HOURS", "3")))
    results_cache: dict = {}
    results_lock = Lock()

    def build_slate(day: date) -> dict:
        """Fit the model for a date and stamp when it was built."""
        from mlb_hr.ensemble import MODEL_VERSION

        # The HR level learns from this model's own settled days; before any
        # exist it is 1.0 and changes nothing.
        level = hr_level(SNAPSHOT_DIR, day, MODEL_VERSION)
        print(f"[app.py] HR level for {day}: x{level:.3f} (odds)")
        data = build_slate_for_date(day, pa_data_path, use_ensemble=True, hr_level=level)
        data["built_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return data

    def start_build(day: date, reason: str) -> bool:
        """Fit a slate on a worker thread. Never called with a lock held.

        A fit takes minutes, so it must not happen inside `slate_lock`: doing
        that blocked the request that triggered it *and* queued every other
        reader behind the same lock, turning one cold date into a page that
        hangs for ten minutes.
        """
        with build_lock:
            if day in building:
                return False
            building.add(day)

        def work():
            try:
                print(f"[app.py] building slate for {day} ({reason})")
                fresh = build_slate(day)
                # Picks in games already under way stay as published. The
                # previous version comes from memory or, after a restart, from
                # the day's saved snapshot.
                with slate_lock:
                    previous = slate_cache.get(day)
                if previous is None:
                    previous = load_slate(str(day), SNAPSHOT_DIR)
                if previous is None:
                    previous = published_picks(
                        str(day), fresh, SNAPSHOT_DIR, pa_data_path
                    )
                if previous is not None:
                    try:
                        started = {
                            pk for pk, g in live_games(day).items()
                            if g.get("state") in ("Live", "Final")
                        }
                        lock_started_picks(previous, fresh, started)
                    except Exception as exc:  # noqa: BLE001
                        print(f"[app.py] could not lock started picks: {exc}")
                with slate_lock:
                    slate_cache[day] = fresh
                    # Keep today and yesterday: the previous day is what gets
                    # served while the new one is still being fitted.
                    for old in sorted(slate_cache)[:-2]:
                        slate_cache.pop(old, None)
                print(f"[app.py] slate ready for {day}")
            except Exception as exc:  # a failed fit must not kill the server
                print(f"[app.py] slate build failed for {day}: {exc}")
            finally:
                with build_lock:
                    building.discard(day)

        Thread(target=work, daemon=True, name=f"slate-build-{day}").start()
        return True

    def slate_target():
        """The date whose games the page should show, cached briefly.

        Resolving this costs a schedule call, and the answer only changes once
        a day, so it is not worth paying for on every request.
        """
        now = datetime.now()
        with target_lock:
            if target_cache and now - target_cache["at"] < timedelta(minutes=5):
                return target_cache["day"]

        today = baseball_day()
        found = None
        # On an off day (the postseason has several) the next slate is what a
        # reader wants, not the one already settled: look a few days ahead
        # before falling back to the most recent day with games.
        offsets = [0, 1, 2, 3] + [-d for d in range(1, 14)]
        for offset in offsets:
            check_date = today + timedelta(days=offset)
            games = schedule(check_date, check_date)
            if any(g.state in ("Preview", "Live", "Final") for g in games):
                found = check_date
                break
        with target_lock:
            target_cache.update({"at": now, "day": found})
        return found

    def latest_slate():
        """The slate to render, without ever waiting on a fit.

        If the target date is not fitted yet, the build starts in the
        background and the previous day's slate is served in the meantime with
        a note saying so. Returning a stale-but-real page in 300ms beats a
        correct one in ten minutes.

        Returns None only when there is nothing cached at all, which the caller
        renders as a warming-up page rather than a hang.
        """
        target = slate_target()
        if target is None:
            return None

        with slate_lock:
            cached = slate_cache.get(target)
            newest_day = max(slate_cache) if slate_cache else None
            newest = slate_cache.get(newest_day) if newest_day else None

        if cached is not None:
            built = datetime.strptime(
                cached.get("built_at", "1970-01-01 00:00:00"), "%Y-%m-%d %H:%M:%S"
            )
            # Once every game is final there is nothing left to re-project.
            if datetime.now() - built > slate_ttl and not is_complete(cached):
                start_build(target, "stale")
            cached.pop("building_note", None)
            slate = cached
        else:
            start_build(target, "not fitted yet")
            if newest is None:
                return None
            slate = newest
            slate["building_note"] = (
                f"Today's slate ({target}) is still being fitted &mdash; this "
                f"takes a few minutes. Showing {slate['date']} until it is ready; "
                "reload to check."
            )

        day = date.fromisoformat(slate["date"])
        attach_results(slate, live_results(day, slate))
        return slate

    def warm_slates() -> None:
        """Fit the current day ahead of the first visitor, and keep doing it.

        The day rolls over at 06:00 Eastern, and without this the first person
        to open the page after that pays for the whole fit. Checking every few
        minutes means the new day is usually ready before anyone asks.
        """
        last_settle = [datetime.min]
        refreshed_for: set = set()

        def work():
            while True:
                try:
                    # Top the feed up through the day, not just when a new
                    # slate appears: on an off day the target never changes,
                    # so last night's games were never pulled and HOMER's
                    # daily replay ran without them. Never during a fit, which
                    # is reading the same file.
                    with build_lock:
                        busy = bool(building)
                    if (not busy and not homer_fitting.is_set()
                            and datetime.now() - last_refresh[0] > feed_ttl):
                        refresh_season_data()

                    target = slate_target()
                    with slate_lock:
                        have = target in slate_cache
                    if target and not have:
                        # A new day: pull yesterday's late games in first, so
                        # the day's fit sees them.
                        if (target not in refreshed_for and datetime.now()
                                - last_refresh[0] > timedelta(hours=1)):
                            refresh_season_data()
                        refreshed_for.add(target)
                        start_build(target, "pre-warming")

                    # Archive every cached day as it progresses, and once more
                    # when all its games are final. Results are pulled here
                    # rather than relying on someone having opened the page,
                    # and yesterday keeps being scored after today's slate
                    # replaces it as the newest. Re-running replaces that
                    # date's rows, so a crashed evening still leaves a record.
                    with slate_lock:
                        cached = dict(slate_cache)
                    for day, slate in sorted(cached.items()):
                        if day in archived:
                            continue
                        attach_results(slate, live_results(day, slate))
                        if not slate.get("results"):
                            continue
                        written = write_snapshot(slate, SNAPSHOT_DIR)
                        if is_complete(slate):
                            archived.add(day)
                            print(f"[app.py] final archive written for {day}: "
                                  f"{written['pdfs']}")

                    # HOMER handicaps the current day once its slate exists,
                    # and again whenever the slate is refit with new lineups.
                    with slate_lock:
                        today_slate = slate_cache.get(target) if target else None
                    if today_slate is not None:
                        card = homer.load_card(str(target), SNAPSHOT_DIR)
                        if homer_stale(card, today_slate):
                            start_homer(target, today_slate)
                        else:
                            grade_and_save(card, live_results(target, today_slate))

                    # Days no longer in memory (restarts, long gaps) are
                    # settled from their saved snapshots.
                    if target and datetime.now() - last_settle[0] > timedelta(minutes=30):
                        last_settle[0] = datetime.now()
                        for done in finalize_pending(SNAPSHOT_DIR, before=target):
                            print(f"[app.py] settled {done['date']} "
                                  f"from {done['source']}: {done['settled']}")
                        settle_homer_cards(target)
                        maybe_refit_homer()
                except Exception as exc:  # noqa: BLE001 - never kill the warmer
                    print(f"[app.py] warmer error: {exc}")
                time.sleep(300)

        Thread(target=work, daemon=True, name="slate-warmer").start()

    def live_results(day: date, slate_data: dict) -> dict:
        """The day's scores and batting lines, cached for a few seconds."""
        with results_lock:
            entry = results_cache.get(day)
            if entry and datetime.now() - entry["at"] < results_ttl:
                return entry["data"]
        try:
            game_pks = [g.get("game_pk") for g in slate_data.get("games", [])]
            data = fetch_results(day, game_pks)
        except Exception as exc:
            # Outcomes are a bonus layer; losing them must not lose the slate.
            print(f"[app.py] could not fetch live results: {exc}")
            return {}
        with results_lock:
            results_cache.clear()
            results_cache[day] = {"at": datetime.now(), "data": data}
        return data

    # ── past days, for the Results tab ─────────────────────────────────────
    review_cache: dict = {}      # day -> slate, only once every game is final
    review_lock = Lock()

    def archived_days(before: date = None) -> list:
        """Dates with a saved slate or a tracker sidecar, oldest first."""
        found = set()
        for pattern in ("*-slate.json.gz", "*-picks.json"):
            for path in SNAPSHOT_DIR.glob(pattern):
                day = path.name[:10]
                try:
                    date.fromisoformat(day)
                except ValueError:
                    continue
                if before is None or day < str(before):
                    found.add(day)
        return sorted(found)

    def review_slate(day: str):
        """A past day's slate with its results, re-scored until it is settled."""
        with review_lock:
            if day in review_cache:
                return review_cache[day]
        slate = load_slate(day, SNAPSHOT_DIR)
        if slate is None:
            return None
        if not is_complete(slate):
            try:
                attach_results(slate, fetch_results(
                    date.fromisoformat(day), [g.get("game_pk") for g in slate.get("games", [])]
                ))
            except Exception as exc:  # noqa: BLE001 - show what was saved
                print(f"[app.py] could not re-score {day}: {exc}")
        if is_complete(slate):
            with review_lock:
                review_cache[day] = slate
                for old in sorted(review_cache)[:-10]:
                    review_cache.pop(old, None)
        return slate

    def picks_sidecar(day: str):
        import json
        path = SNAPSHOT_DIR / f"{day}-picks.json"
        try:
            return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        except (OSError, ValueError):
            return None

    # ── HOMER ──────────────────────────────────────────────────────────────
    # His book is one ~30 second pass over the season feed, built once per day
    # and kept for re-handicapping as lineups post. Only the newest is kept.
    book_cache: dict = {}
    book_lock = Lock()
    homer_building: set = set()
    homer_build_lock = Lock()

    def homer_stale(card, slate) -> bool:
        """Rebuild when there is no card, or the slate was refit after it."""
        if card is None:
            return True
        # A card from before the 5-pick parlays existed is rebuilt once to add
        # them; its published picks are locked, so nothing else moves.
        if "five_picks" not in card:
            return True
        if card.get("record", {}).get("complete"):
            return False
        # A new season replay means new weights: re-handicap what has not started.
        fitted = (homer.load_fit() or {}).get("fitted_at")
        if fitted and fitted != (card.get("evidence") or {}).get("fitted_at"):
            return True
        return str(slate.get("built_at") or "") > str(card.get("built_at") or "")

    def start_homer(day: date, slate: dict) -> bool:
        with homer_build_lock:
            if day in homer_building:
                return False
            homer_building.add(day)

        def work():
            try:
                with book_lock:
                    book = book_cache.get(day)
                if book is None:
                    print(f"[app.py] HOMER is reading the season feed for {day}")
                    book = homer.build_book(pa_data_path, day)
                    with book_lock:
                        book_cache.clear()
                        book_cache[day] = book
                previous = homer.load_card(str(day), SNAPSHOT_DIR)
                results = live_results(day, slate)
                card = homer.build_card(slate, book, previous, results, homer.load_fit())
                homer.grade_card(card, results)
                with card_write_lock:
                    homer.save_card(card, SNAPSHOT_DIR)
                print(f"[app.py] HOMER card ready for {day}")
            except Exception as exc:  # noqa: BLE001 - never kill the server
                import traceback
                print(f"[app.py] HOMER build failed for {day}: {exc}")
                traceback.print_exc()
            finally:
                with homer_build_lock:
                    homer_building.discard(day)

        Thread(target=work, daemon=True, name=f"homer-{day}").start()
        return True

    card_write_lock = Lock()

    homer_fitting = Event()

    def maybe_refit_homer() -> None:
        """Re-run HOMER's season replay daily, never alongside a slate fit.

        HOMER learns from every day of results: once a new day's games are in
        the feed, the whole replay reruns so his weights, grades and parlay
        record include them. It holds the season in memory for several minutes
        and the VM is small, so it waits until no slate is being fitted.
        """
        fit = homer.load_fit()
        fitted = (fit or {}).get("fitted_at", "")
        due = not fitted or datetime.now() - datetime.strptime(
            fitted, "%Y-%m-%d %H:%M:%S") > timedelta(hours=20)
        with build_lock:
            busy = bool(building)
        if not due or busy or homer_fitting.is_set():
            return
        homer_fitting.set()

        def work():
            try:
                from mlb_hr import homer_fit
                print("[app.py] HOMER is replaying the season to relearn his weights")
                homer_fit.run(Path(pa_data_path))
            except Exception as exc:  # noqa: BLE001
                print(f"[app.py] HOMER refit failed: {exc}")
            finally:
                homer_fitting.clear()

        Thread(target=work, daemon=True, name="homer-fit").start()

    def grade_and_save(card: dict, results: dict) -> dict:
        before = card.get("record")
        homer.grade_card(card, results)
        if card.get("record") != before:
            with card_write_lock:
                # A rebuild may have saved a newer card since this one was
                # read; grading must never put the older picks back.
                on_disk = homer.load_card(card["date"], SNAPSHOT_DIR)
                if on_disk is None or on_disk.get("built_at") == card.get("built_at"):
                    homer.save_card(card, SNAPSHOT_DIR)
        return card

    def settle_homer_cards(before: date) -> None:
        """Grade every earlier card that still has open picks."""
        for day in homer.card_days(SNAPSHOT_DIR):
            if day >= str(before):
                continue
            card = homer.load_card(day, SNAPSHOT_DIR)
            if not card or card.get("record", {}).get("complete"):
                continue
            slate = review_slate(day)
            if slate is not None and slate.get("results"):
                grade_and_save(card, homer.results_from_slate(slate))

    def warming_page():
        """Shown only when nothing is cached yet and a fit is underway.

        Auto-refreshes, so a visitor who arrives during a cold start gets the
        slate as soon as it exists instead of staring at a loading spinner for
        ten minutes.
        """
        with build_lock:
            pending = sorted(str(d) for d in building)
        target = ", ".join(pending) or "today"
        body = ui.hero("Cold start", '<span class="spinner"></span>Warming up',
                       f"Fitting the model for <strong>{target}</strong>.") + ui.alert(
            "This runs the comparables ensemble over the full season and takes roughly ten "
            "minutes on a cold start. The page refreshes itself every 20 seconds and will load "
            "as soon as the slate is ready &mdash; no need to wait on this tab.", info=True)
        return ui.page("HR Daily Tracker — warming up", "slate", body,
                       status=ui.status_chip(text="Fitting", busy=True), meta_refresh=20), 503

    def no_games_page():
        body = ui.hero("Off day", "No games on the schedule",
                       "Nothing is scheduled in the next few days. Past slates are under "
                       '<a href="/results">Results</a>.')
        return ui.page("HR Daily Tracker", "slate", body)

    def error_page(exc: Exception):
        import traceback
        print(f"[app.py] Error rendering page: {exc}")
        traceback.print_exc()
        body = ui.hero("Error", "Something broke rendering this page",
                       '<a href="">Retry</a> &middot; the details are in the server log.') + ui.alert(
            f"<code>{escape(str(exc))}</code>")
        return ui.page("HR Daily Tracker — Error", "", body), 500

    # Start fitting immediately so the first visitor does not pay for it.
    warm_slates()

    @app.route("/")
    def index():
        """Render today's HR slate."""
        try:
            slate_data = latest_slate()
            if slate_data is None:
                return warming_page() if building else no_games_page()
            return render_html(slate_data)
        except Exception as e:
            return error_page(e)

    @app.route("/models")
    def models():
        """Side-by-side comparison of every model variant on today's slate."""
        try:
            slate_data = latest_slate()
            if slate_data is None:
                return warming_page() if building else no_games_page()
            return render_models_html(slate_data)
        except Exception as e:
            return error_page(e)

    def requested_day():
        raw = request.args.get("date", "")
        try:
            return str(date.fromisoformat(raw))
        except ValueError:
            return None

    @app.route("/results")
    def results():
        """A previous day's picks and games, scored against what happened."""
        try:
            target = slate_target() or baseball_day()
            days = archived_days(before=target)
            day = requested_day() or (days[-1] if days else str(target - timedelta(days=1)))
            slate = review_slate(day)
            sidecar = None if slate is not None else picks_sidecar(day)
            card = homer.load_card(day, SNAPSHOT_DIR)
            if card and slate is not None and slate.get("results"):
                grade_and_save(card, homer.results_from_slate(slate))
            return render_results_html(day, slate, days, sidecar, card)
        except Exception as e:
            return error_page(e)

    def homer_view():
        """(card, day, status) for the HOMER tab, kicking off a build if due."""
        target = slate_target()
        asked = requested_day()
        if asked and (target is None or asked != str(target)):
            card = homer.load_card(asked, SNAPSHOT_DIR)
            if card and not card.get("record", {}).get("complete"):
                slate = review_slate(asked)
                if slate is not None and slate.get("results"):
                    grade_and_save(card, homer.results_from_slate(slate))
            return card, asked, "" if card else f"HOMER has no card for {asked}."

        slate = latest_slate()
        fitting = slate is None or target is None or bool(slate.get("building_note"))
        if fitting:
            # Today's card may already be saved (e.g. after a restart); show it
            # while the slate refits, and fall back to the newest slate's card.
            today = str(target or baseball_day())
            card = homer.load_card(today, SNAPSHOT_DIR)
            if card:
                return card, today, ("Slate is refitting; HOMER will re-check "
                                     "lineups when it is ready.")
            if slate is None:
                return None, today, "HOMER is waiting for today's slate to finish fitting."
            day = slate["date"]
            return (homer.load_card(day, SNAPSHOT_DIR), day,
                    f"Today's slate ({today}) is still being fitted; HOMER will post "
                    f"his card for it right after. Showing {day}.")
        day = slate["date"]
        card = homer.load_card(day, SNAPSHOT_DIR)
        status = ""
        if homer_stale(card, slate):
            start_homer(date.fromisoformat(day), slate)
            status = ("HOMER is re-handicapping with the latest lineups."
                      if card else f"HOMER is studying film for {day} (about a minute).")
        if card:
            grade_and_save(card, live_results(date.fromisoformat(day), slate))
        return card, day, status

    @app.route("/homer")
    def homer_page():
        """HOMER, the resident expert: his own picks, reasoning and record."""
        try:
            card, day, status = homer_view()
            record = homer.season_record(SNAPSHOT_DIR)
            return render_homer_html(card, record, homer.card_days(SNAPSHOT_DIR), day, status)
        except Exception as e:
            return error_page(e)

    @app.route("/api/homer")
    def api_homer():
        """HOMER's card and season record as JSON."""
        try:
            card, day, status = homer_view()
            return {"date": day, "status": status, "card": card,
                    "record": homer.season_record(SNAPSHOT_DIR)}, 200 if card else 202
        except Exception as e:
            print(f"[app.py] /api/homer failed: {e}")
            return {"error": str(e)}, 500

    @app.route("/snapshot")
    def snapshot():
        """Archive the current slate on demand: PDFs, CSVs and the workbook."""
        try:
            slate_data = latest_slate()
            if slate_data is None:
                return {"error": "no slate available yet"}, 503
            written = write_snapshot(slate_data, SNAPSHOT_DIR)
            return written, 200
        except Exception as e:
            print(f"[app.py] /snapshot failed: {e}")
            return {"error": str(e)}, 500

    @app.route("/api/remaining")
    def api_remaining():
        """Projections for games that are not final yet, as JSON.

        The same data the "Still to Play" section renders, for anyone who
        wants tonight's numbers without the page around them.
        """
        try:
            slate_data = latest_slate()
            if slate_data is None:
                return {"error": "no games scheduled"}, 404
            return {
                "date": slate_data["date"],
                "built_at": slate_data.get("built_at"),
                "fetched_at": (slate_data.get("results") or {}).get("fetched_at"),
                "games": remaining_projections(slate_data),
            }, 200
        except Exception as e:
            print(f"[app.py] /api/remaining failed: {e}")
            return {"error": str(e)}, 500

    @app.route("/health")
    def health():
        """Health check endpoint."""
        return {"status": "ok"}, 200

    return app


if __name__ == "__main__":
    # The banner and several log lines carry non-ASCII. A Windows console (or a
    # redirected stdout) defaults to a legacy codepage, where printing those
    # raises UnicodeEncodeError and kills the server before it ever binds. The
    # container runs UTF-8, so this only bites when the app is run on the host.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass
    app = create_app()
    print("\n" + "=" * 60)
    print("🏟️  HR Daily Tracker — Flask Server")
    print("=" * 60)
    print("Starting on http://localhost:5000")
    print("Press CTRL+C to stop")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
