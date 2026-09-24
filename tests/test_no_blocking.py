"""The page must never wait on a model fit.

A fit takes ten minutes. Before this test existed, the first request for a date
that was not cached ran the fit inline while holding the cache lock, so that
request hung for the whole fit and every other reader queued behind it. These
tests pin the three behaviours that replaced it:

  - a cold start answers immediately with a warming-up page;
  - once yesterday is cached, an unfitted today serves yesterday with a note;
  - concurrent readers are never serialised behind a build.

Run: python -m pytest tests/test_no_blocking.py -v
     (or: python tests/test_no_blocking.py)
"""
from __future__ import annotations

import sys
import threading
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mlb_hr.app as app_module

BUILD_SECONDS = 4.0


class FakeGame:
    def __init__(self, day):
        self.game_pk = 1
        self.date = str(day)
        self.state = "Live"
        self.home = "NYY"
        self.away = "BOS"
        self.home_id = 147
        self.away_id = 111
        self.venue = "Yankee Stadium"


def _install_fakes(monkey_state: dict):
    """Replace everything slow or networked with something instant."""
    app_module.fetch_season = lambda *a, **k: None
    app_module.schedule = lambda start, end, game_type="R": (
        [FakeGame(start)] if start >= monkey_state["first_day"] else []
    )
    app_module.attach_results = lambda slate, results: slate
    app_module.fetch_results = lambda *a, **k: {}
    app_module.render_html = lambda slate: f"SLATE {slate['date']}"
    app_module.render_models_html = lambda slate: f"MODELS {slate['date']}"
    # The warmer settles past days and locks started picks; neither may touch
    # the real snapshot archive or the network from a test.
    app_module.finalize_pending = lambda *a, **k: []
    app_module.load_slate = lambda *a, **k: None
    app_module.published_picks = lambda *a, **k: None
    app_module.live_games = lambda *a, **k: {}
    app_module.write_snapshot = lambda *a, **k: {"pdfs": []}

    def slow_build(day, path, use_ensemble=True):
        time.sleep(BUILD_SECONDS)
        return {"date": str(day), "games": [], "picks_6": [], "hit_picks": []}

    app_module.build_slate_for_date = slow_build


def test_cold_start_does_not_block():
    """A request during the very first fit answers at once, not in 4 seconds."""
    _install_fakes({"first_day": date(2000, 1, 1)})
    client = app_module.create_app("/tmp/none.jsonl").test_client()

    started = time.time()
    resp = client.get("/")
    elapsed = time.time() - started

    assert elapsed < 1.5, f"cold request blocked for {elapsed:.1f}s"
    assert resp.status_code == 503, resp.status_code
    assert b"Fitting the model" in resp.data
    print(f"  cold start answered in {elapsed:.2f}s with a warming page")

    # And once the fit lands, the real slate is served.
    time.sleep(BUILD_SECONDS + 1.5)
    resp = client.get("/")
    assert resp.status_code == 200, resp.status_code
    assert resp.data.startswith(b"SLATE"), resp.data[:40]
    print(f"  after the fit: {resp.data.decode()}")


def test_concurrent_readers_are_not_serialised():
    """Ten readers during a build all answer promptly, not one after another."""
    _install_fakes({"first_day": date(2000, 1, 1)})
    client = app_module.create_app("/tmp/none.jsonl").test_client()

    timings: list[float] = []
    lock = threading.Lock()

    def reader():
        t0 = time.time()
        client.get("/")
        with lock:
            timings.append(time.time() - t0)

    threads = [threading.Thread(target=reader) for _ in range(10)]
    started = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - started

    assert max(timings) < 1.5, f"slowest reader took {max(timings):.1f}s"
    assert wall < BUILD_SECONDS, f"readers serialised: {wall:.1f}s wall clock"
    print(f"  10 concurrent readers: slowest {max(timings):.2f}s, wall {wall:.2f}s")


def test_previous_day_is_served_while_today_builds():
    """With yesterday cached, an unfitted today serves yesterday plus a note."""
    _install_fakes({"first_day": date(2000, 1, 1)})
    application = app_module.create_app("/tmp/none.jsonl")
    client = application.test_client()

    # Let the warmer finish the first day.
    time.sleep(BUILD_SECONDS + 1.5)
    first = client.get("/")
    assert first.status_code == 200, first.status_code
    served_day = first.data.decode().split()[1]

    # Advance the clock a day: the target moves, nothing is cached for it.
    real_baseball_day = app_module.baseball_day
    app_module.baseball_day = lambda now=None: real_baseball_day() + timedelta(days=1)
    try:
        resp = client.get("/")
        assert resp.status_code == 200, "should serve the older slate, not 503"
        assert resp.data.decode().split()[1] == served_day, (
            "expected the previous day's slate while the new one builds"
        )
        print(f"  new day: served {served_day} while today fits")
    finally:
        app_module.baseball_day = real_baseball_day


if __name__ == "__main__":
    failures = 0
    for fn in (
        test_cold_start_does_not_block,
        test_concurrent_readers_are_not_serialised,
        test_previous_day_is_served_while_today_builds,
    ):
        print(f"\n{fn.__name__}")
        try:
            fn()
            print("  PASS")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL: {exc}")
    print(f"\n{'all passed' if not failures else f'{failures} failed'}")
    sys.exit(1 if failures else 0)
