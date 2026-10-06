"""Live pages on Vercel: the published slate, re-scored on every request.

The fit runs on a schedule in GitHub Actions (mlb_hr.publish) and commits the
day's slate to snapshots/. Scores and box scores are two cheap MLB calls, so
this function pulls them per request and renders the page the same way the
Flask server does. The CDN holds each response for 30 seconds, and the pages
already swap in fresh content every 60.

Standard library only -- see requirements.txt. If anything fails, the page
prerendered at publish time is served instead, so the site never goes down
with the MLB API.
"""
from __future__ import annotations

import json
import sys
import traceback
from datetime import date
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

SNAPSHOTS = ROOT / "snapshots"
PUBLIC = ROOT / "public"
PAGES = ("slate", "models", "homer")


def current_slate():
    from mlb_hr.results import attach_results, fetch_results
    from mlb_hr.snapshot import load_slate

    day = json.loads((SNAPSHOTS / "current.json").read_text(encoding="utf-8")).get("date")
    if not day:
        return None, None
    slate = load_slate(day, SNAPSHOTS)
    if slate is None:
        return None, None
    results = fetch_results(date.fromisoformat(day),
                            [g.get("game_pk") for g in slate.get("games", [])])
    attach_results(slate, results)
    return slate, results


def render(page: str) -> str:
    from mlb_hr import homer
    from mlb_hr.publish_links import staticize

    slate, results = current_slate()
    if slate is None:
        raise LookupError("no current slate")
    if page == "slate":
        from mlb_hr.render import render_html
        html = render_html(slate)
    elif page == "models":
        from mlb_hr.render import render_models_html
        html = render_models_html(slate)
    else:
        from mlb_hr.render_review import render_homer_html
        day = slate["date"]
        card = homer.load_card(day, SNAPSHOTS)
        if card:
            homer.grade_card(card, results)
        html = render_homer_html(card, homer.season_record(SNAPSHOTS),
                                 homer.card_days(SNAPSHOTS), day, "")
    return staticize(html)


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        page = (parse_qs(urlparse(self.path).query).get("page") or ["slate"])[0]
        if page not in PAGES:
            page = "slate"
        try:
            body, cache = render(page), "public, s-maxage=30, stale-while-revalidate=60"
        except Exception:  # noqa: BLE001 - fall back to the prerendered page
            traceback.print_exc()
            body = (PUBLIC / "fallback" / f"{page}.html").read_text(encoding="utf-8")
            cache = "public, s-maxage=10"
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", cache)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
