"""The Results page (past days, settled) and HOMER's page.

Both reuse the slate page's stylesheet, tabs and theme toggle from `render`, so
the four tabs read as one site.
"""
from __future__ import annotations

from mlb_hr.homer import EXPECTED_SIGN
from mlb_hr.render import (
    _CSS, _THEME_SCRIPT, _hand_badge, _result_badge, _section_nav, _tabs, _verdict,
)

_EXTRA_CSS = """
        .day-pills { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 14px; }
        .day-pill {
            padding: 5px 11px; border: 1px solid var(--border); border-radius: 999px;
            font-size: 0.82em; color: var(--text-secondary); text-decoration: none;
            background: var(--bg-primary);
        }
        .day-pill:hover { border-color: var(--accent); color: var(--accent); }
        .day-pill.active { background: var(--accent); border-color: var(--accent); color: var(--on-accent); }
        .rv-table { width: 100%; border-collapse: collapse; font-size: 0.9em; }
        .rv-table th, .rv-table td {
            padding: 8px 10px; border-bottom: 1px solid var(--border); text-align: right;
            white-space: nowrap;
        }
        .rv-table th {
            font-size: 0.78em; text-transform: uppercase; letter-spacing: 0.04em;
            color: var(--text-secondary); font-weight: 600;
        }
        .rv-table th.l, .rv-table td.l { text-align: left; }
        .rv-table tr.row-hit td { background: var(--success-light); }
        .rv-table tr.row-miss td { color: var(--text-secondary); }
        .ok { color: var(--success); font-weight: 700; }
        .bad { color: var(--danger); font-weight: 700; }
        .muted { color: var(--text-secondary); }
        .final-line { display: flex; justify-content: space-between; align-items: baseline;
                      font-size: 1.15em; font-weight: 700; margin: 8px 0; }
        .final-line .win { color: var(--text-primary); }
        .final-line .lose { color: var(--text-secondary); font-weight: 500; }
        .homer-hero {
            display: flex; gap: 18px; align-items: center; background: var(--bg-secondary);
            border: 1px solid var(--border); border-radius: 12px; padding: 18px; margin-top: 18px;
        }
        .homer-avatar {
            flex: 0 0 64px; height: 64px; border-radius: 50%; display: grid; place-items: center;
            font-size: 34px; background: var(--accent-light); border: 2px solid var(--accent);
        }
        .homer-quote { font-size: 0.95em; line-height: 1.55; }
        .homer-quote strong { color: var(--accent); }
        .grade {
            display: inline-block; min-width: 30px; text-align: center; padding: 2px 8px;
            border-radius: 6px; font-weight: 800; font-size: 0.85em;
            background: var(--accent-light); color: var(--accent);
        }
        .grade-A { background: var(--success-light); color: var(--success); }
        .take { margin: 10px 0 0 0; padding-left: 18px; font-size: 0.86em; line-height: 1.55; }
        .take li { margin-bottom: 3px; }
        .vs-model { font-size: 0.8em; color: var(--text-secondary); margin-top: 4px; }
        details.method { margin-top: 12px; }
        details.method summary { cursor: pointer; font-weight: 600; color: var(--accent); }
        details.method ul { margin: 10px 0 0 18px; line-height: 1.6; font-size: 0.9em; }
        .tag { font-size: 0.72em; font-weight: 800; letter-spacing: 0.05em; color: var(--accent);
               margin-left: 6px; white-space: nowrap; }
        .plays-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }
        .play-card { border: 1px solid var(--border); border-radius: 12px; padding: 18px;
                     background: var(--bg-secondary); }
        .play-featured { border: 2px solid var(--accent); background: var(--accent-light);
                         grid-column: 1 / -1; }
        .play-head { display: flex; justify-content: space-between; align-items: baseline; gap: 10px;
                     flex-wrap: wrap; }
        .play-name { font-size: 1.2em; font-weight: 800; }
        .play-featured .play-name { font-size: 1.45em; }
        .play-odds { font-size: 1.25em; font-weight: 800; color: var(--accent); }
        .play-blurb { font-size: 0.88em; color: var(--text-secondary); margin: 6px 0 12px; }
        .leg { display: flex; justify-content: space-between; gap: 10px; padding: 7px 0;
               border-top: 1px solid var(--border); font-size: 0.9em; }
        .leg-type { font-size: 0.72em; font-weight: 800; padding: 2px 6px; border-radius: 4px;
                    background: var(--bg-primary); margin-right: 6px; }
        .leg-won { color: var(--success); font-weight: 700; }
        .leg-lost { color: var(--danger); font-weight: 700; text-decoration: line-through; }
        .play-status { font-weight: 800; }
        @media (max-width: 600px) {
            .homer-hero { flex-direction: column; text-align: center; }
        }
"""


def _page(title: str, h1: str, subtitle: str, stats: str, active: str,
          nav: list, body: str, footer: str, refresh: int = 0) -> str:
    reload = f'\n    <meta http-equiv="refresh" content="{refresh}">' if refresh else ""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">{reload}
    <title>{title}</title>
    <style>{_CSS}{_EXTRA_CSS}    </style>{_THEME_SCRIPT}
</head>
<body>
    <div class="container">
        <header>
            <h1>{h1}</h1>
            <p class="subtitle">{subtitle}</p>
            <div class="stats-bar">{stats}
            </div>
            {_tabs(active)}
            {_section_nav(nav) if nav else ''}
        </header>
        {body}
        <footer>
            <p style="opacity: 0.7;">{footer}</p>
        </footer>
    </div>
</body>
</html>
"""


def _stat(label: str, value: str, color: str = "") -> str:
    style = f' style="color: {color};"' if color else ""
    return f"""
                <div class="stat-card">
                    <div class="stat-label">{label}</div>
                    <div class="stat-value"{style}>{value}</div>
                </div>"""


def _pct(p) -> str:
    return "&mdash;" if p is None else f"{p:.1%}"


def _day_pills(days: list, current: str, base: str) -> str:
    if not days:
        return ""
    pills = "".join(
        f'<a class="day-pill{" active" if d == current else ""}" href="{base}?date={d}">'
        f'{d[5:]}</a>'
        for d in sorted(days, reverse=True)[:21]
    )
    return f'<div class="day-pills">{pills}</div>'


# ──────────────────────────────────────────────────────────────── results


def _picks_table(entries: list, key: str) -> str:
    rows = ""
    for i, p in enumerate(entries, 1):
        tone = _verdict(p, key)[0]
        if key == "hr":
            proj = _pct(p.get("prob_hr"))
        else:
            hp = p.get("hits_proj") or {}
            proj = (f"{hp.get('projected_hits', 0):.2f} "
                    f"<span class='muted'>({hp.get('prob_at_least_one', 0):.0%} 1+)</span>")
        badge = _result_badge(p, key) if p.get("result") else '<span class="muted">no line</span>'
        rows += f"""
                    <tr class="{'row-hit' if tone == 'hit' else 'row-miss' if tone == 'miss' else ''}">
                        <td>{i}</td>
                        <td class="l"><strong>{p.get('batter', '')}</strong><br><span class="muted">{p.get('team', '')}</span></td>
                        <td>{proj}</td>
                        <td class="l">{badge}</td>
                    </tr>"""
    unit = "HR prob" if key == "hr" else "Proj. hits"
    return f"""
            <div class="table-wrap">
                <table class="rv-table">
                    <tr><th>#</th><th class="l">Hitter</th><th>{unit}</th><th class="l">Result</th></tr>{rows}
                </table>
            </div>"""


def _homers_table(slate: dict) -> str:
    """Every home run hit by a hitter on the slate, with what the model gave him."""
    everyone = []
    for g in slate.get("games", []):
        for h in g.get("hitters", []):
            everyone.append(h)
    ranked = sorted(everyone, key=lambda h: h.get("prob_hr", 0), reverse=True)
    rank_of = {id(h): i for i, h in enumerate(ranked, 1)}
    pick_ids = {p.get("batter_id") for p in slate.get("picks_6") or []}
    hitters = [h for h in everyone if (h.get("result") or {}).get("hr")]
    hitters.sort(key=lambda h: h.get("prob_hr", 0), reverse=True)
    if not hitters:
        return '<div class="note-box">No home runs recorded for slate hitters.</div>'
    rows = ""
    for h in hitters:
        line = h["result"]
        star = ' <span class="grade grade-A">PICK</span>' if h.get("batter_id") in pick_ids else ""
        rows += f"""
                    <tr>
                        <td class="l"><strong>{h['batter']}</strong>{star}<br><span class="muted">{h.get('team', '')}</span></td>
                        <td>{line.get('hr', 0)}</td>
                        <td>{_pct(h.get('prob_hr'))}</td>
                        <td>#{rank_of[id(h)]} <span class="muted">of {len(ranked)}</span></td>
                        <td class="l muted">{line.get('summary', '')}</td>
                    </tr>"""
    top_decile = sum(1 for h in hitters if rank_of[id(h)] <= max(len(ranked) // 10, 1))
    return f"""
            <p class="section-note">
                {len(hitters)} slate hitters homered. {top_decile} of them were in the
                model's top 10% of the board ({max(len(ranked) // 10, 1)} hitters) &mdash;
                a random ranking would put about {len(hitters) / 10:.1f} there.
            </p>
            <div class="table-wrap">
                <table class="rv-table">
                    <tr><th class="l">Hitter</th><th>HR</th><th>Model prob</th><th>Board rank</th><th class="l">Line</th></tr>{rows}
                </table>
            </div>"""


def _final_cards(slate: dict, homer_games: dict) -> tuple[str, int, int]:
    cards = ""
    right = called = 0
    for g in slate.get("games", []):
        live = g.get("live") or {}
        proj = g.get("projection") or {}
        hs, as_ = live.get("home_score"), live.get("away_score")
        final = live.get("state") == "Final" and hs is not None and as_ is not None
        if final:
            score = f"""
                    <div class="final-line">
                        <span class="{'win' if as_ > hs else 'lose'}">{g['away']} {as_}</span>
                        <span class="{'win' if hs > as_ else 'lose'}">{hs} {g['home']}</span>
                    </div>"""
        else:
            score = f'<div class="game-state"><span class="res res-live">{live.get("detailed") or "Not final"}</span></div>'

        model_line = ""
        if proj:
            fav = proj.get("favorite")
            fav_p = proj.get("favorite_prob", 0)
            verdict = ""
            if final and hs != as_:
                winner = g["home"] if hs > as_ else g["away"]
                ok = winner == fav
                called += 1
                right += ok
                verdict = f' <span class="{"ok" if ok else "bad"}">{"&#10003;" if ok else "&#10007;"}</span>'
            total = ""
            if final:
                diff = (hs + as_) - proj.get("total_runs", 0)
                total = (f' &middot; total {proj.get("total_runs", 0):.1f} proj, '
                         f'{hs + as_} actual ({diff:+.1f})')
            model_line = f"""
                    <div class="ctx-row"><span>Model</span><strong>{fav} {fav_p:.0%}{verdict}</strong></div>
                    <div class="vs-model">Projected {proj.get('away_expected_runs', 0):.1f} &ndash; {proj.get('home_expected_runs', 0):.1f}{total}</div>"""

        homer_line = ""
        hg = homer_games.get(g.get("game_pk"))
        if hg:
            res = hg.get("result") or {}
            mark = ""
            if res.get("final") and not res.get("no_decision"):
                mark = f' <span class="{"ok" if res.get("won") else "bad"}">{"&#10003;" if res.get("won") else "&#10007;"}</span>'
            homer_line = (f'<div class="ctx-row"><span>&#129506; HOMER</span>'
                          f'<strong>{hg["pick"]} {hg["win_prob"]:.0%}{mark}</strong></div>')

        homers = [h for h in g.get("hitters", []) if (h.get("result") or {}).get("hr")]
        hr_list = ", ".join(f"{h['batter']}{' ×' + str(h['result']['hr']) if h['result']['hr'] > 1 else ''}"
                            for h in homers) or "none from slate hitters"
        cards += f"""
                <div class="game-card">
                    <div class="game-matchup">{g['away']} @ {g['home']}</div>
                    <div class="game-venue">{g.get('venue', '')}</div>{score}
                    <div class="game-pitchers">
                        <strong>{g['away']}</strong> SP: {g.get('away_sp') or 'TBD'} {_hand_badge(g.get('away_sp_hand'))}<br>
                        <strong>{g['home']}</strong> SP: {g.get('home_sp') or 'TBD'} {_hand_badge(g.get('home_sp_hand'))}
                    </div>
                    <div class="split-block">{model_line}{homer_line}
                    </div>
                    <div class="top-hitters">
                        <div class="top-hitters-label">Home runs</div>
                        <div class="hitter-list"><div class="hitter-item">{hr_list}</div></div>
                    </div>
                </div>"""
    return cards, right, called


def render_results_html(day: str, slate: dict | None, days: list,
                        picks_only: dict | None = None, homer: dict | None = None) -> str:
    """A settled day: picks against box scores, finals against projections."""
    pills = _day_pills(days, day, "/results")
    homer_games = {g["game_pk"]: g for g in (homer or {}).get("game_picks", [])}

    if slate is None and picks_only is None:
        body = f"""
        <div class="section">
            {pills}
            <div class="note-box" style="margin-top: 16px;">
                Nothing archived for {day}. Days are saved automatically as their
                games are played.
            </div>
        </div>"""
        return _page(f"Results — {day}", "&#128197; Results", f"Nothing archived for {day}",
                     "", "results", [], body, "HR Daily Tracker &mdash; results review")

    if slate is None:
        # Days archived before full slates were saved: the tracker rows only.
        summary = picks_only.get("summary") or {}
        rows = ""
        for r in picks_only.get("picks", []):
            won = r.get("won")
            mark = ('<span class="ok">&#10003;</span>' if won is True
                    else '<span class="bad">&#10007;</span>' if won is False else '<span class="muted">&mdash;</span>')
            proj = _pct(r.get("projected_prob")) if r.get("projected_prob") is not None else r.get("projected")
            rows += f"""
                    <tr><td class="l">{r['kind'].upper()}</td><td>{r['rank']}</td>
                        <td class="l"><strong>{r['batter']}</strong><br><span class="muted">{r['team']}</span></td>
                        <td>{proj}</td><td class="l muted">{r.get('result_line') or ''}</td><td>{mark}</td></tr>"""
        stats = (_stat("HR Picks Hit", f"{summary.get('hr_picks_hit', 0)}/{summary.get('hr_picks_final', 0)}",
                       "var(--success)")
                 + _stat("Hit Picks Hit", f"{summary.get('hit_picks_hit', 0)}/{summary.get('hit_picks_final', 0)}"))
        body = f"""
        <div class="section" id="picks">
            {pills}
            <h2 class="section-title" style="margin-top: 18px;">&#127919; Picks</h2>
            <div class="note-box" style="margin: 0 0 14px 0;">This day was archived before full
                slates were saved, so only the tracked picks are available.</div>
            <div class="table-wrap"><table class="rv-table">
                <tr><th class="l">Kind</th><th>#</th><th class="l">Hitter</th><th>Projected</th><th class="l">Line</th><th>Won</th></tr>{rows}
            </table></div>
        </div>"""
        return _page(f"Results — {day}", "&#128197; Results", f"How the picks did on {day}",
                     stats, "results", [], body, f"Results for {day} &middot; from the tracker CSV")

    res = slate.get("results") or {}
    picks = res.get("picks") or {}
    hits = res.get("hit_picks") or {}
    expected = sum(
        p.get("prob_hr", 0) for p in slate.get("picks_6") or []
        if (p.get("result") or {}).get("final") and not (p.get("result") or {}).get("dnp")
    )
    finals, right, called = _final_cards(slate, homer_games)

    stats = (
        _stat("HR Picks Hit", f"{picks.get('hit', 0)}/{picks.get('scored', 0)}", "var(--success)")
        + _stat("Expected HR", f"{expected:.2f}")
        + _stat("Hit Picks Hit", f"{hits.get('hit', 0)}/{hits.get('scored', 0)}")
        + _stat("Model Winners", f"{right}/{called}")
    )
    homer_block = ""
    nav = [("picks", "🎯 HR Picks"), ("hits", "🥎 Hit Picks"), ("homers", "💣 Every HR"),
           ("finals", "📊 Final Scores")]
    if homer and homer.get("record"):
        rec = homer["record"]
        stats += _stat("HOMER", f"{rec['hr']['hit']}/{rec['hr']['scored']} HR &middot; "
                                f"{rec['games']['won']}-{rec['games']['lost']}")
        rows = ""
        for p in homer.get("hr_picks", []):
            r = p.get("result") or {}
            mark = ("&#10003;" if r.get("hr") else "&#10007;" if r.get("final") and not r.get("dnp") else "&mdash;")
            cls = "ok" if r.get("hr") else "bad" if r.get("final") and not r.get("dnp") else "muted"
            rows += f"""
                    <tr><td class="l"><strong>{p['batter']}</strong><br><span class="muted">{p['team']}</span></td>
                        <td>{p['prob']:.1%}</td><td>{_pct(p.get('model_prob'))}</td>
                        <td class="l muted">{r.get('summary', '')}</td><td class="{cls}">{mark}</td></tr>"""
        homer_block = f"""
        <div class="section" id="homer">
            <h2 class="section-title">&#129506; HOMER's Card</h2>
            <p class="section-note">HOMER went {rec['hr']['hit']}-for-{rec['hr']['scored']} on home runs,
                {rec['games']['won']}-{rec['games']['lost']} picking winners{
                    f", {rec['hits']['hit']}-for-{rec['hits']['scored']} on hit picks" if rec.get('hits') else ''}{
                    f" and {rec['parlays']['won']}-{rec['parlays']['lost']} on his plays" if rec.get('parlays') else ''}.
                <a href="/homer?date={day}">Full card &rarr;</a></p>
            <div class="table-wrap"><table class="rv-table">
                <tr><th class="l">Hitter</th><th>HOMER</th><th>Model</th><th class="l">Line</th><th></th></tr>{rows}
            </table></div>
        </div>"""
        nav.append(("homer", "🧢 HOMER"))

    state = ("all games final" if res.get("final_games") and not res.get("live_games")
             and not res.get("upcoming_games") else
             f"{res.get('final_games', 0)} final, {res.get('live_games', 0)} live, "
             f"{res.get('upcoming_games', 0)} upcoming")
    body = f"""
        <div class="section">
            {pills}
        </div>
        <div class="section" id="picks">
            <h2 class="section-title">&#127919; HR Picks &mdash; how they did</h2>
            <p class="section-note">{picks.get('hit', 0)} of {picks.get('scored', 0)} settled picks homered
                against {expected:.2f} expected from their published probabilities.
                {f"{picks.get('dnp')} did not play." if picks.get('dnp') else ''}</p>
            {_picks_table(slate.get('picks_6') or [], 'hr')}
        </div>
        <div class="section" id="hits">
            <h2 class="section-title">&#129358; Hit Picks</h2>
            {_picks_table(slate.get('hit_picks') or [], 'hits')}
        </div>
        <div class="section" id="homers">
            <h2 class="section-title">&#128163; Every Home Run on the Slate</h2>
            {_homers_table(slate)}
        </div>{homer_block}
        <div class="section" id="finals">
            <h2 class="section-title">&#128202; Final Scores vs Projections</h2>
            <p class="section-note">The model called {right} of {called} winners.
                &#10003; / &#10007; mark whether each projected favorite won.</p>
            <div class="games-grid">{finals}
            </div>
        </div>"""
    footer = (f"Results for {day} &middot; {state} &middot; model built {slate.get('built_at', '')}"
              f" &middot; scores as of {res.get('fetched_at', '')}")
    return _page(f"Results — {day}", "&#128197; Results", f"How the slate played out on {day}",
                 stats, "results", nav, body, footer)


# ────────────────────────────────────────────────────────────────── HOMER


def _homer_intro(record: dict, evidence: dict | None = None) -> str:
    """HOMER in his own words. The brag is his; the numbers are his real record."""
    ev = evidence or {}
    if not record["hr_scored"]:
        receipts = "Card's fresh &mdash; the receipts start tonight."
    else:
        vs = record["hr_hit"] - record["hr_expected"]
        mood = ("I'm running HOT." if vs > 1 else "Right on my numbers, like always."
                if vs > -1 else "Bats are due, and when they come, they come in bunches.")
        bits = [f"{record['hr_hit']}-for-{record['hr_scored']} on home runs",
                f"{record['games_won']}-{record['games_lost']} on winners"]
        if record.get("hits_scored"):
            bits.append(f"{record['hits_hit']}-for-{record['hits_scored']} on hit picks")
        if record.get("parlays_won") or record.get("parlays_lost"):
            bits.append(f"{record['parlays_won']}-{record['parlays_lost']} on parlays")
        receipts = f"Live record: {', '.join(bits)}. {mood}"
    proof = ""
    hr_picks = (ev.get("hr_picks") or {}).get("calibrated")
    if hr_picks:
        proof = (f" Over {hr_picks['days']} replayed days my daily six homered "
                 f"{hr_picks['rate']:.0%} of the time, against {ev.get('board_rate', 0):.0%} for "
                 f"the average bat in the lineup.")
    return f"""
            <div class="homer-hero">
                <div class="homer-avatar">&#129506;</div>
                <div class="homer-quote">
                    <strong>HOMER</strong> &mdash; top-tier MLB expert and professional capper.
                    I look at <em>everything</em>: the arm on the mound, the bat in the box, the wind
                    and the heat, and every player's full body of work &mdash; what he's capable of and
                    <em>when</em> he's capable of doing it. I don't guess and I don't chase. I play to
                    win, because every win is another follower cashing with me. I get sharper with
                    every data pull: my whole season gets re-studied every day.{proof}
                    <br><span class="muted">{receipts}</span>
                </div>
            </div>"""


def _method() -> str:
    return """
            <details class="method">
                <summary>How HOMER does his homework</summary>
                <ul>
                    <li><strong>Expected home runs.</strong> Every batted ball this season is binned by exit
                        velocity and launch angle; the league's HR rate per bin turns a hitter's contact into
                        the home runs it deserved.</li>
                    <li><strong>Form &amp; splits.</strong> Last-21-day xHR, the hitter's own record against today's
                        pitcher hand, barrel rate, and the league platoon effect.</li>
                    <li><strong>Pitchers.</strong> HR and xHR allowed per batter faced, fly-ball rate and recent
                        form for the starter, and the bullpen he hands off to.</li>
                    <li><strong>Park</strong> factor from HR/PA at the venue vs the home team's road games;
                        <strong>weather</strong> from temperature and wind toward the hitter's pull field.</li>
                    <li><strong>He learns what wins.</strong> The season is replayed week by week with his book
                        frozen at each week's start. Weights are fitted only on earlier weeks and graded on the
                        next one. Factors that don't improve those unseen-week predictions are dropped, and the
                        learned weights are used only if they beat his original rule-of-thumb formula.</li>
                    <li><strong>Grades are earned.</strong> A / B+ / B / C bands come from where a pick sits on the
                        replayed board, and each shows how often picks in that band actually homered.</li>
                    <li><strong>Games.</strong> A wOBA runs model (lineup vs the starter and bullpen it faces) and
                        season run differential, weighted the same learned way.</li>
                    <li><strong>Hit picks.</strong> Hit rate, expected hits from contact quality, strikeout and
                        walk rates, and his own split, against the starter's hits allowed and strikeout rate,
                        the bullpen and the park &mdash; learned and tested the same way.</li>
                    <li><strong>HOMER'S PLAYS.</strong> 3- and 5-leg combos from his HR and hit boards, one leg
                        per game. Every play type is rebuilt on every replayed day to check that its quoted
                        odds hold up.</li>
                    <li><strong>Grows every day.</strong> His book refreshes with each data pull, and the whole
                        season replay reruns daily, so new results reshape his weights and grades.</li>
                    <li><strong>Selection rules.</strong> Six HR picks and six hit picks, one per game, 80+ PA, no
                        injury notes, and only hitters in the posted lineup. Picks and plays in started games
                        are locked.</li>
                </ul>
            </details>"""


def _homer_hr_cards(card: dict) -> str:
    cards = ""
    for i, p in enumerate(card.get("hr_picks", []), 1):
        r = p.get("result") or {}
        if r.get("dnp"):
            tone, label = "live", "&#8212; NO RESULT &middot; did not play"
        elif r.get("hr"):
            tone, label = "hit", f"&#10003; HIT &middot; {r['hr']} HR"
        elif r.get("final"):
            tone, label = "miss", "&#10007; MISS"
        elif r:
            tone, label = "live", f"&#9679; LIVE &middot; {r.get('state', '')}"
        else:
            tone, label = "", ""
        ribbon = (f'<div class="verdict verdict-{tone}"><span class="verdict-label">{label}</span>'
                  f'<span class="verdict-detail">{r.get("summary", "")}</span></div>') if tone else ""
        take = "".join(f"<li>{t}</li>" for t in p.get("take", []))
        model = p.get("model_prob")
        edge = ""
        if model:
            diff = p["prob"] - model
            edge = f"House model {model:.1%} &middot; HOMER {'+' if diff >= 0 else ''}{diff * 100:.1f} pts"
        slot = f'<span class="slot-badge">bats {p["slot"]}</span> ' if p.get("slot") else ""
        ev = p.get("evidence")
        evidence = (
            f'<div class="vs-model"><strong>Track record:</strong> grade-{ev["grade"]} picks homered '
            f'{ev["rate"]:.1%} of the time in the season replay ({ev["hit"]} of {ev["n"]:,})</div>'
            if ev else ""
        )
        cards += f"""
                <div class="pick-card{f' card-{tone}' if tone else ''}">
                    {ribbon}
                    <div class="pick-rank">{i}</div>
                    <div class="pick-name">{p['batter']} <span class="grade{' grade-A' if p.get('grade') == 'A' else ''}">{p.get('grade', '')}</span><span class="tag">{p.get('tag', '')}</span></div>
                    <div class="pick-team">{slot}{p['team']} &middot; vs {p['sp_name']} ({p['sp_hand']}HP)</div>
                    <div class="pick-prob">
                        <div class="pick-prob-label">HOMER's HR probability</div>
                        <div class="pick-prob-value">{p['prob']:.1%}</div>
                        <div class="vs-model" style="color: inherit; opacity: 0.8;">{edge}</div>
                    </div>
                    {evidence}
                    <ul class="take">{take}</ul>
                    <div class="split-block">
                        <div class="block-label">HOMER's worksheet</div>
                        <div class="ctx-row"><span>xHR / PA (season)</span><strong>{p['xhr_rate']:.2%}</strong></div>
                        <div class="ctx-row"><span>Last {21} days xHR / PA</span><strong>{p['recent_rate']:.2%}</strong></div>
                        <div class="ctx-row"><span>Barrel rate</span><strong>{p['barrel_rate']:.1%}</strong></div>
                        <div class="ctx-row"><span>Starter HR index</span><strong>{p['sp_index']:.2f}&times;</strong></div>
                        <div class="ctx-row"><span>Bullpen HR index</span><strong>{p['pen_index']:.2f}&times;</strong></div>
                        <div class="ctx-row"><span>Platoon</span><strong>{p['platoon']:.2f}&times;</strong></div>
                        <div class="ctx-row"><span>Park &middot; weather</span><strong>{p['park']:.2f}&times; &middot; {p['weather']:.2f}&times;</strong></div>
                        <div class="ctx-row"><span>Trips vs SP / pen</span><strong>{p['pa_vs_sp']:.1f} / {p['pa_vs_pen']:.1f}</strong></div>
                    </div>
                </div>"""
    return cards


def _homer_hit_cards(card: dict) -> str:
    cards = ""
    for i, p in enumerate(card.get("hit_picks", []), 1):
        r = p.get("result") or {}
        if r.get("dnp"):
            tone, label = "live", "&#8212; NO RESULT &middot; did not play"
        elif r.get("hits"):
            tone, label = "hit", f"&#10003; CASHED &middot; {r['hits']} H"
        elif r.get("final"):
            tone, label = "miss", "&#10007; MISS"
        elif r:
            tone, label = "live", f"&#9679; LIVE &middot; {r.get('state', '')}"
        else:
            tone, label = "", ""
        ribbon = (f'<div class="verdict verdict-{tone}"><span class="verdict-label">{label}</span>'
                  f'<span class="verdict-detail">{r.get("summary", "")}</span></div>') if tone else ""
        ev = p.get("evidence")
        evidence = (f'<div class="vs-model"><strong>Track record:</strong> grade-{ev["grade"]} hit picks '
                    f'cashed {ev["rate"]:.1%} in the replay ({ev["hit"]:,} of {ev["n"]:,})</div>'
                    if ev else "")
        model = p.get("model_prob")
        model_note = f'<div class="vs-model">House model: {model:.0%}</div>' if model else ""
        take = "".join(f"<li>{t}</li>" for t in p.get("take", []))
        cards += f"""
                <div class="hit-card{f' card-{tone}' if tone else ''}">
                    {ribbon}
                    <div class="hit-rank">{i}</div>
                    <div class="pick-name">{p['batter']} <span class="grade{' grade-A' if p.get('grade') == 'A' else ''}">{p.get('grade', '')}</span><span class="tag">{p.get('tag', '')}</span></div>
                    <div class="pick-team">{p['team']} &middot; vs {p['sp_name']} ({p['sp_hand']}HP)</div>
                    <div class="hit-value">{p['prob']:.0%} <span class="hit-unit">1+ hit</span></div>
                    {model_note}{evidence}
                    <ul class="take">{take}</ul>
                </div>"""
    return cards


def _plays(card: dict) -> str:
    """HOMER'S PLAYS: his 3- and 5-leg combos, with honest odds."""
    plays = card.get("parlays") or []
    if not plays:
        return ""
    labels = {"won": ("&#10003; CASHED", "ok"), "lost": ("&#10007; LOST", "bad"),
              "void": ("VOID", "muted"), "pending": ("", "")}
    cards = ""
    for play in sorted(plays, key=lambda p: not p.get("featured")):
        legs = ""
        for leg in play["leg_list"]:
            st = leg.get("status", "pending")
            cls = {"won": "leg-won", "lost": "leg-lost"}.get(st, "")
            mark = {"won": " &#10003;", "lost": " &#10007;", "void": " (void)"}.get(st, "")
            what = "HR" if leg["type"] == "hr" else "1+ HIT"
            book = ""
            if leg.get("book_implied"):
                book = (f' &middot; book ~{leg["book_price"]} ({leg["book_implied"]:.0%})'
                        f' &middot; edge {leg["book_edge"]:+.0%}')
            legs += (f'<div class="leg"><span class="{cls}"><span class="leg-type">{what}</span>'
                     f'{leg["batter"]}{mark}<br><span class="muted">{leg.get("team", "")} vs '
                     f'{leg.get("sp_name", "")}{book}</span></span><strong>{leg["prob"]:.0%}</strong></div>')
        status, cls = labels.get(play.get("status", "pending"), ("", ""))
        ev = play.get("evidence")
        record = ""
        if ev and ev.get("days"):
            # A play that lands every 2.4 days must not read as every 2, so keep a
            # decimal while the cadence is short enough for one to matter.
            cadence = ev.get("days_per_win")
            every = (f', about once every {cadence:.1f} days' if cadence and cadence < 10
                     else f', about once every {cadence:.0f} days' if cadence else "")
            price = f'; typical ticket ~{ev["typical_odds"]}' if ev.get("typical_odds") else ""
            record = (f'<div class="vs-model"><strong>Replay:</strong> this play cashed '
                      f'{ev["won"]} of {ev["days"]} days ({ev["rate"]:.1%}{every}) vs '
                      f'{ev["predicted"]:.1%} predicted{price}</div>')
        book_line = ""
        if play.get("book_odds"):
            verdict = ("HOMER's number beats the price" if play["ev"] > 0
                       else "the book's margin eats this one &mdash; shop for a boost")
            book_line = f"""
                    <div class="leg"><span>Est. book price &mdash; {verdict}</span>
                        <span class="play-odds">{play['book_odds']}</span></div>
                    <div class="leg"><span>HOMER's expected return per $1 at that price</span>
                        <strong>{play['ev']:+.0%}</strong></div>"""
        n_val = play.get("value")
        if n_val is None:
            # Cards saved before the value mix was recorded carry the old
            # all-or-nothing build instead.
            build = "BUILT TO PAY" if play.get("build") == "value" else "BUILT TO CASH"
        elif not n_val:
            build = "BUILT TO CASH"
        elif n_val == play["legs"]:
            build = "BUILT TO PAY"
        else:
            build = f"{play['legs'] - n_val} LIKELY &middot; {n_val} VALUE"
        cards += f"""
                <div class="play-card{' play-featured' if play.get('featured') else ''}">
                    <div class="play-head">
                        <span class="play-name">{'&#129506; ' if play.get('featured') else ''}{play['name']}
                            <span class="muted" style="font-size: 0.6em;">{play['legs']}-leg &middot; {build}</span></span>
                        <span class="play-status {cls}">{status}</span>
                    </div>
                    <div class="play-blurb">{play['blurb']}</div>
                    {legs}
                    <div class="leg"><span>Hits if every leg cashes</span>
                        <strong>{play['prob']:.1%}</strong></div>
                    <div class="leg"><span>Fair odds &mdash; only play it at a better price</span>
                        <span class="play-odds">{play['fair_odds']}</span></div>{book_line}
                    {record}
                </div>"""
    return f"""
        <div class="section" id="plays">
            <h2 class="section-title">&#128176; HOMER'S PLAYS</h2>
            <p class="section-note">Built leg by leg from HOMER's hit board, one leg per game so no single
                rainout or pitcher sinks the ticket. Each play shows its real chance of cashing, the fair
                odds it needs, and what a sportsbook would likely hang on it. Every play on this board
                cleared the same bar in the replay &mdash; it had to cash on at least three days in ten
                &mdash; so what separates them is price, not whether they land. Parlays are still
                high-variance: stake accordingly and never bet more than you can afford to lose.</p>
            <div class="game-card" style="margin-bottom: 14px;">
                <div class="block-label">&#127922; How the book works &mdash; and how HOMER counters</div>
                <ul class="take">
                    <li><strong>The juice compounds.</strong> Every leg carries the book's margin, and a parlay
                        multiplies them. Four hit legs at a normal hold pay about 22% less than their true odds,
                        which is why none of these tickets runs long.</li>
                    <li><strong>No home run legs.</strong> HOMER's best home run bat cashes about one start in
                        five, and a parlay can never beat its weakest leg &mdash; so a ticket carrying one
                        cannot clear the cash bar these plays are held to. His home run board is upstairs;
                        play those straight.</li>
                    <li><strong>The book leans on the season line.</strong> Props are priced mostly off a hitter's
                        rate and name. HOMER prices the matchup: the starter, bullpen, park, weather, lineup slot
                        and contact quality. "Built to pay" plays only take legs where that read clearly beats the
                        season line.</li>
                    <li><strong>One leg per game.</strong> Legs from the same game move together, and books reprice
                        same-game parlays to take that edge back. Spreading legs keeps the math honest.</li>
                    <li><strong>A scratched player voids his leg;</strong> the ticket rides on the rest at a smaller
                        payout.</li>
                    <li><strong>Price is estimated</strong> until live odds are wired in. If your book is shorter
                        than the fair odds, pass or hunt a boost.</li>
                    <li><strong>Size by how often it cashes, not by what it pays.</strong> These plays land
                        between three and four days in ten, so flat stakes suit all of them &mdash; but even a
                        41% ticket goes quiet for a week or more, and the replay line on each card shows its
                        worst run. Read that before you stake it.</li>
                </ul>
            </div>
            <div class="plays-grid">{cards}
            </div>
        </div>"""


def _homer_game_cards(card: dict) -> str:
    cards = ""
    for g in card.get("game_picks", []):
        r = g.get("result") or {}
        if r.get("no_decision"):
            status = f'<span class="res res-live">{r.get("state") or "No decision"}</span>'
        elif r.get("final"):
            status = (f'<span class="res {"res-hit" if r["won"] else "res-miss"}">'
                      f'{"&#10003; WON" if r["won"] else "&#10007; LOST"}</span>'
                      f'<span class="game-score">{g["away"]} {r["away_score"]} &ndash; {r["home_score"]} {g["home"]}</span>')
        elif r:
            status = (f'<span class="res res-live">&#9679; {r.get("state", "Live")}</span>'
                      f'<span class="game-score">{g["away"]} {r.get("away_score")} &ndash; {r.get("home_score")} {g["home"]}</span>')
        else:
            status = ""
        model = g.get("model_pick_prob")
        model_note = ""
        if model is not None:
            agree = "agrees" if model >= 0.5 else "disagrees"
            model_note = f'<div class="vs-model">House model {agree}: {model:.0%} on {g["pick"]}</div>'
        take = "".join(f"<li>{t}</li>" for t in g.get("take", []))
        ev = g.get("evidence")
        if ev and ev.get("rate") is not None:
            model_note += (f'<div class="vs-model"><strong>Track record:</strong> {ev["label"]} calls won '
                           f'{ev["rate"]:.1%} in the replay ({ev["won"]} of {ev["n"]:,})</div>')
        cards += f"""
                <div class="game-card{' card-hit' if r.get('won') else ' card-miss' if r.get('final') and not r.get('no_decision') else ''}">
                    <div class="game-matchup">{g['matchup']}</div>
                    <div class="game-venue">{g.get('venue', '')}</div>
                    {f'<div class="game-state">{status}</div>' if status else ''}
                    <div class="ctx-row" style="margin-top: 10px;"><span>HOMER takes</span>
                        <strong>{g['pick']} <span class="grade{' grade-A' if g['confidence'] == 'Strong' else ''}">{g.get('tag') or g['confidence']}</span></strong></div>
                    <div class="ctx-row"><span>Win probability</span><strong>{g['win_prob']:.0%}</strong></div>
                    <div class="ctx-row"><span>Projected score</span><strong>{g['away_runs']:.1f} &ndash; {g['home_runs']:.1f} &middot; total {g['total']:.1f}</strong></div>
                    {model_note}
                    <ul class="take">{take}</ul>
                </div>"""
    return cards


def _disagreements(card: dict) -> str:
    def table(rows: list, title: str) -> str:
        body = "".join(
            f"""
                    <tr><td class="l"><strong>{c['batter']}</strong><br><span class="muted">{c['team']} &middot; vs {c['sp_name']}</span></td>
                        <td>{c['prob']:.1%}</td><td>{c['model_prob']:.1%}</td>
                        <td class="{'ok' if c['edge'] > 0 else 'bad'}">{c['edge'] * 100:+.1f}</td>
                        <td>{c['barrel_rate']:.1%}</td></tr>"""
            for c in rows
        )
        return f"""
            <h3 style="margin: 16px 0 8px;">{title}</h3>
            <div class="table-wrap"><table class="rv-table">
                <tr><th class="l">Hitter</th><th>HOMER</th><th>Model</th><th>Edge (pts)</th><th>Barrel%</th></tr>{body}
            </table></div>"""

    if not card.get("likes"):
        return ""
    return (table(card["likes"], "HOMER likes more than the model")
            + table(card["fades"], "HOMER is fading"))


def _grade_note(card: dict) -> str:
    bands = (card.get("evidence") or {}).get("grades") or []
    if not bands:
        return "Grades: A &ge; 20%, B+ &ge; 17%, B &ge; 14% (rule of thumb &mdash; no replay yet)."
    parts = ", ".join(f"{b['grade']} homered {b['rate']:.1%}" for b in bands)
    return (f"Grades are earned in the season replay: {parts}. "
            "Bands that didn't outperform the band below were merged.")


def _homework(card: dict) -> str:
    """The season replay behind the card: what HOMER learned and how it tested."""
    ev = card.get("evidence")
    if not ev:
        return """
        <div class="section" id="homework">
            <h2 class="section-title">&#128218; HOMER's Homework</h2>
            <div class="note-box">No season replay yet, so this card uses HOMER's rule-of-thumb
                formula. Run <code>python -m mlb_hr.homer_fit</code> (the app also does it daily).</div>
        </div>"""

    def score_row(label, s, served=False):
        if not s:
            return ""
        return (f'<tr{" class=row-hit" if served else ""}><td class="l">{label}</td>'
                f'<td>{s.get("auc", 0):.3f}</td><td>{s["log_loss"]:.4f}</td><td>{s["n"]:,}</td></tr>')

    hs = ev.get("hr_scores") or {}
    serve = ev.get("hr_serve")
    hr_table = (score_row("HOMER, learned weights", hs.get("learned"), serve == "learned")
                + score_row("HOMER, rule-of-thumb formula", hs.get("rules"), serve == "rules")
                + score_row("Season HR rate only (baseline)", hs.get("baseline")))

    picks = ev.get("hr_picks") or {}

    def pick_row(label, p):
        if not p:
            return ""
        return (f'<tr><td class="l">{label}</td><td>{p["hit"]}/{p["picks"]:,}</td>'
                f'<td><strong>{p["rate"]:.1%}</strong> <span class="muted">&plusmn;{p["rate_se"] * 1.96:.1%}</span></td>'
                f'<td>{p["expected"]:.0f}</td></tr>')

    pick_table = (pick_row("HOMER, learned", picks.get("calibrated") or picks.get("learned"))
                  + pick_row("HOMER, rule-of-thumb", picks.get("rules"))
                  + pick_row("Season HR rate only", picks.get("base")))
    if ev.get("board_rate") is not None:
        pick_table += (f'<tr><td class="l">Any hitter in the lineup</td><td>&mdash;</td>'
                       f'<td><strong>{ev["board_rate"]:.1%}</strong></td><td>&mdash;</td></tr>')

    def is_overlap(w):
        # Older fits did not flag it; any negative HR weight is an overlap correction.
        return w.get("overlap", w["odds_per_sd"] < 0)

    weights = "".join(
        f'<tr><td class="l">{w["label"]}'
        f'{"<br><span class=muted>overlap correction: already counted in expected HR</span>" if is_overlap(w) else ""}</td>'
        f'<td class="{"ok" if w["odds_per_sd"] > 0 else "bad"}">{w["odds_per_sd"] * 100:+.1f}%</td></tr>'
        for w in ev.get("hr_weights") or []
    )
    dropped = ", ".join(
        f'{d["dropped"]}' for d in ev.get("hr_dropped") or []
    ) or "none"
    grades = "".join(
        f'<tr><td class="l"><span class="grade{" grade-A" if g["grade"] == "A" else ""}">{g["grade"]}</span></td>'
        f'<td>{"top " + format(g["top_share"], ".0%") if g["grade"] != "C" else "rest"}</td>'
        f'<td>{g["avg_prob"]:.1%}</td><td><strong>{g["rate"]:.1%}</strong></td><td>{g["n"]:,}</td></tr>'
        for g in ev.get("grades") or []
    )

    gs = ev.get("game_scores") or {}
    gserve = ev.get("games_serve")
    house = ev.get("house_winners")

    def acc_row(label, s, served=False):
        if not s:
            return ""
        return (f'<tr{" class=row-hit" if served else ""}><td class="l">{label}</td>'
                f'<td><strong>{s["accuracy"]:.1%}</strong> <span class="muted">&plusmn;{s["accuracy_se"] * 1.96:.1%}</span></td>'
                f'<td>{s["log_loss"]:.4f}</td><td>{s["n"]:,}</td></tr>')

    game_table = (acc_row("HOMER, learned weights", gs.get("learned"), gserve == "learned")
                  + acc_row("HOMER, rule-of-thumb formula", gs.get("rules"), gserve == "rules")
                  + acc_row("House model (its own walk-forward)", house)
                  + acc_row("Always take the home team", gs.get("home_team")))
    conf = "".join(
        f'<tr><td class="l">{c["label"]}</td><td>{c["n"]:,}</td>'
        f'<td><strong>{c["rate"]:.1%}</strong></td></tr>'
        for c in ev.get("confidence") or [] if c.get("rate") is not None
    )

    hit_block = ""
    hp = ev.get("hit_picks") or {}
    if hp:
        hit_rows = (pick_row("HOMER", hp.get("calibrated"))
                    + pick_row("HOMER, rule-of-thumb", hp.get("rules"))
                    + pick_row("Season hit rate only", hp.get("base")))
        if ev.get("hit_board_rate") is not None:
            hit_rows += (f'<tr><td class="l">Any hitter in the lineup</td><td>&mdash;</td>'
                         f'<td><strong>{ev["hit_board_rate"]:.1%}</strong></td><td>&mdash;</td></tr>')
        hit_grades = "".join(
            f'<tr><td class="l"><span class="grade">{g["grade"]}</span></td>'
            f'<td>{g["avg_prob"]:.1%}</td><td><strong>{g["rate"]:.1%}</strong></td><td>{g["n"]:,}</td></tr>'
            for g in ev.get("hit_grades") or []
        )
        hit_weights = "".join(
            f'<tr><td class="l">{w["label"]}'
            f'{"<br><span class=muted>overlap correction</span>" if EXPECTED_SIGN.get(w["feature"], 1) * w["odds_per_sd"] < 0 else ""}</td>'
            f'<td>{w["odds_per_sd"] * 100:+.1f}%</td></tr>'
            for w in ev.get("hit_weights") or []
        )
        hit_dropped = ", ".join(d["dropped"] for d in ev.get("hit_dropped") or []) or "none"
        hit_block = f"""
                <div class="game-card">
                    <div class="block-label">Daily 6 hit picks, replayed ({'learned' if ev.get('hits_serve') == 'learned' else 'rule-of-thumb'})</div>
                    <div class="table-wrap"><table class="rv-table">
                        <tr><th class="l">Method</th><th>Cashed</th><th>Rate</th><th>Expected</th></tr>{hit_rows}
                    </table></div>
                    <div class="table-wrap" style="margin-top: 10px;"><table class="rv-table">
                        <tr><th class="l">Grade</th><th>Avg prob</th><th>Got a hit</th><th>n</th></tr>{hit_grades}
                    </table></div>
                </div>
                <div class="game-card">
                    <div class="block-label">What moves HOMER's hit needle</div>
                    <div class="table-wrap"><table class="rv-table">
                        <tr><th class="l">Factor</th><th>Odds per 1 SD</th></tr>{hit_weights}
                    </table></div>
                    <div class="vs-model">Tested and dropped: {hit_dropped}</div>
                </div>"""

    parlay_block = ""
    if ev.get("parlays"):
        rows = "".join(
            f'<tr><td class="l">{p["name"]}</td><td>{p["won"]}/{p["days"]}</td>'
            f'<td><strong>{p["rate"]:.1%}</strong></td><td>{p["predicted"]:.1%}</td>'
            f'<td>{p.get("typical_odds") or "&mdash;"}</td>'
            f'<td>{"%+.0f%%" % (100 * p["est_roi"]) if p.get("est_roi") is not None else "&mdash;"}</td></tr>'
            for p in sorted(ev["parlays"].values(), key=lambda p: -p.get("rate", 0)) if p.get("days")
        )
        legs = (ev.get("book") or {}).get("legs") or {}
        names = {"hit_prob": "Top 5 hits", "hit_value": "Value hits",
                 "hr_prob": "Top 5 HR", "hr_value": "Value HR"}
        leg_rows = "".join(
            f'<tr><td class="l">{names[k]}</td><td><strong>{v["actual"]:.1%}</strong></td>'
            f'<td>{v["market"]:.1%}</td><td>{v["break_even"]:.1%}</td><td>{v["roi"]:+.1%}</td></tr>'
            for k, v in legs.items() if k in names
        )
        leg_table = f"""
                    <div class="block-label" style="margin-top: 12px;">Leg by leg vs the book</div>
                    <div class="table-wrap"><table class="rv-table">
                        <tr><th class="l">Legs</th><th>Cashed</th><th>Season line</th><th>Book needs</th><th>Straight ROI</th></tr>{leg_rows}
                    </table></div>""" if leg_rows else ""
        parlay_block = f"""
                <div class="game-card">
                    <div class="block-label">HOMER's plays, replayed every day</div>
                    <div class="table-wrap"><table class="rv-table">
                        <tr><th class="l">Play</th><th>Cashed</th><th>Rate</th><th>Predicted</th><th>Typical price</th><th>Est. ROI</th></tr>{rows}
                    </table></div>{leg_table}
                    <div class="vs-model">Same construction rules as today's plays. When the actual rate
                        tracks the predicted one, the odds HOMER quotes can be trusted. Prices and ROI use a
                        modeled season-rate book (no live odds feed yet), so treat them as estimates; a sharper
                        book pays less. With only a handful of wins per play, ROI swings widely.</div>
                </div>"""

    return f"""
        <div class="section" id="homework">
            <h2 class="section-title">&#128218; HOMER's Homework</h2>
            <p class="section-note">
                Season replay through {ev.get('through')} (refit {ev.get('fitted_at')}). Every number below
                is from weeks HOMER had not seen when he made the call. He is currently using
                <strong>{'learned weights' if serve == 'learned' else 'his rule-of-thumb formula'}</strong>
                for home runs and <strong>{'learned weights' if gserve == 'learned' else 'his rule-of-thumb formula'}</strong>
                for winners &mdash; whichever tested better.
            </p>
            <div class="games-grid">{parlay_block}
                <div class="game-card">
                    <div class="block-label">Daily 6 HR picks, replayed</div>
                    <div class="table-wrap"><table class="rv-table">
                        <tr><th class="l">Method</th><th>Hit</th><th>Rate</th><th>Expected</th></tr>{pick_table}
                    </table></div>
                    <div class="vs-model">Same selection rules as today's card, every replayed day.</div>
                </div>
                <div class="game-card">
                    <div class="block-label">Ranking every hitter-game</div>
                    <div class="table-wrap"><table class="rv-table">
                        <tr><th class="l">Method</th><th>AUC</th><th>Log loss</th><th>Games</th></tr>{hr_table}
                    </table></div>
                    <div class="vs-model">AUC: how often a hitter who homered was ranked above one who didn't
                        (0.5 = coin flip). Lower log loss = better-calibrated probabilities.</div>
                </div>
                <div class="game-card">
                    <div class="block-label">What each grade has earned</div>
                    <div class="table-wrap"><table class="rv-table">
                        <tr><th class="l">Grade</th><th>Board</th><th>Avg prob</th><th>Homered</th><th>n</th></tr>{grades}
                    </table></div>
                </div>
                <div class="game-card">
                    <div class="block-label">What moves HOMER's needle</div>
                    <div class="table-wrap"><table class="rv-table">
                        <tr><th class="l">Factor</th><th>Odds per 1 SD</th></tr>{weights}
                    </table></div>
                    <div class="vs-model">Tested and dropped (didn't help on unseen weeks): {dropped}</div>
                </div>{hit_block}
                <div class="game-card">
                    <div class="block-label">Picking winners, replayed</div>
                    <div class="table-wrap"><table class="rv-table">
                        <tr><th class="l">Method</th><th>Correct</th><th>Log loss</th><th>Games</th></tr>{game_table}
                    </table></div>
                    <div class="table-wrap" style="margin-top: 10px;"><table class="rv-table">
                        <tr><th class="l">Confidence</th><th>Games</th><th>Won</th></tr>{conf}
                    </table></div>
                </div>
            </div>
        </div>"""


def render_homer_html(card: dict | None, record: dict, days: list, day: str,
                      status: str = "") -> str:
    """HOMER's tab: today's card (or a past one), his reasoning and his record."""
    pills = _day_pills(days, day, "/homer")
    hr_rate = f"{record['hr_hit']}/{record['hr_scored']}" if record["hr_scored"] else "0/0"
    stats = (
        _stat("Season HR Picks", hr_rate, "var(--success)")
        + _stat("Hit Picks", f"{record.get('hits_hit', 0)}/{record.get('hits_scored', 0)}")
        + _stat("Parlays W-L", f"{record.get('parlays_won', 0)}-{record.get('parlays_lost', 0)}")
        + _stat("Winners W-L", f"{record['games_won']}-{record['games_lost']}")
    )
    subtitle = "Top-tier MLB expert &middot; professional capper &middot; plays built to win"

    if card is None:
        body = f"""
        <div class="section">{_homer_intro(record)}{_method()}{pills}
            <div class="note-box" style="margin-top: 16px; border-left: 3px solid var(--accent);">
                &#128218; {status or 'HOMER is studying film for ' + day + '. Check back in a minute.'}
            </div>
        </div>"""
        return _page(f"HOMER — {day}", "&#129506; HOMER", subtitle,
                     stats, "homer", [], body,
                     "HOMER builds his card after the day's slate is ready.", refresh=45)

    rec = card.get("record") or {}

    history = ""
    if record["days"]:
        rows = "".join(
            f"""
                    <tr><td class="l"><a href="/homer?date={d['date']}">{d['date']}</a></td>
                        <td>{d['hr_hit']}/{d['hr_scored']}</td><td>{d['hr_expected']:.2f}</td>
                        <td>{d.get('hits_hit', 0)}/{d.get('hits_scored', 0)}</td>
                        <td>{d.get('parlays_won', 0)}-{d.get('parlays_lost', 0)}</td>
                        <td>{d['games_won']}-{d['games_lost']}</td>
                        <td class="l muted">{'final' if d['complete'] else 'in progress'}{' &middot; backfilled' if d.get('backfilled') else ''}</td></tr>"""
            for d in record["days"]
        )
        history = f"""
        <div class="section" id="record">
            <h2 class="section-title">&#128210; HOMER's Record</h2>
            <div class="table-wrap"><table class="rv-table">
                <tr><th class="l">Date</th><th>HR hit</th><th>Expected</th><th>Hit picks</th><th>Parlays</th><th>Winners</th><th class="l"></th></tr>{rows}
            </table></div>
        </div>"""

    lg = card.get("league") or {}
    backfill = (
        '<div class="note-box" style="margin-top: 14px;">&#9888;&#65039; Backfilled card: '
        'built after the fact from the published slate, using only games played before '
        'this date. HOMER did not post it before first pitch.</div>'
        if card.get("backfilled") else ""
    )
    hit_section = ""
    if card.get("hit_picks"):
        bands = (card.get("evidence") or {}).get("hit_grades") or []
        note = ", ".join(f"{b['grade']} cashed {b['rate']:.0%}" for b in bands)
        hit_section = f"""
        <div class="section" id="hits">
            <h2 class="section-title">&#129358; HOMER's 6 Hit Picks</h2>
            <p class="section-note">At least one hit, one per game, ranked by HOMER's own probability.
                {f'Grades earned in the replay: {note}.' if note else ''}</p>
            <div class="hits-grid">{_homer_hit_cards(card)}
            </div>
        </div>"""
    body = f"""
        <div class="section">{_homer_intro(record, card.get('evidence'))}{_method()}{pills}{backfill}
        </div>{_plays(card)}
        <div class="section" id="hr">
            <h2 class="section-title">&#128163; HOMER's 6 Home Run Picks</h2>
            <p class="section-note">One per game, ranked by HOMER's own per-game probability.
                {_grade_note(card)}</p>
            <div class="picks-grid">{_homer_hr_cards(card)}
            </div>
        </div>{hit_section}
        <div class="section" id="games">
            <h2 class="section-title">&#9918; HOMER's Winners</h2>
            <p class="section-note">Every game, strongest opinion first.</p>
            <div class="games-grid">{_homer_game_cards(card)}
            </div>
        </div>
        <div class="section" id="vs">
            <h2 class="section-title">&#129354; HOMER vs the Model</h2>
            <p class="section-note">Same hitters, same day &mdash; where HOMER's numbers and the house
                ensemble part ways the most.</p>
            {_disagreements(card)}
        </div>{_homework(card)}{history}"""
    nav = ([("plays", "💰 HOMER'S PLAYS")] if card.get("parlays") else []) + [("hr", "💣 HR Picks")]
    if hit_section:
        nav.append(("hits", "🥎 Hit Picks"))
    nav += [("games", "⚾ Winners"), ("vs", "🥊 vs the Model"), ("homework", "📚 Homework")]
    if history:
        nav.append(("record", "📒 Record"))
    footer = (f"HOMER's card for {card.get('date')} &middot; built {card.get('built_at')} "
              f"&middot; book through {card.get('book_through')} "
              f"({lg.get('pa', 0):,} PA, league {lg.get('hr_rate', 0):.2%} HR/PA)"
              + (f" &middot; graded {rec.get('graded_at')}" if rec.get('graded_at') else "")
              + (f" &middot; {status}" if status else ""))
    return _page(f"HOMER — {card.get('date')}", "&#129506; HOMER",
                 f"{subtitle} &middot; {card.get('date')}",
                 stats, "homer", nav, body, footer)
