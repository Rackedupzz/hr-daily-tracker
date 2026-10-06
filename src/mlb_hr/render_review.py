"""The Results page (past days, settled) and HOMER's page.

Both are built from the same pieces as the slate page (`mlb_hr.ui`), so the four
tabs read as one site.
"""
from __future__ import annotations

from datetime import date
from html import escape

from mlb_hr.homer import EXPECTED_SIGN
from mlb_hr.render import _result_badge, _verdict
from mlb_hr.ui import (
    abbr, alert, avatar, calib_chart, chance, color, day_pills, fair_odds, hand, head, hero, logo,
    mini, page, pct, person, ring, short, status_chip, subnav, tile, tiles, winbar,
)


def _pct(p) -> str:
    return "&mdash;" if p is None else f"{p:.1%}"


def _long_date(day: str) -> str:
    try:
        d = date.fromisoformat(str(day))
    except ValueError:
        return escape(str(day))
    return f"{d:%A}, {d:%B} {d.day}"


def _grade(g) -> str:
    if not g:
        return ""
    g = str(g)
    return f'<span class="grade g-{escape(g[:1])}">{escape(g)}</span>'


def _tag(t) -> str:
    return f'<span class="tag">{t}</span>' if t else ""


# ──────────────────────────────────────────────────────────────── results


def _picks_table(entries: list, key: str) -> str:
    if not entries:
        return '<div class="card muted">No picks were published for this day.</div>'
    top = max((p.get("prob_hr", 0) for p in entries), default=0) or 1
    rows = ""
    for i, p in enumerate(entries, 1):
        tone = _verdict(p, key)[0]
        if key == "hr":
            proj = f"{_pct(p.get('prob_hr'))}{mini(p.get('prob_hr', 0) / top)}"
        else:
            hp = p.get("hits_proj") or {}
            proj = (f"{hp.get('projected_hits', 0):.2f} "
                    f"<span class='muted'>({hp.get('prob_at_least_one', 0):.0%} 1+)</span>")
        badge = _result_badge(p, key) if p.get("result") else '<span class="muted">no line</span>'
        line = escape(str((p.get("result") or {}).get("summary") or ""))
        cls = ' class="hitrow"' if tone == "hit" else ' class="dim"' if tone == "miss" else ""
        rows += (f"<tr{cls}><td class='muted'>{i}</td>"
                 f"<td>{person(p.get('batter_id'), p.get('batter', ''), p.get('team'), abbr(p.get('team')), 'sm')}</td>"
                 f"<td class='n'>{proj}</td><td>{badge}</td><td class='muted'>{line}</td></tr>")
    unit = "HR prob" if key == "hr" else "Proj. hits"
    return (f"<div class='card scroll'><table class='t'><thead><tr><th>#</th><th>Hitter</th><th class='n'>{unit}</th>"
            f"<th>Result</th><th>Line</th></tr></thead><tbody>{rows}</tbody></table></div>")


def _homers_table(slate: dict) -> str:
    """Every home run hit by a hitter on the slate, with what the model gave him."""
    everyone = [h for g in slate.get("games", []) for h in g.get("hitters", [])]
    ranked = sorted(everyone, key=lambda h: h.get("prob_hr", 0), reverse=True)
    rank_of = {id(h): i for i, h in enumerate(ranked, 1)}
    pick_ids = {p.get("batter_id") for p in slate.get("picks_6") or []}
    hitters = [h for h in everyone if (h.get("result") or {}).get("hr")]
    hitters.sort(key=lambda h: h.get("prob_hr", 0), reverse=True)
    if not hitters:
        return '<div class="card muted">No home runs recorded for slate hitters.</div>'
    decile = max(len(ranked) // 10, 1)
    rows = ""
    for h in hitters:
        line = h["result"]
        star = ' <span class="pill strong">PICK</span>' if h.get("batter_id") in pick_ids else ""
        r = rank_of[id(h)]
        rows += (f"<tr{' class=hitrow' if h.get('batter_id') in pick_ids else ''}>"
                 f"<td><div style='display:flex;align-items:center;gap:6px'>{person(h.get('batter_id'), h['batter'], h.get('team'), abbr(h.get('team')), 'sm')}{star}</div></td>"
                 f"<td class='n'><b>{line.get('hr', 0)}</b></td><td class='n'>{_pct(h.get('prob_hr'))}</td>"
                 f"<td class='n'>#{r} <span class='muted'>of {len(ranked)}</span>{mini(1 - (r - 1) / max(len(ranked), 1), 'good' if r <= decile else '')}</td>"
                 f"<td class='muted'>{escape(str(line.get('summary', '')))}</td></tr>")
    top_decile = sum(1 for h in hitters if rank_of[id(h)] <= decile)
    note = (f"{len(hitters)} slate hitters homered. <b>{top_decile}</b> of them were in the model's top 10% of the "
            f"board ({decile} hitters) &mdash; a random ranking would put about {len(hitters) / 10:.1f} there.")
    return (f"<p class='note'>{note}</p><div class='card scroll'><table class='t'><thead><tr><th>Hitter</th>"
            f"<th class='n'>HR</th><th class='n'>Model prob</th><th class='n'>Board rank</th><th>Line</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>")


def _final_cards(slate: dict, homer_games: dict) -> tuple[str, int, int]:
    cards = ""
    right = called = 0
    for g in slate.get("games", []):
        live = g.get("live") or {}
        proj = g.get("projection") or {}
        hs, as_ = live.get("home_score"), live.get("away_score")
        final = live.get("state") == "Final" and hs is not None and as_ is not None
        if final:
            mid = f"<b>{as_}&ndash;{hs}</b>Final"
            a_cls, h_cls = ("", "lose") if as_ > hs else ("lose", "") if hs > as_ else ("", "")
        else:
            mid = f"<b>&mdash;</b>{escape(str(live.get('detailed') or 'Not final'))}"
            a_cls = h_cls = ""
        rows = ""
        tone = ""
        if proj:
            fav = proj.get("favorite")
            mark = ""
            if final and hs != as_:
                winner = g["home"] if hs > as_ else g["away"]
                ok = winner == fav
                called += 1
                right += ok
                mark = f'<span class="badge {"hit" if ok else "miss"}">{"&#10003;" if ok else "&#10007;"}</span>'
            total = ""
            if final:
                diff = (hs + as_) - proj.get("total_runs", 0)
                total = f" &middot; total {proj.get('total_runs', 0):.1f} vs {hs + as_} ({diff:+.1f})"
            rows += (f'<div class="rowx"><span>Model <span class="muted">proj {proj.get("away_expected_runs", 0):.1f}&ndash;'
                     f'{proj.get("home_expected_runs", 0):.1f}{total}</span></span>'
                     f'<span class="r">{logo(fav, "sm")} {abbr(fav)} {proj.get("favorite_prob", 0):.0%}{mark}</span></div>')
        hg = homer_games.get(g.get("game_pk"))
        if hg:
            res = hg.get("result") or {}
            mark = ""
            if res.get("final") and not res.get("no_decision"):
                mark = f'<span class="badge {"hit" if res.get("won") else "miss"}">{"&#10003;" if res.get("won") else "&#10007;"}</span>'
            rows += (f'<div class="rowx"><span>&#129506; HOMER</span><span class="r">{logo(hg["pick"], "sm")} '
                     f'{abbr(hg["pick"])} {hg["win_prob"]:.0%}{mark}</span></div>')
        homers = [h for h in g.get("hitters", []) if (h.get("result") or {}).get("hr")]
        hr_list = " ".join(
            f'<span class="chip">{avatar(h.get("batter_id"), h["batter"], h.get("team"), "xs")}<b>{escape(h["batter"])}</b>'
            f'{" &times;" + str(h["result"]["hr"]) if h["result"]["hr"] > 1 else ""}</span>' for h in homers)
        cards += (f'<article class="card gcard{tone}">{_ghead_final(g, mid, a_cls, h_cls)}'
                  f'<div class="sps"><div>{escape(str(g.get("away_sp") or "TBD"))} {hand(g.get("away_sp_hand"))}</div>'
                  f'<div>{hand(g.get("home_sp_hand"))} {escape(str(g.get("home_sp") or "TBD"))}</div></div>'
                  f'<div class="rows">{rows}</div><div><div class="label">Home runs</div>'
                  f'<div class="chips" style="margin:0">{hr_list or "<span class=muted>none from slate hitters</span>"}</div></div></article>')
    return cards, right, called


def _ghead_final(g: dict, mid: str, a_cls: str, h_cls: str) -> str:
    return (f'<div class="ghead"><div class="tm {a_cls}">{logo(g["away"])}<span>{escape(short(g["away"]))}</span></div>'
            f'<div class="c">{mid}</div><div class="tm h {h_cls}">{logo(g["home"])}<span>{escape(short(g["home"]))}</span></div></div>')


def render_results_html(day: str, slate: dict | None, days: list,
                        picks_only: dict | None = None, homer: dict | None = None) -> str:
    """A settled day: picks against box scores, finals against projections."""
    pills = day_pills(days, day, "/results")
    homer_games = {g["game_pk"]: g for g in (homer or {}).get("game_picks", [])}
    intro = "Every pick as it was published before first pitch, graded against the box scores."

    if slate is None and picks_only is None:
        body = (hero("Results", _long_date(day), "", pills)
                + alert(f"Nothing archived for {day}. Days are saved automatically as their games are played."))
        return page(f"Results — {day}", "results", body, "<div>HR Daily Tracker &middot; results review</div>")

    if slate is None:
        # Days archived before full slates were saved: the tracker rows only.
        summary = picks_only.get("summary") or {}
        rows = ""
        for r in picks_only.get("picks", []):
            won = r.get("won")
            mark = ('<span class="badge hit">&#10003;</span>' if won is True
                    else '<span class="badge miss">&#10007;</span>' if won is False else '<span class="muted">&mdash;</span>')
            proj = _pct(r.get("projected_prob")) if r.get("projected_prob") is not None else r.get("projected")
            rows += (f"<tr{' class=hitrow' if won is True else ''}><td><span class='pill slot'>{r['kind'].upper()}</span></td>"
                     f"<td class='n'>{r['rank']}</td><td>{person(None, r['batter'], r['team'], abbr(r['team']), 'xs')}</td>"
                     f"<td class='n'>{proj}</td><td class='muted'>{escape(str(r.get('result_line') or ''))}</td><td>{mark}</td></tr>")
        hr_n, hr_h = summary.get("hr_picks_final", 0), summary.get("hr_picks_hit", 0)
        hit_n, hit_h = summary.get("hit_picks_final", 0), summary.get("hit_picks_hit", 0)
        body = (hero("Results", _long_date(day), intro, pills)
                + tiles([tile(f"{hr_h}/{hr_n}", "HR picks hit", hr_h / hr_n if hr_n else None, "good"),
                         tile(f"{hit_h}/{hit_n}", "hit picks hit", hit_h / hit_n if hit_n else None)])
                + alert("This day was archived before full slates were saved, so only the tracked picks are available.", info=True)
                + f'<section class="section" id="picks">{head("Picks", "&#127919;")}'
                  f"<div class='card scroll'><table class='t'><thead><tr><th>Kind</th><th class='n'>#</th><th>Hitter</th>"
                  f"<th class='n'>Projected</th><th>Line</th><th>Won</th></tr></thead><tbody>{rows}</tbody></table></div></section>")
        return page(f"Results — {day}", "results", body, f"<div>Results for {day} &middot; from the tracker CSV</div>")

    res = slate.get("results") or {}
    picks = res.get("picks") or {}
    hits = res.get("hit_picks") or {}
    expected = sum(
        p.get("prob_hr", 0) for p in slate.get("picks_6") or []
        if (p.get("result") or {}).get("final") and not (p.get("result") or {}).get("dnp")
    )
    finals, right, called = _final_cards(slate, homer_games)
    cells = [tile(f"{picks.get('hit', 0)}/{picks.get('scored', 0)}", "HR picks homered",
                  picks.get("hit", 0) / picks["scored"] if picks.get("scored") else None, "good"),
             tile(f"{expected:.2f}", "home runs expected from them", tone="accent"),
             tile(f"{hits.get('hit', 0)}/{hits.get('scored', 0)}", "hit picks with a hit",
                  hits.get("hit", 0) / hits["scored"] if hits.get("scored") else None),
             tile(f"{right}/{called}", "winners called", right / called if called else None)]
    nav = [("picks", "HR Picks"), ("hits", "Hit Picks"), ("homers", "Every HR"), ("finals", "Final Scores")]
    homer_block = ""
    if homer and homer.get("record"):
        rec = homer["record"]
        cells.append(tile(f"{rec['hr']['hit']}/{rec['hr']['scored']}",
                          f"HOMER HR picks &middot; {rec['games']['won']}-{rec['games']['lost']} on winners", tone="gold"))
        rows = ""
        for p in homer.get("hr_picks", []):
            r = p.get("result") or {}
            if r.get("hr"):
                mark = '<span class="badge hit">&#10003; HR</span>'
            elif r.get("final") and not r.get("dnp"):
                mark = '<span class="badge miss">&#10007;</span>'
            else:
                mark = '<span class="muted">&mdash;</span>'
            rows += (f"<tr{' class=hitrow' if r.get('hr') else ''}><td>{person(p.get('batter_id'), p['batter'], p['team'], abbr(p['team']), 'sm')}</td>"
                     f"<td class='n'><b>{p['prob']:.1%}</b></td><td class='n'>{_pct(p.get('model_prob'))}</td>"
                     f"<td>{mark}</td><td class='muted'>{escape(str(r.get('summary', '')))}</td></tr>")
        extra = (f", {rec['hits']['hit']}-for-{rec['hits']['scored']} on hit picks" if rec.get("hits") else "") + \
                (f" and {rec['parlays']['won']}-{rec['parlays']['lost']} on his parlays" if rec.get("parlays") else "")
        note = (f"HOMER went {rec['hr']['hit']}-for-{rec['hr']['scored']} on home runs, "
                f"{rec['games']['won']}-{rec['games']['lost']} picking winners{extra}. "
                f"<a href=\"/homer?date={day}\">Full card &rarr;</a>")
        table = (f"<div class='card scroll'><table class='t'><thead><tr><th>Hitter</th><th class='n'>HOMER</th>"
                 f"<th class='n'>Model</th><th>Result</th><th>Line</th></tr></thead><tbody>{rows}</tbody></table></div>"
                 if rows else "")
        homer_block = f'<section class="section" id="homer">{head("HOMER&rsquo;s Card", "&#129506;", note)}{table}</section>'
        nav.append(("homer", "HOMER"))

    state = ("all games final" if res.get("final_games") and not res.get("live_games")
             and not res.get("upcoming_games") else
             f"{res.get('final_games', 0)} final, {res.get('live_games', 0)} live, "
             f"{res.get('upcoming_games', 0)} upcoming")
    pick_note = (f"{picks.get('hit', 0)} of {picks.get('scored', 0)} settled picks homered against {expected:.2f} "
                 f"expected from their published probabilities."
                 + (f" {picks.get('dnp')} did not play." if picks.get("dnp") else ""))
    body = (hero("Results", _long_date(day), intro, pills) + subnav(nav) + tiles(cells)
            + f'<section class="section" id="picks">{head("HR Picks &mdash; how they did", "&#128163;", pick_note)}'
              f'{_picks_table(slate.get("picks_6") or [], "hr")}</section>'
            + f'<section class="section" id="hits">{head("Hit Picks", "&#129358;")}{_picks_table(slate.get("hit_picks") or [], "hits")}</section>'
            + f'<section class="section" id="homers">{head("Every Home Run on the Slate", "&#127878;")}{_homers_table(slate)}</section>'
            + homer_block
            + f'<section class="section" id="finals">{head("Final Scores vs Projections", "&#9918;", f"The model called {right} of {called} winners; the badge marks whether each projected favorite won.")}'
              f'<div class="grid wide">{finals}</div></section>')
    footer = (f"<div>Results for {day} &middot; {state} &middot; model built {slate.get('built_at', '')}"
              f" &middot; scores as of {res.get('fetched_at', '')}</div>")
    return page(f"Results — {day}", "results", body, footer)


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
    return f"""<div class="card persona"><div class="face">&#129506;</div><div style="min-width:0">
<div class="q"><strong>HOMER</strong> &mdash; top-tier MLB expert and professional capper.
I look at <em>everything</em>: the arm on the mound, the bat in the box, the wind and the heat, and every player's
full body of work &mdash; what he's capable of and <em>when</em> he's capable of doing it. I don't guess and I don't
chase. I play to win, because every win is another follower cashing with me. I get sharper with every data pull: my
whole season gets re-studied every day.{proof}</div>
<div class="rcpt">{receipts}</div><div style="margin-top:10px">{_method()}</div></div></div>"""


def _method() -> str:
    items = """
<li><strong>Expected home runs.</strong> Every batted ball this season is binned by exit velocity and launch angle;
the league's HR rate per bin turns a hitter's contact into the home runs it deserved.</li>
<li><strong>Form &amp; splits.</strong> Last-21-day xHR, the hitter's own record against today's pitcher hand, barrel
rate, and the league platoon effect.</li>
<li><strong>Pitchers.</strong> HR and xHR allowed per batter faced, fly-ball rate and recent form for the starter,
and the bullpen he hands off to.</li>
<li><strong>Park</strong> factor from HR/PA at the venue vs the home team's road games; <strong>weather</strong> from
temperature and wind toward the hitter's pull field.</li>
<li><strong>He learns what wins.</strong> The season is replayed week by week with his book frozen at each week's
start. Weights are fitted only on earlier weeks and graded on the next one. Factors that don't improve those
unseen-week predictions are dropped, and the learned weights are used only if they beat his original rule-of-thumb
formula.</li>
<li><strong>Grades are earned.</strong> A / B+ / B / C bands come from where a pick sits on the replayed board, and
each shows how often picks in that band actually homered.</li>
<li><strong>Games.</strong> A wOBA runs model (lineup vs the starter and bullpen it faces) and season run
differential, weighted the same learned way.</li>
<li><strong>Hit picks.</strong> Hit rate, expected hits from contact quality, strikeout and walk rates, and his own
split, against the starter's hits allowed and strikeout rate, the bullpen and the park &mdash; learned and tested the
same way.</li>
<li><strong>Parlays.</strong> The house rules on HOMER's own numbers: five-leg core tickets, one leg per game, posted
only if the rule cashed 30+ times when rebuilt on every replayed day, plus the Early and Late Ten long shots. Every leg
is a line DraftKings posts, with its price.</li>
<li><strong>Grows every day.</strong> His book refreshes with each data pull, and the whole season replay reruns
daily, so new results reshape his weights and grades.</li>
<li><strong>Selection rules.</strong> Six HR picks and six hit picks, one per game, 80+ PA, no injury notes, and only
hitters in the posted lineup. Picks and plays in started games are locked.</li>"""
    return (f'<details class="why" id="homer-method"><summary>How HOMER does his homework</summary>'
            f'<ul>{items}</ul></details>')


def _homer_badge(r: dict, key: str) -> tuple:
    """(card tone, badge html) for one of HOMER's picks."""
    if not r:
        return "", ""
    if r.get("dnp"):
        return "", '<span class="badge void">&#8212; NO RESULT &middot; did not play</span>'
    got = r.get(key)
    summary = escape(str(r.get("summary", "")))
    if got:
        word = "HIT" if key == "hr" else "CASHED"
        unit = "HR" if key == "hr" else "H"
        return " hit", f'<span class="badge hit">&#10003; {word} &middot; {got} {unit} <small>&middot; {summary}</small></span>'
    if r.get("final"):
        return " miss", f'<span class="badge miss">&#10007; MISS <small>&middot; {summary}</small></span>'
    return "", f'<span class="badge live">&#9679; LIVE &middot; {escape(str(r.get("state", "")))} <small>&middot; {summary}</small></span>'


def _homer_hr_cards(card: dict) -> str:
    cards = ""
    for i, p in enumerate(card.get("hr_picks", []), 1):
        tone, badge = _homer_badge(p.get("result") or {}, "hr")
        take = "".join(f"<li>{t}</li>" for t in p.get("take", []))
        model = p.get("model_prob")
        edge = ""
        if model:
            diff = p["prob"] - model
            edge = (f'<span class="chip{" up" if diff >= 0 else " down"}">vs model <b>{diff * 100:+.1f} pts</b></span>'
                    f'<span class="chip">model <b>{model:.1%}</b></span>')
        slot = f"<span class='chip'>bats <b>#{p['slot']}</b></span>" if p.get("slot") else ""
        ev = p.get("evidence")
        evidence = (f'<div class="evid"><b>Track record:</b> grade-{escape(str(ev["grade"]))} picks homered '
                    f'<b>{ev["rate"]:.1%}</b> of the time in the season replay ({ev["hit"]} of {ev["n"]:,})</div>'
                    if ev else "")
        work = "".join(
            f'<div class="mult"><b>{v}</b>{k}</div>' for k, v in (
                ("xHR / PA season", f"{p['xhr_rate']:.2%}"), ("xHR / PA last 21d", f"{p['recent_rate']:.2%}"),
                ("Barrel rate", f"{p['barrel_rate']:.1%}"), ("Starter HR idx", f"{p['sp_index']:.2f}&times;"),
                ("Bullpen HR idx", f"{p['pen_index']:.2f}&times;"), ("Platoon", f"{p['platoon']:.2f}&times;"),
                ("Park &middot; weather", f"{p['park']:.2f} &middot; {p['weather']:.2f}"),
                ("Trips vs SP / pen", f"{p['pa_vs_sp']:.1f} / {p['pa_vs_pen']:.1f}")))
        cards += f"""<article class="card pick{tone}" style="--tc:{color(p.get('team'))}">
<div class="pick-top">{avatar(p.get('batter_id'), p['batter'], p.get('team'))}<div class="who">
<div class="name">{escape(p['batter'])}{_grade(p.get('grade'))}{_tag(p.get('tag'))}</div>
<div class="sub">{logo(p.get('team'), 'sm')}{abbr(p.get('team'))} &middot; vs {escape(str(p['sp_name']))} {hand(p.get('sp_hand'))}</div></div>
<div class="rank">#{i}</div></div>
<div class="pick-main">{ring(p['prob'], 'var(--gold)')}<div><div class="big">{p['prob']:.1%}</div><div class="unit">HOMER's HR probability</div>
<div class="chips"><span class="chip">fair <b>{fair_odds(p['prob'])}</b></span>{slot}{edge}</div></div></div>
{evidence}{f'<ul class="take">{take}</ul>' if take else ''}
<details class="why" id="hw-{p.get('batter_id') or i}"><summary>HOMER's worksheet</summary><div class="mults">{work}</div></details>
{badge}</article>"""
    return cards


def _homer_hit_cards(card: dict) -> str:
    cards = ""
    for i, p in enumerate(card.get("hit_picks", []), 1):
        tone, badge = _homer_badge(p.get("result") or {}, "hits")
        ev = p.get("evidence")
        evidence = (f'<div class="evid"><b>Track record:</b> grade-{escape(str(ev["grade"]))} hit picks cashed '
                    f'<b>{ev["rate"]:.1%}</b> in the replay ({ev["hit"]:,} of {ev["n"]:,})</div>' if ev else "")
        model = p.get("model_prob")
        model_chip = f"<span class='chip'>model <b>{model:.0%}</b></span>" if model else ""
        take = "".join(f"<li>{t}</li>" for t in p.get("take", []))
        cards += f"""<article class="card pick{tone}" style="--tc:{color(p.get('team'))}">
<div class="pick-top">{avatar(p.get('batter_id'), p['batter'], p.get('team'))}<div class="who">
<div class="name">{escape(p['batter'])}{_grade(p.get('grade'))}{_tag(p.get('tag'))}</div>
<div class="sub">{logo(p.get('team'), 'sm')}{abbr(p.get('team'))} &middot; vs {escape(str(p['sp_name']))} {hand(p.get('sp_hand'))}</div></div>
<div class="rank">#{i}</div></div>
<div class="pick-main">{ring(p['prob'], 'var(--s1)')}<div><div class="big">{p['prob']:.0%}</div><div class="unit">chance of 1+ hit</div>
<div class="chips"><span class="chip">fair <b>{fair_odds(p['prob'])}</b></span>{model_chip}</div></div></div>
{evidence}{f'<ul class="take">{take}</ul>' if take else ''}{badge}</article>"""
    return cards


def _chance(prob: float) -> str:
    """A ticket's chance: a percentage, or '1 in N' once it is too small to read."""
    return chance(prob)


def _parlays(card: dict) -> str:
    """HOMER's parlays: the house rules (mlb_hr.parlays) on his own numbers,
    with his replay record for every rule."""
    from mlb_hr.render import _render_parlays_section
    replay = {k: {f: v for f, v in r.items() if f != "wins_on"}
              for k, r in ((card.get("evidence") or {}).get("parlays") or {}).items()
              if isinstance(r, dict) and "rate" in r and "kind" not in r}
    lead = ("HOMER&rsquo;s own reads &mdash; his home run and hit chances and his winners &mdash; run through the "
            "house parlay rules, with his own replay record. ")
    return _render_parlays_section({"parlays": card.get("parlays") or [], "parlay_replay": replay},
                                   title="HOMER&rsquo;s Parlays", sid="plays", lead=lead, icon="&#128176;")


def _homer_game_cards(card: dict) -> str:
    cards = ""
    for g in card.get("game_picks", []):
        r = g.get("result") or {}
        away, home = g.get("away"), g.get("home")
        pick_home = g.get("pick_side") == "home" or (g.get("pick_side") is None and g.get("pick") == home)
        ph = g["win_prob"] if pick_home else 1 - g["win_prob"]
        tone = ""
        if r.get("no_decision"):
            mid = f"<b>&mdash;</b>{escape(str(r.get('state') or 'No decision'))}"
            badge = ""
        elif r.get("final"):
            mid = f"<b>{r['away_score']}&ndash;{r['home_score']}</b>Final"
            badge = f'<span class="badge {"hit" if r["won"] else "miss"}">{"&#10003; WON" if r["won"] else "&#10007; LOST"}</span>'
            tone = " hit" if r["won"] else " miss"
        elif r:
            mid = f'<b class="live">{r.get("away_score")}&ndash;{r.get("home_score")}</b>&#9679; {escape(str(r.get("state", "Live")))}'
            badge = ""
        else:
            mid = f"<b>@</b>proj {g['away_runs']:.1f}&ndash;{g['home_runs']:.1f}"
            badge = ""
        model = g.get("model_pick_prob")
        notes = ""
        if model is not None:
            agree = "agrees" if model >= 0.5 else "disagrees"
            notes += f"<span class='chip{' up' if model >= 0.5 else ' down'}'>house model {agree} <b>{model:.0%}</b></span>"
        notes += f"<span class='chip'>total <b>{g['total']:.1f}</b></span>"
        ev = g.get("evidence")
        evid = ""
        if ev and ev.get("rate") is not None:
            evid = (f'<div class="evid"><b>Track record:</b> {escape(str(ev["label"]))} calls won <b>{ev["rate"]:.1%}</b> '
                    f'in the replay ({ev["won"]} of {ev["n"]:,})</div>')
        take = "".join(f"<li>{t}</li>" for t in g.get("take", []))
        conf = g.get("tag") or g["confidence"]
        cards += f"""<article class="card gcard{tone}">
<div class="ghead"><div class="tm">{logo(away)}<span>{escape(short(away))}</span></div><div class="c">{mid}</div>
<div class="tm h">{logo(home)}<span>{escape(short(home))}</span></div></div>
{winbar(away, home, 1 - ph, ph)}
<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap"><span class="person">{logo(g['pick'])}<span class="nm">HOMER takes {escape(short(g['pick']))}</span></span>
<span class="pill{' strong' if g['confidence'] == 'Strong' else ' lean'}">{escape(str(conf))}</span>{badge}</div>
<div class="chips" style="margin:0">{notes}</div>{evid}{f'<ul class="take" style="margin:0">{take}</ul>' if take else ''}</article>"""
    return cards


def _disagreements(card: dict) -> str:
    def table(rows: list, title: str) -> str:
        body = "".join(
            f"<tr><td>{person(c.get('batter_id'), c['batter'], c.get('team'), abbr(c.get('team')) + ' &middot; vs ' + escape(str(c['sp_name'])), 'xs')}</td>"
            f"<td class='n'><b>{c['prob']:.1%}</b></td><td class='n'>{c['model_prob']:.1%}</td>"
            f"<td class='n {'ok' if c['edge'] > 0 else 'bad'}'>{c['edge'] * 100:+.1f}</td>"
            f"<td class='n'>{c['barrel_rate']:.1%}</td></tr>"
            for c in rows)
        return (f"<div class='card scroll'><div class='label'>{title}</div><table class='t'><thead><tr><th>Hitter</th>"
                f"<th class='n'>HOMER</th><th class='n'>Model</th><th class='n'>Edge (pts)</th><th class='n'>Barrel%</th>"
                f"</tr></thead><tbody>{body}</tbody></table></div>")

    if not card.get("likes"):
        return "<div class='card muted'>No disagreements on this card.</div>"
    return (f'<div class="grid lab">{table(card["likes"], "HOMER likes more than the model")}'
            f'{table(card["fades"], "HOMER is fading")}</div>')


def _grade_note(card: dict) -> str:
    bands = (card.get("evidence") or {}).get("grades") or []
    if not bands:
        return "Grades: A &ge; 20%, B+ &ge; 17%, B &ge; 14% (rule of thumb &mdash; no replay yet)."
    parts = ", ".join(f"{b['grade']} homered {b['rate']:.1%}" for b in bands)
    return (f"Grades are earned in the season replay: {parts}. "
            "Bands that didn't outperform the band below were merged.")


def _tbl(head_cells: str, rows: str) -> str:
    return f"<div class='scroll'><table class='t'><thead><tr>{head_cells}</tr></thead><tbody>{rows}</tbody></table></div>"


def _homework(card: dict) -> str:
    """The season replay behind the card: what HOMER learned and how it tested."""
    ev = card.get("evidence")
    if not ev:
        return (f'<section class="section" id="homework">{head("HOMER&rsquo;s Homework", "&#128218;")}'
                + alert("No season replay yet, so this card uses HOMER's rule-of-thumb formula. Run "
                        "<code>python -m mlb_hr.homer_fit</code> (the app also does it daily).", info=True)
                + "</section>")

    def score_row(label, s, served=False):
        if not s:
            return ""
        return (f'<tr{" class=served" if served else ""}><td>{label}</td>'
                f'<td class="n">{s.get("auc", 0):.3f}</td><td class="n">{s["log_loss"]:.4f}</td><td class="n">{s["n"]:,}</td></tr>')

    hs = ev.get("hr_scores") or {}
    serve = ev.get("hr_serve")
    hr_table = (score_row("HOMER, learned weights", hs.get("learned"), serve == "learned")
                + score_row("HOMER, rule-of-thumb formula", hs.get("rules"), serve == "rules")
                + score_row("Season HR rate only (baseline)", hs.get("baseline")))
    picks = ev.get("hr_picks") or {}

    def pick_row(label, p):
        if not p:
            return ""
        return (f'<tr><td>{label}</td><td class="n">{p["hit"]}/{p["picks"]:,}</td>'
                f'<td class="n"><b>{p["rate"]:.1%}</b> <span class="muted">&plusmn;{p["rate_se"] * 1.96:.1%}</span></td>'
                f'<td class="n">{p["expected"]:.0f}</td></tr>')

    pick_table = (pick_row("HOMER, learned", picks.get("calibrated") or picks.get("learned"))
                  + pick_row("HOMER, rule-of-thumb", picks.get("rules"))
                  + pick_row("Season HR rate only", picks.get("base")))
    if ev.get("board_rate") is not None:
        pick_table += (f'<tr class="dim"><td>Any hitter in the lineup</td><td class="n">&mdash;</td>'
                       f'<td class="n"><b>{ev["board_rate"]:.1%}</b></td><td class="n">&mdash;</td></tr>')

    def is_overlap(w):
        # Older fits did not flag it; any negative HR weight is an overlap correction.
        return w.get("overlap", w["odds_per_sd"] < 0)

    def weight_rows(ws, overlap_fn, note):
        top = max((abs(w["odds_per_sd"]) for w in ws), default=0) or 1
        return "".join(
            f'<tr><td class="wrap">{w["label"]}{"<div class=sub>" + note + "</div>" if overlap_fn(w) else ""}</td>'
            f'<td class="n {"ok" if w["odds_per_sd"] > 0 else "bad"}">{w["odds_per_sd"] * 100:+.1f}%'
            f'{mini(abs(w["odds_per_sd"]) / top, "good" if w["odds_per_sd"] > 0 else "bad")}</td></tr>'
            for w in ws)

    weights = weight_rows(ev.get("hr_weights") or [], is_overlap, "overlap correction: already counted in expected HR")
    dropped = ", ".join(f'{d["dropped"]}' for d in ev.get("hr_dropped") or []) or "none"
    grades_list = ev.get("grades") or []
    grades = "".join(
        f'<tr><td>{_grade(g["grade"])}</td>'
        f'<td class="n">{"top " + format(g["top_share"], ".0%") if g["grade"] != "C" else "rest"}</td>'
        f'<td class="n">{g["avg_prob"]:.1%}</td><td class="n"><b>{g["rate"]:.1%}</b></td><td class="n">{g["n"]:,}</td></tr>'
        for g in grades_list)
    grade_chart = calib_chart([(g["grade"], g["avg_prob"], g["rate"]) for g in grades_list], ("Avg prob", "Homered"))

    gs = ev.get("game_scores") or {}
    gserve = ev.get("games_serve")
    house = ev.get("house_winners")

    def acc_row(label, s, served=False):
        if not s:
            return ""
        return (f'<tr{" class=served" if served else ""}><td>{label}</td>'
                f'<td class="n"><b>{s["accuracy"]:.1%}</b> <span class="muted">&plusmn;{s["accuracy_se"] * 1.96:.1%}</span></td>'
                f'<td class="n">{s["log_loss"]:.4f}</td><td class="n">{s["n"]:,}</td></tr>')

    game_table = (acc_row("HOMER, learned weights", gs.get("learned"), gserve == "learned")
                  + acc_row("HOMER, rule-of-thumb formula", gs.get("rules"), gserve == "rules")
                  + acc_row("House model (its own walk-forward)", house)
                  + acc_row("Always take the home team", gs.get("home_team")))
    conf = "".join(
        f'<tr><td>{c["label"]}</td><td class="n">{c["n"]:,}</td><td class="n"><b>{c["rate"]:.1%}</b>{mini(c["rate"])}</td></tr>'
        for c in ev.get("confidence") or [] if c.get("rate") is not None)

    cards = []
    legs = (ev.get("book") or {}).get("legs") or {}
    names = {"hit_prob": "Top 5 hits", "hit_value": "Value hits", "hr_prob": "Top 5 HR", "hr_value": "Value HR"}
    leg_rows = "".join(
        f'<tr><td>{names[k]}</td><td class="n"><b>{v["actual"]:.1%}</b></td>'
        f'<td class="n">{v["market"]:.1%}</td><td class="n">{v["break_even"]:.1%}</td><td class="n">{v["roi"]:+.1%}</td></tr>'
        for k, v in legs.items() if k in names)
    if leg_rows:
        cards.append(
            '<div class="card" style="grid-column:1/-1"><div class="label">Leg by leg vs a season-rate book</div>'
            + _tbl('<th>Legs</th><th class="n">Cashed</th><th class="n">Season line</th><th class="n">Book needs</th>'
                   '<th class="n">Straight ROI</th>', leg_rows)
            + '<p class="note" style="margin:10px 0 0">HOMER&rsquo;s daily five legs of each kind bet straight at a '
              "modeled season-rate book, every replayed day. A diagnostic only: the parlays price their legs at "
              "DraftKings&rsquo; own lines, and their replay record is in the Parlays section.</p></div>")
    cards.append(f'<div class="card"><div class="label">Daily 6 HR picks, replayed</div>'
                 + _tbl('<th>Method</th><th class="n">Hit</th><th class="n">Rate</th><th class="n">Expected</th>', pick_table)
                 + '<p class="note" style="margin:10px 0 0">Same selection rules as today\'s card, every replayed day.</p></div>')
    cards.append(f'<div class="card"><div class="label">Ranking every hitter-game</div>'
                 + _tbl('<th>Method</th><th class="n">AUC</th><th class="n">Log loss</th><th class="n">Games</th>', hr_table)
                 + '<p class="note" style="margin:10px 0 0">AUC: how often a hitter who homered was ranked above one who '
                   "didn't (0.5 = coin flip). Lower log loss = better-calibrated probabilities.</p></div>")
    cards.append(f'<div class="card"><div class="label">What each grade has earned</div>{grade_chart}'
                 + _tbl('<th>Grade</th><th class="n">Board</th><th class="n">Avg prob</th><th class="n">Homered</th><th class="n">n</th>', grades)
                 + "</div>")
    cards.append(f'<div class="card"><div class="label">What moves HOMER&rsquo;s needle</div>'
                 + _tbl('<th>Factor</th><th class="n">Odds per 1 SD</th>', weights)
                 + f'<p class="note" style="margin:10px 0 0">Tested and dropped (didn\'t help on unseen weeks): {dropped}</p></div>')
    hp = ev.get("hit_picks") or {}
    if hp:
        hit_rows = (pick_row("HOMER", hp.get("calibrated"))
                    + pick_row("HOMER, rule-of-thumb", hp.get("rules"))
                    + pick_row("Season hit rate only", hp.get("base")))
        if ev.get("hit_board_rate") is not None:
            hit_rows += (f'<tr class="dim"><td>Any hitter in the lineup</td><td class="n">&mdash;</td>'
                         f'<td class="n"><b>{ev["hit_board_rate"]:.1%}</b></td><td class="n">&mdash;</td></tr>')
        hit_grades = ev.get("hit_grades") or []
        hg_rows = "".join(
            f'<tr><td>{_grade(g["grade"])}</td><td class="n">{g["avg_prob"]:.1%}</td>'
            f'<td class="n"><b>{g["rate"]:.1%}</b></td><td class="n">{g["n"]:,}</td></tr>' for g in hit_grades)
        hit_weights = weight_rows(ev.get("hit_weights") or [],
                                  lambda w: EXPECTED_SIGN.get(w["feature"], 1) * w["odds_per_sd"] < 0,
                                  "overlap correction")
        hit_dropped = ", ".join(d["dropped"] for d in ev.get("hit_dropped") or []) or "none"
        cards.append(f'<div class="card"><div class="label">Daily 6 hit picks, replayed '
                     f'({"learned" if ev.get("hits_serve") == "learned" else "rule-of-thumb"})</div>'
                     + _tbl('<th>Method</th><th class="n">Cashed</th><th class="n">Rate</th><th class="n">Expected</th>', hit_rows)
                     + '<div style="margin-top:12px"></div>'
                     + calib_chart([(g["grade"], g["avg_prob"], g["rate"]) for g in hit_grades], ("Avg prob", "Got a hit"))
                     + _tbl('<th>Grade</th><th class="n">Avg prob</th><th class="n">Got a hit</th><th class="n">n</th>', hg_rows)
                     + "</div>")
        cards.append(f'<div class="card"><div class="label">What moves HOMER&rsquo;s hit needle</div>'
                     + _tbl('<th>Factor</th><th class="n">Odds per 1 SD</th>', hit_weights)
                     + f'<p class="note" style="margin:10px 0 0">Tested and dropped: {hit_dropped}</p></div>')
    cards.append(f'<div class="card"><div class="label">Picking winners, replayed</div>'
                 + _tbl('<th>Method</th><th class="n">Correct</th><th class="n">Log loss</th><th class="n">Games</th>', game_table)
                 + '<div style="margin-top:12px"></div>'
                 + _tbl('<th>Confidence</th><th class="n">Games</th><th class="n">Won</th>', conf) + "</div>")
    note = (f"Season replay through {ev.get('through')} (refit {ev.get('fitted_at')}). Every number below is from weeks "
            f"HOMER had not seen when he made the call. He is currently using "
            f"<b>{'learned weights' if serve == 'learned' else 'his rule-of-thumb formula'}</b> for home runs and "
            f"<b>{'learned weights' if gserve == 'learned' else 'his rule-of-thumb formula'}</b> for winners &mdash; "
            "whichever tested better. Highlighted rows are the ones in use.")
    return (f'<section class="section" id="homework">{head("HOMER&rsquo;s Homework", "&#128218;", note)}'
            f'<div class="grid lab">{"".join(cards)}</div></section>')


def _record_tiles(record: dict) -> str:
    hr_n = record["hr_scored"]
    cells = [tile(f"{record['hr_hit']}/{hr_n}" if hr_n else "0/0", "season HR picks",
                  record["hr_hit"] / hr_n if hr_n else None, "gold"),
             tile(f"{record.get('hits_hit', 0)}/{record.get('hits_scored', 0)}", "hit picks",
                  record.get("hits_hit", 0) / record["hits_scored"] if record.get("hits_scored") else None),
             tile(f"{record.get('parlays_won', 0)}-{record.get('parlays_lost', 0)}", "parlays W-L")]
    if record.get("five_won") or record.get("five_lost"):
        cells.append(tile(f"{record['five_won']}-{record['five_lost']}", "old 5-pick parlays W-L"))
    gw, gl = record["games_won"], record["games_lost"]
    cells.append(tile(f"{gw}-{gl}", "winners W-L", gw / (gw + gl) if gw + gl else None))
    return tiles(cells)


def render_homer_html(card: dict | None, record: dict, days: list, day: str,
                      status: str = "") -> str:
    """HOMER's tab: today's card (or a past one), his reasoning and his record."""
    pills = day_pills(days, day, "/homer")
    subtitle = "Top-tier MLB expert &middot; professional capper &middot; plays built to win"

    if card is None:
        body = (hero(f"HOMER &middot; {day}", "HOMER&rsquo;s Card", subtitle, pills)
                + _homer_intro(record) + _record_tiles(record)
                + alert('<span class="spinner"></span>'
                        + (escape(status) if status else f"HOMER is studying film for {day}. Check back in a minute."),
                        info=True))
        return page(f"HOMER — {day}", "homer", body,
                    "<div>HOMER builds his card after the day's slate is ready.</div>",
                    status=status_chip(text="Studying film", busy=True), meta_refresh=45)

    rec = card.get("record") or {}
    history = ""
    if record["days"]:
        rows = "".join(
            f"<tr><td><a href=\"/homer?date={d['date']}\">{d['date']}</a></td>"
            f"<td class='n'><b>{d['hr_hit']}</b>/{d['hr_scored']}</td><td class='n'>{d['hr_expected']:.2f}</td>"
            f"<td class='n'>{d.get('hits_hit', 0)}/{d.get('hits_scored', 0)}</td>"
            f"<td class='n'>{d.get('parlays_won', 0)}-{d.get('parlays_lost', 0)}</td>"
            f"<td class='n'>{d['games_won']}-{d['games_lost']}</td>"
            f"<td class='muted'>{'final' if d['complete'] else 'in progress'}"
            f"{' <span class=inj>backfilled</span>' if d.get('backfilled') else ''}</td></tr>"
            for d in record["days"])
        history = (f'<section class="section" id="record">{head("HOMER&rsquo;s Record", "&#128210;")}'
                   f"<div class='card scroll'><table class='t'><thead><tr><th>Date</th><th class='n'>HR hit</th>"
                   f"<th class='n'>Expected</th><th class='n'>Hit picks</th><th class='n'>Parlays</th><th class='n'>Winners</th>"
                   f"<th></th></tr></thead><tbody>{rows}</tbody></table></div></section>")

    lg = card.get("league") or {}
    alerts = ""
    if card.get("backfilled"):
        alerts += alert("Backfilled card: built after the fact from the published slate, using only games played "
                        "before this date. HOMER did not post it before first pitch.")
    if status:
        alerts += alert(escape(status), info=True)
    hit_section = ""
    if card.get("hit_picks"):
        bands = (card.get("evidence") or {}).get("hit_grades") or []
        gnote = ", ".join(f"{b['grade']} cashed {b['rate']:.0%}" for b in bands)
        hit_section = (f'<section class="section" id="hits">{head("HOMER&rsquo;s 6 Hit Picks", "&#129358;", "At least one hit, one per game, ranked by HOMER&rsquo;s own probability." + (f" Grades earned in the replay: {gnote}." if gnote else ""))}'
                       f'<div class="grid">{_homer_hit_cards(card)}</div></section>')
    nav = (([("plays", "Parlays")] if card.get("parlays") or (card.get("evidence") or {}).get("parlays") else [])
           + [("hr", "HR Picks")])
    if hit_section:
        nav.append(("hits", "Hit Picks"))
    nav += [("games", "Winners"), ("vs", "vs the Model"), ("homework", "Homework")]
    if history:
        nav.append(("record", "Record"))

    live = sum(1 for g in card.get("game_picks", []) if (g.get("result") or {}) and not (g.get("result") or {}).get("final")
               and not (g.get("result") or {}).get("no_decision"))
    body = (hero(f"HOMER &middot; {_long_date(card.get('date'))}", "HOMER&rsquo;s Card", subtitle, pills)
            + alerts + _homer_intro(record, card.get("evidence")) + subnav(nav) + _record_tiles(record)
            + _parlays(card)
            + f'<section class="section" id="hr">{head("HOMER&rsquo;s 6 Home Run Picks", "&#128163;", "One per game, ranked by HOMER&rsquo;s own per-game probability. " + _grade_note(card))}'
              f'<div class="grid">{_homer_hr_cards(card)}</div></section>'
            + hit_section
            + f'<section class="section" id="games">{head("HOMER&rsquo;s Winners", "&#9918;", "Every game, strongest opinion first.")}'
              f'<div class="grid wide">{_homer_game_cards(card)}</div></section>'
            + f'<section class="section" id="vs">{head("HOMER vs the Model", "&#129354;", "Same hitters, same day &mdash; where HOMER&rsquo;s numbers and the house ensemble part ways the most.")}'
              f'{_disagreements(card)}</section>'
            + _homework(card) + history)
    footer = (f"<div>HOMER's card for {card.get('date')} &middot; built {card.get('built_at')} "
              f"&middot; book through {card.get('book_through')} "
              f"({lg.get('pa', 0):,} PA, league {lg.get('hr_rate', 0):.2%} HR/PA)"
              + (f" &middot; graded {rec.get('graded_at')}" if rec.get("graded_at") else "") + "</div>")
    return page(f"HOMER — {card.get('date')}", "homer", body, footer,
                status=status_chip(live) if live else status_chip(text="Card posted"), live=60)
