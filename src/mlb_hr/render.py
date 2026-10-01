"""Render slate data to beautiful, responsive HTML."""
from __future__ import annotations

import json
from datetime import date


def _hand_badge(hand: str | None) -> str:
    """RHP/LHP badge for a pitcher. Handedness drives the platoon split, so it
    belongs next to the name rather than buried in the model."""
    if hand not in ("L", "R"):
        return ""
    label = "LHP" if hand == "L" else "RHP"
    return (
        f'<span class="hand-badge hand-{hand.lower()}">{label}</span>'
    )


def _platoon_note(hitter: dict) -> str:
    """This hitter's home runs against the hand he faces today."""
    pa = hitter.get("pa_vs_facing")
    if not pa:
        return ""
    hand = "LHP" if hitter.get("facing_hand") == "L" else "RHP"
    return (
        f'<span class="platoon-note">{hitter.get("hr_vs_facing", 0)} HR '
        f'/ {pa} PA vs {hand}</span>'
    )


def _render_sp_strikeouts(game: dict) -> str:
    """Projected strikeouts for both starters, with the terms behind each."""
    rows = []
    for side, team in (("away", game.get("away")), ("home", game.get("home"))):
        proj = game.get(f"{side}_sp_k")
        if not proj:
            continue
        hand = "LHP" if proj["hand"] == "L" else "RHP"
        rows.append(f"""
                        <div class="k-row">
                            <div class="k-name">{proj['name']} <span class="k-hand">{hand}</span></div>
                            <div class="k-value">{proj['projected_k']:.1f} K</div>
                            <div class="k-range">{proj['low']}&ndash;{proj['high']}</div>
                            <div class="k-detail">
                                {proj['sp_k_rate']:.1%} K rate &times; {proj['expected_bf']:.0f} batters faced
                                &middot; {team} lineup whiffs {proj['opp_k_rate']:.1%} vs {hand}
                                &middot; matchup {proj['matchup_k_rate']:.1%}
                            </div>
                        </div>""")

    if not rows:
        return ""
    return f"""
                    <div class="k-block">
                        <div class="block-label">Projected Strikeouts</div>{''.join(rows)}
                    </div>"""


def _render_projection(game: dict) -> str:
    """Win/loss projection with the run components that produced it."""
    proj = game.get("projection")
    if not proj:
        return ""

    c = proj["components"]
    home_win = proj["home_win_prob"]
    away_win = proj["away_win_prob"]
    home_rec = proj["records"]["home"]
    away_rec = proj["records"]["away"]
    home_leads = home_win >= 0.5

    return f"""
                    <div class="proj-block">
                        <div class="block-label">Projected Outcome</div>
                        <div class="proj-bar">
                            <div class="proj-fill" style="width: {home_win:.1%};"></div>
                        </div>
                        <div class="proj-teams">
                            <span class="{'proj-fav' if not home_leads else ''}">{proj['away_team']} {away_win:.0%}</span>
                            <span class="{'proj-fav' if home_leads else ''}">{proj['home_team']} {home_win:.0%}</span>
                        </div>
                        <div class="proj-score">
                            Projected score {proj['away_expected_runs']:.1f} &ndash; {proj['home_expected_runs']:.1f}
                            &middot; total {proj['total_runs']:.1f}
                        </div>
                        <table class="proj-table">
                            <tr><th></th><th>{proj['away_team']}</th><th>{proj['home_team']}</th></tr>
                            <tr><td>Record</td><td>{away_rec['wins']}&ndash;{away_rec['losses']}</td><td>{home_rec['wins']}&ndash;{home_rec['losses']}</td></tr>
                            <tr><td>Run diff</td><td>{away_rec['run_differential']:+d}</td><td>{home_rec['run_differential']:+d}</td></tr>
                            <tr><td>Pythag win%</td><td>{away_rec['pythagorean_win_pct']:.3f}</td><td>{home_rec['pythagorean_win_pct']:.3f}</td></tr>
                            <tr><td>Offense</td><td>{c['away_offense_index']:.2f}&times;</td><td>{c['home_offense_index']:.2f}&times;</td></tr>
                            <tr><td>Starter</td><td>{c['away_sp_index']:.2f}&times;</td><td>{c['home_sp_index']:.2f}&times;</td></tr>
                            <tr><td>Bullpen</td><td>{c['away_bullpen_index']:.2f}&times;</td><td>{c['home_bullpen_index']:.2f}&times;</td></tr>
                        </table>
                        <div class="proj-note">
                            Indices are multiples of league average ({proj['league_rpg']:.2f} runs/game);
                            below 1.00 suppresses runs. The starter carries
                            {c['starter_share']:.0%} of run prevention, the bullpen the rest.
                            Park {c['park_runs_factor']:.2f}&times;, home field {c['home_field']:.2f}&times;.
                        </div>
                    </div>"""


def _render_conditions(game: dict) -> str:
    """Weather and plate umpire for this game."""
    w = game.get("weather") or {}
    if not w or w.get("temp_f") is None:
        return ""
    bits = []
    if w.get("temp_f") is not None:
        bits.append(f"{w['temp_f']:.0f}&deg;F")
    if w.get("condition"):
        bits.append(w["condition"])
    if w.get("wind_mph") is not None and w.get("wind_dir"):
        bits.append(f"wind {w['wind_mph']:.0f} mph {w['wind_dir']}")
    factor = w.get("hr_factor", 1.0)
    tone = "cond-up" if factor > 1.02 else ("cond-down" if factor < 0.98 else "")
    ump = f" &middot; HP {w['ump_hp']}" if w.get("ump_hp") else ""
    return f"""
                    <div class="cond-block">
                        <span class="cond-text">{' &middot; '.join(bits)}{ump}</span>
                        <span class="cond-factor {tone}">{factor:.2f}&times; HR</span>
                    </div>"""


def _render_bullpen(game: dict, bullpens: dict, league: dict) -> str:
    """Both bullpens, which cover the plate appearances the starter does not."""
    if not bullpens:
        return ""
    lg_hr = league.get("hr_rate") or 0.03
    rows = []
    for team in (game.get("away"), game.get("home")):
        pen = bullpens.get(team)
        if not pen:
            continue
        index = pen["hr_index"]
        tone = "pen-hot" if index > 1.06 else ("pen-cold" if index < 0.94 else "")
        rows.append(f"""
                        <div class="pen-row">
                            <span class="pen-team">{team}</span>
                            <span class="pen-index {tone}">{index:.2f}&times; HR</span>
                            <span class="pen-detail">{pen['hr_rate']:.2%} HR &middot; {pen['k_rate']:.1%} K &middot; {pen['bf']:,} BF</span>
                        </div>""")
    if not rows:
        return ""
    return f"""
                    <div class="pen-block">
                        <div class="block-label">Bullpens &middot; league {lg_hr:.2%} HR/PA</div>{''.join(rows)}
                        <div class="pen-note">
                            Relief arms throw 43% of all plate appearances. Index is
                            home runs allowed vs league; above 1.00 helps hitters late.
                        </div>
                    </div>"""


def _render_hits_section(hit_picks: list) -> str:
    """Projected hits, the same matchup logic applied to contact instead of power."""
    if not hit_picks:
        return ""

    cards = []
    for i, pick in enumerate(hit_picks, 1):
        proj = pick["hits_proj"]
        hand = "LHP" if pick.get("facing_hand") == "L" else "RHP"
        tone = _verdict(pick, "hits")[0]
        cards.append(f"""
                <div class="hit-card{f' card-{tone}' if tone else ''}">
                    {_verdict_ribbon(pick, f"{proj['projected_hits']:.2f}", "hits")}
                    <div class="hit-rank">{i}</div>
                    <div class="pick-name">{pick['batter']}</div>
                    <div class="pick-team">{pick['team']}</div>
                    <div class="hit-value">{proj['projected_hits']:.2f} <span class="hit-unit">hits</span></div>
                    <div class="hit-sub">{proj['prob_at_least_one']:.0%} chance of at least one</div>
                    <table class="hit-table">
                        <tr><td>Season rate</td><td>{proj['hitter_rate']:.1%}</td></tr>
                        <tr><td>vs starter ({hand})</td><td>{proj['rate_vs_sp']:.1%} &times; {proj['pa_vs_sp']:.1f} PA</td></tr>
                        <tr><td>vs bullpen</td><td>{proj['rate_vs_pen']:.1%} &times; {proj['pa_vs_pen']:.1f} PA</td></tr>
                        <tr><td>Starter allows</td><td>{proj['sp_hit_rate']:.1%}</td></tr>{
                            f'<tr><td>Head-to-head</td><td>'
                            f'{pick["matchup"]["hits"]}-for-{pick["matchup"]["pa"]}'
                            f' &middot; {pick["matchup"]["hit_rate"]:.3f}</td></tr>'
                            if pick.get("matchup") else ''
                        }
                    </table>{
                        f'<div class="res-row">{_result_badge(pick, "hits")}</div>'
                        if pick.get("result") else ''
                    }
                </div>""")

    return f"""
        <div class="section" id="hits">
            <h2 class="section-title">🥎 Projected Hits</h2>
            <p class="section-note">
                Expected hits from the hitter's rate and his contact quality, the
                starter's hits allowed and strikeout rate, the bullpen, the park,
                and his plate appearances for his lineup slot, each weighted by
                what the 2026 season replay showed it is worth. The rows below
                show the matchup pieces.
            </p>
            <div class="hits-grid">{''.join(cards)}
            </div>
        </div>
"""


def _chance(prob: float) -> str:
    """A ticket's chance: a percentage, or '1 in N' once it is too small to read."""
    if prob >= 0.01:
        return f"{prob:.1%}"
    return f"1 in {round(1 / prob):,}" if prob > 0 else "&mdash;"


def _render_parlays_section(slate_data: dict) -> str:
    """The model's two 5-pick parlays (mlb_hr.parlays), graded as games finish."""
    plays = slate_data.get("parlays") or []
    if not plays:
        return ""
    badges = {"won": ("&#10003; CASHED", "won"), "lost": ("&#10007; LOST", "lost"),
              "void": ("VOID", "void"), "pending": ("OPEN", "pending")}
    cards = []
    for play in plays:
        legs = ""
        for leg in play["leg_list"]:
            st = leg.get("status", "pending")
            mark = {"won": " &#10003;", "lost": " &#10007;", "void": " (void)"}.get(st, "")
            line = (leg.get("result") or {}).get("summary") or ""
            hand = "LHP" if leg.get("facing_hand") == "L" else "RHP"
            vs = f"vs {leg['opp_sp']} ({hand})" if leg.get("opp_sp") else f"vs {hand}"
            legs += f"""
                    <div class="parlay-leg {st}">
                        <span><span class="leg-type">{'HR' if leg['type'] == 'hr' else '1+ HIT'}</span>{leg['batter']}{mark}
                            <span class="leg-sub">{leg.get('team', '')} &middot; {vs}{
                                f' &middot; {line}' if line else ''}</span></span>
                        <strong>{leg['prob']:.0%}</strong>
                    </div>"""
        label, cls = badges.get(play.get("status", "pending"), badges["pending"])
        ev = play.get("evidence")
        record = ""
        if ev and ev.get("days"):
            if ev["won"]:
                record = (f"In the 2026 replay this ticket, built this way every day, cashed "
                          f"{ev['won']} of {ev['days']} days ({ev['rate']:.1%}) &mdash; the model "
                          f"expected {ev['predicted']:.1%}.")
            else:
                record = (f"In the 2026 replay this ticket, built this way every day, cashed 0 of "
                          f"{ev['days']} days &mdash; the model expected "
                          f"{ev['expected_wins']:.2f} wins in all that time.")
        backfill = (' <span class="tag-backfill">built after the fact</span>'
                    if play.get("backfilled") else "")
        cards.append(f"""
                <div class="parlay-card">
                    <div class="parlay-head">
                        <span class="parlay-name">{play['name']}{backfill}</span>
                        <span class="parlay-status {cls}">{label}</span>
                    </div>
                    <div class="parlay-blurb">{play['blurb']}</div>{legs}
                    <div class="parlay-total"><span>Cashes only if all five land</span>
                        <span>{_chance(play['prob'])}</span></div>
                    <div class="parlay-total"><span>Fair odds &mdash; the price it needs to break even</span>
                        <span class="parlay-odds">{play['fair_odds']}</span></div>{
                        f'<div class="parlay-record">{record}</div>' if record else ''}
                </div>""")
    backfilled = any(p.get("backfilled") for p in plays)
    return f"""
        <div class="section" id="parlays">
            <h2 class="section-title">🎰 5-Pick Parlays</h2>
            <p class="section-note">
                The model's best judgment in two tickets: its five likeliest home run
                bats and its five likeliest hit bats, one leg per game so the legs are
                independent and the chance is simply their product. Tickets lock the
                moment any of their games starts; a scratched player's leg is void and
                the rest ride. Parlays are entertainment with a steep house edge &mdash;
                never stake what you cannot afford to lose.{
                    ' This day was over before the parlays existed, so they were built from'
                    ' that morning&rsquo;s projections and graded against what happened.'
                    if backfilled else ''}
            </p>
            <div class="parlay-grid">{''.join(cards)}
            </div>
        </div>
"""


def _render_matchups_section(matchups: list) -> str:
    """Notable batter-vs-pitcher histories, with the sample-size caveat."""
    if not matchups:
        return ""

    rows = []
    for h in matchups:
        m = h["matchup"]
        tone = "mu-hot" if m["verdict"] == "hot" else "mu-cold"
        rows.append(f"""
                        <tr>
                            <td><strong>{h['batter']}</strong><br><span class="mu-team">{h['team']}</span></td>
                            <td>{m.get('pitcher') or 'starter'}</td>
                            <td class="{tone}">{m['hits']}-for-{m['pa']}</td>
                            <td>{m['hit_rate']:.3f}</td>
                            <td>{m['baseline_hit_rate']:.3f}</td>
                            <td>{m['home_runs']}</td>
                            <td>{m['strikeouts']}</td>
                        </tr>""")

    return f"""
        <div class="section" id="bvp">
            <h2 class="section-title">🔍 Batter vs Pitcher History</h2>
            <p class="section-note">
                Hitters with a notable line against today's opposing starter.
            </p>
            <div class="table-scroll">
                <table class="mu-table">
                    <tr>
                        <th>Hitter</th><th>Starter</th><th>Line</th>
                        <th>Rate</th><th>Season rate</th><th>HR</th><th>K</th>
                    </tr>{''.join(rows)}
                </table>
            </div>
            <div class="mu-caveat">
                <strong>Context only &mdash; this feeds none of the projections above.</strong>
                The largest batter-vs-pitcher sample all season is 13 plate
                appearances, and only about 520 pairs reach even 8. At that size the
                standard error on a hit rate is roughly .16, wider than the entire
                spread of true talent between major-league hitters. A 4-for-9 line is
                noise that looks like a trend, so it is shown with its sample size
                attached and deliberately kept out of the model.
            </div>
        </div>
"""




def _verdict(entry: dict, key: str = "hr") -> tuple:
    """Did this projection come in? Returns (tone, label, detail).

    The page publishes a probability in the morning and the box score answers
    it at night; this is the one place that decides which of the two states a
    projection is in, so the card, the ribbon and the summary cannot disagree.
    """
    line = entry.get("result")
    if not line:
        return ("", "", "")
    got = line.get(key, 0)
    detail = line.get("summary") or f"{line.get('hits', 0)}-for-{line.get('ab', 0)}"
    if line.get("dnp"):
        return ("live", f"&#8212; NO RESULT &middot; {line.get('detailed')}", detail)
    if got:
        unit = "HR" if key == "hr" else "H"
        return ("hit", f"&#10003; HIT &middot; {got} {unit}", detail)
    if line.get("final"):
        return ("miss", "&#10007; MISS", detail)
    # Still batting: not a miss yet, so it reads as live rather than wrong.
    where = line.get("detailed") or "In progress"
    return ("live", f"&#9679; LIVE &middot; {where}", detail)


def _verdict_ribbon(entry: dict, projected: float, key: str = "hr") -> str:
    """Outcome banner across the top of a pick card.

    Projected and actual sit on the same line deliberately: the number the
    model published is only meaningful next to what happened.
    """
    tone, label, detail = _verdict(entry, key)
    if not tone:
        return ""
    unit = "HR" if key == "hr" else "hits"
    return (
        f'<div class="verdict verdict-{tone}">'
        f'<span class="verdict-label">{label}</span>'
        f'<span class="verdict-detail">projected {projected} {unit} '
        f'&rarr; {detail}</span>'
        f'</div>'
    )


def _section_nav(items: list) -> str:
    """Jump links to the sections further down a long page."""
    links = "".join(
        f'<a class="jump" href="#{anchor}">{label}</a>' for anchor, label in items
    )
    return f'<div class="jumps">{links}</div>'


def _result_badge(entry: dict, key: str = "hr") -> str:
    """Hit / miss / in-progress badge for one projected hitter.

    A pick is only a miss once his game is final -- a hitless third inning is
    not a wrong prediction yet -- so an unfinished game shows the line so far
    rather than a verdict.
    """
    line = entry.get("result")
    if not line:
        return ""
    got = line.get(key, 0)
    detail = line.get("summary") or f"{line.get('hits', 0)}-for-{line.get('ab', 0)}"
    if line.get("dnp"):
        return f'<span class="res res-live">&#8212; {line.get("detailed")}</span>'
    if got:
        label = f"{got} HR" if key == "hr" else f"{got} H"
        return (
            f'<span class="res res-hit">&#10003; {label}</span>'
            f'<span class="res-line">{detail}</span>'
        )
    if line.get("final"):
        return (
            f'<span class="res res-miss">&#10007; none</span>'
            f'<span class="res-line">{detail}</span>'
        )
    state = line.get("detailed") or "In progress"
    return (
        f'<span class="res res-live">&#9679; {state}</span>'
        f'<span class="res-line">{detail}</span>'
    )




def remaining_games(slate_data: dict) -> list:
    """Games that have not finished, live ones first.

    Once part of a slate is in the books the only projections still worth
    acting on are the ones attached to games that are still to be played, so
    they are pulled out rather than left mixed in with settled results.
    """
    out = []
    for game in slate_data.get("games", []):
        live = game.get("live") or {}
        if live.get("state") == "Final":
            continue
        out.append(game)
    # Underway before not-yet-started, then by scheduled order.
    out.sort(
        key=lambda g: (
            0 if (g.get("live") or {}).get("state") == "Live" else 1,
            g.get("game_pk", 0),
        )
    )
    return out


def remaining_projections(slate_data: dict, per_game: int = 3) -> list:
    """Top HR and hit projections for each unfinished game.

    Hitters already in the lineup card carry whatever they have done so far,
    so a hitter with a home run in the third is reported as such rather than
    being offered again as a fresh projection.
    """
    hit_by_id = {
        p.get("batter_id"): p for p in (slate_data.get("hit_picks") or [])
    }
    out = []
    for game in remaining_games(slate_data):
        hitters = game.get("hitters") or []
        by_hr = sorted(hitters, key=lambda h: h.get("prob_hr", 0), reverse=True)
        by_hits = sorted(
            (h for h in hitters if h.get("hits_proj")),
            key=lambda h: h["hits_proj"]["projected_hits"],
            reverse=True,
        )
        live = game.get("live") or {}
        out.append({
            "game_pk": game.get("game_pk"),
            "away": game.get("away"),
            "home": game.get("home"),
            "venue": game.get("venue"),
            "away_sp": game.get("away_sp"),
            "home_sp": game.get("home_sp"),
            "away_sp_hand": game.get("away_sp_hand"),
            "home_sp_hand": game.get("home_sp_hand"),
            "state": live.get("state") or "Preview",
            "detailed": live.get("detailed") or "Scheduled",
            "inning": live.get("inning_ordinal"),
            "away_score": live.get("away_score"),
            "home_score": live.get("home_score"),
            "hr": [
                {
                    "batter": h["batter"], "team": h.get("team"),
                    "prob_hr": h.get("prob_hr", 0),
                    "facing_hand": h.get("facing_hand"),
                    "season": f"{h.get('season_hrs', 0)} HR / {h.get('season_pas', 0)} PA",
                    "result": h.get("result"),
                }
                for h in by_hr[:per_game]
            ],
            "hits": [
                {
                    "batter": h["batter"], "team": h.get("team"),
                    "projected_hits": h["hits_proj"]["projected_hits"],
                    "prob_at_least_one": h["hits_proj"]["prob_at_least_one"],
                    "result": h.get("result"),
                    # The head-to-head line, when this pair has any history.
                    "matchup": (hit_by_id.get(h.get("batter_id")) or h).get("matchup"),
                }
                for h in by_hits[:per_game]
            ],
        })
    return out


def _render_remaining_section(slate_data: dict) -> str:
    """Home run and hit projections for the games still to be played."""
    # Without the live layer there is no way to know what has finished, and a
    # section claiming the whole slate is still to play would be worse than
    # showing nothing.
    if not slate_data.get("results"):
        return ""
    games = remaining_projections(slate_data)
    if not games:
        return ""

    cards = ""
    for g in games:
        if g["state"] == "Live":
            tone, when = "res-live", f"{g['detailed']} &middot; {g['inning'] or ''}"
            score = (
                f'<span class="game-score">{g["away"]} {g["away_score"]} &ndash;'
                f' {g["home_score"]} {g["home"]}</span>'
            )
        else:
            tone, when, score = "res-miss", g["detailed"], ""

        hr_rows = ""
        for h in g["hr"]:
            hand = "LHP" if h["facing_hand"] == "L" else "RHP"
            # A hitter who already homered in a game still underway keeps his
            # projection on screen, with what he has done attached to it.
            so_far = _result_badge(h)
            hr_rows += (
                f'<div class="ctx-row"><span>{h["batter"]} '
                f'<span class="cmp-meta">{h["team"]} &middot; vs {hand} '
                f'&middot; {h["season"]}</span></span>'
                f'<strong>{h["prob_hr"]:.1%} {so_far}</strong></div>'
            )

        hit_rows = ""
        for h in g["hits"]:
            mu = h.get("matchup")
            h2h = (
                f' &middot; H2H {mu["hits"]}-for-{mu["pa"]}' if mu else ""
            )
            hit_rows += (
                f'<div class="ctx-row"><span>{h["batter"]} '
                f'<span class="cmp-meta">{h["team"]}{h2h}</span></span>'
                f'<strong>{h["projected_hits"]:.2f} H '
                f'<span class="cmp-meta">{h["prob_at_least_one"]:.0%} for 1+</span>'
                f'</strong></div>'
            )

        cards += f"""
                <div class="game-card">
                    <div class="game-matchup">{g['away']} @ {g['home']}</div>
                    <div class="game-venue">{g['venue']}</div>
                    <div class="game-state"><span class="res {tone}">{when}</span>{score}</div>
                    <div class="game-pitchers">
                        <strong>{g['away']}</strong> SP: {g['away_sp'] or 'TBD'} {_hand_badge(g.get('away_sp_hand'))}<br>
                        <strong>{g['home']}</strong> SP: {g['home_sp'] or 'TBD'} {_hand_badge(g.get('home_sp_hand'))}
                    </div>
                    <div class="split-block">
                        <div class="block-label">Home run &mdash; top {len(g['hr'])}</div>
                        {hr_rows}
                    </div>
                    <div class="split-block">
                        <div class="block-label">Hits &mdash; top {len(g['hits'])}</div>
                        {hit_rows}
                    </div>
                </div>"""

    live_count = sum(1 for g in games if g["state"] == "Live")
    return f"""
        <div class="section" id="remaining">
            <h2 class="section-title">&#127765; Still to Play</h2>
            <p class="section-note">
                {len(games)} game(s) not yet final{f', {live_count} underway' if live_count else ''}
                &mdash; home run probability and projected hits for the hitters who
                still have at-bats coming. Settled games are excluded, so this is
                the part of the slate that is still actionable.
            </p>
            <div class="games-grid">{cards}
            </div>
        </div>
"""

def _freshness(slate_data: dict) -> str:
    """What is from the cached model build and what was just pulled.

    The two have very different ages -- the slate is refit on a timer, the
    scores come down on every load -- and a single "last updated" line would
    misstate both.
    """
    parts = [f"Slate for {slate_data.get('date', '')}"]
    if slate_data.get("built_at"):
        parts.append(f"model built {slate_data['built_at']}")
    live = (slate_data.get("results") or {}).get("fetched_at")
    if live:
        parts.append(f"live data {live}")
    return " &middot; ".join(parts)


def _results_banner(slate_data: dict) -> str:
    """Slate-level scoreboard, shown once anything has actually happened."""
    res = slate_data.get("results")
    if not res or not res.get("any"):
        return ""

    picks = res.get("picks", {})
    hits = res.get("hit_picks", {})
    scored = picks.get("scored", 0)
    parts = []
    if scored:
        parts.append(
            f'<div class="analytics-item"><div class="analytics-value">'
            f'{picks.get("hit", 0)}/{scored}</div>'
            f'<div class="analytics-label">HR picks that connected '
            f'(games final)</div></div>'
        )
    if hits.get("scored"):
        parts.append(
            f'<div class="analytics-item"><div class="analytics-value">'
            f'{hits.get("hit", 0)}/{hits["scored"]}</div>'
            f'<div class="analytics-label">Hit picks with at least one hit</div></div>'
        )
    parts.append(
        f'<div class="analytics-item"><div class="analytics-value">'
        f'{res.get("final_games", 0)}&thinsp;/&thinsp;{res.get("live_games", 0)}'
        f'&thinsp;/&thinsp;{res.get("upcoming_games", 0)}</div>'
        f'<div class="analytics-label">Games final / live / upcoming</div></div>'
    )

    pending = picks.get("pending", 0)
    pending_note = f" {pending} still to be decided." if pending else ""
    return f"""
        <div class="section" id="results">
            <h2 class="section-title">&#127942; Results</h2>
            <div class="analytics">
                <div class="analytics-grid">{''.join(parts)}
                </div>
                <div class="note-box">
                    Scored against the day's box scores, refreshed each time the
                    page loads (as of {res.get('fetched_at', '')}). A pick counts
                    as decided only once its game is final.{pending_note}
                </div>
            </div>
        </div>"""


def _game_score(game: dict) -> str:
    """Live score line for a game card, once it has started."""
    live = game.get("live")
    if not live or live.get("state") not in ("Live", "Final"):
        detailed = (live or {}).get("detailed")
        # Delays and postponements matter before first pitch; "Scheduled" does
        # not, since the card already reads as a preview.
        if detailed and detailed not in ("Scheduled", "Pre-Game", "Warmup"):
            return f'<div class="game-state">{detailed}</div>'
        return ""

    away, home = live.get("away_score"), live.get("home_score")
    if live.get("state") == "Final":
        when = "Final"
    else:
        ordinal = live.get("inning_ordinal") or ""
        half = (live.get("inning_state") or "")[:3]
        when = f"{half} {ordinal}".strip()
    tone = "res-hit" if live.get("state") == "Final" else "res-live"
    return (
        f'<div class="game-state"><span class="res {tone}">{when}</span>'
        f'<span class="game-score">{game["away"]} {away} &ndash;'
        f' {home} {game["home"]}</span></div>'
    )

# Page styling is shared by both pages: the slate at "/" and the model
# comparison at "/models". It lives at module level as a plain string so
# neither renderer has to double every CSS brace inside an f-string.
_CSS = """
        /* Light by default, dark when the system asks for it, and either one
           when the reader picks it with the toggle (data-theme on <html>,
           remembered in localStorage). The dark palette is repeated for the
           two dark cases so the toggle wins in both directions. */
        :root {
            color-scheme: light;
            --bg-primary: #ffffff;
            --bg-secondary: #f5f5f5;
            --text-primary: #1a1a1a;
            --text-secondary: #666;
            --border: #e0e0e0;
            --accent: #0066cc;
            --accent-light: #e6f0ff;
            --success: #22863a;
            --success-light: #f0f9f4;
            --on-accent: #ffffff;
            --danger: #c0392b;
        }

        @media (prefers-color-scheme: dark) {
            :root:not([data-theme="light"]) {
                color-scheme: dark;
                --bg-primary: #1a1a1a;
                --bg-secondary: #2a2a2a;
                --text-primary: #f0f0f0;
                --text-secondary: #999;
                --border: #444;
                --accent: #4da6ff;
                --accent-light: #0d2d5c;
                --success: #34a148;
                --success-light: #0f3d1f;
                --on-accent: #0a1628;
                --danger: #ff6b5e;
            }
        }

        :root[data-theme="dark"] {
            color-scheme: dark;
            --bg-primary: #1a1a1a;
            --bg-secondary: #2a2a2a;
            --text-primary: #f0f0f0;
            --text-secondary: #999;
            --border: #444;
            --accent: #4da6ff;
            --accent-light: #0d2d5c;
            --success: #34a148;
            --success-light: #0f3d1f;
            --on-accent: #0a1628;
            --danger: #ff6b5e;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 20px;
            transition: background-color 0.3s, color 0.3s;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
        }

        header {
            margin-bottom: 40px;
            border-bottom: 2px solid var(--border);
            padding-bottom: 20px;
        }

        h1 {
            font-size: 2.5em;
            font-weight: 600;
            margin-bottom: 8px;
            letter-spacing: -0.5px;
        }

        .subtitle {
            color: var(--text-secondary);
            font-size: 1.1em;
            margin-bottom: 12px;
        }

        .stats-bar {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }

        .stat-card {
            background: var(--bg-secondary);
            padding: 16px;
            border-radius: 8px;
            border: 1px solid var(--border);
        }

        .stat-label {
            color: var(--text-secondary);
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }

        .stat-value {
            font-size: 2em;
            font-weight: 600;
            color: var(--accent);
        }

        .section {
            margin-bottom: 50px;
        }

        .section-title {
            font-size: 1.8em;
            font-weight: 600;
            margin-bottom: 24px;
            color: var(--text-primary);
        }

        .picks-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .pick-card {
            background: var(--bg-secondary);
            border: 2px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .pick-card:hover {
            border-color: var(--accent);
            box-shadow: 0 8px 24px rgba(0, 102, 204, 0.1);
        }

        .pick-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, var(--accent), transparent);
        }

        .pick-rank {
            display: inline-block;
            background: var(--accent);
            color: var(--on-accent);
            width: 32px;
            height: 32px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            margin-bottom: 12px;
        }

        .pick-name {
            font-size: 1.4em;
            font-weight: 600;
            margin-bottom: 4px;
        }

        .pick-team {
            color: var(--text-secondary);
            font-size: 0.95em;
            margin-bottom: 16px;
        }

        .pick-prob {
            background: var(--success-light);
            border-left: 4px solid var(--success);
            padding: 12px;
            border-radius: 6px;
            margin-bottom: 12px;
        }

        .pick-prob-label {
            color: var(--text-secondary);
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 0.3px;
            margin-bottom: 4px;
        }

        .pick-prob-value {
            font-size: 1.8em;
            font-weight: 600;
            color: var(--success);
        }

        .pick-stats {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            font-size: 0.9em;
        }

        .pick-stat {
            color: var(--text-secondary);
        }

        .pick-stat strong {
            color: var(--text-primary);
            font-weight: 600;
        }

        .games-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 20px;
        }

        .game-card {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 20px;
        }

        .game-matchup {
            font-size: 1.2em;
            font-weight: 600;
            margin-bottom: 4px;
        }

        .game-venue {
            color: var(--text-secondary);
            font-size: 0.9em;
            margin-bottom: 16px;
        }

        .game-pitchers {
            background: var(--bg-primary);
            padding: 12px;
            border-radius: 6px;
            font-size: 0.85em;
            color: var(--text-secondary);
            line-height: 1.6;
        }

        .top-hitters {
            margin-top: 16px;
            padding-top: 16px;
            border-top: 1px solid var(--border);
        }

        .top-hitters-label {
            font-size: 0.85em;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.3px;
            margin-bottom: 8px;
        }

        .hitter-list {
            font-size: 0.85em;
            color: var(--text-secondary);
            line-height: 1.8;
        }

        .hitter-item {
            padding: 4px 0;
        }

        .hitter-name {
            color: var(--text-primary);
            font-weight: 500;
        }

        .analytics {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 30px;
        }

        .analytics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 30px;
        }

        .analytics-item {
            text-align: center;
        }

        .analytics-value {
            font-size: 2.5em;
            font-weight: 600;
            color: var(--accent);
            margin-bottom: 8px;
        }

        .analytics-label {
            color: var(--text-secondary);
            font-size: 0.95em;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }

        footer {
            text-align: center;
            margin-top: 60px;
            padding-top: 20px;
            border-top: 1px solid var(--border);
            color: var(--text-secondary);
            font-size: 0.9em;
        }

        .refresh-btn {
            background: var(--accent);
            color: var(--on-accent);
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.95em;
            font-weight: 600;
            transition: all 0.3s;
            display: inline-block;
            margin-bottom: 20px;
        }

        .refresh-btn:hover {
            background: var(--accent);
            opacity: 0.9;
            transform: translateY(-2px);
        }

        @media (max-width: 768px) {
            h1 { font-size: 2em; }
            .section-title { font-size: 1.5em; }
            .picks-grid { grid-template-columns: 1fr; }
            .games-grid { grid-template-columns: 1fr; }
            body { padding: 16px; }
        }
        /* Pitcher handedness, strikeout and game-projection blocks */
        .hand-badge {
            display: inline-block;
            font-size: 0.75em;
            font-weight: 700;
            letter-spacing: 0.5px;
            padding: 1px 6px;
            border-radius: 4px;
            margin-left: 4px;
            vertical-align: middle;
        }

        .hand-r {
            background: var(--accent-light);
            color: var(--accent);
        }

        .hand-l {
            background: var(--success-light);
            color: var(--success);
        }

        .block-label {
            font-size: 0.75em;
            text-transform: uppercase;
            letter-spacing: 0.6px;
            color: var(--text-secondary);
            margin-bottom: 8px;
        }

        .proj-block, .k-block {
            margin-top: 16px;
            padding-top: 16px;
            border-top: 1px solid var(--border);
        }

        .proj-bar {
            height: 8px;
            border-radius: 4px;
            background: var(--accent-light);
            overflow: hidden;
            display: flex;
            flex-direction: row-reverse;
        }

        .proj-fill {
            height: 100%;
            background: var(--accent);
        }

        .proj-teams {
            display: flex;
            justify-content: space-between;
            font-size: 0.85em;
            margin-top: 6px;
            color: var(--text-secondary);
        }

        .proj-fav {
            color: var(--text-primary);
            font-weight: 700;
        }

        .proj-score {
            font-size: 0.85em;
            color: var(--text-secondary);
            margin-top: 6px;
        }

        .proj-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 12px;
            font-size: 0.8em;
            font-variant-numeric: tabular-nums;
        }

        .proj-table th {
            text-align: right;
            font-weight: 600;
            color: var(--text-secondary);
            padding: 3px 0;
            border-bottom: 1px solid var(--border);
        }

        .proj-table th:first-child {
            text-align: left;
        }

        .proj-table td {
            text-align: right;
            padding: 3px 0;
            color: var(--text-primary);
        }

        .proj-table td:first-child {
            text-align: left;
            color: var(--text-secondary);
        }

        .proj-note {
            font-size: 0.75em;
            color: var(--text-secondary);
            line-height: 1.6;
            margin-top: 10px;
        }

        .k-row {
            display: grid;
            grid-template-columns: 1fr auto auto;
            gap: 4px 10px;
            align-items: baseline;
            margin-bottom: 10px;
        }

        .k-name {
            font-weight: 600;
            font-size: 0.9em;
        }

        .k-hand {
            font-size: 0.8em;
            color: var(--text-secondary);
            font-weight: 500;
        }

        .k-value {
            font-weight: 700;
            color: var(--accent);
            font-variant-numeric: tabular-nums;
        }

        .k-range {
            font-size: 0.8em;
            color: var(--text-secondary);
            font-variant-numeric: tabular-nums;
        }

        .k-detail {
            grid-column: 1 / -1;
            font-size: 0.75em;
            color: var(--text-secondary);
            line-height: 1.5;
        }

        .platoon-note {
            display: block;
            font-size: 0.75em;
            color: var(--text-secondary);
        }

        .split-block {
            margin-top: 14px;
            padding-top: 12px;
            border-top: 1px solid var(--border);
        }

        .split-row {
            display: grid;
            grid-template-columns: 4.5em auto 1fr;
            gap: 8px;
            align-items: baseline;
            padding: 3px 6px;
            border-radius: 4px;
            font-size: 0.85em;
            font-variant-numeric: tabular-nums;
            color: var(--text-secondary);
        }

        .split-row strong {
            color: var(--text-primary);
        }

        .split-facing {
            background: var(--accent-light);
            color: var(--text-primary);
        }

        .split-rate {
            text-align: right;
        }

        .split-note {
            font-size: 0.72em;
            color: var(--text-secondary);
            margin-top: 6px;
        }

        /* Bullpen, hits and batter-vs-pitcher sections */
        .section-note {
            color: var(--text-secondary);
            font-size: 0.9em;
            margin-top: -12px;
            margin-bottom: 20px;
            max-width: 70ch;
        }

        .pen-block {
            margin-top: 16px;
            padding-top: 16px;
            border-top: 1px solid var(--border);
        }

        .pen-row {
            display: grid;
            grid-template-columns: 1fr auto;
            gap: 2px 10px;
            font-size: 0.85em;
            margin-bottom: 8px;
        }

        .pen-team {
            font-weight: 600;
        }

        .pen-index {
            font-weight: 700;
            font-variant-numeric: tabular-nums;
            color: var(--text-secondary);
        }

        .pen-hot { color: var(--danger); }
        .pen-cold { color: var(--success); }

        .pen-detail {
            grid-column: 1 / -1;
            font-size: 0.85em;
            color: var(--text-secondary);
            font-variant-numeric: tabular-nums;
        }

        .pen-note {
            font-size: 0.75em;
            color: var(--text-secondary);
            line-height: 1.5;
            margin-top: 8px;
        }

        .hits-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 20px;
        }

        .parlay-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(min(320px, 100%), 1fr));
            gap: 20px;
        }

        .parlay-card {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 18px;
        }

        .parlay-head {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            gap: 10px;
            margin-bottom: 6px;
        }

        .parlay-name { font-size: 1.15em; font-weight: 700; }

        .parlay-blurb {
            font-size: 0.85em;
            color: var(--text-secondary);
            line-height: 1.5;
            margin-bottom: 10px;
        }

        .parlay-leg {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
            padding: 8px 0;
            border-top: 1px solid var(--border);
            font-size: 0.92em;
        }

        .parlay-leg .leg-type {
            display: inline-block;
            min-width: 3.6em;
            font-size: 0.72em;
            font-weight: 700;
            color: var(--accent);
        }

        .parlay-leg .leg-sub {
            display: block;
            font-size: 0.8em;
            color: var(--text-secondary);
            margin-top: 2px;
        }

        .parlay-leg strong { font-variant-numeric: tabular-nums; }
        .parlay-leg.won strong, .parlay-leg.won > span { color: var(--success); }
        .parlay-leg.lost strong { color: var(--danger); }
        .parlay-leg.void { opacity: 0.6; }

        .parlay-total {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            padding: 9px 0 0;
            margin-top: 6px;
            border-top: 2px solid var(--border);
            font-weight: 600;
            font-variant-numeric: tabular-nums;
        }

        .parlay-odds { color: var(--accent); font-weight: 700; }

        .parlay-status {
            font-size: 0.72em;
            font-weight: 700;
            padding: 3px 9px;
            border-radius: 12px;
            border: 1px solid var(--border);
            color: var(--text-secondary);
            white-space: nowrap;
        }

        .parlay-status.won {
            background: var(--success-light);
            border-color: var(--success);
            color: var(--success);
        }

        .parlay-status.lost { border-color: var(--danger); color: var(--danger); }

        .parlay-record {
            margin-top: 12px;
            font-size: 0.8em;
            color: var(--text-secondary);
            line-height: 1.5;
        }

        .tag-backfill {
            font-size: 0.6em;
            font-weight: 600;
            color: var(--text-secondary);
            border: 1px dashed var(--border);
            padding: 2px 6px;
            border-radius: 10px;
            margin-left: 6px;
            vertical-align: middle;
        }

        .hit-card {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 18px;
        }

        .hit-rank {
            display: inline-block;
            font-size: 0.75em;
            font-weight: 700;
            color: var(--text-secondary);
            margin-bottom: 8px;
        }

        .hit-value {
            font-size: 1.9em;
            font-weight: 600;
            color: var(--accent);
            font-variant-numeric: tabular-nums;
            margin-top: 10px;
        }

        .hit-unit {
            font-size: 0.45em;
            font-weight: 500;
            color: var(--text-secondary);
            letter-spacing: 0.5px;
        }

        .hit-sub {
            font-size: 0.85em;
            color: var(--text-secondary);
            margin-bottom: 12px;
        }

        .hit-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.8em;
            font-variant-numeric: tabular-nums;
        }

        .hit-table td {
            padding: 3px 0;
            color: var(--text-primary);
            text-align: right;
        }

        .hit-table td:first-child {
            text-align: left;
            color: var(--text-secondary);
        }

        .table-scroll {
            overflow-x: auto;
        }

        .mu-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85em;
            font-variant-numeric: tabular-nums;
            min-width: 560px;
        }

        .mu-table th {
            text-align: right;
            font-weight: 600;
            color: var(--text-secondary);
            padding: 8px 10px;
            border-bottom: 2px solid var(--border);
            white-space: nowrap;
        }

        .mu-table th:first-child, .mu-table th:nth-child(2) {
            text-align: left;
        }

        .mu-table td {
            text-align: right;
            padding: 8px 10px;
            border-bottom: 1px solid var(--border);
        }

        .mu-table td:first-child, .mu-table td:nth-child(2) {
            text-align: left;
        }

        .mu-team {
            font-size: 0.85em;
            color: var(--text-secondary);
        }

        .mu-hot { color: var(--success); font-weight: 700; }
        .mu-cold { color: var(--danger); font-weight: 700; }

        .mu-caveat {
            margin-top: 16px;
            padding: 14px;
            border-radius: 8px;
            background: var(--bg-secondary);
            border-left: 4px solid var(--text-secondary);
            font-size: 0.85em;
            color: var(--text-secondary);
            line-height: 1.7;
            max-width: 78ch;
        }

        .slot-badge {
            display: inline-block;
            font-size: 0.75em;
            font-weight: 700;
            padding: 1px 6px;
            border-radius: 4px;
            background: var(--accent-light);
            color: var(--accent);
            margin-right: 4px;
        }

        .cond-block {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            gap: 10px;
            margin-top: 14px;
            padding: 8px 10px;
            border-radius: 6px;
            background: var(--bg-primary);
            font-size: 0.8em;
            color: var(--text-secondary);
        }

        .cond-factor {
            font-weight: 700;
            font-variant-numeric: tabular-nums;
            white-space: nowrap;
        }

        .cond-up { color: var(--danger); }
        .cond-down { color: var(--success); }

        .ctx-row {
            display: flex;
            justify-content: space-between;
            font-size: 0.85em;
            padding: 3px 6px;
            color: var(--text-secondary);
            font-variant-numeric: tabular-nums;
        }

        .ctx-row strong {
            color: var(--text-primary);
        }


        /* ---- shared nav tabs (slate page / model lab page) ---- */
        .tabs {
            display: flex;
            gap: 8px;
            justify-content: center;
            margin-top: 20px;
        }

        .tab {
            padding: 8px 18px;
            border-radius: 999px;
            border: 1px solid var(--border);
            background: var(--bg-primary);
            color: var(--text-secondary);
            text-decoration: none;
            font-size: 0.9em;
            font-weight: 600;
        }

        .tab:hover { border-color: var(--accent); color: var(--accent); }

        .tab.active {
            background: var(--accent);
            border-color: var(--accent);
            color: var(--on-accent);
        }

        .theme-toggle {
            cursor: pointer;
            font-family: inherit;
            line-height: inherit;
        }

        .theme-toggle:focus-visible {
            outline: 2px solid var(--accent);
            outline-offset: 2px;
        }

        /* ---- model comparison table ---- */
        .table-wrap {
            overflow-x: auto;
            border: 1px solid var(--border);
            border-radius: 8px;
            background: var(--bg-primary);
        }

        .cmp-table {
            border-collapse: collapse;
            width: 100%;
            font-size: 0.88em;
            font-variant-numeric: tabular-nums;
        }

        .cmp-table th,
        .cmp-table td {
            padding: 8px 12px;
            text-align: right;
            white-space: nowrap;
            border-bottom: 1px solid var(--border);
        }

        .cmp-table th {
            position: sticky;
            top: 0;
            background: var(--bg-secondary);
            color: var(--text-secondary);
            font-size: 0.9em;
            text-align: right;
        }

        .cmp-table th:first-child,
        .cmp-table td:first-child { text-align: left; }

        .cmp-table tbody tr:hover { background: var(--bg-secondary); }

        .cmp-table td.served {
            background: var(--accent-light);
            font-weight: 700;
            color: var(--text-primary);
        }

        .cmp-table th.served { color: var(--accent); }

        .cmp-name { font-weight: 600; color: var(--text-primary); }

        .cmp-meta { color: var(--text-secondary); font-size: 0.85em; }

        .rank-in { color: var(--success); font-weight: 700; }

        .rank-out { color: var(--text-secondary); }

        .note-box {
            margin-top: 16px;
            font-size: 0.85em;
            color: var(--text-secondary);
            line-height: 1.8;
            background: var(--bg-primary);
            padding: 12px;
            border-radius: 6px;
        }

        .agree-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px;
        }

        /* ---- outcomes, once games start ---- */
        .res {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 999px;
            font-size: 0.78em;
            font-weight: 700;
            white-space: nowrap;
        }

        .res-hit { background: var(--success-light); color: var(--success); }

        .res-miss { background: var(--bg-secondary); color: var(--text-secondary); }

        .res-live { background: var(--accent-light); color: var(--accent); }

        .res-line {
            margin-left: 8px;
            font-size: 0.78em;
            color: var(--text-secondary);
            font-variant-numeric: tabular-nums;
        }

        .res-row {
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid var(--border);
        }

        .game-state {
            display: flex;
            align-items: center;
            gap: 8px;
            margin: 8px 0;
            flex-wrap: wrap;
        }

        .game-score {
            font-size: 0.85em;
            font-weight: 600;
            font-variant-numeric: tabular-nums;
            color: var(--text-primary);
        }

        .live-note {
            margin-top: 10px;
            font-size: 0.8em;
            color: var(--text-secondary);
        }

        /* ---- verdict ribbon: the projection, judged ---- */
        .verdict {
            display: flex;
            flex-wrap: wrap;
            align-items: baseline;
            gap: 4px 10px;
            margin: -20px -20px 16px -20px;
            padding: 10px 20px;
            border-bottom: 1px solid var(--border);
        }

        .verdict-label {
            font-weight: 800;
            font-size: 0.95em;
            letter-spacing: 0.02em;
        }

        .verdict-detail {
            font-size: 0.8em;
            color: var(--text-secondary);
            font-variant-numeric: tabular-nums;
        }

        .verdict-hit {
            background: var(--success-light);
            border-bottom-color: var(--success);
        }

        .verdict-hit .verdict-label { color: var(--success); }

        .verdict-miss { background: var(--bg-primary); }

        .verdict-miss .verdict-label { color: var(--text-secondary); }

        .verdict-live {
            background: var(--accent-light);
            border-bottom-color: var(--accent);
        }

        .verdict-live .verdict-label { color: var(--accent); }

        /* The ribbon bleeds to the card edge, so it has to match that card's
           own padding; the two card types differ by two pixels. */
        .hit-card { overflow: hidden; }

        .hit-card .verdict { margin: -18px -18px 14px -18px; padding: 9px 18px; }

        .card-hit { border-color: var(--success); }

        .card-miss { opacity: 0.72; }

        .card-live { border-color: var(--accent); }

        /* ---- in-page jump links ---- */
        .jumps {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            justify-content: center;
            margin-top: 10px;
        }

        .jump {
            padding: 5px 12px;
            border-radius: 999px;
            border: 1px solid var(--border);
            background: var(--bg-secondary);
            color: var(--text-secondary);
            text-decoration: none;
            font-size: 0.8em;
            font-weight: 600;
        }

        .jump:hover { border-color: var(--accent); color: var(--accent); }

        .section { scroll-margin-top: 16px; }

        /* ---- print / PDF snapshot ---- */
        @media print {
            /* The PDF renderer has no grid engine to speak of, and a screen
               grid collapses to one item per row when it is ignored. Explicit
               two-column flex keeps the cards side by side on paper. */
            .picks-grid, .games-grid, .hits-grid, .analytics-grid, .agree-grid {
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
            }

            .pick-card, .game-card, .hit-card, .analytics-item {
                width: 48%;
                break-inside: avoid;
                page-break-inside: avoid;
            }

            /* Navigation is meaningless on paper. */
            .tabs, .jumps { display: none; }

            .section {
                break-inside: auto;
                margin-bottom: 18px;
            }

            .section-title {
                break-after: avoid;
                page-break-after: avoid;
            }

            .table-wrap, .table-scroll { overflow: visible; }

            .cmp-table { font-size: 0.72em; }

            .cmp-table th { position: static; }

            body {
                background: #fff;
                color: #111;
                font-size: 10pt;
            }

            .container { max-width: none; padding: 0; }

            a { text-decoration: none; color: inherit; }
        }

        @page {
            size: A4 landscape;
            margin: 12mm 10mm;
        }
"""


# Runs in <head>, before first paint, so a saved dark choice never flashes
# light. Storage can be unavailable (private windows); the page then simply
# follows the system theme.
_THEME_SCRIPT = """
    <script>
        (function () {
            try {
                var saved = localStorage.getItem("theme");
                if (saved === "light" || saved === "dark") {
                    document.documentElement.setAttribute("data-theme", saved);
                }
            } catch (e) {}
        })();

        function currentTheme() {
            var set = document.documentElement.getAttribute("data-theme");
            if (set) return set;
            return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
        }

        function syncThemeToggle() {
            var dark = currentTheme() === "dark";
            document.querySelectorAll(".theme-toggle").forEach(function (b) {
                // The label names the mode a click switches to.
                b.innerHTML = dark ? "&#9728;&#65039; Light" : "&#127769; Dark";
                b.setAttribute("aria-pressed", dark ? "true" : "false");
            });
        }

        function toggleTheme() {
            var next = currentTheme() === "dark" ? "light" : "dark";
            document.documentElement.setAttribute("data-theme", next);
            try { localStorage.setItem("theme", next); } catch (e) {}
            syncThemeToggle();
        }

        document.addEventListener("DOMContentLoaded", syncThemeToggle);
        // Follow system changes until the reader makes a choice of their own.
        window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", syncThemeToggle);
    </script>"""


def _tabs(active: str) -> str:
    """Nav between the slate page and the model-comparison page, plus the
    light/dark toggle."""
    links = (
        ("/", "slate", "&#127919; Slate"),
        ("/results", "results", "&#128197; Results"),
        ("/homer", "homer", "&#129506; HOMER"),
        ("/models", "models", "&#9878; Model Lab"),
    )
    items = "".join(
        f'<a class="tab{" active" if key == active else ""}" href="{href}">{label}</a>'
        for href, key, label in links
    )
    toggle = (
        '<button type="button" class="tab theme-toggle" onclick="toggleTheme()" '
        'aria-label="Switch between light and dark mode">&#127769; Dark</button>'
    )
    return f'<div class="tabs">{items}{toggle}</div>'


def render_html(slate_data: dict) -> str:
    """Generate modern, user-friendly HTML from slate data."""

    date_str = slate_data["date"]
    games_count = slate_data["games_count"]
    picks = slate_data["picks_6"]
    games = slate_data["games"]
    ensemble_metrics = slate_data.get("ensemble_metrics")
    bullpens = slate_data.get("bullpens") or {}
    league_rates = slate_data.get("league_rates") or {}
    hit_picks = slate_data.get("hit_picks") or []
    highlighted = slate_data.get("highlighted_matchups") or []

    # Calculate some analytics
    total_hrs_today = sum(
        len([h for h in g["hitters"] if h["prob_hr"] >= 0.05])
        for g in games
    )
    avg_prob = (
        sum(p["prob_hr"] for p in picks) / len(picks)
        if picks else 0
    )

    # Once anything is final the slate has a record, and that belongs beside
    # the forecast stats rather than only at the bottom of the page.
    results = slate_data.get("results") or {}
    scored = (results.get("picks") or {}).get("scored", 0)
    record_card = ""
    if scored:
        hit = results["picks"]["hit"]
        record_card = f"""<div class="stat-card">
                    <div class="stat-label">Picks Hit (final)</div>
                    <div class="stat-value" style="color: var(--success);">{hit}/{scored}</div>
                </div>"""

    # Jump links: the page is long, and the hit and batter-vs-pitcher tables
    # sit below the fold where readers were not finding them.
    jump_targets = [("picks", "🎯 Picks")]
    if results.get("any"):
        jump_targets.append(("results", "🏆 Results"))
    if results and remaining_projections(slate_data):
        jump_targets.append(("remaining", "🌙 Still to Play"))
    if hit_picks:
        jump_targets.append(("hits", "🥎 Projected Hits"))
    if slate_data.get("parlays"):
        jump_targets.append(("parlays", "🎰 5-Pick Parlays"))
    if highlighted:
        jump_targets.append(("bvp", "🔍 Batter vs Pitcher"))
    jump_targets += [("games", "📊 Games"), ("analytics", "📈 Analytics")]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>HR Daily Tracker — {date_str}</title>
    <style>{_CSS}    </style>{_THEME_SCRIPT}
</head>
<body>
    <div class="container">
        <header>
            <h1>🏟️ HR Daily Tracker</h1>
            <p class="subtitle">Home run predictions for {date_str}</p>
            <div class="stats-bar">
                <div class="stat-card">
                    <div class="stat-label">Games Today</div>
                    <div class="stat-value">{games_count}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Top Picks</div>
                    <div class="stat-value">{len(picks)}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Avg Prob</div>
                    <div class="stat-value">{avg_prob:.1%}</div>
                </div>
                {record_card}
            </div>
            {_tabs("slate")}
            {_section_nav(jump_targets)}
        </header>

        <div class="section" id="picks">
            <h2 class="section-title">🎯 6-Pick HR Slate</h2>
            <div class="picks-grid">
"""

    for i, pick in enumerate(picks, 1):
        tone = _verdict(pick)[0]
        html += f"""
                <div class="pick-card{f' card-{tone}' if tone else ''}">
                    {_verdict_ribbon(pick, f"{pick['prob_hr']:.1%}")}
                    <div class="pick-rank">{i}</div>
                    <div class="pick-name">{pick['batter']}</div>
                    <div class="pick-team">{
                        f'<span class="slot-badge">bats {pick["lineup_slot"]}</span> '
                        if pick.get('lineup_slot') else ''
                    }{pick['team']}{
                        f' &middot; <span style="color: var(--accent); font-weight: 600;">{pick["injury_note"]}</span>'
                        if pick.get('injury_note') else ''
                    }</div>
                    <div class="pick-prob">
                        <div class="pick-prob-label">HR Probability (this game)</div>
                        <div class="pick-prob-value">{pick['prob_hr']:.1%}</div>
                        <div style="font-size: 0.75em; opacity: 0.75; margin-top: 4px;">
                            {pick.get('prob_ensemble_per_pa', 0):.2%} per PA
                            &times; {pick.get('prob_expected_pa', 0):.1f} PA
                            ({pick.get('prob_pa_vs_sp', 0):.1f} vs SP
                            + {pick.get('prob_pa_vs_pen', 0):.1f} vs pen)
                        </div>
                    </div>
                    <div class="pick-stats">
                        <div class="pick-stat">
                            <strong>{pick['season_hrs']}</strong> HRs this season
                        </div>
                        <div class="pick-stat">
                            <strong>{pick['season_pas']}</strong> plate appearances
                        </div>
                    </div>
                    <div class="split-block">
                        <div class="block-label">HRs by pitcher hand{
                            ' &middot; switch-hits' if pick.get('bat_side') == 'S'
                            else f" &middot; bats {pick['bat_side']}" if pick.get('bat_side') else ''
                        }</div>
                        <div class="split-row{' split-facing' if pick.get('facing_hand') == 'R' else ''}">
                            <span>vs RHP</span>
                            <strong>{pick.get('hr_vs_rhp', 0)} HR</strong>
                            <span class="split-rate">{pick.get('pa_vs_rhp', 0)} PA{
                                f" &middot; {pick['hr_vs_rhp'] / pick['pa_vs_rhp']:.1%}" if pick.get('pa_vs_rhp') else ''
                            }</span>
                        </div>
                        <div class="split-row{' split-facing' if pick.get('facing_hand') == 'L' else ''}">
                            <span>vs LHP</span>
                            <strong>{pick.get('hr_vs_lhp', 0)} HR</strong>
                            <span class="split-rate">{pick.get('pa_vs_lhp', 0)} PA{
                                f" &middot; {pick['hr_vs_lhp'] / pick['pa_vs_lhp']:.1%}" if pick.get('pa_vs_lhp') else ''
                            }</span>
                        </div>
                        <div class="split-note">Highlighted row is today's starter ({
                            'LHP' if pick.get('facing_hand') == 'L' else 'RHP'
                        })</div>
                    </div>
                    <div class="split-block">
                        <div class="block-label">Context multipliers</div>
                        <div class="ctx-row"><span>Park (vs {_side_vs_starter(pick)}HB)</span><strong>{pick.get('prob_park_side', 1):.2f}&times;</strong></div>
                        <div class="ctx-row"><span>Pull rate</span><strong>{pick.get('prob_pull_rate', 0):.0%}</strong></div>
                        <div class="ctx-row"><span>Park &times; pull</span><strong>{pick.get('prob_park_effective', 1):.2f}&times;</strong></div>
                        <div class="ctx-row"><span>Weather</span><strong>{pick.get('prob_weather_factor', 1):.2f}&times;</strong></div>
                        <div class="ctx-row"><span>Opp bullpen</span><strong>{pick.get('prob_pen_hr_index', 1):.2f}&times;</strong></div>{
                            f'<div class="ctx-row"><span>Vs this starter</span>'
                            f'<strong>{pick["matchup"]["hits"]}-for-{pick["matchup"]["pa"]}'
                            f'</strong></div>'
                            if pick.get("matchup") else ''
                        }
                    </div>{
                        f'<div class="res-row">{_result_badge(pick)}</div>'
                        if pick.get("result") else ''
                    }
                </div>
"""

    html += """
            </div>
        </div>
"""

    # Outcomes go directly under the picks: the first thing a reader wants
    # once games are underway is whether the six names delivered.
    # Say so plainly when this is yesterday's slate standing in for one that is
    # still being fitted; a stale page that admits it beats a silent one.
    if slate_data.get("building_note"):
        html += f"""
        <div class="section">
            <div class="note-box" style="border-left: 3px solid var(--accent);">
                &#9881;&#65039; {slate_data['building_note']}
            </div>
        </div>"""

    html += _results_banner(slate_data)
    # What is still live comes before the settled parts of the page: once the
    # afternoon games are in the book, tonight's are the only actionable ones.
    html += _render_remaining_section(slate_data)
    html += _render_hits_section(hit_picks)
    html += _render_parlays_section(slate_data)
    html += _render_matchups_section(highlighted)

    html += """
        <div class="section" id="games">
            <h2 class="section-title">📊 Today's Games</h2>
            <div class="games-grid">
"""

    for game in games:
        home = game["home"]
        away = game["away"]
        venue = game["venue"]
        home_sp = game.get("home_sp", "TBD")
        away_sp = game.get("away_sp", "TBD")

        # Get top 3 hitters for this game
        top_hitters = sorted(
            game["hitters"],
            key=lambda h: h["prob_hr"],
            reverse=True
        )[:3]

        html += f"""
                <div class="game-card">
                    <div class="game-matchup">{away} @ {home}</div>
                    <div class="game-venue">{venue}</div>
{_game_score(game)}
                    <div class="game-pitchers">
                        <strong>{away}</strong> SP: {away_sp} {_hand_badge(game.get("away_sp_hand"))}<br>
                        <strong>{home}</strong> SP: {home_sp} {_hand_badge(game.get("home_sp_hand"))}
                    </div>
{_render_conditions(game)}{_render_projection(game)}{_render_sp_strikeouts(game)}{_render_bullpen(game, bullpens, league_rates)}
                    <div class="top-hitters">
                        <div class="top-hitters-label">Highest HR Probs</div>
                        <div class="hitter-list">
"""

        for hitter in top_hitters:
            outcome = (
                f' {_result_badge(hitter)}' if hitter.get("result") else ""
            )
            html += (
                '                            <div class="hitter-item">'
                f'<span class="hitter-name">{hitter["batter"]}</span> '
                f'{hitter["prob_hr"]:.1%}'
                f'{_platoon_note(hitter)}{outcome}</div>\n'
            )

        html += """
                        </div>
                    </div>
                </div>
"""

    html += """
            </div>
        </div>

        <div class="section" id="analytics">
            <h2 class="section-title">📈 Analytics & Model Comparison</h2>
            <div class="analytics">
                <div class="analytics-grid">
                    <div class="analytics-item">
                        <div class="analytics-value">165K+</div>
                        <div class="analytics-label">Plate Appearances Analyzed</div>
                    </div>
                    <div class="analytics-item">
                        <div class="analytics-value">654</div>
                        <div class="analytics-label">Unique Batters in Model</div>
                    </div>
                    <div class="analytics-item">
                        <div class="analytics-value">2026</div>
                        <div class="analytics-label">Full Season Data</div>
                    </div>
                </div>
            </div>"""

    # Add model comparison metrics if ensemble was used
    if ensemble_metrics:
        cutoff = ensemble_metrics.get("cutoff_date", "")
        n_test = ensemble_metrics.get("n_test_samples", 0)
        holdout_rate = ensemble_metrics.get("holdout_hr_rate", 0)
        html += f"""
            <div class="analytics" style="margin-top: 20px;">
                <h3 style="font-size: 1.2em; margin-bottom: 4px;">Model Performance (held-out)</h3>
                <div style="font-size: 0.85em; color: var(--text-secondary); margin-bottom: 16px;">
                    Fitted on plate appearances before {cutoff}, scored on the
                    {n_test:,} that came after (actual HR rate {holdout_rate:.2%}).
                    Walk-forward: no outcome in this test window was visible during fitting.
                </div>
                <div class="analytics-grid">"""

        for model_name, key in [
            ("Flat league prior", "empirical_bayes"),
            ("KNN comparables", "knn"),
            ("SVR", "svm"),
            ("Random forest", "rf"),
            ("Core model (served)", "core"),
            ("Forest model", "forest"),
        ]:
            metrics = ensemble_metrics.get(key, {})
            if metrics:
                html += f"""
                    <div class="analytics-item">
                        <div style="font-size: 0.9em; color: var(--text-secondary); margin-bottom: 8px;"><strong>{model_name}</strong></div>
                        <div style="font-size: 0.85em; color: var(--text-secondary); line-height: 1.8;">
                            AUC: <strong>{metrics.get('auc', 0):.4f}</strong><br>
                            Top-decile lift: {metrics.get('top_decile_lift', 0):.2f}&times;<br>
                            Brier: {metrics.get('brier', 0):.5f}<br>
                            Log Loss: {metrics.get('log_loss', 0):.5f}<br>
                            Cal. RMSE: {metrics.get('calibration_rmse', 0):.5f}
                        </div>
                    </div>"""

        html += """
                </div>
                <div style="margin-top: 16px; font-size: 0.85em; color: var(--text-secondary); line-height: 1.8; background: var(--bg-primary); padding: 12px; border-radius: 6px;">
                    <strong>Read AUC first.</strong> The slate ranks hitters and takes the top six,
                    so ordering is what matters; AUC asks whether home runs were ranked above
                    non-home-runs. Per-PA Brier and log loss are dominated by the ~3% base rate
                    and barely separate the models. Metrics are per plate appearance, while the
                    probabilities above are per game.
                </div>
            </div>"""

    espn = slate_data.get("espn")
    if espn:
        excluded = slate_data.get("excluded_injured") or []
        coverage = espn.get("bio_coverage", {})
        sample = ", ".join(
            f"{e['batter']} ({e['status']})" for e in excluded[:8]
        )
        more = f" +{len(excluded) - 8} more" if len(excluded) > 8 else ""
        html += f"""
            <div class="analytics" style="margin-top: 20px;">
                <h3 style="font-size: 1.2em; margin-bottom: 16px;">ESPN Data</h3>
                <div class="analytics-grid">
                    <div class="analytics-item">
                        <div class="analytics-value">{espn.get('excluded_injured', 0)}</div>
                        <div class="analytics-label">Injured players excluded</div>
                    </div>
                    <div class="analytics-item">
                        <div class="analytics-value">{espn.get('injuries', 0)}</div>
                        <div class="analytics-label">League injury entries</div>
                    </div>
                    <div class="analytics-item">
                        <div class="analytics-value">{coverage.get('rate', 0):.0%}</div>
                        <div class="analytics-label">Bio match rate ({coverage.get('matched', 0)}/{coverage.get('total', 0)})</div>
                    </div>
                </div>
                <div style="margin-top: 16px; font-size: 0.85em; color: var(--text-secondary); line-height: 1.8; background: var(--bg-primary); padding: 12px; border-radius: 6px;">
                    Height, weight and age feed the comparables model; the injury
                    report removes anyone on an IL variant from the pick pool.
                    {f'<br><strong>Excluded today:</strong> {sample}{more}' if sample else ''}
                </div>
            </div>"""

    html += f"""
        </div>

        <footer>
            <p>HR Daily Tracker — PA-based HR probability model with per-batter exposure</p>
            <p style="margin-top: 12px; opacity: 0.7;">{_freshness(slate_data)}</p>
        </footer>
    </div>

    <script>
        // Auto-refresh option
        function refreshPage() {{
            location.reload();
        }}

        // Optional: refresh every 5 minutes
        // setInterval(refreshPage, 5 * 60 * 1000);
    </script>
</body>
</html>
"""
    return html


# Every prior the ensemble carries, in the order they are shown on the model
# page: internal key on the hitter record, column label, and a one-line note on
# what the variant is. The served number is whichever of these the ensemble is
# configured to serve, so it appears twice -- once under its own name and once
# as the highlighted "served" column.
_VARIANTS = [
    ("served", "Served (full season)",
     "KNN comparables refit on every PA to date, through the stacked model -- the published number."),
    ("empirical_bayes", "Flat league", "League HR rate as the prior for every hitter."),
    ("knn", "KNN comps", "Prior is the pooled rate of the nearest comparables."),
    ("svm", "SVR", "Support-vector regression on the same feature matrix."),
    ("rf", "Random forest", "Forest regression; a learned similarity metric."),
    ("core", "Core (KNN+SVR)", "Blend of KNN and SVR, no forest."),
    ("forest", "Forest blend", "KNN + SVR + forest, weights fitted out-of-fold."),
    ("linear", "Linear regression", "Ordinary least squares on the same features."),
    ("logistic", "Logistic regression",
     "Binomial logistic on the per-PA home run outcome."),
    ("neural", "Neural net (torch)",
     "Two-layer PyTorch MLP, PA-weighted, trained on the same features."),
    ("xgboost", "XGBoost",
     "Gradient-boosted trees, depth 3 with subsampling and L2."),
]

# Which of the above the ensemble's serve_variant name corresponds to.
_SERVED_KEY_FOR = {
    "league": "empirical_bayes", "knn": "knn", "svr": "svm",
    "rf": "rf", "core": "core", "ensemble": "forest", "served": "served",
}


def _side_vs_starter(pick: dict) -> str:
    """The side a hitter bats from against today's starter (a switch hitter
    turns around to face him)."""
    side = pick.get("bat_side") or "R"
    if side == "S":
        return "R" if pick.get("facing_hand") == "L" else "L"
    return side

_METRIC_COLUMNS = [
    ("auc", "AUC", "{:.4f}"),
    ("top_decile_lift", "Top-decile lift", "{:.2f}&times;"),
    ("brier", "Brier", "{:.5f}"),
    ("log_loss", "Log loss", "{:.5f}"),
    ("calibration_rmse", "Cal. RMSE", "{:.5f}"),
]


def _all_hitters(slate_data: dict) -> list:
    """Every eligible hitter on the slate, flattened out of the games."""
    return [h for g in slate_data.get("games", []) for h in g.get("hitters", [])]


def _served_key(slate_data: dict) -> str:
    """Which variant column the slate's served probability came from."""
    hitters = _all_hitters(slate_data)
    name = next(
        (h.get("prob_served_variant") for h in hitters if h.get("prob_served_variant")),
        None,
    )
    # Falling back to the ensemble's own default keeps the highlighted column
    # honest when an older slate carries no served-variant marker.
    return _SERVED_KEY_FOR.get(name, "core")


def _present_variants(slate_data: dict) -> list:
    """Variants this slate actually carries numbers for.

    An older model object, or the plain empirical-Bayes fallback used when the
    ensemble could not be fitted, produces fewer columns. Rendering only what
    exists keeps the page honest rather than printing a wall of zeros.
    """
    hitters = _all_hitters(slate_data)
    metrics = slate_data.get("ensemble_metrics") or {}
    return [
        v for v in _VARIANTS
        if any(f"prob_{v[0]}" in h for h in hitters) or v[0] in metrics
    ]


def _metrics_table(metrics: dict, variants: list, served: str) -> str:
    """Held-out scores, one row per prior."""
    head = "".join(f"<th>{label}</th>" for _, label, _ in _METRIC_COLUMNS)
    rows = ""
    for key, label, blurb in variants:
        m = metrics.get(key) or {}
        if not m:
            continue
        cells = "".join(
            f"<td>{fmt.format(m.get(mk, 0))}</td>" for mk, _, fmt in _METRIC_COLUMNS
        )
        tag = ' <span class="cmp-meta">(served)</span>' if key == served else ""
        rows += f"""
                        <tr>
                            <td><span class="cmp-name">{label}</span>{tag}
                                <div class="cmp-meta">{blurb}</div></td>
                            {cells}
                        </tr>"""
    if not rows:
        return ""
    return f"""
            <div class="table-wrap">
                <table class="cmp-table">
                    <thead><tr><th>Prior</th>{head}</tr></thead>
                    <tbody>{rows}
                    </tbody>
                </table>
            </div>"""


def _variant_top6(hitters: list, key: str) -> list:
    """That variant's own top six, ranked by its own probability."""
    scored = [h for h in hitters if h.get(f"prob_{key}") is not None]
    scored.sort(key=lambda h: h[f"prob_{key}"], reverse=True)
    return scored[:6]


def _agreement_section(slate_data: dict, variants: list, served: str) -> str:
    """How much each variant's own top six overlaps the served slate.

    This is the question the metrics table cannot answer: two priors can score
    almost identically per plate appearance and still hand you a different six
    names, and the six names are the only output anyone acts on.
    """
    hitters = _all_hitters(slate_data)
    if not hitters:
        return ""

    served_names = {h["batter"] for h in _variant_top6(hitters, served)}
    cards = ""
    for key, label, _ in variants:
        top = _variant_top6(hitters, key)
        if not top:
            continue
        overlap = len(served_names.intersection({h["batter"] for h in top}))
        listing = ""
        for h in top:
            line = h.get("result") or {}
            # A name that already homered is marked here rather than only in
            # the table below, so the card reads as a scorecard once games end.
            mark = " &#10003;" if line.get("hr") else ""
            listing += (
                f'<div class="ctx-row">'
                f'<span class="{"rank-in" if h["batter"] in served_names else "rank-out"}">'
                f'{h["batter"]}{mark}</span>'
                f'<strong>{h["prob_" + key]:.1%}</strong></div>'
            )
        # Only finished games count: a hitter still batting is not yet a miss.
        final = [h for h in top if (h.get("result") or {}).get("final")]
        hit = sum(1 for h in final if h["result"]["hr"])
        scoreline = (
            f'<span class="res {"res-hit" if hit else "res-miss"}">'
            f'{hit}/{len(final)} hit</span>' if final else ""
        )
        cards += f"""
                    <div class="analytics-item">
                        <div style="font-size: 0.9em; margin-bottom: 8px;">
                            <strong>{label}</strong>
                            <span class="cmp-meta">&middot; {overlap}/6 shared</span>
                            {scoreline}
                        </div>
                        {listing}
                    </div>"""

    return f"""
            <div class="agree-grid">{cards}
            </div>
            <div class="note-box">
                Each card is that prior's own top six, ranked by its own number.
                Names in green also appear in the served prior's six; the count
                is the overlap. Priors that look alike on aggregate metrics can
                still disagree on the names, and that disagreement is the part
                that changes what you would play.
            </div>"""


def _side_by_side_table(
    slate_data: dict, variants: list, served: str, limit: int
) -> str:
    """Per-hitter probabilities from every prior, ranked by the served one."""
    hitters = sorted(
        _all_hitters(slate_data), key=lambda h: h.get("prob_hr", 0), reverse=True
    )[:limit]
    if not hitters:
        return ""

    pick_names = {p["batter"] for p in slate_data.get("picks_6", [])}
    scored = any(h.get("result") for h in hitters)
    head = "".join(
        f'<th class="{"served" if key == served else ""}">{label}</th>'
        for key, label, _ in variants
    ) + ("<th>Result</th>" if scored else "")
    rows = ""
    for i, h in enumerate(hitters, 1):
        cells = ""
        for key, _, _ in variants:
            value = h.get(f"prob_{key}")
            cls = "served" if key == served else ""
            cells += (
                f'<td class="{cls}">{value:.1%}</td>' if value is not None
                else f'<td class="{cls}">&ndash;</td>'
            )
        if scored:
            cells += f'<td>{_result_badge(h) or "&ndash;"}</td>'
        badge = (
            ' <span class="slot-badge">slate</span>'
            if h["batter"] in pick_names else ""
        )
        hand = "LHP" if h.get("facing_hand") == "L" else "RHP"
        rows += f"""
                        <tr>
                            <td>{i}. <span class="cmp-name">{h['batter']}</span>{badge}
                                <div class="cmp-meta">{h.get('team', '')} &middot; vs {hand}
                                    &middot; {h.get('season_hrs', 0)} HR /
                                    {h.get('season_pas', 0)} PA</div></td>
                            {cells}
                        </tr>"""

    return f"""
            <div class="table-wrap">
                <table class="cmp-table">
                    <thead><tr><th>Hitter</th>{head}</tr></thead>
                    <tbody>{rows}
                    </tbody>
                </table>
            </div>
            <div class="note-box">
                Per-game probabilities: each prior's per-PA rate run through this
                hitter's own plate appearances, park, weather, bullpen and
                times-through-the-order terms. Only the prior changes across the
                columns &mdash; every other term is held fixed, so a left-to-right
                spread is the prior disagreeing about the hitter, not the context.
            </div>"""


def render_models_html(slate_data: dict, limit: int = 40) -> str:
    """Model Lab: every prior's numbers side by side, for the same slate.

    The slate page shows one probability per hitter. This page shows the same
    hitters under all of the priors the ensemble fits, plus the held-out scores
    that decided which one is served.
    """
    date_str = slate_data["date"]
    metrics = slate_data.get("ensemble_metrics") or {}
    variants = _present_variants(slate_data)
    served = _served_key(slate_data)
    served_label = next(
        (label for key, label, _ in _VARIANTS if key == served), served
    )
    n_test = metrics.get("n_test_samples", 0)

    if not variants:
        body = """
        <div class="section">
            <h2 class="section-title">No model comparison available</h2>
            <div class="note-box">
                This slate was built without the ensemble, so there is only one
                set of probabilities to show. The slate page has them.
            </div>
        </div>"""
    else:
        cutoff = metrics.get("cutoff_date", "n/a")
        holdout_rate = metrics.get("holdout_hr_rate", 0)

        low_pa = metrics.get("low_pa") or {}
        low_pa_block = ""
        if low_pa:
            low_pa_block = f"""
        <div class="section" id="thin">
            <h2 class="section-title">&#128300; Thin-Sample Subset</h2>
            <div class="note-box" style="margin-top: 0; margin-bottom: 16px;">
                The same held-out window, restricted to hitters with fewer than
                {low_pa.get('threshold_pa', 0)} plate appearances before the cutoff
                ({low_pa.get('n_test_samples', 0):,} PA, actual HR rate
                {low_pa.get('holdout_hr_rate', 0):.2%}). For a regular with 600 PA the
                observed record swamps any prior, so the priors mostly separate here.
            </div>
            {_metrics_table(low_pa, variants, served)}
        </div>"""

        weights = metrics.get("prior_weights") or {}
        k_shrink = metrics.get("shrinkage_k") or {}
        config = ""
        if weights or k_shrink:
            weight_str = ", ".join(f"{k.upper()} {v:.2f}" for k, v in weights.items())
            k_str = ", ".join(f"{k} {v:.0f}" for k, v in k_shrink.items())
            config = f"""
            <div class="note-box">
                <strong>Blend weights</strong> (forest variant): {weight_str or 'n/a'}<br>
                <strong>Shrinkage K</strong> per prior: {k_str or 'n/a'} &mdash; estimated
                by method of moments on out-of-fold residuals, not hand-tuned.
            </div>"""

        body = f"""
        <div class="section" id="scores">
            <h2 class="section-title">&#128200; Held-Out Scores</h2>
            <div class="note-box" style="margin-top: 0; margin-bottom: 16px;">
                Fitted on plate appearances before {cutoff}, scored on the {n_test:,}
                that came after (actual HR rate {holdout_rate:.2%}). No outcome in the
                test window was visible during fitting. Metrics are per plate
                appearance; the probabilities lower down are per game.
                <br><strong>Read AUC first</strong> &mdash; the slate ranks hitters and
                takes the top six, so ordering is what matters. Brier and log loss are
                dominated by the ~3% base rate and barely separate the priors.
            </div>
            {_metrics_table(metrics, variants, served)}
            {config}
        </div>
        {low_pa_block}
        <div class="section" id="agreement">
            <h2 class="section-title">&#129309; Slate Agreement</h2>
            {_agreement_section(slate_data, variants, served)}
        </div>

        <div class="section" id="hitters">
            <h2 class="section-title">&#9878; Hitter by Hitter</h2>
            {_side_by_side_table(slate_data, variants, served, limit)}
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Model Lab &mdash; {date_str}</title>
    <style>{_CSS}    </style>{_THEME_SCRIPT}
</head>
<body>
    <div class="container">
        <header>
            <h1>&#9878; Model Lab</h1>
            <p class="subtitle">Every prior, same slate &mdash; {date_str}</p>
            <div class="stats-bar">
                <div class="stat-card">
                    <div class="stat-label">Served Prior</div>
                    <div class="stat-value" style="font-size: 1.3em;">{served_label}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Priors Compared</div>
                    <div class="stat-value">{len(variants)}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Held-Out PA</div>
                    <div class="stat-value">{n_test:,}</div>
                </div>
            </div>
            {_tabs("models")}
            {_section_nav(
                [("scores", "📈 Held-Out Scores")]
                + ([("thin", "🔬 Thin Sample")] if metrics.get("low_pa") else [])
                + [("agreement", "🤝 Slate Agreement"), ("hitters", "⚖️ Hitter by Hitter")]
            ) if variants else ''}
        </header>
        {body}

        <footer>
            <p>HR Daily Tracker &mdash; model comparison</p>
            <p style="margin-top: 12px; opacity: 0.7;">{_freshness(slate_data)}</p>
        </footer>
    </div>
</body>
</html>
"""
