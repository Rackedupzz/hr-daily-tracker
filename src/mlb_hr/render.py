"""The slate page ("/") and the Model Lab ("/models").

Markup only; the stylesheet, page shell and shared pieces live in `mlb_hr.ui`.
"""
from __future__ import annotations

from html import escape

from mlb_hr.ui import (
    abbr, alert, avatar, chance, color, fair_odds, hand, head, hero, is_print, logo, mini, more, page,
    pct, person, ring, short, status_chip, subnav, tile, tiles, winbar,
)


def _hand_badge(h: str | None) -> str:
    """RHP/LHP badge for a pitcher. Handedness drives the platoon split, so it
    belongs next to the name rather than buried in the model."""
    return hand(h)


def _games_by_pk(slate_data: dict) -> dict:
    return {g.get("game_pk"): g for g in slate_data.get("games", [])}


def _opponent(h: dict, games: dict) -> tuple:
    """(opposing team, opposing starter, his hand) for a hitter."""
    g = games.get(h.get("game_pk")) or {}
    side = h.get("side")
    if side not in ("home", "away"):
        return None, None, h.get("facing_hand")
    other = "away" if side == "home" else "home"
    return g.get(other), g.get(f"{other}_sp"), g.get(f"{other}_sp_hand") or h.get("facing_hand")


def _vs_line(h: dict, games: dict) -> str:
    opp, sp, sp_hand = _opponent(h, games)
    team = h.get("team")
    lead = f"{logo(team, 'sm')}{abbr(team)}"
    if opp:
        lead += f" {'vs' if h.get('side') == 'home' else '@'} {abbr(opp)}"
    if sp:
        return f"{lead} &middot; {escape(sp)} {hand(sp_hand)}"
    return f"{lead} &middot; vs {hand(h.get('facing_hand'))}"


def _inj(h: dict) -> str:
    return f'<span class="inj">{escape(str(h["injury_note"]))}</span>' if h.get("injury_note") else ""


# ------------------------------------------------------------------ verdicts
def _verdict(entry: dict, key: str = "hr") -> tuple:
    """Did this projection come in? Returns (tone, label, detail).

    The page publishes a probability in the morning and the box score answers
    it at night; this is the one place that decides which of the two states a
    projection is in, so the card, the badge and the summary cannot disagree.
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


def _badge(entry: dict, key: str = "hr") -> str:
    """The outcome pill on a pick card: what the box score says so far."""
    tone, label, detail = _verdict(entry, key)
    if not tone:
        return ""
    return f'<span class="badge {tone}">{label} <small>&middot; {escape(str(detail))}</small></span>'


def _result_badge(entry: dict, key: str = "hr") -> str:
    """Compact hit / miss / in-progress badge for a row.

    A pick is only a miss once his game is final -- a hitless third inning is
    not a wrong prediction yet -- so an unfinished game shows the line so far
    rather than a verdict.
    """
    line = entry.get("result")
    if not line:
        return ""
    got = line.get(key, 0)
    detail = escape(str(line.get("summary") or f"{line.get('hits', 0)}-for-{line.get('ab', 0)}"))
    if line.get("dnp"):
        return f'<span class="badge void">{line.get("detailed")}</span>'
    if got:
        label = f"{got} HR" if key == "hr" else f"{got} H"
        return f'<span class="badge hit" title="{detail}">&#10003; {label}</span>'
    if line.get("final"):
        return f'<span class="badge miss" title="{detail}">&#10007; {detail}</span>'
    return f'<span class="badge live" title="{line.get("detailed") or ""}">&#9679; {detail}</span>'


def _tone(entry: dict, key: str = "hr") -> str:
    t = _verdict(entry, key)[0]
    return {"hit": " hit", "miss": " miss"}.get(t, "")


# ------------------------------------------------------------- pick cards
def _mult(label: str, value: float, fmt: str = "{:.2f}&times;", neutral: float = 1.0) -> str:
    tone = " up" if value > neutral * 1.02 else (" down" if value < neutral * 0.98 else "")
    return f'<div class="mult{tone}"><b>{fmt.format(value)}</b>{label}</div>'


def _side_vs_starter(pick: dict) -> str:
    """The side a hitter bats from against today's starter (a switch hitter
    turns around to face him)."""
    side = pick.get("bat_side") or "R"
    if side == "S":
        return "R" if pick.get("facing_hand") == "L" else "L"
    return side


def _split_rows(pick: dict) -> str:
    rows = ""
    for h in ("R", "L"):
        hr, pa = pick.get(f"hr_vs_{h.lower()}hp", 0), pick.get(f"pa_vs_{h.lower()}hp", 0)
        on = ' class="on"' if pick.get("facing_hand") == h else ""
        rate = f" &middot; {hr / pa:.1%}" if pa else ""
        rows += f"<dt{on}>vs {h}HP</dt><dd{on}>{hr} HR &middot; {pa} PA{rate}</dd>"
    return rows


def _hr_card(i: int, pick: dict, games: dict) -> str:
    p = pick["prob_hr"]
    side = pick.get("bat_side")
    bats = " &middot; switch-hits" if side == "S" else (f" &middot; bats {side}" if side else "")
    slot = f"<span class='chip'>bats <b>#{pick['lineup_slot']}</b></span>" if pick.get("lineup_slot") else ""
    mults = (_mult(f"Park vs {_side_vs_starter(pick)}HB", pick.get("prob_park_side", 1))
             + _mult("Park &times; pull", pick.get("prob_park_effective", 1))
             + _mult("Weather", pick.get("prob_weather_factor", 1))
             + _mult("Opp bullpen", pick.get("prob_pen_hr_index", 1))
             + f'<div class="mult"><b>{pick.get("prob_pull_rate", 0):.0%}</b>Pull rate</div>')
    if pick.get("matchup"):
        m = pick["matchup"]
        mults += f'<div class="mult"><b>{m["hits"]}-for-{m["pa"]}</b>Vs this starter</div>'
    pid = pick.get("batter_id") or i
    return f"""<article class="card pick{_tone(pick)}" style="--tc:{color(pick.get('team'))}">
<div class="pick-top">{avatar(pick.get('batter_id'), pick['batter'], pick.get('team'))}<div class="who"><div class="name">{escape(pick['batter'])}{_inj(pick)}</div>
<div class="sub">{_vs_line(pick, games)}</div></div><div class="rank">#{i}</div></div>
<div class="pick-main">{ring(p)}<div><div class="big">{pct(p)}</div><div class="unit">chance to homer</div>
<div class="chips"><span class="chip">fair <b>{fair_odds(p)}</b></span>{slot}</div></div></div>
<dl class="kv"><dt>Per PA &times; PA</dt><dd>{pick.get('prob_ensemble_per_pa', 0):.2%} &times; {pick.get('prob_expected_pa', 0):.1f}</dd>
<dt>PA vs starter / pen</dt><dd>{pick.get('prob_pa_vs_sp', 0):.1f} / {pick.get('prob_pa_vs_pen', 0):.1f}</dd>
{_split_rows(pick)}<dd class="full">Season: {pick['season_hrs']} HR in {pick['season_pas']} PA{bats}</dd></dl>
<details class="why" id="why-hr-{pid}"><summary>Context multipliers</summary><div class="mults">{mults}</div></details>
{_badge(pick)}</article>"""


def _hit_card(i: int, pick: dict, games: dict) -> str:
    proj = pick["hits_proj"]
    hand_l = "LHP" if pick.get("facing_hand") == "L" else "RHP"
    p1 = proj["prob_at_least_one"]
    h2h = ""
    if pick.get("matchup"):
        m = pick["matchup"]
        h2h = f"<dt>Head-to-head</dt><dd>{m['hits']}-for-{m['pa']} &middot; {m['hit_rate']:.3f}</dd>"
    slot = f"<span class='chip'>bats <b>#{pick['lineup_slot']}</b></span>" if pick.get("lineup_slot") else ""
    return f"""<article class="card pick{_tone(pick, 'hits')}" style="--tc:{color(pick.get('team'))}">
<div class="pick-top">{avatar(pick.get('batter_id'), pick['batter'], pick.get('team'))}<div class="who"><div class="name">{escape(pick['batter'])}{_inj(pick)}</div>
<div class="sub">{_vs_line(pick, games)}</div></div><div class="rank">#{i}</div></div>
<div class="pick-main">{ring(p1, 'var(--s1)')}<div><div class="big">{proj['projected_hits']:.2f}<span class="unit">&nbsp;hits</span></div>
<div class="unit">{p1:.0%} chance of at least one</div>
<div class="chips"><span class="chip">fair 1+ <b>{fair_odds(p1)}</b></span>{slot}</div></div></div>
<dl class="kv"><dt>Season hit rate</dt><dd>{proj['hitter_rate']:.1%}</dd>
<dt>vs starter ({hand_l})</dt><dd>{proj['rate_vs_sp']:.1%} &times; {proj['pa_vs_sp']:.1f} PA</dd>
<dt>vs bullpen</dt><dd>{proj['rate_vs_pen']:.1%} &times; {proj['pa_vs_pen']:.1f} PA</dd>
<dt>Starter allows</dt><dd>{proj['sp_hit_rate']:.1%}</dd>{h2h}</dl>
{_badge(pick, 'hits')}</article>"""


def _render_hits_section(hit_picks: list, games: dict | None = None) -> str:
    """Projected hits, the same matchup logic applied to contact instead of power."""
    if not hit_picks:
        return ""
    cards = "".join(_hit_card(i, p, games or {}) for i, p in enumerate(hit_picks, 1))
    note = ("Expected hits from the hitter's rate and his contact quality, the starter's hits allowed "
            "and strikeout rate, the bullpen, the park, and his plate appearances for his lineup slot, "
            "each weighted by what the 2026 season replay showed it is worth.")
    return f'<section class="section" id="hits">{head("Projected Hits", "&#129358;", note)}<div class="grid">{cards}</div></section>'


# ---------------------------------------------------------------- parlays
def _chance(prob: float) -> str:
    """A ticket's chance: a percentage, or '1 in N' once it is too small to read."""
    return chance(prob)


def _leg_kind(leg: dict) -> str:
    """The leg's market as a short tag: HR, 1+ HIT, 2+ HITS, WIN."""
    if leg["type"] == "hr":
        return "HR"
    if leg["type"] == "win":
        return "WIN"
    n = int(leg.get("line") or 1)
    return f"{n}+ HIT{'S' if n != 1 else ''}"


def _dk(price) -> str:
    """' &middot; DK +450' for a DraftKings price (an int); HOMER's older
    tickets carry a modeled estimate as text, which is not shown as the book's."""
    return f" &middot; DK {price:+d}" if isinstance(price, int) else ""


def _ticket_leg(leg: dict, extra: str = "") -> str:
    st = leg.get("status", "pending")
    mark = {"won": "&#10003;", "lost": "&#10007;"}.get(st, "")
    line = (leg.get("result") or {}).get("summary") or ""
    void = " (void)" if st == "void" else ""
    if leg["type"] == "win":
        sub = f"over {escape(leg.get('opp') or '')}{_dk(leg.get('book_price'))}{' &middot; ' + escape(line) if line else ''}{extra}"
        face = logo(leg.get("team"), "sm")
    else:
        sp = leg.get("opp_sp") or leg.get("sp_name")
        vs = f"vs {escape(sp)} {hand(leg.get('facing_hand'))}" if sp else f"vs {hand(leg.get('facing_hand'))}"
        sub = f"{abbr(leg.get('team', ''))} &middot; {vs}{_dk(leg.get('book_price'))}{' &middot; ' + escape(line) if line else ''}{extra}"
        face = avatar(leg.get("batter_id"), leg["batter"], leg.get("team"), "xs")
    tone = "" if leg["type"] == "hr" else " hit"
    return (f'<li class="leg {st}"><span class="st">{mark}</span>{face}'
            f'<span class="lp"><b><span class="ltype{tone}">{_leg_kind(leg)}</span>{escape(leg["batter"])}{void}</b>'
            f'<small>{sub}</small></span><span class="lpct">{leg["prob"]:.0%}</span></li>')


_TICKET_BADGES = {"won": ("&#10003; CASHED", "hit"), "lost": ("&#10007; LOST", "miss"),
                  "void": ("VOID", "void"), "pending": ("&#9679; OPEN", "live")}


def _ticket_card(play: dict) -> str:
    legs = "".join(_ticket_leg(leg) for leg in play["leg_list"])
    status = play.get("status", "pending")
    text, cls = _TICKET_BADGES.get(status, _TICKET_BADGES["pending"])
    ev = play.get("evidence") or {}
    record = ""
    if ev.get("days"):
        expected = (f" (model expected {ev['expected_wins']:.1f})"
                    if ev.get("expected_wins") is not None else "")
        record = (f"Season replay: built this way every day, cashed <b>{ev['won']}</b> of {ev['days']} days"
                  f"{expected}.")
    late = "<div>Built after the fact from that morning's projections.</div>" if play.get("backfilled") else ""
    if play.get("note"):
        late += f"<div>{escape(play['note'])}</div>"
    long = " long" if play["prob"] < 0.10 else ""
    if play.get("book_decimal"):
        payout = f"<b>${10 * play['book_decimal']:,.0f}</b> at DK"
    elif play["prob"] > 0:
        payout = f"<b>${10 / play['prob']:,.0f}</b> fair"
    else:
        payout = "<b>&mdash;</b> fair"
    if play.get("book_odds"):
        book = f"<span class='chip'>DraftKings <b>{play['book_odds']}</b></span>"
    elif play.get("book_priced"):
        book = f"<span class='chip'>DK priced <b>{play['book_priced']}</b> of {len(play['leg_list'])} legs</span>"
    else:
        book = ""
    return f"""<article class="card ticket{long} st-{status}">
<div class="thead"><div><div class="tname">{play['name']}</div><div class="blurb">{play['blurb']}</div></div>
<div class="tchance"><b>{_chance(play['prob'])}</b><small>chance</small></div></div>
<div class="tstats"><span class="chip"><b>{len(play['leg_list'])}</b> legs</span><span class="chip">fair <b>{play['fair_odds']}</b></span>{book}
<span class="chip">$10 pays {payout}</span></div>
<ol class="legs">{legs}</ol><div class="tfoot">{f'<div>{record}</div>' if record else ''}{late}<span class="badge {cls}">{text}</span></div></article>"""


def _parlay_replay_table(replay: dict) -> str:
    """Every house rule rebuilt on every replayed day: offered, below the bar, or a long shot."""
    from mlb_hr.parlays import CORE_MIN_CASHED, PARLAYS, qualifies
    rows = ""
    for spec in PARLAYS:
        r = (replay or {}).get(spec["key"])
        if not r or not r.get("days"):
            continue
        status = ("<span class='chip'>long shot</span>" if spec.get("longshot")
                  else "<span class='chip'><b>&#10003; offered</b></span>" if qualifies(spec, r)
                  else f"<span class='chip'>below {CORE_MIN_CASHED}</span>")
        legs = f"{r['leg_rate']:.0%} / {r['leg_projected']:.0%}" if r.get("leg_rate") is not None else "&mdash;"
        rows += (f"<tr><td><b>{escape(spec['name'])}</b></td><td class='n'>{r['legs']}</td><td class='n'>{r['days']}</td>"
                 f"<td class='n'><b>{r['won']}</b></td><td class='n'>{r['expected_wins']:.1f}</td>"
                 f"<td class='n'>{(r['rate'] or 0):.1%}</td><td class='n'>{(r['predicted'] or 0):.1%}</td>"
                 f"<td class='n'>{legs}</td><td>{status}</td></tr>")
    if not rows:
        return ""
    note = ("Every ticket rebuilt by the live rules on every day of the walk-forward 2026 replay, from projections "
            f"that had only seen earlier days. A core ticket is offered only if it cashed {CORE_MIN_CASHED}+ times. "
            "The replay has no past prop prices (ESPN keeps none), so it scores the model's legs without "
            "DraftKings' lines and prices or the injury report, and prices winners without the park factor.")
    return (head("House parlays in the replay", "", note, tag="h3")
            + "<div class='card scroll'><table class='t'><tr><th>Ticket</th><th class='n'>Legs</th><th class='n'>Days</th>"
              "<th class='n'>Cashed</th><th class='n'>Model expected</th><th class='n'>Actual rate</th>"
              "<th class='n'>Predicted rate</th><th class='n'>Legs landed / projected</th><th></th></tr>"
            + rows + "</table></div>")


def _render_parlays_section(slate_data: dict, title: str = "House Parlays", sid: str = "parlays",
                            lead: str = "", icon: str = "&#127903;") -> str:
    """The house parlays (mlb_hr.parlays): core tickets, the long shots, and
    tickets posted under earlier rules, graded as games finish. HOMER's page
    shows his own (same rules, his numbers) through this with its own title."""
    from mlb_hr.parlays import CORE_MIN_CASHED, CORE_MIN_LEGS
    plays = slate_data.get("parlays") or []
    replay = slate_data.get("parlay_replay") or {}
    if not plays and not replay:
        return ""
    core = [p for p in plays if not p.get("longshot") and not p.get("retired")]
    longshots = [p for p in plays if p.get("longshot") and not p.get("retired")]
    retired = [p for p in plays if p.get("retired")]
    backfilled = any(p.get("backfilled") for p in plays)
    note = (lead + f"Core tickets: at least {CORE_MIN_LEGS} legs, one per game, and only rules that cashed "
            f"{CORE_MIN_CASHED} or more times when rebuilt on every day of the walk-forward season replay. Each "
            "game's single likeliest leg, then the best of those, so the legs are independent and the ticket's "
            "chance is the product of its legs. Every leg is a line DraftKings posts (as ESPN carries it) for a "
            "hitter cleared by the injury report, with DraftKings' price where ESPN shows one; a scratched leg is "
            "void and the rest ride. Tickets lock the moment any of their games starts. Fair odds are the price a "
            "ticket needs to break even; parlays carry a steep house edge &mdash; never stake what you cannot "
            "afford to lose."
            + (" This day was over before the parlays existed, so they were built from that "
               "morning&rsquo;s projections and graded against what happened." if backfilled else ""))
    from mlb_hr.parlays import PARLAYS, qualifies
    earned = [spec for spec in PARLAYS if not spec.get("longshot") and qualifies(spec, replay.get(spec["key"]))]
    empty = ("<div class='card'>No core ticket can be built from the games still to start.</div>" if earned else
             f"<div class='card'>No core ticket today: no core rule has cashed {CORE_MIN_CASHED} or more times in the "
             "season replay yet (see the replay table below). The long shots still post.</div>")
    parts = [f'<section class="section" id="{sid}">{head(title, icon, note)}'
             f'<div class="grid wide">{"".join(_ticket_card(p) for p in core) or empty}</div>']
    if longshots:
        parts.append(head("Long shots: +1000 or more", "", "Ten legs from one first-pitch window at DraftKings' "
                          "lines and prices. "
                          f"These pay +1000 or more and do <b>not</b> meet the {CORE_MIN_CASHED}-cash bar "
                          "&mdash; their replay record is on each card.", tag="h3")
                     + f'<div class="grid wide">{"".join(_ticket_card(p) for p in longshots)}</div>')
    if retired:
        parts.append(head("Posted earlier, still settling", "", "Tickets posted under the earlier rules before "
                          "their games started. They stay here until they settle.", tag="h3")
                     + f'<div class="grid wide">{"".join(_ticket_card(p) for p in retired)}</div>')
    parts.append(_parlay_replay_table(replay))
    parts.append("</section>")
    return "".join(parts)


# --------------------------------------------------------------- matchups
def _render_matchups_section(matchups: list, games: dict | None = None) -> str:
    """Notable batter-vs-pitcher histories, with the sample-size caveat."""
    if not matchups:
        return ""
    rows = ""
    for h in matchups:
        m = h["matchup"]
        tone = "ok" if m["verdict"] == "hot" else "bad"
        rows += (f"<tr><td>{person(h.get('batter_id'), h['batter'], h.get('team'), abbr(h.get('team')))}</td>"
                 f"<td>{escape(str(m.get('pitcher') or 'starter'))}</td><td class='n {tone}'>{m['hits']}-for-{m['pa']}</td>"
                 f"<td class='n'>{m['hit_rate']:.3f}</td><td class='n'>{m['baseline_hit_rate']:.3f}</td>"
                 f"<td class='n'>{m['home_runs']}</td><td class='n'>{m['strikeouts']}</td></tr>")
    caveat = ("The largest batter-vs-pitcher sample all season is 13 plate appearances, and only about 520 "
              "pairs reach even 8. At that size the standard error on a hit rate is roughly .16, wider than "
              "the entire spread of true talent between major-league hitters. A 4-for-9 line is noise that "
              "looks like a trend, so it is shown with its sample size attached and deliberately kept out of "
              "the model.")
    return f"""<section class="section" id="bvp">{head("Batter vs Pitcher", "&#128269;",
        "Hitters with a notable line against today's opposing starter. <b>Context only &mdash; this feeds none of the projections.</b>")}
<div class="card scroll"><table class="t"><thead><tr><th>Hitter</th><th>Starter</th><th class="n">Line</th><th class="n">Rate</th>
<th class="n">Season rate</th><th class="n">HR</th><th class="n">K</th></tr></thead><tbody>{rows}</tbody></table></div>
<div style="margin-top:10px">{more("Why it is kept out of the model", caveat)}</div></section>"""


# ---------------------------------------------------------- still to play
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
                    "batter": h["batter"], "batter_id": h.get("batter_id"), "team": h.get("team"),
                    "prob_hr": h.get("prob_hr", 0),
                    "facing_hand": h.get("facing_hand"),
                    "season": f"{h.get('season_hrs', 0)} HR / {h.get('season_pas', 0)} PA",
                    "result": h.get("result"),
                }
                for h in by_hr[:per_game]
            ],
            "hits": [
                {
                    "batter": h["batter"], "batter_id": h.get("batter_id"), "team": h.get("team"),
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


def _ghead(away, home, mid: str, away_cls: str = "", home_cls: str = "") -> str:
    return (f'<div class="ghead"><div class="tm {away_cls}">{logo(away)}<span>{escape(short(away))}</span></div>'
            f'<div class="c">{mid}</div><div class="tm h {home_cls}">{logo(home)}<span>{escape(short(home))}</span></div></div>')


def _sps(g: dict) -> str:
    return (f'<div class="sps"><div>{escape(str(g.get("away_sp") or "TBD"))} {hand(g.get("away_sp_hand"))}</div>'
            f'<div>{hand(g.get("home_sp_hand"))} {escape(str(g.get("home_sp") or "TBD"))}</div></div>')


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
            mid = (f'<b class="live">{g["away_score"]}&ndash;{g["home_score"]}</b>'
                   f'&#9679; {escape(str(g["inning"] or g["detailed"]))}')
        else:
            mid = f'<b>@</b>{escape(str(g["detailed"]))}'
        hr_rows = "".join(
            f'<div class="rowx">{person(h.get("batter_id"), h["batter"], h["team"], abbr(h["team"]) + " &middot; " + h["season"], "xs")}'
            f'<span class="r">{_result_badge(h)}{h["prob_hr"]:.1%}</span></div>' for h in g["hr"])
        hit_rows = ""
        for h in g["hits"]:
            mu = h.get("matchup")
            h2h = f' &middot; H2H {mu["hits"]}-for-{mu["pa"]}' if mu else ""
            hit_rows += (f'<div class="rowx">{person(h.get("batter_id"), h["batter"], h["team"], abbr(h["team"]) + h2h, "xs")}'
                         f'<span class="r">{_result_badge(h, "hits")}{h["projected_hits"]:.2f} H '
                         f'<span class="muted" style="font-weight:500">{h["prob_at_least_one"]:.0%}</span></span></div>')
        cards += (f'<article class="card gcard">{_ghead(g["away"], g["home"], mid)}{_sps(g)}'
                  f'<div><div class="label">Home run &mdash; top {len(g["hr"])}</div><div class="rows">{hr_rows}</div></div>'
                  f'<div><div class="label">Hits &mdash; top {len(g["hits"])}</div><div class="rows">{hit_rows}</div></div></article>')
    live_count = sum(1 for g in games if g["state"] == "Live")
    note = (f"{len(games)} game{'s' if len(games) != 1 else ''} not yet final"
            f"{f', {live_count} underway' if live_count else ''} &mdash; home run probability and projected hits "
            "for the hitters who still have at-bats coming. Settled games are left out, so this is the part of "
            "the slate that is still actionable.")
    return f'<section class="section" id="remaining">{head("Still to Play", "&#127769;", note)}<div class="grid wide">{cards}</div></section>'


# ------------------------------------------------------------------ games
def _render_projection(game: dict) -> str:
    """Win/loss projection with the run components that produced it."""
    proj = game.get("projection")
    if not proj:
        return ""
    c = proj["components"]
    hr_, ar = proj["records"]["home"], proj["records"]["away"]
    a, h = game.get("away"), game.get("home")
    note = (f"Indices are multiples of league average ({proj['league_rpg']:.2f} runs/game); below 1.00 "
            f"suppresses runs. The starter carries {c['starter_share']:.0%} of run prevention, the bullpen the "
            f"rest. Park {c['park_runs_factor']:.2f}&times;, home field {c['home_field']:.2f}&times;.")
    return f"""<div><div class="label">Projected outcome</div>
{winbar(a, h, proj['away_win_prob'], proj['home_win_prob'])}
<div class="meta" style="margin-top:8px">Projected {proj['away_expected_runs']:.1f} &ndash; {proj['home_expected_runs']:.1f} &middot; total {proj['total_runs']:.1f}</div>
<div class="scroll"><table class="t" style="margin-top:10px"><thead><tr><th></th><th class="n">{logo(a, 'sm')} {abbr(a)}</th><th class="n">{logo(h, 'sm')} {abbr(h)}</th></tr></thead><tbody>
<tr><td>Record</td><td class="n">{ar['wins']}&ndash;{ar['losses']}</td><td class="n">{hr_['wins']}&ndash;{hr_['losses']}</td></tr>
<tr><td>Run diff</td><td class="n">{ar['run_differential']:+d}</td><td class="n">{hr_['run_differential']:+d}</td></tr>
<tr><td>Pythag win%</td><td class="n">{ar['pythagorean_win_pct']:.3f}</td><td class="n">{hr_['pythagorean_win_pct']:.3f}</td></tr>
<tr><td>Offense</td><td class="n">{c['away_offense_index']:.2f}&times;</td><td class="n">{c['home_offense_index']:.2f}&times;</td></tr>
<tr><td>Starter</td><td class="n">{c['away_sp_index']:.2f}&times;</td><td class="n">{c['home_sp_index']:.2f}&times;</td></tr>
<tr><td>Bullpen</td><td class="n">{c['away_bullpen_index']:.2f}&times;</td><td class="n">{c['home_bullpen_index']:.2f}&times;</td></tr>
</tbody></table></div><p class="note" style="margin:8px 0 0;font-size:12.5px">{note}</p></div>"""


def _render_conditions(game: dict) -> str:
    """Weather and plate umpire for this game."""
    w = game.get("weather") or {}
    if not w or w.get("temp_f") is None:
        return ""
    bits = [f"{w['temp_f']:.0f}&deg;F"]
    if w.get("condition"):
        bits.append(escape(str(w["condition"])))
    if w.get("wind_mph") is not None and w.get("wind_dir"):
        bits.append(f"wind {w['wind_mph']:.0f} mph {escape(str(w['wind_dir']))}")
    if w.get("ump_hp"):
        bits.append(f"HP {escape(str(w['ump_hp']))}")
    factor = w.get("hr_factor", 1.0)
    tone = " up" if factor > 1.02 else (" down" if factor < 0.98 else "")
    return (f'<div class="cond"><span>{" &middot; ".join(bits)}</span>'
            f'<span class="chip{tone}">weather <b>{factor:.2f}&times; HR</b></span></div>')


def _render_sp_strikeouts(game: dict) -> str:
    """Projected strikeouts for both starters, with the terms behind each."""
    rows = ""
    for side in ("away", "home"):
        proj = game.get(f"{side}_sp_k")
        if not proj:
            continue
        team = game.get("home" if side == "away" else "away")  # the lineup he faces
        rows += (f'<div class="prow"><span>{escape(proj["name"])} {hand(proj["hand"])}</span>'
                 f'<span class="v">{proj["projected_k"]:.1f} K <span class="muted" style="font-weight:500">'
                 f'{proj["low"]}&ndash;{proj["high"]}</span></span>'
                 f'<span class="d">{proj["sp_k_rate"]:.1%} K rate &times; {proj["expected_bf"]:.0f} batters faced &middot; '
                 f'{abbr(team)} whiffs {proj["opp_k_rate"]:.1%} vs {proj["hand"]}HP &middot; matchup {proj["matchup_k_rate"]:.1%}</span></div>')
    return f'<div><div class="label">Projected strikeouts</div>{rows}</div>' if rows else ""


def _render_bullpen(game: dict, bullpens: dict, league: dict) -> str:
    """Both bullpens, which cover the plate appearances the starter does not."""
    if not bullpens:
        return ""
    lg_hr = league.get("hr_rate") or 0.03
    rows = ""
    for team in (game.get("away"), game.get("home")):
        pen = bullpens.get(team)
        if not pen:
            continue
        idx = pen["hr_index"]
        tone = "bad" if idx > 1.06 else ("ok" if idx < 0.94 else "")
        rows += (f'<div class="prow"><span>{logo(team, "sm")} {abbr(team)} bullpen</span>'
                 f'<span class="v {tone}">{idx:.2f}&times; HR</span>'
                 f'<span class="d">{pen["hr_rate"]:.2%} HR &middot; {pen["k_rate"]:.1%} K &middot; {pen["bf"]:,} BF</span></div>')
    if not rows:
        return ""
    return (f'<div><div class="label">Bullpens &middot; league {lg_hr:.2%} HR/PA</div>{rows}'
            f'<p class="note" style="margin:6px 0 0;font-size:12.5px">Relief arms throw 43% of all plate '
            f'appearances. The index is home runs allowed against league; above 1.00 helps hitters late.</p></div>')


def _game_mid(game: dict) -> str:
    live = game.get("live") or {}
    state = live.get("state")
    proj = game.get("projection") or {}
    if state in ("Live", "Final"):
        if state == "Final":
            when = "Final"
        else:
            when = f"&#9679; {(live.get('inning_state') or '')[:3]} {live.get('inning_ordinal') or ''}".strip()
        mid = (f'<div class="score{" live" if state == "Live" else ""}">{live.get("away_score", 0)} &ndash; '
               f'{live.get("home_score", 0)}</div><div class="meta">{when}</div>')
    else:
        detailed = live.get("detailed")
        label = detailed if detailed and detailed not in ("Pre-Game", "Warmup") else "Scheduled"
        mid = f'<div class="when">{escape(str(label))}</div>'
    if proj:
        mid += f'<div class="meta">proj {proj["away_expected_runs"]:.1f}&ndash;{proj["home_expected_runs"]:.1f}</div>'
    return mid


def _game_card(game: dict, bullpens: dict, league: dict, pick_ids: set) -> str:
    proj = game.get("projection") or {}
    home_fav = bool(proj) and proj["home_win_prob"] >= 0.5

    def side(team, sp, sp_hand, prob, is_home, fav):
        wp = f' &middot; <span class="wp">{prob:.0%}</span>' if prob is not None else ""
        return (f'<div class="side{" home" if is_home else ""}{" fav" if fav else ""}">{logo(team, "lg")}'
                f'<div><div class="tn">{escape(short(team))}</div><div class="tp">{escape(str(sp or "TBD"))} {hand(sp_hand)}{wp}</div></div></div>')
    summary = (side(game["away"], game.get("away_sp"), game.get("away_sp_hand"), proj.get("away_win_prob"), False,
                    bool(proj) and not home_fav)
               + f'<div class="mid">{_game_mid(game)}</div>'
               + side(game["home"], game.get("home_sp"), game.get("home_sp_hand"), proj.get("home_win_prob"), True, home_fav)
               + '<span class="chev">&#9662;</span>')
    hitters = sorted(game.get("hitters", []), key=lambda h: h.get("prob_hr", 0), reverse=True)[:8]
    top = max((h.get("prob_hr", 0) for h in hitters), default=0) or 1
    rows = ""
    for h in hitters:
        star = ' <span class="pill lean">pick</span>' if h.get("batter_id") in pick_ids else ""
        hp = h.get("hits_proj") or {}
        pa = h.get("pa_vs_facing")
        split = (f"{h.get('hr_vs_facing', 0)} / {pa}" if pa else "&mdash;")
        rows += (f"<tr><td>{person(h.get('batter_id'), h['batter'], h.get('team'), abbr(h.get('team')) + (' &middot; #' + str(h['lineup_slot']) if h.get('lineup_slot') else ''), 'xs')}</td>"
                 f"<td class='n hot'>{h.get('prob_hr', 0):.1%}{mini(h.get('prob_hr', 0) / top)}{star}</td>"
                 f"<td class='n'>{split}</td>"
                 f"<td class='n'>{hp.get('projected_hits', 0):.2f}</td>"
                 f"<td>{_result_badge(h) or ''}</td></tr>")
    left = "".join(x for x in (_render_conditions(game), _render_projection(game), _render_sp_strikeouts(game),
                               _render_bullpen(game, bullpens, league)) if x)
    return f"""<details class="game" id="game-{game.get('game_pk')}"><summary>{summary}</summary>
<div class="gbody"><div class="meta" style="margin-top:12px">{escape(str(game.get('venue', '')))}</div>
<div class="cols"><div class="stack">{left}</div><div class="scroll"><div class="label">Highest HR probabilities</div>
<table class="t"><thead><tr><th>Hitter</th><th class="n">HR</th><th class="n">HR / PA vs hand</th><th class="n">Hits</th><th>Result</th></tr></thead>
<tbody>{rows}</tbody></table></div></div></div></details>"""


# --------------------------------------------------------------- page bits
def _freshness(slate_data: dict) -> str:
    """What is from the cached model build and what was just pulled.

    The two have very different ages -- the slate is refit on a timer, the
    scores come down on every load -- and a single "last updated" line would
    misstate both.
    """
    parts = [f"Slate for {slate_data.get('date', '')}"]
    if slate_data.get("model_version"):
        parts.append(f"model {escape(str(slate_data['model_version']))}")
    if slate_data.get("built_at"):
        parts.append(f"built {slate_data['built_at']}")
    live = (slate_data.get("results") or {}).get("fetched_at")
    if live:
        parts.append(f"live data {live}")
    return "<div>" + " &middot; ".join(parts) + "</div>"


def _live_count(slate_data: dict) -> int:
    return sum(1 for g in slate_data.get("games", []) if (g.get("live") or {}).get("state") == "Live")


def _status(slate_data: dict) -> str:
    live = _live_count(slate_data)
    if live:
        return status_chip(live)
    if slate_data.get("building_note"):
        return status_chip(text="Refitting", busy=True)
    res = slate_data.get("results") or {}
    if res and res.get("final_games") and not res.get("live_games") and not res.get("upcoming_games"):
        return status_chip(text="All final")
    return status_chip()


def _results_tiles(slate_data: dict) -> str:
    """Slate-level scoreboard, beside the forecast numbers."""
    picks = slate_data.get("picks_6") or []
    res = slate_data.get("results") or {}
    exp = sum(p.get("prob_hr", 0) for p in picks)
    cells = [tile(str(slate_data.get("games_count", len(slate_data.get("games", [])))), "games on the slate"),
             tile(f"{exp:.2f}", f"home runs expected from the {len(picks)} picks", tone="accent")]
    if res.get("any"):
        pk, hp = res.get("picks") or {}, res.get("hit_picks") or {}
        if pk.get("scored"):
            cells.append(tile(f"{pk.get('hit', 0)}/{pk['scored']}",
                              "HR picks connected" + (f" &middot; {pk['pending']} open" if pk.get("pending") else ""),
                              pk.get("hit", 0) / pk["scored"], tone="good"))
        if hp.get("scored"):
            cells.append(tile(f"{hp.get('hit', 0)}/{hp['scored']}", "hit picks with a hit",
                              hp.get("hit", 0) / hp["scored"]))
        cells.append(tile(f"{res.get('final_games', 0)}&thinsp;/&thinsp;{res.get('live_games', 0)}&thinsp;/&thinsp;"
                          f"{res.get('upcoming_games', 0)}", "games final / live / to come"))
    else:
        avg = exp / len(picks) if picks else 0
        cells.append(tile(pct(avg), "average pick probability"))
        cells.append(tile(str(len(slate_data.get("hit_picks") or [])), "hit picks"))
    return tiles(cells)


def _model_section(slate_data: dict) -> str:
    """The held-out scores in brief, and the ESPN inputs."""
    m = slate_data.get("ensemble_metrics") or {}
    parts = []
    if m:
        served = _served_key(slate_data)
        rows = ""
        best = max((m.get(k, {}).get("auc", 0) for k, _, _ in _VARIANTS if isinstance(m.get(k), dict)), default=1) or 1
        for key, label, _ in _VARIANTS:
            s = m.get(key)
            if not isinstance(s, dict) or "auc" not in s:
                continue
            cls = ' class="served"' if key == served else ""
            rows += (f"<tr{cls}><td>{label}</td><td class='n'>{s.get('auc', 0):.4f}{mini((s.get('auc', 0) - 0.5) / max(best - 0.5, 1e-9))}</td>"
                     f"<td class='n'>{s.get('top_decile_lift', 0):.2f}&times;</td><td class='n'>{s.get('brier', 0):.5f}</td>"
                     f"<td class='n'>{s.get('log_loss', 0):.5f}</td></tr>")
        note = (f"Fitted on plate appearances before {m.get('cutoff_date', '')}, scored on the "
                f"{m.get('n_test_samples', 0):,} that came after (actual HR rate {m.get('holdout_hr_rate', 0):.2%}). "
                "<b>Read AUC first</b>: the slate ranks hitters and takes the top six, so ordering is what matters. "
                "Metrics are per plate appearance; the probabilities above are per game. The highlighted row is "
                "served. Every prior side by side is in the <a href='/models'>Model Lab</a>.")
        parts.append(head("Model check", "&#128200;", note)
                     + f"<div class='card scroll'><table class='t'><thead><tr><th>Prior</th><th class='n'>AUC</th>"
                       f"<th class='n'>Top-decile lift</th><th class='n'>Brier</th><th class='n'>Log loss</th></tr></thead>"
                       f"<tbody>{rows}</tbody></table></div>")
    espn = slate_data.get("espn")
    if espn:
        excluded = slate_data.get("excluded_injured") or []
        cov = espn.get("bio_coverage", {})
        sample = ", ".join(f"{escape(e['batter'])} ({escape(str(e['status']))})" for e in excluded[:8])
        extra = f" +{len(excluded) - 8} more" if len(excluded) > 8 else ""
        parts.append(head("ESPN inputs", "", "Height, weight and age feed the comparables model; the injury report "
                          "removes anyone on an IL variant from the pick pool.", tag="h3")
                     + tiles([tile(str(espn.get("excluded_injured", 0)), "injured players excluded"),
                              tile(str(espn.get("injuries", 0)), "league injury entries"),
                              tile(f"{cov.get('rate', 0):.0%}", f"bio match rate ({cov.get('matched', 0)}/{cov.get('total', 0)})")])
                     + (f"<p class='note'><b>Excluded today:</b> {sample}{extra}</p>" if sample else ""))
    if not parts:
        return ""
    return '<section class="section" id="model">' + "".join(parts) + "</section>"


def _print_table(head_cells: list, rows: list) -> str:
    th = "".join(f"<th{' class=n' if n else ''}>{h}</th>" for h, n in head_cells)
    return f"<table class='t'><thead><tr>{th}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _print_slate(slate_data: dict) -> str:
    """The PDF archive: the same day as dense tables. WeasyPrint lays out card
    grids poorly and slowly, and an archive is read as a record, not browsed."""
    by_pk = _games_by_pk(slate_data)
    picks = slate_data["picks_6"]
    res = slate_data.get("results") or {}

    def outcome(h, key="hr"):
        tone, label, detail = _verdict(h, key)
        return f"<span class='{ {'hit': 'ok', 'miss': 'bad'}.get(tone, 'muted') }'>{label}</span> <span class='muted'>{escape(str(detail))}</span>" if tone else ""

    def vs(h):
        opp, sp, sp_hand = _opponent(h, by_pk)
        return f"{abbr(opp) if opp else ''} &middot; {escape(str(sp or ''))} {hand(sp_hand)}"

    parts = [hero(f"{slate_data['date']} &middot; {slate_data['games_count']} games", "HR Daily Tracker &mdash; daily archive"),
             _results_tiles(slate_data)]
    rows = [f"<tr><td>{i}</td><td><b>{escape(p['batter'])}</b>{_inj(p)}</td><td>{abbr(p.get('team'))}</td><td>{vs(p)}</td>"
            f"<td class='n'><b>{pct(p['prob_hr'])}</b></td><td class='n'>{fair_odds(p['prob_hr'])}</td>"
            f"<td class='n'>{p.get('prob_ensemble_per_pa', 0):.2%} &times; {p.get('prob_expected_pa', 0):.1f}</td>"
            f"<td class='n'>{p['season_hrs']} / {p['season_pas']}</td><td>{outcome(p)}</td></tr>"
            for i, p in enumerate(picks, 1)]
    parts.append(head("Home Run Picks") + _print_table(
        [("#", 0), ("Hitter", 0), ("Team", 0), ("Opponent &middot; starter", 0), ("HR prob", 1), ("Fair", 1),
         ("Per PA &times; PA", 1), ("Season HR / PA", 1), ("Result", 0)], rows))
    hit_rows = [f"<tr><td>{i}</td><td><b>{escape(p['batter'])}</b></td><td>{abbr(p.get('team'))}</td><td>{vs(p)}</td>"
                f"<td class='n'><b>{p['hits_proj']['projected_hits']:.2f}</b></td><td class='n'>{p['hits_proj']['prob_at_least_one']:.0%}</td>"
                f"<td class='n'>{p['hits_proj']['hitter_rate']:.1%}</td><td>{outcome(p, 'hits')}</td></tr>"
                for i, p in enumerate(slate_data.get("hit_picks") or [], 1)]
    if hit_rows:
        parts.append(head("Projected Hits") + _print_table(
            [("#", 0), ("Hitter", 0), ("Team", 0), ("Opponent &middot; starter", 0), ("Proj. hits", 1), ("1+ hit", 1),
             ("Season rate", 1), ("Result", 0)], hit_rows))
    for play in slate_data.get("parlays") or []:
        legs = [f"<tr><td>{_leg_kind(l)}</td><td><b>{escape(l['batter'])}</b></td>"
                f"<td>{abbr(l.get('team'))}</td><td class='n'>{l['prob']:.0%}</td><td>{escape(l.get('status', 'pending'))}</td></tr>"
                for l in play["leg_list"]]
        parts.append(head(f"{play['name']} &middot; {_chance(play['prob'])} &middot; fair {play['fair_odds']} "
                          f"&middot; {escape(play.get('status', 'pending'))}", tag="h3")
                     + _print_table([("Leg", 0), ("Hitter", 0), ("Team", 0), ("Prob", 1), ("Status", 0)], legs))
    game_rows = []
    for g in slate_data.get("games", []):
        p = g.get("projection") or {}
        live = g.get("live") or {}
        score = (f"{live.get('away_score')}&ndash;{live.get('home_score')} {live.get('state')}"
                 if live.get("state") in ("Live", "Final") else "")
        top = sorted(g.get("hitters", []), key=lambda h: h.get("prob_hr", 0), reverse=True)[:3]
        bats = ", ".join(f"{escape(h['batter'])} {h['prob_hr']:.0%}{' &#10003;' if (h.get('result') or {}).get('hr') else ''}" for h in top)
        game_rows.append(
            f"<tr><td><b>{abbr(g['away'])} @ {abbr(g['home'])}</b></td>"
            f"<td>{escape(str(g.get('away_sp') or 'TBD'))} {hand(g.get('away_sp_hand'))} / {escape(str(g.get('home_sp') or 'TBD'))} {hand(g.get('home_sp_hand'))}</td>"
            f"<td class='n'>{pct(p.get('away_win_prob'), 0)} / {pct(p.get('home_win_prob'), 0)}</td>"
            f"<td class='n'>{p.get('away_expected_runs', 0):.1f}&ndash;{p.get('home_expected_runs', 0):.1f}</td>"
            f"<td class='n'>{(g.get('weather') or {}).get('hr_factor', 1):.2f}&times;</td>"
            f"<td class='wrap'>{bats}</td><td>{score}</td></tr>")
    parts.append(head("Games") + _print_table(
        [("Game", 0), ("Starters", 0), ("Win away / home", 1), ("Proj. runs", 1), ("Weather HR", 1),
         ("Top HR bats", 0), ("Score", 0)], game_rows))
    parts.append(_model_section(slate_data))
    return page(f"HR Daily Tracker — {slate_data['date']}", "slate", "".join(parts), _freshness(slate_data))


def render_html(slate_data: dict) -> str:
    """The slate page: today's picks, what is still to play, parlays and games."""
    if is_print():
        return _print_slate(slate_data)
    date_str = slate_data["date"]
    games_count = slate_data["games_count"]
    picks = slate_data["picks_6"]
    games = slate_data["games"]
    by_pk = _games_by_pk(slate_data)
    hit_picks = slate_data.get("hit_picks") or []
    highlighted = slate_data.get("highlighted_matchups") or []
    results = slate_data.get("results") or {}

    lineups = slate_data.get("lineups_posted")
    lineup_note = (f" Lineups are posted for {lineups} of {games_count * 2} teams; the rest use projected lineups."
                   if isinstance(lineups, int) and 0 < lineups < games_count * 2 else "")
    parts = [hero(f"{date_str} &middot; {games_count} game{'s' if games_count != 1 else ''}",
                  "Today&rsquo;s Home Run Board",
                  "Six home run picks and six hit picks from the stacked season-replay model, with every game "
                  "projected underneath. Picks lock as their games start; the rest of the board keeps refreshing."
                  + lineup_note)]
    # Say so plainly when this is yesterday's slate standing in for one that is
    # still being fitted; a stale page that admits it beats a silent one.
    if slate_data.get("building_note"):
        parts.append(alert(slate_data["building_note"], info=True))
    # A postseason day can be one game. Picks are one per team and parlay legs
    # one per game, so a short slate posts fewer of both; say why, or the page
    # just looks like it lost four picks and a section.
    if 0 < games_count < 5:
        parts.append(alert(
            f"Short slate: {games_count} game{'' if games_count == 1 else 's'} today. Picks are one per team, so "
            f"there {'is' if len(picks) == 1 else 'are'} {len(picks)} home run pick{'' if len(picks) == 1 else 's'} "
            f"instead of six, and the core parlays need five different games, so none "
            f"{'is' if not slate_data.get('parlays') else 'may be'} posted."))

    nav = [("picks", "HR Picks")]
    remaining = _render_remaining_section(slate_data)
    if remaining:
        nav.append(("remaining", "Still to Play"))
    if hit_picks:
        nav.append(("hits", "Hits"))
    if slate_data.get("parlays") or slate_data.get("parlay_replay"):
        nav.append(("parlays", "Parlays"))
    if highlighted:
        nav.append(("bvp", "Batter vs Pitcher"))
    nav.append(("games", "Games"))
    if slate_data.get("ensemble_metrics") or slate_data.get("espn"):
        nav.append(("model", "Model"))
    parts.append(subnav(nav))
    parts.append(_results_tiles(slate_data))

    cards = "".join(_hr_card(i, p, by_pk) for i, p in enumerate(picks, 1))
    note = ("Chance of at least one home run in this game: the hitter's per-PA rate from the stacked model "
            "(comparables prior, contact quality, the starter, platoon split, park, weather and bullpen), run "
            "through his expected plate appearances against the starter and the pen. One pick per team.")
    parts.append(f'<section class="section" id="picks">{head("Home Run Picks", "&#128163;", note)}'
                 f'<div class="grid">{cards or "<div class=card>No picks for this slate.</div>"}</div></section>')
    # What is still live comes before the settled parts of the page: once the
    # afternoon games are in the book, tonight's are the only actionable ones.
    parts.append(remaining)
    parts.append(_render_hits_section(hit_picks, by_pk))
    parts.append(_render_parlays_section(slate_data))
    parts.append(_render_matchups_section(highlighted, by_pk))

    pick_ids = {p.get("batter_id") for p in picks}
    order = sorted(games, key=lambda g: ({"Live": 0, "Preview": 1}.get((g.get("live") or {}).get("state"), 1)
                                         if (g.get("live") or {}).get("state") != "Final" else 2, g.get("game_pk", 0)))
    game_html = "".join(_game_card(g, slate_data.get("bullpens") or {}, slate_data.get("league_rates") or {}, pick_ids)
                        for g in order)
    parts.append(f'<section class="section" id="games">'
                 + head("Games", "&#9918;", "Live games first, then the rest of the slate, finals last. Tap a game "
                        "for the win projection, conditions, strikeout projections, bullpens and its top bats.")
                 + f"{game_html}</section>")
    parts.append(_model_section(slate_data))
    return page(f"HR Daily Tracker — {date_str}", "slate", "".join(parts), _freshness(slate_data),
                status=_status(slate_data), live=60)


# --------------------------------------------------------------- model lab
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
    """Held-out scores, one row per prior, with AUC drawn as a bar."""
    present = [(k, l, b) for k, l, b in variants if metrics.get(k)]
    if not present:
        return ""
    aucs = [metrics[k].get("auc", 0) for k, _, _ in present]
    lo, hi = min(aucs), max(aucs)
    head_cells = "".join(f"<th class='n'>{label}</th>" for _, label, _ in _METRIC_COLUMNS)
    rows = ""
    for key, label, blurb in present:
        m = metrics[key]
        cells = ""
        for mk, _, fmt in _METRIC_COLUMNS:
            bar = mini((m.get(mk, 0) - lo) / (hi - lo) if hi > lo else 1) if mk == "auc" else ""
            cells += f"<td class='n'>{fmt.format(m.get(mk, 0))}{bar}</td>"
        cls = ' class="served"' if key == served else ""
        tag = ' <span class="pill lean">served</span>' if key == served else ""
        rows += (f"<tr{cls}><td class='wrap'><b>{label}</b>{tag}<div class='sub'>{blurb}</div></td>{cells}</tr>")
    return (f"<div class='card scroll'><table class='t'><thead><tr><th>Prior</th>{head_cells}</tr></thead>"
            f"<tbody>{rows}</tbody></table></div>")


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
    if is_print():
        rows = ""
        for key, label, _ in variants:
            top = _variant_top6(hitters, key)
            if top:
                shared = len(served_names.intersection({h["batter"] for h in top}))
                names = ", ".join(f"{escape(h['batter'])} {h['prob_' + key]:.1%}" for h in top)
                rows += f"<tr><td><b>{label}</b></td><td class='n'>{shared}/6</td><td class='wrap'>{names}</td></tr>"
        return ("<table class='t'><thead><tr><th>Prior</th><th class='n'>Shared</th><th>Its top six</th></tr></thead>"
                f"<tbody>{rows}</tbody></table>")
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
            mark = ' <span class="ok">&#10003;</span>' if line.get("hr") else ""
            inside = h["batter"] in served_names
            listing += (f'<div class="rowx"><span style="{"" if inside else "color:var(--mute)"}">'
                        f'{"<b>" if inside else ""}{escape(h["batter"])}{"</b>" if inside else ""}{mark}</span>'
                        f'<span class="r">{h["prob_" + key]:.1%}</span></div>')
        # Only finished games count: a hitter still batting is not yet a miss.
        final = [h for h in top if (h.get("result") or {}).get("final")]
        hit = sum(1 for h in final if h["result"]["hr"])
        score = (f'<span class="badge {"hit" if hit else "miss"}">{hit}/{len(final)} homered</span>' if final else "")
        cards += (f'<article class="card{" pick" if key == served else ""}" style="--tc:var(--accent)">'
                  f'<div style="display:flex;justify-content:space-between;gap:8px;align-items:center;margin-bottom:6px">'
                  f'<b>{label}</b><span class="chip"><b>{overlap}</b>/6 shared</span></div>'
                  f'<div class="rows">{listing}</div>{f"<div style=margin-top:8px>{score}</div>" if score else ""}</article>')
    return (f'<div class="grid">{cards}</div><p class="note" style="margin-top:12px">Each card is that prior\'s own '
            "top six, ranked by its own number. Names in bold also appear in the served prior's six; the count is "
            "the overlap. Priors that look alike on aggregate metrics can still disagree on the names, and that "
            "disagreement is the part that changes what you would play.</p>")


def _side_by_side_table(slate_data: dict, variants: list, served: str, limit: int) -> str:
    """Per-hitter probabilities from every prior, ranked by the served one, shaded by size."""
    hitters = sorted(_all_hitters(slate_data), key=lambda h: h.get("prob_hr", 0), reverse=True)[:limit]
    if not hitters:
        return ""
    pick_names = {p["batter"] for p in slate_data.get("picks_6", [])}
    scored = any(h.get("result") for h in hitters)
    values = [h.get(f"prob_{k}") for h in hitters for k, _, _ in variants if h.get(f"prob_{k}") is not None]
    top = max(values, default=0) or 1
    head_cells = "".join(f'<th class="n{" sv" if key == served else ""}">{label}</th>' for key, label, _ in variants)
    head_cells += "<th>Result</th>" if scored else ""
    rows = ""
    for i, h in enumerate(hitters, 1):
        cells = ""
        for key, _, _ in variants:
            v = h.get(f"prob_{key}")
            if v is None:
                cells += f'<td class="n{" sv" if key == served else ""}">&ndash;</td>'
            elif key == served:
                cells += f'<td class="n sv">{v:.1%}</td>'
            else:
                cells += f'<td class="n heat" style="--h:{v / top:.2f}">{v:.1%}</td>'
        if scored:
            cells += f"<td>{_result_badge(h) or '&ndash;'}</td>"
        badge = ' <span class="pill lean">slate</span>' if h["batter"] in pick_names else ""
        sub = (f"{abbr(h.get('team', ''))} &middot; vs {hand(h.get('facing_hand'))} &middot; "
               f"{h.get('season_hrs', 0)} HR / {h.get('season_pas', 0)} PA")
        rows += (f"<tr><td><div style='display:flex;gap:8px;align-items:center'><span class='muted' style='width:22px'>{i}</span>"
                 f"{person(h.get('batter_id'), h['batter'], h.get('team'), sub, 'xs') if not is_print() else '<b>' + escape(h['batter']) + '</b> <span class=muted>' + abbr(h.get('team', '')) + '</span>'}"
                 f"{badge}</div></td>{cells}</tr>")
    return (f"<div class='card scroll'><table class='t'><thead><tr><th>Hitter</th>{head_cells}</tr></thead>"
            f"<tbody>{rows}</tbody></table></div>"
            "<p class='note' style='margin-top:12px'>Per-game probabilities: each prior's per-PA rate run through "
            "this hitter's own plate appearances, park, weather, bullpen and times-through-the-order terms. Only the "
            "prior changes across the columns &mdash; every other term is held fixed, so a left-to-right spread is "
            "the prior disagreeing about the hitter, not the context. Shading grows with the probability.</p>")


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
    served_label = next((label for key, label, _ in _VARIANTS if key == served), served)
    n_test = metrics.get("n_test_samples", 0)
    parts = [hero(f"Model Lab &middot; {date_str}", "Every prior, same slate",
                  "The held-out scores that decided which prior is served, then how each one would have "
                  "ranked today's hitters.")]
    if not variants:
        parts.append(alert("This slate was built without the ensemble, so there is only one set of "
                           "probabilities to show. The slate page has them."))
        return page(f"Model Lab — {date_str}", "models", "".join(parts), _freshness(slate_data))

    parts.append(subnav([("scores", "Held-out scores")] + ([("thin", "Thin sample")] if metrics.get("low_pa") else [])
                        + [("agreement", "Slate agreement"), ("hitters", "Hitter by hitter")]))
    best = max(((k, metrics[k].get("auc", 0)) for k, _, _ in variants if metrics.get(k)), key=lambda kv: kv[1], default=None)
    parts.append(tiles([tile(served_label, "served prior", small=True, tone="accent"),
                        tile(str(len(variants)), "priors compared"),
                        tile(f"{n_test:,}", "held-out plate appearances"),
                        tile(f"{metrics.get('holdout_hr_rate', 0):.2%}", "held-out HR rate"),
                        tile(f"{best[1]:.4f}" if best else "&mdash;",
                             f"best AUC &middot; {next((l for k, l, _ in _VARIANTS if best and k == best[0]), '')}")]))
    weights = metrics.get("prior_weights") or {}
    k_shrink = metrics.get("shrinkage_k") or {}
    config = ""
    if weights or k_shrink:
        w = " ".join(f"<span class='chip'>{k.upper()} <b>{v:.2f}</b></span>" for k, v in weights.items())
        ks = " ".join(f"<span class='chip'>{escape(str(k))} <b>{v:.0f}</b></span>" for k, v in k_shrink.items())
        config = (f"<div class='grid lab' style='margin-top:14px'><div class='card'><div class='label'>Blend weights (forest variant)</div>"
                  f"<div class='chips'>{w or 'n/a'}</div></div><div class='card'><div class='label'>Shrinkage K per prior</div>"
                  f"<div class='chips'>{ks or 'n/a'}</div><p class='note' style='margin:8px 0 0'>Estimated by method of "
                  f"moments on out-of-fold residuals, not hand-tuned.</p></div></div>")
    note = (f"Fitted on plate appearances before {metrics.get('cutoff_date', 'n/a')}, scored on the {n_test:,} that "
            f"came after (actual HR rate {metrics.get('holdout_hr_rate', 0):.2%}). No outcome in the test window was "
            "visible during fitting. <b>Read AUC first</b> &mdash; the slate ranks hitters and takes the top six, so "
            "ordering is what matters. Brier and log loss are dominated by the ~3% base rate and barely separate the "
            "priors. Metrics are per plate appearance; the probabilities lower down are per game.")
    parts.append(f'<section class="section" id="scores">{head("Held-out scores", "&#128200;", note)}'
                 f'{_metrics_table(metrics, variants, served)}{config}</section>')
    low_pa = metrics.get("low_pa") or {}
    if low_pa:
        note = (f"The same held-out window, restricted to hitters with fewer than {low_pa.get('threshold_pa', 0)} "
                f"plate appearances before the cutoff ({low_pa.get('n_test_samples', 0):,} PA, actual HR rate "
                f"{low_pa.get('holdout_hr_rate', 0):.2%}). For a regular with 600 PA the observed record swamps any "
                "prior, so the priors mostly separate here.")
        parts.append(f'<section class="section" id="thin">{head("Thin-sample subset", "&#128300;", note)}'
                     f'{_metrics_table(low_pa, variants, served)}</section>')
    parts.append(f'<section class="section" id="agreement">{head("Slate agreement", "&#129309;")}'
                 f'{_agreement_section(slate_data, variants, served)}</section>')
    if is_print():
        # The hitter-by-hitter table runs eleven priors wide.
        parts.append("<style>@page{size:Letter landscape}</style>")
    parts.append(f'<section class="section" id="hitters">{head("Hitter by hitter", "&#9878;", f"The top {limit} bats by the served probability.")}'
                 f'{_side_by_side_table(slate_data, variants, served, limit)}</section>')
    return page(f"Model Lab — {date_str}", "models", "".join(parts), _freshness(slate_data),
                status=_status(slate_data))
