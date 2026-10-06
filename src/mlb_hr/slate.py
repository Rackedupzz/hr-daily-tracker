"""Build today's game slate with 6-pick HR randomizer."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import pandas as pd

from mlb_hr import parlays as parlay_mod
from mlb_hr.context import build_context_model
from mlb_hr.fetch import (
    active_rosters,
    game_context,
    lineups as fetch_lineups,
    pitcher_hands,
    probable_pitchers,
    schedule,
    team_standings,
)
from mlb_hr.parks import build_park_factors, summarize as summarize_parks
from mlb_hr.game import project_game
from mlb_hr.features import expected_pa_for_slot, recent_lineups
from mlb_hr.model import HRModel, BatterStats
from mlb_hr.pitching import (
    build_pitching_profiles,
    matchup_note,
    project_hits,
    project_strikeouts,
)

if TYPE_CHECKING:
    from mlb_hr.ensemble import EnsembleHRModel

# Before lineups post, a hitter must have started this many of his team's last
# RECENT_TEAM_GAMES to be eligible for a pick.
MIN_RECENT_STARTS = 4
RECENT_TEAM_GAMES = 7

# Recency half-life for the hitter model: 45 days beat 90, 21 and no decay on
# held-out AUC (0.5800 vs 0.5780 with equal weighting). A modest gain, but
# consistent across every value tried.
HALF_LIFE_DAYS = 45.0


def data_before(model_path: str, today: date) -> str:
    """A PA file holding only games before `today`.

    The season feed tops up with every final game, so by evening it holds
    today's afternoon games. A slate for today refitted then would be trained on
    the outcomes it is projecting. When the file reaches today, the earlier
    rows are copied to a per-day side file (reused across rebuilds) and that is
    what the slate is built from.
    """
    src = Path(model_path)
    cutoff = str(today)
    out = src.parent / "slate_cache" / f"season_pa_before_{cutoff}.jsonl"
    if out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
        return str(out)

    def row_date(line: str) -> str:
        at = line.find('"date": "')
        return line[at + 9:at + 19] if at >= 0 else ""

    # Games are appended as they finish, not in date order, so the whole file
    # is checked -- by string slice, which is fast enough for every rebuild.
    with open(src) as fh:
        if not any(row_date(line) >= cutoff for line in fh):
            return model_path
    out.parent.mkdir(parents=True, exist_ok=True)
    for old in out.parent.glob("season_pa_before_*.jsonl"):
        if old != out:
            # Housekeeping only: on a Windows bind mount a stale file can be
            # briefly locked, and that must not fail the day's build.
            try:
                old.unlink(missing_ok=True)
            except OSError as exc:
                print(f"[slate.py] could not remove {old.name} ({exc}); leaving it")
    tmp = out.with_suffix(".tmp")
    kept = 0
    with open(src) as fh, open(tmp, "w") as dst:
        for line in fh:
            if not line.strip() or row_date(line) >= cutoff:
                continue
            dst.write(line)
            kept += 1
    tmp.replace(out)
    print(f"[slate.py] season feed reaches {today}; using {kept:,} PAs before it")
    return str(out)


class GameSlate:
    """Today's games and hitters playing."""

    def __init__(
        self,
        today: date,
        model: HRModel,
        ensemble: Optional[EnsembleHRModel] = None,
        espn=None,
        pitchers: Optional[dict] = None,
        team_batting: Optional[dict] = None,
        league=None,
        standings: Optional[dict] = None,
        bullpens: Optional[dict] = None,
        matchups: Optional[dict] = None,
        lineups: Optional[dict] = None,
        context=None,
        recent_starts: Optional[dict] = None,
        usual_slots: Optional[dict] = None,
        hr_level: float = 1.0,
    ):
        self.date = today
        self.model = model
        self.ensemble = ensemble
        self.espn = espn
        self.pitchers = pitchers or {}
        self.team_batting = team_batting or {}
        self.league = league
        self.standings = standings or {}
        self.bullpens = bullpens or {}
        self.matchups = matchups or {}
        self.lineups = lineups or {}
        self.context = context
        # None means "unknown" and disables the bench filter, rather than
        # treating every hitter as a non-starter.
        self.recent_starts = recent_starts
        self.usual_slots = usual_slots or {}
        # Odds multiplier from the tracker's recent actual-over-projected
        # home runs (snapshot.hr_level); 1.0 when there is no record yet.
        self.hr_level = hr_level
        self.excluded_bench: list[dict] = []
        self.game_context: dict = {}
        self.lineups_used = 0
        self.games: list[dict] = []
        self.active_hitters: dict[str, BatterStats] = {}
        self.excluded_injured: list[dict] = []
        self._build()

    def _build(self) -> None:
        """Fetch today's games, probable pitchers, active rosters."""
        # The API's abstractGameState is one of Preview / Live / Final -- there
        # is no "Scheduled". Filtering on that name dropped every game that had
        # not finished, so an evening slate showed only the afternoon results
        # and no night games at all.
        games = schedule(self.date, self.date)
        games = [g for g in games if g.state in ("Preview", "Live", "Final")]

        if not games:
            print(f"No games found for {self.date}")
            return

        # Fetch probable pitchers
        probs = probable_pitchers(self.date)
        pp_by_pk = {p["game_pk"]: p for p in probs}

        # Get all teams playing today
        team_ids = set()
        for g in games:
            team_ids.add(g.home_id)
            team_ids.add(g.away_id)

        # Fetch active rosters
        rosters = active_rosters(team_ids, 2026)
        active_names = {}
        for tid, roster in rosters.items():
            for r in roster:
                active_names[r["name"]] = r["id"]

        # Fetch pitcher hands
        pitcher_ids = set()
        for p in probs:
            if p.get("home_sp_id"):
                pitcher_ids.add(p["home_sp_id"])
            if p.get("away_sp_id"):
                pitcher_ids.add(p["away_sp_id"])
        hands = pitcher_hands(pitcher_ids)

        # Conditions for these specific games, not a season average.
        if self.context is not None:
            try:
                self.game_context = game_context([g.game_pk for g in games])
            except Exception as e:  # noqa: BLE001 - fall back to neutral
                print(f"[slate.py] Warning: game conditions unavailable ({e})")
                self.game_context = {}

        # Build game records with hitters
        for g in games:
            pp = pp_by_pk.get(g.game_pk, {})
            home_sp_id = pp.get("home_sp_id")
            away_sp_id = pp.get("away_sp_id")
            home_sp_hand = hands.get(home_sp_id, "R")
            away_sp_hand = hands.get(away_sp_id, "R")

            game_record = {
                "game_pk": g.game_pk,
                "date": g.date,
                "start": pp.get("start", ""),
                "venue": g.venue,
                "home": g.home,
                "away": g.away,
                "home_id": g.home_id,
                "away_id": g.away_id,
                "home_sp": pp.get("home_sp"),
                "away_sp": pp.get("away_sp"),
                "home_sp_hand": home_sp_hand,
                "away_sp_hand": away_sp_hand,
                "hitters": [],
            }

            # Starting-pitcher strikeout projections. Each starter is matched
            # against the lineup he actually faces, not a generic opponent.
            home_sp_profile = self.pitchers.get(home_sp_id)
            away_sp_profile = self.pitchers.get(away_sp_id)
            if self.league is not None:
                if home_sp_profile:
                    game_record["home_sp_k"] = project_strikeouts(
                        home_sp_profile, self.team_batting.get(g.away), self.league
                    )
                if away_sp_profile:
                    game_record["away_sp_k"] = project_strikeouts(
                        away_sp_profile, self.team_batting.get(g.home), self.league
                    )

            # Win/loss projection with its supporting terms.
            if self.standings and self.league is not None:
                game_record["projection"] = project_game(
                    home_id=g.home_id,
                    away_id=g.away_id,
                    home_team=g.home,
                    away_team=g.away,
                    standings=self.standings,
                    home_sp=home_sp_profile,
                    away_sp=away_sp_profile,
                    league=self.league,
                    park_factor=self.model.park_factors.get(g.venue, 1.0),
                )

            # Weather and order-turn adjustments for this game. The starter is
            # met on trips one to three, relief afterwards, so the two exposures
            # carry different order-turn factors.
            ctx = (self.game_context or {}).get(g.game_pk, {})
            weather_factor = 1.0
            tto_factors = (1.0, 1.0)
            if self.context is not None:
                weather_factor = self.context.game_hr_factor(
                    ctx.get("temp_f"), ctx.get("wind_mph"), ctx.get("wind_dir")
                )
                tto_factors = (
                    (self.context.tto_hr_factor(1)
                     + self.context.tto_hr_factor(2)
                     + self.context.tto_hr_factor(3)) / 3.0,
                    self.context.tto_hr_factor(1),
                )
            game_record["weather"] = {
                **ctx, "hr_factor": round(weather_factor, 3),
            }

            # Add eligible hitters: in active roster + has PA data in model
            # A hitter faces the *other* club's starter, so the home lineup
            # takes the away starter's hand. This used to hand each lineup its
            # own starter's hand: in any game with a lefty against a righty,
            # both lineups got the wrong platoon split and park side.
            for side, team_id, sp_hand in [
                ("home", g.home_id, away_sp_hand),
                ("away", g.away_id, home_sp_hand),
            ]:
                # A posted lineup replaces the roster: those nine are the only
                # hitters who will bat, and their slot sets their exposure.
                # Without it every active-roster bat competes, bench included.
                lineup = self.lineups.get(g.game_pk, {})
                lineup_ids = lineup.get(side) or []
                slot_by_id = {pid: i + 1 for i, pid in enumerate(lineup_ids)}
                # Hitters are matched to the PA feed by MLB person id, which the
                # lineup and roster both carry. Matching by name picked the
                # first hitter with that name and dropped anyone whose name was
                # spelled differently between the two endpoints.
                if lineup_ids:
                    id_to_name = {
                        r["id"]: r["name"] for r in rosters.get(team_id, [])
                    }
                    names = lineup.get(f"{side}_names") or []
                    team_roster = [
                        (pid, id_to_name.get(pid) or name)
                        for pid, name in zip(lineup_ids, names)
                    ]
                    self.lineups_used += 1
                else:
                    team_roster = [
                        (r["id"], r["name"]) for r in rosters.get(team_id, [])
                    ]

                for player_id, hitter_name in team_roster:
                    # ESPN injury report: players on an IL variant cannot appear
                    # today, so they must not be eligible for a pick.
                    if self.espn and self.espn.is_unavailable(hitter_name):
                        self.excluded_injured.append({
                            "batter": hitter_name,
                            "status": self.espn.injury_status(hitter_name),
                            "team": g.home if side == "home" else g.away,
                        })
                        continue

                    b = self.model.get_batter(player_id) if player_id else None
                    if b is None and not player_id:
                        b = self.model.get_batter_by_name(hitter_name)
                    # No lineup yet: keep to hitters who have been starting.
                    # A bench bat can out-rate a regular and win a pick for a
                    # game he never plays in.
                    if (b and not lineup_ids and self.recent_starts is not None
                            and self.recent_starts.get(b.batter_id, 0)
                            < MIN_RECENT_STARTS):
                        self.excluded_bench.append({
                            "batter": hitter_name,
                            "team": g.home if side == "home" else g.away,
                            "recent_starts": self.recent_starts.get(b.batter_id, 0),
                        })
                        continue
                    if b and b.pa_total >= 2:  # Only hitters with at least 2 PAs
                        # The bullpen a hitter meets late is the *opposing*
                        # club's, which is the team he is not on.
                        opp_team = g.away if side == "home" else g.home
                        opp_pen = self.bullpens.get(opp_team)
                        opp_sp_id = away_sp_id if side == "home" else home_sp_id

                        slot = slot_by_id.get(b.batter_id)
                        # Until the lineup posts, his usual recent slot stands
                        # in: the stacked model weights exposure heavily, and a
                        # per-game average that counts pinch-hit appearances
                        # would mark a regular down for hours.
                        use_slot = slot or (
                            None if lineup_ids else self.usual_slots.get(b.batter_id)
                        )
                        slot_pa = (
                            expected_pa_for_slot(use_slot, None) if use_slot else None
                        )
                        opp_sp_profile = self.pitchers.get(opp_sp_id)
                        sp_hr_index = (
                            opp_sp_profile.hr_index(self.league.hr_rate)
                            if opp_sp_profile and self.league else 1.0
                        )

                        # Use ensemble if available, otherwise empirical-Bayes
                        if self.ensemble:
                            probs_dict = self.ensemble.pr_hr_ensemble(
                                b, -1, g.venue, sp_hand,
                                bullpen=opp_pen,
                                league_hr_rate=(
                                    self.league.hr_rate if self.league else None
                                ),
                                expected_pa=slot_pa,
                                weather_factor=weather_factor,
                                tto_factors=tto_factors,
                                sp_hr_index=sp_hr_index,
                                level=self.hr_level,
                            )
                            prob = probs_dict.get("ensemble", probs_dict.get("empirical_bayes", 0))
                        else:
                            prob = self.model.pr_hr_today(
                                b, -1, g.venue, sp_hand
                            )
                            probs_dict = {"empirical_bayes": prob}

                        hitter_record = {
                            # Carried on the pick itself so results can be
                            # settled per game, including "did not play".
                            "game_pk": g.game_pk,
                            "batter_id": b.batter_id,
                            "batter": b.batter,
                            "team": g.home if side == "home" else g.away,
                            "prob_hr": prob,
                            "season_hrs": b.hr_total,
                            "season_pas": b.pa_total,
                            "side": side,
                            # Platoon detail: how this hitter's home runs split
                            # by the handedness of the pitcher who threw them,
                            # and which of those splits today's starter falls in.
                            "bat_side": b.hand,
                            "lineup_slot": slot,
                            "hr_vs_rhp": b.hr_vs_r,
                            "pa_vs_rhp": b.pa_vs_r,
                            "hr_vs_lhp": b.hr_vs_l,
                            "pa_vs_lhp": b.pa_vs_l,
                            "facing_hand": sp_hand,
                            "hr_vs_facing": b.hr_vs_l if sp_hand == "L" else b.hr_vs_r,
                            "pa_vs_facing": b.pa_vs_l if sp_hand == "L" else b.pa_vs_r,
                            # Day-to-day players stay eligible but are flagged.
                            "injury_note": (
                                self.espn.injury_status(hitter_name)
                                if self.espn else None
                            ),
                        }
                        # Include all model predictions for comparison
                        hitter_record.update({
                            f"prob_{k}": v for k, v in probs_dict.items()
                        })

                        # Hits projection and batter-vs-pitcher history, both
                        # against the starter this hitter actually faces.
                        profile = (
                            self.ensemble.profiles.get(b.batter_id)
                            if self.ensemble else None
                        )
                        if profile and self.league:
                            hitter_record["hits_proj"] = project_hits(
                                batter_hit_rate=profile.hit_rate,
                                batter_pa=profile.pa,
                                pitcher=self.pitchers.get(opp_sp_id),
                                bullpen=opp_pen,
                                league=self.league,
                                expected_pa=probs_dict.get("expected_pa", 4.1),
                                batter_xhit=profile.xhit,
                                batter_k=profile.strikeouts,
                                park_hit=self.model.hit_park_factor_for(
                                    g.venue, profile.side_vs(sp_hand)
                                ),
                                coef=self.ensemble.hit_coef,
                                top_cal=self.ensemble.hit_top_cal,
                            )
                            note = matchup_note(
                                self.matchups.get((b.batter_id, opp_sp_id)),
                                batter_hit_rate=profile.hit_rate,
                                batter_hr_rate=profile.hr_rate,
                            )
                            if note:
                                note["pitcher"] = (
                                    pp.get("away_sp") if side == "home"
                                    else pp.get("home_sp")
                                )
                                hitter_record["matchup"] = note
                        game_record["hitters"].append(hitter_record)
                        self.active_hitters[hitter_name] = b

            self.games.append(game_record)

    def random_6_pick_slate(self) -> list[dict]:
        """Generate 6 HR picks: one per game where possible, all from today's games.

        Returns list of hitters with their probabilities and context.
        """
        return choose_hr_picks(self.games)

    def top_hit_projections(self, limit: int = 6) -> list[dict]:
        """Highest projected hit totals, one per team.

        Ranked on expected hits rather than probability of a hit, since that is
        the quantity the projection actually estimates; the at-least-one
        probability rides along for readers who think in those terms.
        """
        return choose_hit_picks(self.games, limit=limit)

    def highlighted_matchups(self, limit: int = 8) -> list[dict]:
        """Batter-vs-pitcher histories worth surfacing, hottest first."""
        found = [
            h for g in self.games for h in g["hitters"] if h.get("matchup")
        ]
        found.sort(
            key=lambda h: (
                h["matchup"]["verdict"] == "hot",
                h["matchup"]["hit_rate"],
                h["matchup"]["pa"],
            ),
            reverse=True,
        )
        return found[:limit]


def choose_hr_picks(
    games: list[dict], locked: list[dict] = (), closed: set = frozenset(),
    n: int = 6,
) -> list[dict]:
    """Six HR picks by probability, one per team, one per game where possible.

    `locked` picks are kept first, as published; `closed` game_pks have started
    and can neither lose nor gain a pick, so a rebuild during the evening
    cannot swap in a hitter from a game already underway.
    """
    venue_of = {g["game_pk"]: g["venue"] for g in games}
    picks = list(locked)
    venues_used = {venue_of.get(p.get("game_pk"), p.get("game_pk")) for p in picks}
    teams_used = {p["team"] for p in picks}
    names = {p["batter"] for p in picks}

    hitters = sorted(
        (h for g in games for h in g["hitters"]),
        key=lambda h: h["prob_hr"], reverse=True,
    )
    for h in hitters:
        if len(picks) >= n:
            break
        if h.get("game_pk") in closed or h["batter"] in names:
            continue
        venue, team = venue_of.get(h.get("game_pk")), h["team"]
        # Prefer one hitter per game/venue; after three picks allow a second
        # from the same venue if the team differs.
        if venue not in venues_used and team not in teams_used:
            picks.append(h)
            venues_used.add(venue)
            teams_used.add(team)
        elif team not in teams_used and len(picks) >= 3:
            picks.append(h)
            teams_used.add(team)
    # A slate of one or two games never reaches three picks, so the other
    # club's best bat was never allowed in: a one-game playoff day posted a
    # single pick. Once every game has its pick, the remaining teams fill in.
    for h in hitters:
        if len(picks) >= n:
            break
        if h.get("game_pk") in closed or h["batter"] in names or h["team"] in teams_used:
            continue
        picks.append(h)
        teams_used.add(h["team"])
    return picks


def choose_hit_picks(
    games: list[dict], locked: list[dict] = (), closed: set = frozenset(),
    limit: int = 6,
) -> list[dict]:
    """Highest projected hit totals, one per team; see choose_hr_picks."""
    candidates = [
        h for g in games for h in g["hitters"]
        if h.get("hits_proj") and h.get("game_pk") not in closed
    ]
    # Sort on the unrounded value: three hitters showing 1.18 are not tied.
    candidates.sort(
        key=lambda h: h["hits_proj"].get(
            "projected_hits_raw", h["hits_proj"]["projected_hits"]
        ),
        reverse=True,
    )
    picks = list(locked)
    teams_used = {p["team"] for p in picks}
    for h in candidates:
        if len(picks) >= limit:
            break
        if h["team"] in teams_used:
            continue
        picks.append(h)
        teams_used.add(h["team"])
    return picks


def lock_started_picks(previous: dict, fresh: dict, started: set) -> dict:
    """Carry published picks for games that have started into a rebuilt slate.

    The slate is refitted through the day as lineups post. Before first pitch
    that is an improvement; after it, replacing a pick is grading the model on
    a choice it never made in time. So picks in started games are frozen as
    they were, and only the open slots are re-chosen from games not yet begun.
    """
    if not previous or not started:
        return fresh
    for key, chooser in (("picks_6", choose_hr_picks), ("hit_picks", choose_hit_picks)):
        kept = [p for p in previous.get(key) or [] if p.get("game_pk") in started]
        fresh[key] = chooser(fresh.get("games", []), locked=kept, closed=started)
    # A parlay is one ticket: once any leg's game is under way the whole
    # ticket stands as posted (parlays.choose).
    evidence = fresh.get("parlay_replay") or {p["key"]: p.get("evidence") for p in fresh.get("parlays") or []}
    fresh["parlays"] = parlay_mod.choose(
        fresh.get("games", []), previous=previous.get("parlays"), started=started,
        evidence=evidence,
    )
    fresh["locked_games"] = sorted(started)
    return fresh


def build_slate_for_date(
    today: date,
    model_path: str,
    use_ensemble: bool = True,
    use_espn: bool = True,
    hr_level: float = 1.0,
) -> dict:
    """Full slate build: load model, fit ensemble, fetch games, pick six.

    Args:
        today: Date to build slate for
        model_path: Path to season PA JSONL
        use_ensemble: If True, fit the comparables-prior ensemble (KNN + SVR);
            if False, use flat-prior empirical-Bayes only
        use_espn: If True, pull ESPN bio + injury data for features and for
            filtering unavailable players out of the pick pool
        hr_level: odds multiplier on every HR probability, from the tracker's
            own recent record (snapshot.hr_level)
    """
    data_dir = Path(model_path).parent
    # Nothing from `today` or later may inform today's projections.
    model_path = data_before(model_path, today)
    model = HRModel(model_path)

    # Measured, handedness-split park factors replace the hand-entered table,
    # which covered 15 of 34 venues and left the rest neutral.
    park_factors = build_park_factors(model_path)
    model.set_park_model(park_factors)
    print(summarize_parks(park_factors, top=3))

    # Weather, order-turn and umpire effects, measured within park.
    context = build_context_model(model_path, park_factors)
    print(context.summary())

    # ESPN supplies comparables features (height/weight/age) and the injury
    # report. A failure here degrades the model rather than breaking the page.
    espn = None
    if use_espn:
        try:
            from mlb_hr.espn import ESPNData
            cache = str(data_dir / "espn_cache.json")
            espn = ESPNData(cache).load()
        except Exception as e:
            print(f"[slate.py] Warning: ESPN data unavailable ({e}); continuing without it")

    # Fit ensemble if requested
    ensemble = None
    if use_ensemble:
        try:
            from mlb_hr.ensemble import EnsembleHRModel
            print("[slate.py] Fitting comparables-prior ensemble (KNN + SVR)...")
            ensemble = EnsembleHRModel(
                model, model_path, espn=espn, half_life_days=HALF_LIFE_DAYS,
            )
        except ImportError:
            print("[slate.py] Warning: scikit-learn not installed, using empirical-Bayes only")
        except Exception as e:
            print(f"[slate.py] Warning: could not fit ensemble: {e}")
            print("[slate.py] Proceeding with empirical-Bayes only")

    # Pitcher/team profiles come from the same PA feed as the hitter model, so
    # strikeout projections and lineup contact rates are measured on identical
    # terms. Standings supply the run totals the PA feed does not carry.
    pitching = build_pitching_profiles(model_path)
    pitchers, team_batting, league = pitching.pitchers, pitching.teams, pitching.league
    print(
        f"[slate.py] pitching profiles: {len(pitchers):,} pitchers, "
        f"{len(pitching.bullpens)} bullpens, {len(team_batting)} clubs, "
        f"league K {league.k_rate:.3f}, hit {league.hit_rate:.3f}, "
        f"{league.bf_per_start:.1f} BF/start, "
        f"{len(pitching.matchups):,} batter-pitcher pairs"
    )
    try:
        standings = team_standings(today.year)
        print(f"[slate.py] standings loaded for {len(standings)} teams")
    except Exception as e:
        print(f"[slate.py] Warning: standings unavailable ({e}); no win projections")
        standings = {}

    # Confirmed batting orders, when posted.
    try:
        todays_lineups = fetch_lineups(today)
        print(f"[slate.py] lineups posted for {len(todays_lineups)} games")
    except Exception as e:
        print(f"[slate.py] Warning: lineups unavailable ({e})")
        todays_lineups = {}

    try:
        starts, usual_slots = recent_lineups(model_path, str(today), RECENT_TEAM_GAMES)
    except Exception as e:  # noqa: BLE001 - the filter is a refinement
        print(f"[slate.py] Warning: recent starts unavailable ({e}); no bench filter")
        starts, usual_slots = None, {}

    slate = GameSlate(
        today, model, ensemble, espn,
        pitchers=pitchers,
        team_batting=team_batting,
        league=league,
        standings=standings,
        bullpens=pitching.bullpens,
        matchups=pitching.matchups,
        lineups=todays_lineups,
        context=context,
        recent_starts=starts,
        usual_slots=usual_slots,
        hr_level=hr_level,
    )
    if slate.excluded_bench:
        print(f"[slate.py] {len(slate.excluded_bench)} hitters held out pending "
              f"lineups (<{MIN_RECENT_STARTS} of last {RECENT_TEAM_GAMES} started)")

    picks = slate.random_6_pick_slate()
    hit_picks = slate.top_hit_projections()
    highlighted = slate.highlighted_matchups()

    espn_summary = None
    if espn:
        rostered = list(slate.active_hitters.keys())
        espn_summary = {
            "bios": len(espn.bios),
            "injuries": len(espn.injuries),
            "excluded_injured": len(slate.excluded_injured),
            "bio_coverage": espn.coverage(rostered),
        }

    from mlb_hr.ensemble import MODEL_VERSION, load_model_fit

    # The house parlays, with the season replay's record for each; the record
    # of every rule, offered or not, goes to the page's replay table.
    replay = load_model_fit().get("parlays") or {}
    # DraftKings' lines and prices as ESPN carries them: every parlay leg is a
    # line the book posts (parlays.offered). Without them the legs go unpriced.
    try:
        from mlb_hr import draftkings
        n_books = draftkings.attach(slate.games, today)
        print(f"[slate.py] DraftKings lines for {n_books} of {len(slate.games)} games")
    except Exception as e:  # noqa: BLE001 - the book is a refinement
        print(f"[slate.py] Warning: DraftKings lines unavailable ({e})")
    parlays = parlay_mod.choose(slate.games, evidence=replay)
    parlay_replay = {k: {f: v for f, v in r.items() if f != "wins_on"} for k, r in replay.items()}

    return {
        "date": str(today),
        "games_count": len(slate.games),
        "picks_6": picks,
        "games": slate.games,
        "parlay_replay": parlay_replay,
        # Which model produced these numbers, and the level it applied: the
        # tracker only learns its level from days the current model projected.
        "model_version": MODEL_VERSION if ensemble else "empirical_bayes",
        "hr_level": hr_level,
        "ensemble_metrics": ensemble.calibration if ensemble else None,
        "espn": espn_summary,
        "excluded_injured": slate.excluded_injured,
        "excluded_bench": slate.excluded_bench,
        "lineups_posted": len(todays_lineups),
        "lineups_used": slate.lineups_used,
        "context": {
            "fitted": context.fitted,
            "temp_per_degree": context.temp_per_degree,
            "wind_hr": context.wind_hr,
            "tto_hr": context.tto_hr,
            "tto_k": context.tto_k,
            "umpires_rated": len(context.ump_k),
        },
        "park_factors": {
            v: {"factor": p.factor, "lhb": p.factor_vs_lhb, "rhb": p.factor_vs_rhb,
                "measured": p.measured}
            for v, p in park_factors.items()
        },
        "hit_picks": hit_picks,
        "parlays": parlays,
        "highlighted_matchups": highlighted,
        "bullpens": {
            team: {
                "bf": pen.bf,
                "hr_rate": pen.hr_rate,
                "k_rate": pen.k_rate,
                "hit_rate": pen.hit_rate,
                "hr_index": pen.hr_index(league.hr_rate),
                "hit_index": pen.hit_index(league.hit_rate),
            }
            for team, pen in pitching.bullpens.items()
        },
        "league_rates": {
            "hr_rate": league.hr_rate,
            "k_rate": league.k_rate,
            "hit_rate": league.hit_rate,
            "bf_per_start": league.bf_per_start,
        },
    }
