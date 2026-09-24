"""Pitcher and team-batting profiles, and the starting-pitcher strikeout projection.

Built from the same plate-appearance feed the hitter model uses, so a starter's
strikeout rate and the lineup he faces are measured on identical terms.

Two ideas carry the projection:

1. **Strikeouts are a matchup, not a pitcher constant.** A high-strikeout arm
   against a contact lineup lands somewhere between the two rates, and the
   odds-ratio (log5) formula is the standard way to combine them. Using the
   pitcher's rate alone systematically over-projects him against good contact
   teams and under-projects him against free swingers.
2. **Workload is its own estimate.** Strikeouts are a rate times an exposure,
   and the exposure -- batters faced in a start -- varies more between pitchers
   than the rate does. Both are shrunk toward league norms.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Run values per plate appearance, on the standard linear-weights scale
# (runs above average). Used to turn a pitcher's outcome mix into a single
# quality number for the game projection.
RUN_VALUES = {
    "home_run": 1.44,
    "triple": 1.03,
    "double": 0.75,
    "single": 0.47,
    "hit_by_pitch": 0.33,
    "walk": 0.31,
    "field_error": 0.30,
    "sac_fly": -0.22,
    "sac_bunt": -0.22,
    "field_out": -0.26,
    "force_out": -0.26,
    "strikeout": -0.27,
    "grounded_into_double_play": -0.50,
    "double_play": -0.50,
}
DEFAULT_RUN_VALUE = -0.26  # unrecognised events are almost always outs

HIT_EVENTS = {"single", "double", "triple", "home_run"}

# A starter faces about 21.3 batters, which is roughly 2.4 turns through a
# nine-man order. Everything past that belongs to the bullpen -- for a hitter
# with 4+ plate appearances that is most often his last two trips.
STARTER_TIMES_THROUGH_ORDER = 21.3 / 9.0

# Shrinkage strengths, in plate appearances of equivalent prior weight.
K_RATE_PRIOR_BF = 250.0
BF_PER_START_PRIOR = 6.0
TEAM_K_PRIOR_PA = 400.0

# Runs allowed regresses far harder than strikeout rate: a season of outcomes
# carries the defense behind the pitcher and a large helping of batted-ball
# luck. With a 300-BF prior the spread of starter quality came out roughly
# double the real one (a 3.0 run/9 ace, where ~1.5 is the honest figure), so
# the prior is deliberately heavier than the strikeout one.
RUN_VALUE_PRIOR_BF = 900.0

# A hitter's hit rate is mostly balls-in-play luck over a few hundred PA. Scored
# forward (to-date rate vs the next day's PAs) the log loss bottoms out at
# 400-600 PA of league-average weight; the old 200 let a 129-PA hot streak make
# Brett Bateman the top hit projection two days running.
HIT_PRIOR_PA = 500.0

# Starter home-run rates, shrunk this hard, carry through to the matchup at
# almost exactly odds-ratio strength: a forward logistic fit gave 1.13 on
# log(starter rate / league) with the batter's own rate controlled. The
# backtest residual correction flagged the missing term (odds x1.05 per SD).
SP_HR_PRIOR_BF = 900.0

# A starter who is removed early still had a full lineup's worth of exposure
# projected; clamp to plausible bounds.
MIN_BF_PER_START = 12.0
MAX_BF_PER_START = 30.0


@dataclass
class PitcherProfile:
    """Season profile for one pitcher, with batter-handedness splits."""
    pitcher_id: int
    name: str = ""
    hand: str = "R"

    bf: int = 0                 # batters faced
    strikeouts: int = 0
    walks: int = 0
    home_runs: int = 0
    hits_allowed: int = 0
    run_value_sum: float = 0.0
    inning_sum: int = 0

    starts: int = 0
    bf_as_starter: int = 0

    # Splits by the *batter's* side, which is what a lineup presents.
    bf_vs_l: int = 0
    k_vs_l: int = 0
    bf_vs_r: int = 0
    k_vs_r: int = 0

    _start_games: set = field(default_factory=set, repr=False)

    @property
    def k_rate(self) -> float:
        return self.strikeouts / self.bf if self.bf else 0.0

    @property
    def bb_rate(self) -> float:
        return self.walks / self.bf if self.bf else 0.0

    @property
    def hr_rate(self) -> float:
        return self.home_runs / self.bf if self.bf else 0.0

    @property
    def hit_rate(self) -> float:
        return self.hits_allowed / self.bf if self.bf else 0.0

    @property
    def run_value_per_bf(self) -> float:
        return self.run_value_sum / self.bf if self.bf else 0.0

    @property
    def bf_per_start(self) -> float:
        return self.bf_as_starter / self.starts if self.starts else 0.0

    def hr_index(self, league_hr_rate: float) -> float:
        """Shrunk HR rate allowed relative to league, as a per-PA multiplier."""
        if league_hr_rate <= 0:
            return 1.0
        shrunk = _shrink(self.home_runs, self.bf, league_hr_rate, SP_HR_PRIOR_BF)
        return float(np.clip(shrunk / league_hr_rate, 0.65, 1.45))

    def k_rate_vs(self, bat_side: str) -> tuple[int, int]:
        """(strikeouts, batters faced) against the given batter side."""
        if bat_side == "L":
            return self.k_vs_l, self.bf_vs_l
        return self.k_vs_r, self.bf_vs_r


@dataclass
class TeamBattingProfile:
    """Season batting profile for one club, split by pitcher handedness."""
    team: str
    pa: int = 0
    strikeouts: int = 0
    home_runs: int = 0
    run_value_sum: float = 0.0

    pa_vs_l: int = 0
    k_vs_l: int = 0
    pa_vs_r: int = 0
    k_vs_r: int = 0

    @property
    def k_rate(self) -> float:
        return self.strikeouts / self.pa if self.pa else 0.0

    def k_rate_vs(self, pitcher_hand: str) -> tuple[int, int]:
        """(strikeouts, plate appearances) against the given pitcher hand."""
        if pitcher_hand == "L":
            return self.k_vs_l, self.pa_vs_l
        return self.k_vs_r, self.pa_vs_r


@dataclass
class BullpenProfile:
    """Relief-corps profile for one club.

    Every plate appearance thrown by someone other than that game's starter.
    A hitter's last one or two trips are against these arms, so a slate that
    models only the starter is describing barely half the exposure -- and
    bullpens vary far more than rotations in what they give up.
    """
    team: str
    bf: int = 0
    strikeouts: int = 0
    walks: int = 0
    home_runs: int = 0
    hits_allowed: int = 0
    run_value_sum: float = 0.0

    # Split by the batter's side, since a pen can be lefty- or righty-heavy.
    bf_vs_l: int = 0
    hr_vs_l: int = 0
    bf_vs_r: int = 0
    hr_vs_r: int = 0

    # Late-inning core: the arms that actually pitch the seventh onward. A pen
    # averaged whole treats the closer and the mop-up man as one pitcher.
    late_bf: int = 0
    late_hr: int = 0
    late_k: int = 0

    @property
    def hr_rate(self) -> float:
        return self.home_runs / self.bf if self.bf else 0.0

    @property
    def late_hr_rate(self) -> float:
        return self.late_hr / self.late_bf if self.late_bf else self.hr_rate

    @property
    def late_k_rate(self) -> float:
        return self.late_k / self.late_bf if self.late_bf else self.k_rate

    @property
    def k_rate(self) -> float:
        return self.strikeouts / self.bf if self.bf else 0.0

    @property
    def hit_rate(self) -> float:
        return self.hits_allowed / self.bf if self.bf else 0.0

    def hr_rate_vs(self, bat_side: str) -> tuple[int, int]:
        """(home runs allowed, batters faced) against the given batter side."""
        if bat_side == "L":
            return self.hr_vs_l, self.bf_vs_l
        return self.hr_vs_r, self.bf_vs_r

    def hr_index(self, league_hr_rate: float, prior_bf: float = 1500.0,
                 late_only: bool = False) -> float:
        """Home runs allowed relative to league, shrunk toward 1.0.

        With late_only, uses just the arms that work the seventh inning on --
        the ones a hitter's final plate appearance actually meets. A pen
        averaged whole blends the closer with the mop-up man.
        """
        hr, bf = (
            (self.late_hr, self.late_bf) if late_only and self.late_bf > 400
            else (self.home_runs, self.bf)
        )
        if bf == 0 or league_hr_rate <= 0:
            return 1.0
        shrunk = _shrink(hr, bf, league_hr_rate, prior_bf)
        return float(np.clip(shrunk / league_hr_rate, 0.65, 1.45))

    def hit_index(self, league_hit_rate: float, prior_bf: float = 1500.0) -> float:
        """Hits allowed relative to league, shrunk toward 1.0."""
        if self.bf == 0 or league_hit_rate <= 0:
            return 1.0
        shrunk = _shrink(self.hits_allowed, self.bf, league_hit_rate, prior_bf)
        return float(np.clip(shrunk / league_hit_rate, 0.75, 1.30))


@dataclass
class MatchupHistory:
    """One batter's career-to-date line against one pitcher, from this feed."""
    pa: int = 0
    hits: int = 0
    home_runs: int = 0
    strikeouts: int = 0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.pa if self.pa else 0.0

    @property
    def hr_rate(self) -> float:
        return self.home_runs / self.pa if self.pa else 0.0


@dataclass
class PitchingData:
    """Everything one pass over the PA feed yields about pitching."""
    pitchers: dict
    teams: dict
    league: "LeagueRates"
    bullpens: dict
    matchups: dict


@dataclass
class LeagueRates:
    """League-wide baselines used as shrinkage targets."""
    k_rate: float = 0.22
    bb_rate: float = 0.086
    hr_rate: float = 0.030
    hit_rate: float = 0.225
    run_value_per_bf: float = 0.0
    bf_per_start: float = 21.6


def build_pitching_profiles(pa_jsonl_path: str, min_matchup_pa: int = 3) -> PitchingData:
    """One pass over the PA feed for pitcher, bullpen, team and matchup profiles.

    The starter for each half of a game is the pitcher who threw its first
    recorded plate appearance, which is how batters-faced-per-start is measured
    rather than assumed. Everyone else that half is bullpen, which is also how
    relief innings get separated without a second pass over the file: per-game
    per-pitcher counters are rolled up once the openers are known.
    """
    pitchers: dict[int, PitcherProfile] = {}
    teams: dict[str, TeamBattingProfile] = {}
    bullpens: dict[str, BullpenProfile] = {}
    matchups: dict[tuple, MatchupHistory] = {}
    # (game_pk, pitching_team) -> (lowest pa_index seen, pitcher_id)
    openers: dict[tuple, tuple[int, int]] = {}
    bf_by_game: dict[tuple, int] = {}
    # (game_pk, pitching_team, pitcher_id) -> counters, rolled up after the pass
    per_appearance: dict[tuple, list] = {}

    total_bf = total_k = total_bb = total_hr = total_hits = 0
    total_run_value = 0.0

    with open(pa_jsonl_path) as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            pitcher_id = record.get("pitcher_id")
            if not pitcher_id:
                continue

            event = record.get("event_type") or ""
            run_value = RUN_VALUES.get(event, DEFAULT_RUN_VALUE)
            is_k = event == "strikeout"
            is_bb = event == "walk"
            is_hit = event in HIT_EVENTS
            is_hr = int(record.get("is_hr", 0) or 0)
            bat_side = record.get("bat_side") or "R"
            pitch_hand = "L" if record.get("pitch_hand") == "L" else "R"

            profile = pitchers.get(pitcher_id)
            if profile is None:
                profile = PitcherProfile(pitcher_id=pitcher_id)
                pitchers[pitcher_id] = profile
            profile.name = record.get("pitcher", profile.name)
            profile.hand = pitch_hand
            profile.bf += 1
            profile.strikeouts += is_k
            profile.walks += is_bb
            profile.home_runs += is_hr
            profile.hits_allowed += is_hit
            profile.run_value_sum += run_value
            if bat_side == "L":
                profile.bf_vs_l += 1
                profile.k_vs_l += is_k
            else:
                profile.bf_vs_r += 1
                profile.k_vs_r += is_k

            # Batter-versus-pitcher history, for context only (see note in
            # matchup_note): these samples are far too small to project from.
            batter_id = record.get("batter_id")
            if batter_id:
                history = matchups.get((batter_id, pitcher_id))
                if history is None:
                    history = MatchupHistory()
                    matchups[(batter_id, pitcher_id)] = history
                history.pa += 1
                history.hits += is_hit
                history.home_runs += is_hr
                history.strikeouts += is_k

            # Per-appearance counters, rolled into bullpens once openers are known.
            inning = record.get("inning") or 0
            profile.inning_sum += inning
            is_late = inning >= 7

            appearance_key = (record.get("game_pk"), record.get("pitching_team"), pitcher_id)
            counters = per_appearance.get(appearance_key)
            if counters is None:
                counters = [0] * 13
                per_appearance[appearance_key] = counters
            if is_late:
                counters[10] += 1
                counters[11] += is_hr
                counters[12] += is_k
            counters[0] += 1                       # bf
            counters[1] += is_k
            counters[2] += is_bb
            counters[3] += is_hr
            counters[4] += is_hit
            counters[5] += run_value
            if bat_side == "L":
                counters[6] += 1
                counters[7] += is_hr
            else:
                counters[8] += 1
                counters[9] += is_hr

            team_name = record.get("batting_team") or ""
            if team_name:
                team = teams.get(team_name)
                if team is None:
                    team = TeamBattingProfile(team=team_name)
                    teams[team_name] = team
                team.pa += 1
                team.strikeouts += is_k
                team.home_runs += is_hr
                team.run_value_sum += run_value
                if pitch_hand == "L":
                    team.pa_vs_l += 1
                    team.k_vs_l += is_k
                else:
                    team.pa_vs_r += 1
                    team.k_vs_r += is_k

            key = (record.get("game_pk"), record.get("pitching_team"))
            index = record.get("pa_index", 0)
            if key not in openers or index < openers[key][0]:
                openers[key] = (index, pitcher_id)
            bf_by_game[(record.get("game_pk"), pitcher_id)] = (
                bf_by_game.get((record.get("game_pk"), pitcher_id), 0) + 1
            )

            total_bf += 1
            total_k += is_k
            total_bb += is_bb
            total_hr += is_hr
            total_hits += is_hit
            total_run_value += run_value

    # Credit each opener with a start and the batters he faced in it.
    for (game_pk, _team), (_index, pitcher_id) in openers.items():
        profile = pitchers.get(pitcher_id)
        if profile is None:
            continue
        profile.starts += 1
        profile.bf_as_starter += bf_by_game.get((game_pk, pitcher_id), 0)
        profile._start_games.add(game_pk)

    # Anyone who was not that half's opener pitched in relief.
    for (game_pk, team, pitcher_id), c in per_appearance.items():
        opener = openers.get((game_pk, team))
        if opener and opener[1] == pitcher_id:
            continue
        if not team:
            continue
        pen = bullpens.get(team)
        if pen is None:
            pen = BullpenProfile(team=team)
            bullpens[team] = pen
        pen.bf += c[0]
        pen.strikeouts += c[1]
        pen.walks += c[2]
        pen.home_runs += c[3]
        pen.hits_allowed += c[4]
        pen.run_value_sum += c[5]
        pen.bf_vs_l += c[6]
        pen.hr_vs_l += c[7]
        pen.bf_vs_r += c[8]
        pen.hr_vs_r += c[9]
        pen.late_bf += c[10]
        pen.late_hr += c[11]
        pen.late_k += c[12]

    # Drop one- and two-PA matchup pairs: too thin to display, and they would
    # dominate the dictionary.
    matchups = {k: v for k, v in matchups.items() if v.pa >= min_matchup_pa}

    starter_bf = [p.bf_per_start for p in pitchers.values() if p.starts >= 5]
    league = LeagueRates(
        k_rate=total_k / total_bf if total_bf else 0.22,
        bb_rate=total_bb / total_bf if total_bf else 0.086,
        hr_rate=total_hr / total_bf if total_bf else 0.030,
        hit_rate=total_hits / total_bf if total_bf else 0.225,
        run_value_per_bf=total_run_value / total_bf if total_bf else 0.0,
        bf_per_start=float(np.mean(starter_bf)) if starter_bf else 21.6,
    )
    return PitchingData(
        pitchers=pitchers,
        teams=teams,
        league=league,
        bullpens=bullpens,
        matchups=matchups,
    )


def odds_ratio_rate(pitcher_rate: float, batter_rate: float, league_rate: float) -> float:
    """Combine a pitcher rate and a batter rate into a matchup rate (log5).

    Neither side's rate alone describes the matchup. The odds-ratio form is the
    standard resolution: convert both to odds relative to league, multiply, and
    convert back.
    """
    eps = 1e-6
    p = min(max(pitcher_rate, eps), 1 - eps)
    b = min(max(batter_rate, eps), 1 - eps)
    lg = min(max(league_rate, eps), 1 - eps)

    odds = (p / (1 - p)) * (b / (1 - b)) / (lg / (1 - lg))
    return float(odds / (1 + odds))


def _shrink(successes: float, trials: float, prior_rate: float, prior_weight: float) -> float:
    return (successes + prior_rate * prior_weight) / (trials + prior_weight)


def project_strikeouts(
    pitcher: PitcherProfile,
    opponent: Optional[TeamBattingProfile],
    league: LeagueRates,
) -> dict:
    """Projected strikeouts for one start, with the terms that produced it.

    Returns the point estimate plus the components, so the projection can be
    shown as a chain of reasoning rather than a bare number.
    """
    # Pitcher's strikeout rate, shrunk toward the league rate.
    sp_k_rate = _shrink(
        pitcher.strikeouts, pitcher.bf, league.k_rate, K_RATE_PRIOR_BF
    )

    # Opposing lineup's strikeout rate against this pitcher's hand.
    if opponent is not None:
        opp_k, opp_pa = opponent.k_rate_vs(pitcher.hand)
        opp_k_rate = _shrink(opp_k, opp_pa, league.k_rate, TEAM_K_PRIOR_PA)
    else:
        opp_k_rate = league.k_rate

    matchup_k_rate = odds_ratio_rate(sp_k_rate, opp_k_rate, league.k_rate)

    # Expected batters faced, shrunk toward the league starter workload.
    expected_bf = _shrink(
        pitcher.bf_as_starter,
        pitcher.starts,
        league.bf_per_start,
        BF_PER_START_PRIOR,
    )
    expected_bf = float(np.clip(expected_bf, MIN_BF_PER_START, MAX_BF_PER_START))

    projected = expected_bf * matchup_k_rate
    # Binomial spread around the point estimate, for an honest range.
    sd = math.sqrt(expected_bf * matchup_k_rate * (1 - matchup_k_rate))

    return {
        "projected_k": round(projected, 1),
        "low": max(0, int(round(projected - sd))),
        "high": int(round(projected + sd)),
        "expected_bf": round(expected_bf, 1),
        "matchup_k_rate": matchup_k_rate,
        "sp_k_rate": sp_k_rate,
        "sp_k_rate_raw": pitcher.k_rate,
        "opp_k_rate": opp_k_rate,
        "league_k_rate": league.k_rate,
        "sp_strikeouts": pitcher.strikeouts,
        "sp_bf": pitcher.bf,
        "sp_starts": pitcher.starts,
        "hand": pitcher.hand,
        "name": pitcher.name,
    }


def pitcher_quality_index(
    pitcher: Optional[PitcherProfile],
    league: LeagueRates,
    league_rpg: float = 4.4,
) -> float:
    """Runs-allowed multiplier for a starter, relative to a league-average arm.

    Below 1.0 suppresses runs. Derived from the pitcher's outcome mix on the
    linear-weights scale, shrunk hard toward league average, then scaled by a
    full team-game of plate appearances so the number lands in runs-per-game
    terms.
    """
    if pitcher is None or pitcher.bf == 0:
        return 1.0

    rv = _shrink(
        pitcher.run_value_sum,
        pitcher.bf,
        league.run_value_per_bf,
        RUN_VALUE_PRIOR_BF,
    )
    # ~38 plate appearances is one team's full game, so this is runs per 9.
    runs_above_average = (rv - league.run_value_per_bf) * 38.0
    return float(np.clip((league_rpg + runs_above_average) / league_rpg, 0.70, 1.30))


def split_exposure(expected_pa: float) -> tuple[float, float]:
    """Divide a hitter's plate appearances between the starter and the bullpen.

    A starter faces roughly 2.4 turns through the order, so the first two or
    three trips are his and the rest belong to relief. Modelling every PA
    against the starter -- as the slate did before -- attributes the late-game
    exposure to the wrong arms, which matters because bullpens and rotations
    give up home runs at meaningfully different rates.
    """
    vs_starter = min(expected_pa, STARTER_TIMES_THROUGH_ORDER)
    return vs_starter, max(0.0, expected_pa - vs_starter)


def project_hits(
    batter_hit_rate: float,
    batter_pa: int,
    pitcher: Optional[PitcherProfile],
    bullpen: Optional[BullpenProfile],
    league: LeagueRates,
    expected_pa: float,
) -> dict:
    """Projected hits for one hitter, starter and bullpen exposure separated.

    Same odds-ratio matchup logic as the strikeout projection: a hitter's rate
    and the pitcher's rate are combined relative to league rather than one
    being used alone.
    """
    hitter_rate = _shrink(
        batter_hit_rate * batter_pa, batter_pa, league.hit_rate, HIT_PRIOR_PA
    )

    if pitcher is not None and pitcher.bf > 0:
        sp_rate = _shrink(
            pitcher.hits_allowed, pitcher.bf, league.hit_rate, 400.0
        )
    else:
        sp_rate = league.hit_rate
    vs_sp = odds_ratio_rate(sp_rate, hitter_rate, league.hit_rate)

    pen_index = bullpen.hit_index(league.hit_rate) if bullpen else 1.0
    vs_pen = float(np.clip(hitter_rate * pen_index, 0.0, 0.9))

    pa_sp, pa_pen = split_exposure(expected_pa)
    projected = pa_sp * vs_sp + pa_pen * vs_pen

    return {
        "projected_hits": round(projected, 2),
        "projected_hits_raw": float(projected),
        "hitter_rate": hitter_rate,
        "sp_hit_rate": sp_rate,
        "rate_vs_sp": vs_sp,
        "rate_vs_pen": vs_pen,
        "pen_hit_index": pen_index,
        "pa_vs_sp": round(pa_sp, 1),
        "pa_vs_pen": round(pa_pen, 1),
        "expected_pa": round(expected_pa, 1),
        # Probability of at least one hit, which is how the outcome is usually
        # framed, computed from the two exposures separately.
        "prob_at_least_one": float(
            1.0 - (1.0 - vs_sp) ** pa_sp * (1.0 - vs_pen) ** pa_pen
        ),
    }


def matchup_note(
    history: Optional[MatchupHistory],
    batter_hit_rate: float,
    batter_hr_rate: float,
    min_pa: int = 6,
) -> Optional[dict]:
    """Flag a batter-versus-pitcher history worth showing.

    Deliberately NOT fed into any projection. In this feed the largest
    batter-vs-pitcher sample all season is 13 plate appearances, and only ~520
    pairs reach even 8. At that size the standard error on a hit rate is around
    .16 -- wider than the entire spread of true talent between major-league
    hitters -- so a 4-for-9 line is noise wearing the costume of a trend. It is
    shown as history with its sample size attached, and every projection on
    this page ignores it.
    """
    if history is None or history.pa < min_pa:
        return None

    hot = history.hit_rate > batter_hit_rate * 1.25 or history.home_runs >= 2
    cold = history.hit_rate < batter_hit_rate * 0.6 and history.pa >= 8

    if not (hot or cold):
        return None

    return {
        "pa": history.pa,
        "hits": history.hits,
        "home_runs": history.home_runs,
        "strikeouts": history.strikeouts,
        "hit_rate": history.hit_rate,
        "baseline_hit_rate": batter_hit_rate,
        "baseline_hr_rate": batter_hr_rate,
        "verdict": "hot" if hot else "cold",
    }
