"""Game win/loss projection.

Built as a runs model rather than a direct win-probability fit, because the
question "who wins?" is only useful here if it comes with the reasoning. Each
side's expected runs is assembled from named parts -- offense, opposing starter,
opposing bullpen, park -- and the win probability falls out of those. Every term
is reported alongside the result so the projection can be read as an argument.

Team strength comes from Pythagorean expectation on runs scored/allowed rather
than raw W-L: run differential is a steadier estimate of true talent, since a
record absorbs bullpen luck and one-run-game variance.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from mlb_hr.pitching import (
    LeagueRates,
    PitcherProfile,
    TeamBattingProfile,
    pitcher_quality_index,
)

# Pythagorean exponent for baseball run differential (Davenport/Smyth).
PYTHAGOREAN_EXPONENT = 1.83

# A modern starter covers roughly 5.4 of 9 innings; the bullpen covers the rest.
STARTER_INNINGS_SHARE = 0.60

# Home teams win about 54% of games. Applied as a symmetric nudge to expected
# runs so it flows through the same Pythagorean step as everything else.
HOME_RUNS_MULTIPLIER = 1.04
AWAY_RUNS_MULTIPLIER = 0.96

# Shrinkage for team scoring rates, in games of equivalent prior weight.
TEAM_PRIOR_GAMES = 30.0

DEFAULT_LEAGUE_RPG = 4.4


def league_runs_per_game(standings: dict[int, dict]) -> float:
    """League average runs scored per team-game."""
    runs = sum(t.get("runs_scored", 0) for t in standings.values())
    games = sum(t.get("games_played", 0) for t in standings.values())
    return runs / games if games else DEFAULT_LEAGUE_RPG


def _team_index(
    total: float, games: float, league_rpg: float, prior_games: float = TEAM_PRIOR_GAMES
) -> float:
    """Runs-per-game relative to league, shrunk toward 1.0."""
    if games <= 0:
        return 1.0
    shrunk = (total + league_rpg * prior_games) / (games + prior_games)
    return float(np.clip(shrunk / league_rpg, 0.6, 1.5))


def pythagorean_win_pct(runs_scored: float, runs_allowed: float) -> float:
    """Expected win percentage from run totals."""
    if runs_scored <= 0 and runs_allowed <= 0:
        return 0.5
    rs = max(runs_scored, 1e-6) ** PYTHAGOREAN_EXPONENT
    ra = max(runs_allowed, 1e-6) ** PYTHAGOREAN_EXPONENT
    return float(rs / (rs + ra))


def project_game(
    home_id: int,
    away_id: int,
    home_team: str,
    away_team: str,
    standings: dict[int, dict],
    home_sp: Optional[PitcherProfile],
    away_sp: Optional[PitcherProfile],
    league: LeagueRates,
    park_factor: float = 1.0,
) -> Optional[dict]:
    """Projected runs and win probability for one game.

    Returns None when either club is missing from the standings, so the caller
    can omit the projection rather than show an invented one.
    """
    home_record = standings.get(home_id)
    away_record = standings.get(away_id)
    if not home_record or not away_record:
        return None

    league_rpg = league_runs_per_game(standings) or DEFAULT_LEAGUE_RPG

    # Offense and team run prevention, as multipliers on league average.
    home_offense = _team_index(
        home_record["runs_scored"], home_record["games_played"], league_rpg
    )
    away_offense = _team_index(
        away_record["runs_scored"], away_record["games_played"], league_rpg
    )
    home_defense = _team_index(
        home_record["runs_allowed"], home_record["games_played"], league_rpg
    )
    away_defense = _team_index(
        away_record["runs_allowed"], away_record["games_played"], league_rpg
    )

    # The starter covers most of the game; the rest reverts to the club's own
    # run prevention, which is dominated by its bullpen.
    home_sp_index = pitcher_quality_index(home_sp, league, league_rpg)
    away_sp_index = pitcher_quality_index(away_sp, league, league_rpg)
    home_run_prevention = (
        STARTER_INNINGS_SHARE * home_sp_index
        + (1 - STARTER_INNINGS_SHARE) * home_defense
    )
    away_run_prevention = (
        STARTER_INNINGS_SHARE * away_sp_index
        + (1 - STARTER_INNINGS_SHARE) * away_defense
    )

    # Park scaling is milder for runs than for home runs specifically.
    park_runs = 1.0 + (park_factor - 1.0) * 0.5

    home_runs_exp = (
        league_rpg * home_offense * away_run_prevention * park_runs * HOME_RUNS_MULTIPLIER
    )
    away_runs_exp = (
        league_rpg * away_offense * home_run_prevention * park_runs * AWAY_RUNS_MULTIPLIER
    )

    home_win = pythagorean_win_pct(home_runs_exp, away_runs_exp)
    home_win = float(np.clip(home_win, 0.05, 0.95))

    favorite, underdog = (
        (home_team, away_team) if home_win >= 0.5 else (away_team, home_team)
    )

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_win_prob": home_win,
        "away_win_prob": 1.0 - home_win,
        "favorite": favorite,
        "underdog": underdog,
        "favorite_prob": max(home_win, 1.0 - home_win),
        "home_expected_runs": round(home_runs_exp, 2),
        "away_expected_runs": round(away_runs_exp, 2),
        "total_runs": round(home_runs_exp + away_runs_exp, 2),
        "run_line": round(home_runs_exp - away_runs_exp, 2),
        "league_rpg": round(league_rpg, 2),
        # Supporting detail, so the projection can be read as an argument.
        "components": {
            "home_offense_index": round(home_offense, 3),
            "away_offense_index": round(away_offense, 3),
            "home_bullpen_index": round(home_defense, 3),
            "away_bullpen_index": round(away_defense, 3),
            "home_sp_index": round(home_sp_index, 3),
            "away_sp_index": round(away_sp_index, 3),
            "park_runs_factor": round(park_runs, 3),
            "home_field": HOME_RUNS_MULTIPLIER,
            "starter_share": STARTER_INNINGS_SHARE,
        },
        "records": {
            "home": _record_summary(home_record),
            "away": _record_summary(away_record),
        },
    }


def _record_summary(record: dict) -> dict:
    """Record plus the Pythagorean expectation implied by run differential."""
    games = max(record.get("games_played", 0), 1)
    return {
        "wins": record.get("wins", 0),
        "losses": record.get("losses", 0),
        "runs_scored": record.get("runs_scored", 0),
        "runs_allowed": record.get("runs_allowed", 0),
        "run_differential": record.get("run_differential", 0),
        "runs_per_game": round(record.get("runs_scored", 0) / games, 2),
        "runs_allowed_per_game": round(record.get("runs_allowed", 0) / games, 2),
        "pythagorean_win_pct": round(
            pythagorean_win_pct(
                record.get("runs_scored", 0), record.get("runs_allowed", 0)
            ),
            3,
        ),
        "streak": record.get("streak", ""),
        "division_rank": record.get("division_rank", ""),
    }
