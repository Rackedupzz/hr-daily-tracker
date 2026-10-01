"""The season replay must reproduce the live model exactly.

The replay (mlb_hr.replay) reimplements the live arithmetic in vectorized form
so a whole season can be replayed in minutes; the stacked models are fitted on
its rows. If it ever computes a component differently from the live code, the
coefficients are fitted to a model that is not the one being served. This
builds a synthetic season, then for one day compares every starter's replayed
HR probability and hit projection with what the live pipeline -- profiles,
serving prior, parks, context, pitching, EnsembleHRModel.pr_hr_ensemble,
project_hits -- produces from the same plate appearances.

Run: python -m pytest tests/test_replay.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthetic import write_season  # noqa: E402

DAY = "2026-05-06"


@pytest.fixture(scope="module")
def season(tmp_path_factory):
    from mlb_hr import replay

    root = tmp_path_factory.mktemp("season")
    feed = root / "season.jsonl"
    write_season(feed)
    rows = replay.replay(str(feed), start=DAY, espn=None, log=lambda *a: None)
    before = root / "before.jsonl"
    with open(feed) as src, open(before, "w") as dst:
        for line in src:
            if json.loads(line)["date"] < DAY:
                dst.write(line)
    return {"rows": rows, "before": before, "feed": feed, "root": root}


def _live(before: Path, fit_path: Path):
    from mlb_hr.context import build_context_model
    from mlb_hr.ensemble import ComparablesKNN, EnsembleHRModel, ServingPrior, load_model_fit
    from mlb_hr.features import build_profiles, platoon_factors
    from mlb_hr.model import HRModel
    from mlb_hr.parks import build_park_factors
    from mlb_hr.pitching import build_pitching_profiles
    from mlb_hr.slate import HALF_LIFE_DAYS

    model = HRModel(str(before))
    parks = build_park_factors(str(before))
    model.set_park_model(parks)
    profiles = build_profiles(str(before), half_life_days=HALF_LIFE_DAYS)
    pitching = build_pitching_profiles(str(before))
    # The served path of EnsembleHRModel without its (slow, unserved)
    # validation fits: every other variant falls back to the league rate.
    ens = EnsembleHRModel.__new__(EnsembleHRModel)
    ens.model, ens.profiles = model, profiles
    ens.platoon = platoon_factors(profiles)
    ens.serving = ServingPrior(profiles, None)
    ens.knn = ComparablesKNN()
    ens.league_rate = pitching.league.hr_rate
    ens.k_shrink, ens._prior_cache, ens.serve_variant = {}, {}, "served"
    fit = load_model_fit(fit_path)
    ens.stack_coef, ens.hit_coef = fit["hr_stack"]["coef"], fit["hit_stack"]["coef"]
    ens.top_cal, ens.hit_top_cal = fit["hr_top_cal"]["coef"], fit["hit_top_cal"]["coef"]
    return {"model": model, "parks": parks, "profiles": profiles, "pitching": pitching,
            "context": build_context_model(str(before), parks), "ens": ens}


def test_replay_reproduces_the_live_projection(season):
    from mlb_hr import replay
    from mlb_hr.ensemble import stacked_game_prob
    from mlb_hr.features import SLOT_PA
    from mlb_hr.pitching import HIT_STACK, HIT_TOP_CAL, project_hits

    rows = season["rows"]
    today = rows[rows["date"] == DAY].reset_index(drop=True)
    assert len(today) >= 80, "every game's starters are projected"
    live = _live(season["before"], season["root"] / "no_fit.json")
    ens, pitching, context = live["ens"], live["pitching"], live["context"]
    league = pitching.league
    games = {}
    for line in open(season["feed"]):
        r = json.loads(line)
        games.setdefault(r["game_pk"], r)

    X, off = replay.hit_design(today)
    hit_coef = np.array([HIT_STACK["intercept"]] + [HIT_STACK[t] for t in replay.HIT_TERMS])
    replay_hits = replay.apply_hit_top_cal(replay._predict_hits(hit_coef, X, off), HIT_TOP_CAL)
    switch_seen = 0
    for i, r in enumerate(today.itertuples(index=False)):
        batter = live["model"].get_batter(r.batter_id)
        profile = live["profiles"][r.batter_id]
        assert profile.bats == r.bats
        switch_seen += r.bats == "S"
        sp = pitching.pitchers.get(r.sp_id)
        pen = pitching.bullpens.get(r.opp)
        g = games[r.game_pk]
        weather = context.game_hr_factor(g["temp_f"], g["wind_mph"], g["wind_dir"])
        result = ens.pr_hr_ensemble(
            batter, -1, r.venue, r.sp_hand, bullpen=pen, league_hr_rate=league.hr_rate,
            expected_pa=SLOT_PA[r.slot], weather_factor=weather,
            sp_hr_index=sp.hr_index(league.hr_rate) if sp else 1.0,
        )
        replayed = stacked_game_prob(r.r_hand, r.r_all, r.park, r.weather, r.sp_index,
                                     r.pen_index, r.exposure, r.xhr_rate, r.lg_hr, ens.stack_coef,
                                     1.0, ens.top_cal)
        assert result["ensemble"] == pytest.approx(replayed, rel=1e-6), r.batter_id
        assert ens.per_pa_rate(profile, "ALL", "served") == pytest.approx(r.r_all, rel=1e-6)

        hits = project_hits(
            profile.hit_rate, profile.pa, sp, pen, league, SLOT_PA[r.slot],
            batter_xhit=profile.xhit, batter_k=profile.strikeouts,
            park_hit=live["model"].hit_park_factor_for(r.venue, profile.side_vs(r.sp_hand)),
            coef=ens.hit_coef, top_cal=ens.hit_top_cal,
        )
        assert hits["model"] == "stack"
        assert hits["projected_hits_raw"] == pytest.approx(replay_hits[i], rel=1e-6), r.batter_id
    assert switch_seen, "the synthetic season's switch hitters were projected"


def test_parks_were_measured_and_both_lineups_count(season):
    """Enough games to measure every park, so the park term was really exercised."""
    live = _live(season["before"], season["root"] / "no_fit.json")
    measured = [p for p in live["parks"].values() if p.measured]
    assert len(measured) == 10
    assert any(abs(p.factor - 1.0) > 0.02 for p in measured)
    assert any(abs(p.hit_factor - 1.0) > 0.0 for p in measured)


def test_fit_recovers_a_usable_model(season):
    """The stack fit runs on replay rows and every coefficient is finite."""
    from mlb_hr import replay

    rows = season["rows"]
    X, off = replay.hr_design(rows)
    b = replay.fit_logistic(X, rows["y_hr"].values.astype(float), off)
    assert np.all(np.isfinite(b)) and len(b) == len(replay.HR_TERMS) + 1
    Xh, oh = replay.hit_design(rows)
    bh = replay.fit_poisson(Xh, rows["y_hits"].values.astype(float), oh)
    assert np.all(np.isfinite(bh)) and len(bh) == len(replay.HIT_TERMS) + 1
