"""Season replay of the live projection, and the fit of its stacked layers.

Every morning the slate refits on the plate appearances before that day and
projects the day's hitters. This module does the same for every day of the
season at once: for each day it rebuilds, from the PAs strictly before it,
exactly the components the live model combines --

  * the hitter's recency-weighted home runs, split by the starter's hand, and
    the full-season comparables prior they are shrunk toward (ServingPrior,
    K x SHRINK_MULT), with platoon factors by batting hand;
  * expected home runs and hits from his contact (features.contact_bin);
  * the handed, pull-leveraged park factor (home vs the club's road games);
  * the weather factor, measured within park;
  * the opposing starter's and bullpen's HR indices; the starter's hit and
    strikeout rates; the bullpen's hit index;
  * the starter's plate appearances for his lineup slot (features.SLOT_PA),

-- and records what actually happened. Only the day's starters are projected
(the first hitter in each lineup slot): substitutes inherit a slot in the feed
and would read as starters who went hitless in one plate appearance.

Those rows are what the stacked layers are fitted on (ensemble.HR_STACK and
pitching.HIT_STACK), and the fit is validated the only honest way: walk
forward, refitting weekly on the weeks before, scoring the week after.

It reimplements the live arithmetic in vectorized form -- refitting the full
pipeline for each of ~165 days would take hours -- but takes every constant
and helper from the live modules, and tests/test_replay.py checks it against
the live code on a synthetic season, so the two cannot drift apart unnoticed.
The first version of this replay reproduced the published 9/12-9/27 slates:
per-PA rates correlated 0.997-1.000 day by day, park/starter/bullpen indices
matched to four decimals.

Run: python -m mlb_hr.replay            # replay, fit, validate, write data/model_fit.json
     python -m mlb_hr.replay --no-write # report only
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from mlb_hr.context import (
    REFERENCE_TEMP_F, TEMP_CLAMP, WIND_CLAMP, WIND_PRIOR_PA, wind_category,
)
from mlb_hr.ensemble import (
    FIT_PATH, HR_STACK, MODEL_VERSION, RELIABLE_PA, SHRINK_MULT, XHR_PRIOR_PA,
    _estimate_k,
)
from mlb_hr.features import (
    BARREL_LA_RANGE, BARREL_MIN_EV, FLY_BALL_LA_RANGE, HARD_HIT_EV, HIT_EVENT_TYPES,
    PULL_ANGLE_MIN, SLOT_PA, SWITCH_MIN_PA, SWITCH_MIN_SHARE, XHR_EV_BINS, XHR_EV_LO,
    XHR_EV_STEP, XHR_LA_BINS, XHR_LA_LO, XHR_LA_STEP, _LEAGUE_DEFAULTS,
)
from mlb_hr.model import DEFAULT_PARK_FACTORS
from mlb_hr.parks import PARK_PRIOR_PA, PARK_PRIOR_PA_SPLIT, _regress
from mlb_hr.pitching import (
    BATTER_K_PRIOR_PA, HIT_PRIOR_PA, HIT_STACK, PEN_HR_PRIOR_BF, SP_HR_PRIOR_BF,
    SP_K_PRIOR_BF, XHIT_PRIOR_PA,
)
from mlb_hr.slate import HALF_LIFE_DAYS

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_START = "04-15"          # month-day; three weeks of warm-up before scoring
PEN_HIT_PRIOR_BF = 1500.0        # BullpenProfile.hit_index default
SP_HIT_PRIOR_BF = 400.0          # project_hits' starter hit prior
HR_TERMS = [k for k in HR_STACK if k != "intercept"]
HIT_TERMS = [k for k in HIT_STACK if k != "intercept"]
_FEED_FIELDS = (
    "game_pk", "date", "venue", "batting_team", "pitching_team", "is_home",
    "batter_id", "batter", "bat_side", "pitcher_id", "pitch_hand", "pa_index",
    "event_type", "is_hr", "launch_speed", "launch_angle", "pull_angle",
    "lineup_slot", "temp_f", "wind_mph", "wind_dir",
)


# ------------------------------------------------------------------ loading

def load_feed(pa_path: str) -> pd.DataFrame:
    """The season feed as a frame, in file order (the order profiles see it)."""
    rows = []
    with open(pa_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append([r.get(k) for k in _FEED_FIELDS])
    df = pd.DataFrame(rows, columns=list(_FEED_FIELDS))
    df["file_order"] = np.arange(len(df))
    df = df[df["batter_id"].notna() & df["pitcher_id"].notna()].copy()
    df["batter_id"] = df["batter_id"].astype(int)
    df["pitcher_id"] = df["pitcher_id"].astype(int)
    df["is_hr"] = df["is_hr"].fillna(0).astype(int)
    df["pitch_hand"] = np.where(df["pitch_hand"] == "L", "L", "R")
    df["bat_side"] = df["bat_side"].fillna("")
    df["event_type"] = df["event_type"].fillna("")
    return df


def _pool_rates(nn: NearestNeighbors, Xq: np.ndarray, q_ids, ids, hr, pa, k, league):
    """ComparablesKNN.prior for many rows: pooled HR/PA of the k nearest, self excluded."""
    _, idx = nn.kneighbors(Xq)
    out = np.empty(len(Xq))
    for i in range(len(Xq)):
        nb = [j for j in idx[i] if ids[j] != q_ids[i]][:k]
        h, p = hr[nb].sum(), pa[nb].sum()
        out[i] = h / p if nb and p > 0 else league
    return out


class VectorPrior:
    """ServingPrior, vectorized: same scaling, weights, k search and K estimate."""

    CANDIDATES = (10, 15, 20, 30, 40, 60)

    def __init__(self, X_raw: np.ndarray, hr: np.ndarray, pa: np.ndarray, ids: np.ndarray,
                 default_k: int = 40):
        from mlb_hr.ensemble import _feature_weights

        self.ids, self.hr, self.pa = ids, hr, pa
        self.league = float(hr.sum() / pa.sum())
        self.scaler = StandardScaler().fit(X_raw)
        X = self.scaler.transform(X_raw)
        self.weights = _feature_weights(X, hr, pa)
        X = X * self.weights
        self.X = X
        rates = hr / np.maximum(pa, 1)
        mask = pa >= RELIABLE_PA
        best_k, best = default_k, float("inf")
        if mask.sum() >= 30:
            for k in self.CANDIDATES:
                if k >= len(X):
                    continue
                nn = NearestNeighbors(n_neighbors=min(k + 1, len(X))).fit(X)
                prior = _pool_rates(nn, X, ids, ids, hr, pa, k, self.league)
                mse = float(np.average((prior[mask] - rates[mask]) ** 2, weights=pa[mask]))
                if mse < best:
                    best_k, best = k, mse
        k = min(best_k, max(1, len(X) - 1))
        self.k = k
        self.nn = NearestNeighbors(n_neighbors=min(k + 1, len(X))).fit(X)
        oof = _pool_rates(self.nn, X, ids, ids, hr, pa, k, self.league)
        k_league = _estimate_k(hr, pa, np.full(len(hr), self.league)) or 300.0
        self.K = _estimate_k(hr, pa, np.clip(oof, 0.002, 0.15)) or k_league

    def prior(self, X_raw_q: np.ndarray, q_ids: np.ndarray) -> np.ndarray:
        Xq = self.scaler.transform(X_raw_q) * self.weights
        return _pool_rates(self.nn, Xq, q_ids, self.ids, self.hr, self.pa, self.k, self.league)


# --------------------------------------------------------------- the season

class Season:
    """Cumulative, day-indexed tallies of the feed, for as-of-morning lookups."""

    def __init__(self, feed: pd.DataFrame, espn=None):
        pa = feed.sort_values(["game_pk", "pa_index"]).reset_index(drop=True)
        self.pa = pa
        self.days = sorted(pa["date"].unique())
        self.day_ix = {d: i for i, d in enumerate(self.days)}
        self.ordinal = np.array([date.fromisoformat(d).toordinal() for d in self.days])
        nd = self.ND = len(self.days)
        di = pa["date"].map(self.day_ix).values
        self.DI = di

        self.bat_ids = np.array(sorted(pa["batter_id"].unique()))
        bix = {b: i for i, b in enumerate(self.bat_ids)}
        bi = pa["batter_id"].map(bix).values
        self.bix = bix
        pit_ids = sorted(pa["pitcher_id"].unique())
        self.pix = {p: i for i, p in enumerate(pit_ids)}
        pi = pa["pitcher_id"].map(self.pix).values
        teams = sorted(set(pa["batting_team"].dropna()) | set(pa["pitching_team"].dropna()))
        self.tix = {t: i for i, t in enumerate(teams)}
        self.teams = teams
        venues = sorted(pa["venue"].fillna("").unique())
        self.vix = {v: i for i, v in enumerate(venues)}
        self.venues = venues
        vi = pa["venue"].fillna("").map(self.vix).values
        nb, np_, nt, nv = len(self.bat_ids), len(pit_ids), len(teams), len(venues)

        et = pa["event_type"].values
        ev = pa["launch_speed"].astype(float).values
        la = pa["launch_angle"].astype(float).values
        pull = pa["pull_angle"].astype(float).values
        has_ev, has_la = ~np.isnan(ev), ~np.isnan(la)
        is_hr = pa["is_hr"].values.astype(float)
        is_hit = np.isin(et, list(HIT_EVENT_TYPES)).astype(float)
        is_k = (et == "strikeout").astype(float)
        hand_l = (pa["pitch_hand"].values == "L").astype(float)
        side = pa["bat_side"].values
        ones = np.ones(len(pa))

        def daily(values, rows, n):
            out = np.zeros((n, nd))
            np.add.at(out, (rows, di), values)
            c = np.zeros((n, nd + 1))
            c[:, 1:] = np.cumsum(out, axis=1)
            return out, c

        # --- batters: daily counts (for recency weights) and cumulative sums
        spray = has_ev & ~np.isnan(pull)
        la_ok = has_ev & has_la
        stats = {
            "pa": ones, "hr": is_hr, "pa_L": hand_l, "hr_L": is_hr * hand_l,
            "pa_R": 1 - hand_l, "hr_R": is_hr * (1 - hand_l), "hits": is_hit, "k": is_k,
            "bb": (et == "walk").astype(float),
            "xbh": np.isin(et, ["double", "triple"]).astype(float),
            "bip": has_ev.astype(float), "ev_sum": np.where(has_ev, ev, 0.0),
            "hard": (has_ev & (ev >= HARD_HIT_EV)).astype(float),
            "barrel": (la_ok & (ev >= BARREL_MIN_EV) & (la >= BARREL_LA_RANGE[0])
                       & (la <= BARREL_LA_RANGE[1])).astype(float),
            "fb": (la_ok & (la >= FLY_BALL_LA_RANGE[0]) & (la <= FLY_BALL_LA_RANGE[1])).astype(float),
            "la_sum": np.where(la_ok, la, 0.0),
            "spray": spray.astype(float), "pulled": (spray & (pull > PULL_ANGLE_MIN)).astype(float),
            "as_l": (side == "L").astype(float), "as_r": (side == "R").astype(float),
        }
        self.B, self.CB = {}, {}
        for name, vals in stats.items():
            self.B[name], self.CB[name] = daily(vals, bi, nb)
        games = pa.groupby([bi, di])["game_pk"].nunique()
        g = np.zeros((nb, nd))
        g[games.index.get_level_values(0), games.index.get_level_values(1)] = games.values
        self.CB["games"] = np.concatenate([np.zeros((nb, 1)), np.cumsum(g, axis=1)], axis=1)

        # last-seen side and name per batter (file order), for fallbacks and ESPN
        last = pa.sort_values("file_order").groupby(["batter_id", "date"])["bat_side"].last()
        self.last_side: dict[int, list] = defaultdict(list)
        for (b, d), s in last.items():
            self.last_side[bix[b]].append((self.day_ix[d], s))
        names = pa.sort_values("file_order").groupby("batter_id")["batter"].last()
        self.names = {bix[b]: n for b, n in names.items()}

        # EV lists for the 90th percentile
        evs = pa.loc[has_ev, ["batter_id", "date", "launch_speed"]]
        self.ev_by_b = {}
        for b, grp in evs.groupby("batter_id"):
            grp = grp.assign(d=grp["date"].map(self.day_ix)).sort_values("d")
            self.ev_by_b[bix[b]] = (grp["d"].values, grp["launch_speed"].astype(float).values)

        # contact bins for xHR / xhit (batted balls with both EV and LA)
        e = np.clip((ev - XHR_EV_LO) // XHR_EV_STEP, 0, XHR_EV_BINS - 1)
        a = np.clip((la - XHR_LA_LO) // XHR_LA_STEP, 0, XHR_LA_BINS - 1)
        m = la_ok
        self.bip_b, self.bip_d = bi[m], di[m]
        self.bip_bin = (e[m] * XHR_LA_BINS + a[m]).astype(int)
        self.bip_hr, self.bip_hit = is_hr[m], is_hit[m]
        self.NBIN = XHR_EV_BINS * XHR_LA_BINS

        # ESPN bio (height, weight, age), with the profile defaults
        self.bio = np.tile([_LEAGUE_DEFAULTS["height_in"], _LEAGUE_DEFAULTS["weight_lb"],
                            _LEAGUE_DEFAULTS["age"]], (nb, 1)).astype(float)
        if espn is not None:
            for b in range(nb):
                x = espn.bio_for(self.names.get(b, ""))
                if x is None:
                    continue
                for j, attr in enumerate(("height_in", "weight_lb", "age")):
                    v = getattr(x, attr, None)
                    if v is not None:
                        self.bio[b, j] = float(v)

        # --- pitchers, bullpens (relief = not the half's first pitcher), league
        openers = pa.groupby(["game_pk", "pitching_team"])["pitcher_id"].first()
        self.openers = openers.to_dict()
        opener_arr = np.array([self.openers[(gk, t)] for gk, t in
                               zip(pa["game_pk"], pa["pitching_team"])])
        relief = (pa["pitcher_id"].values != opener_arr).astype(float)
        pt = pa["pitching_team"].map(self.tix).values
        self.P = {n: daily(v, pi, np_)[1] for n, v in
                  (("bf", ones), ("hr", is_hr), ("hits", is_hit), ("k", is_k))}
        self.PEN = {n: daily(v * relief, pt, nt)[1] for n, v in
                    (("bf", ones), ("hr", is_hr), ("hits", is_hit))}
        self.LG = {n: np.concatenate([[0.0], np.cumsum(np.bincount(di, weights=v, minlength=nd))])
                   for n, v in (("pa", ones), ("hr", is_hr), ("hits", is_hit), ("k", is_k))}
        self.hand_of = pa.groupby("pitcher_id")["pitch_hand"].agg(lambda s: s.mode().iloc[0]).to_dict()

        # --- parks: venue totals by side, and each club's games (home/away) by venue
        side_lr = np.where(side == "L", 0, 1)
        home_team = pa.loc[pa["is_home"] == 1].groupby("game_pk")["batting_team"].first()
        away_team = pa.loc[pa["is_home"] != 1].groupby("game_pk")["batting_team"].first()
        h_arr = pa["game_pk"].map(home_team).map(self.tix).fillna(-1).astype(int).values
        a_arr = pa["game_pk"].map(away_team).map(self.tix).fillna(-1).astype(int).values
        self.VEN = {}
        self.CLUB_AWAY, self.CLUB_HOME = {}, {}
        for n, v in (("pa", ones), ("hr", is_hr), ("hits", is_hit)):
            ven = np.zeros((nv, 2, nd))
            np.add.at(ven, (vi, side_lr, di), v)
            self.VEN[n] = np.concatenate([np.zeros((nv, 2, 1)), np.cumsum(ven, 2)], 2)
            for store, teams_ in ((self.CLUB_AWAY, a_arr), (self.CLUB_HOME, h_arr)):
                ok = teams_ >= 0
                arr = np.zeros((nt, nv, 2, nd))
                np.add.at(arr, (teams_[ok], vi[ok], side_lr[ok], di[ok]), v[ok])
                store[n] = np.concatenate([np.zeros((nt, nv, 2, 1)), np.cumsum(arr, 3)], 3)
        home_games = pa.loc[pa["is_home"] == 1].groupby(["game_pk"]).agg(
            team=("batting_team", "first"), venue=("venue", "first"), date=("date", "first"))
        hg = np.zeros((nt, nv, nd))
        for _, r in home_games.iterrows():
            hg[self.tix[r["team"]], self.vix[r["venue"] or ""], self.day_ix[r["date"]]] += 1
        self.home_games = np.concatenate([np.zeros((nt, nv, 1)), np.cumsum(hg, 2)], 2)

        # --- weather cells, measured within park (context.build_context_model)
        temp = pa["temp_f"].astype(float).values
        self.temp_bucket = np.where(np.isnan(temp), -999,
                                    np.round(np.nan_to_num(temp) / 5.0) * 5).astype(int)
        has_wind = (pa["wind_dir"].notna() | pa["wind_mph"].notna()).values
        self.wind_key = np.where(has_wind, [wind_category(w) for w in pa["wind_dir"].values],
                                 "__none__")
        self.VI, self.is_hr = vi, is_hr

        # --- starters and outcomes
        slotted = pa[pa["lineup_slot"].notna()]
        first = slotted.groupby(["game_pk", "batting_team", "lineup_slot"])["batter_id"].first()
        starters = pd.DataFrame({"game_pk": first.index.get_level_values(0),
                                 "team": first.index.get_level_values(1),
                                 "slot": first.index.get_level_values(2).astype(int),
                                 "batter_id": first.values})
        per_game = pa.groupby(["game_pk", "batter_id"]).agg(
            y_pa=("is_hr", "size"), y_hr=("is_hr", "sum"),
            y_hits=("event_type", lambda s: int(s.isin(HIT_EVENT_TYPES).sum())))
        info = pa.groupby("game_pk").agg(date=("date", "first"), venue=("venue", "first"),
                                         temp=("temp_f", "first"), wind=("wind_dir", "first"))
        s = starters.merge(per_game, left_on=["game_pk", "batter_id"], right_index=True)
        s = s.merge(info, left_on="game_pk", right_index=True)
        s["opp"] = [home_team.get(g) if home_team.get(g) != t else away_team.get(g)
                    for g, t in zip(s["game_pk"], s["team"])]
        s["is_home"] = (s["team"] == s["game_pk"].map(home_team)).astype(int)
        s["sp_id"] = [self.openers.get((g, o)) for g, o in zip(s["game_pk"], s["opp"])]
        s["sp_hand"] = s["sp_id"].map(self.hand_of).fillna("R")
        s["di"] = s["date"].map(self.day_ix)
        s["bi"] = s["batter_id"].map(bix)
        self.starters = s.sort_values(["di", "game_pk", "team", "slot"]).reset_index(drop=True)

    # ---------------------------------------------------------- as of day d

    def features_at(self, d: int, rows: np.ndarray) -> np.ndarray:
        """BatterProfile.feature_vector for `rows`, from PAs before day index d."""
        g = {k: self.CB[k][rows, d] for k in ("pa", "bip", "ev_sum", "hard", "barrel", "fb",
                                               "la_sum", "k", "bb", "xbh", "spray", "pulled")}
        bip = g["bip"]
        pos = bip > 0
        safe = np.maximum(bip, 1)
        ev90 = np.full(len(rows), _LEAGUE_DEFAULTS["ev90"])
        for i, b in enumerate(rows):
            if pos[i]:
                dd, vals = self.ev_by_b[b]
                ev90[i] = np.percentile(vals[:np.searchsorted(dd, d, side="left")], 90)
        p1 = np.maximum(g["pa"], 1)
        return np.column_stack([
            np.log1p(g["pa"]),
            np.where(pos, g["ev_sum"] / safe, _LEAGUE_DEFAULTS["avg_ev"]),
            ev90,
            np.where(pos, g["hard"] / safe, _LEAGUE_DEFAULTS["hard_hit_rate"]),
            np.where(pos, g["barrel"] / safe, _LEAGUE_DEFAULTS["barrel_rate"]),
            np.where(pos, g["fb"] / safe, _LEAGUE_DEFAULTS["fly_ball_rate"]),
            np.where(pos, g["la_sum"] / safe, _LEAGUE_DEFAULTS["avg_la"]),
            g["bip"] / p1, g["k"] / p1, g["bb"] / p1, g["xbh"] / p1,
            np.where(g["spray"] >= 20, g["pulled"] / np.maximum(g["spray"], 1),
                     _LEAGUE_DEFAULTS["pull_rate"]),
            self.bio[rows],
        ])

    def bats(self, d: int, rows) -> dict:
        """BatterProfile.bats as of day d."""
        out = {}
        for b in rows:
            l, r = self.CB["as_l"][b, d], self.CB["as_r"][b, d]
            low, high = sorted((l, r))
            if low >= SWITCH_MIN_PA and low / (low + high) >= SWITCH_MIN_SHARE:
                out[b] = "S"
            elif l == r:
                seen = [s for dd, s in self.last_side[b] if dd < d]
                out[b] = seen[-1] if seen and seen[-1] in ("L", "R") else "R"
            else:
                out[b] = "L" if l > r else "R"
        return out

    def platoon(self, d: int, bats: dict) -> dict:
        """features.platoon_factors as of day d."""
        tot = defaultdict(lambda: [0.0, 0.0])
        by = defaultdict(lambda: [0.0, 0.0])
        for b, s in bats.items():
            if s == "S":
                continue
            for hand in ("L", "R"):
                h, n = self.CB[f"hr_{hand}"][b, d], self.CB[f"pa_{hand}"][b, d]
                tot[(s, hand)][0] += h
                tot[(s, hand)][1] += n
                by[s][0] += h
                by[s][1] += n
        f = {}
        for key, (h, n) in tot.items():
            sh, sn = by[key[0]]
            f[key] = 1.0 if n < 500 or sn == 0 or sh == 0 else float(
                np.clip((h / n) / (sh / sn), 0.75, 1.35))
        f[("S", "L")] = f[("S", "R")] = 1.0
        return f

    def parks(self, d: int) -> dict:
        """parks.build_park_factors as of day d: venue -> (hr L, hr R, hr all, hit L, hit R, hit all)."""
        hp = {}
        counts = self.home_games[:, :, d]
        for t in range(len(self.teams)):
            if counts[t].sum() > 0:
                hp[t] = int(np.argmax(counts[t]))    # most home games; ties -> first venue
        owner = {v: t for t, v in hp.items()}
        out = {}
        for v, vi in self.vix.items():
            vpa, vhr, vhit = (self.VEN[n][vi, :, d] for n in ("pa", "hr", "hits"))
            t = owner.get(vi)
            default = DEFAULT_PARK_FACTORS.get(v, 1.0)
            if t is None:
                out[v] = (default, default, default, 1.0, 1.0, 1.0)
                continue
            road = {}
            for n in ("pa", "hr", "hits"):
                arr = self.CLUB_AWAY[n][t, :, :, d] + self.CLUB_HOME[n][t, :, :, d]
                arr[hp[t]] = 0.0                     # games in his own park are not road games
                road[n] = arr.sum(axis=0)            # by side
            rpa, rhr, rhit = road["pa"], road["hr"], road["hits"]
            if rpa.sum() > 1000 and vpa.sum() > 500:
                f = _regress(vhr.sum() / vpa.sum(), rhr.sum() / rpa.sum(), vpa.sum(), PARK_PRIOR_PA)
                fh = _regress(vhit.sum() / vpa.sum(), rhit.sum() / rpa.sum(), vpa.sum(), PARK_PRIOR_PA)
                sides, hsides = [], []
                for s in (0, 1):
                    ok = vpa[s] > 300 and rpa[s] > 500
                    sides.append(_regress(vhr[s] / vpa[s], rhr[s] / rpa[s], vpa[s], PARK_PRIOR_PA_SPLIT)
                                 if ok and rhr[s] > 0 else f)
                    hsides.append(_regress(vhit[s] / vpa[s], rhit[s] / rpa[s], vpa[s], PARK_PRIOR_PA_SPLIT)
                                  if ok and rhit[s] > 0 else fh)
                out[v] = (sides[0], sides[1], f, hsides[0], hsides[1], fh)
            else:
                out[v] = (default, default, default, 1.0, 1.0, 1.0)
        return out

    def weather_model(self, d: int) -> tuple[float, dict]:
        """(temp per degree, wind factors) from context.build_context_model as of d."""
        m = self.DI < d
        vdf = pd.DataFrame({"v": self.VI[m], "hr": self.is_hr[m]})
        vt = vdf.groupby("v")["hr"].agg(["sum", "count"])
        vrate = (vt["sum"] / vt["count"])[vt["count"] > 500].to_dict()

        def pool(keys):
            cells = pd.DataFrame({"v": self.VI[m], "g": keys[m], "hr": self.is_hr[m]}).groupby(
                ["v", "g"])["hr"].agg(["sum", "count"]).reset_index()
            cells = cells[cells["v"].isin(vrate.keys()) & (cells["count"] >= 150)]
            cells["exp"] = cells["v"].map(vrate) * cells["count"]
            return cells.groupby("g")[["sum", "exp", "count"]].sum()

        tpd = 0.0
        tp = pool(self.temp_bucket)
        tp = tp[(tp.index != -999) & (tp["count"] >= 2000) & (tp["exp"] > 0)]
        if len(tp) >= 4:
            slope, icpt = np.polyfit(tp.index.values.astype(float), (tp["sum"] / tp["exp"]).values,
                                     1, w=np.sqrt(tp["count"].values))
            at = slope * REFERENCE_TEMP_F + icpt
            if at > 0:
                tpd = float(slope / at)
        wind = {}
        for g, row in pool(self.wind_key).iterrows():
            if g == "__none__" or row["count"] < 2000:
                continue
            raw = row["sum"] / row["exp"]
            w = row["count"] / (row["count"] + WIND_PRIOR_PA)
            wind[g] = float(np.clip(1 + (raw - 1) * w, *WIND_CLAMP))
        return tpd, wind

    def expected_contact(self, d: int, weights_by_day: np.ndarray):
        """(xHR, recency-weighted xHR, xhit) per batter from contact before d."""
        m = self.bip_d < d
        n = np.bincount(self.bip_bin[m], minlength=self.NBIN)
        h = np.bincount(self.bip_bin[m], weights=self.bip_hr[m], minlength=self.NBIN)
        t = np.bincount(self.bip_bin[m], weights=self.bip_hit[m], minlength=self.NBIN)
        tot = max(n.sum(), 1)
        hr_table = (h + 2 * h.sum() / tot) / (n + 2)
        hit_table = (t + 2 * t.sum() / tot) / (n + 2)
        nb = len(self.bat_ids)
        b, cells = self.bip_b[m], self.bip_bin[m]
        xhr = np.bincount(b, weights=hr_table[cells], minlength=nb)
        xhr_w = np.bincount(b, weights=hr_table[cells] * weights_by_day[self.bip_d[m]], minlength=nb)
        xhit = np.bincount(b, weights=hit_table[cells], minlength=nb)
        return xhr, xhr_w, xhit


def _shrink(s, n, prior, w):
    return (s + prior * w) / (n + w)


def replay(pa_path: str, start: Optional[str] = None, espn=None, log=print) -> pd.DataFrame:
    """One row per starter per game from `start`, every input as of that morning."""
    t0 = time.perf_counter()
    season = Season(load_feed(pa_path), espn)
    days = season.days
    start = start or f"{days[0][:4]}-{DEFAULT_START}"
    first_d = next((i for i, d in enumerate(days) if d >= start), len(days))
    log(f"[replay] {len(season.pa):,} PA, {len(season.bat_ids)} hitters, "
        f"{len(days)} days; projecting starters from {days[min(first_d, len(days) - 1)]}")
    rows = []
    prior_cache: dict = {}
    for d in range(max(first_d, 1), season.ND):
        today = season.starters[season.starters["di"] == d]
        # The slate only projects hitters with 2+ plate appearances on record.
        today = today[season.CB["pa"][today["bi"].values, d] >= 2]
        if today.empty:
            continue
        anchor = season.ordinal[d - 1]
        age = anchor - season.ordinal[:d]
        w_day = np.where(age <= 0, 1.0, 0.5 ** (age / HALF_LIFE_DAYS))
        w_day_full = np.zeros(season.ND)
        w_day_full[:d] = w_day

        fit_rows = np.flatnonzero(season.CB["pa"][:, d] >= 1)
        if d not in prior_cache:
            prior_cache.clear()
            prior_cache[d] = VectorPrior(
                season.features_at(d, fit_rows), season.CB["hr"][fit_rows, d],
                season.CB["pa"][fit_rows, d], season.bat_ids[fit_rows])
        vp = prior_cache[d]
        q = today["bi"].values
        prior = vp.prior(season.features_at(d, q), season.bat_ids[q])
        K = vp.K * SHRINK_MULT
        W = {k: season.B[k][q, :d] @ w_day for k in ("pa", "hr", "pa_L", "hr_L", "pa_R", "hr_R")}
        bats = season.bats(d, fit_rows)
        plat = season.platoon(d, bats)
        parks = season.parks(d)
        tpd, wind = season.weather_model(d)
        xhr, xhr_w, xhit = season.expected_contact(d, w_day_full)
        lg_pa = season.LG["pa"][d]
        lg_hr, lg_hit, lg_k = (season.LG[n][d] / lg_pa for n in ("hr", "hits", "k"))

        for i, r in enumerate(today.itertuples(index=False)):
            b = r.bi
            hand = r.sp_hand
            side = bats.get(b, "R")
            vs = ("R" if hand == "L" else "L") if side == "S" else side
            hr_h, pa_h = W[f"hr_{hand}"][i], W[f"pa_{hand}"][i]
            if pa_h <= 0:
                hr_h, pa_h = W["hr"][i], W["pa"][i]
            r_hand = float(np.clip(_shrink(hr_h, pa_h, prior[i] * plat.get((side, hand), 1.0), K),
                                   0.0005, 0.15))
            r_all = float(np.clip(_shrink(W["hr"][i], W["pa"][i], prior[i], K), 0.0005, 0.15))
            pk = parks.get(r.venue, (1.0,) * 6)
            park_side = pk[0] if vs == "L" else pk[1]
            spray = season.CB["spray"][b, d]
            pull = season.CB["pulled"][b, d] / spray if spray >= 20 else _LEAGUE_DEFAULTS["pull_rate"]
            park = 1.0 + (park_side - 1.0) * float(np.clip(pull / 0.40, 0.7, 1.3))
            temp = r.temp
            temp_f = 1.0 if temp is None or pd.isna(temp) else float(
                np.clip(1.0 + tpd * (float(temp) - REFERENCE_TEMP_F), *TEMP_CLAMP))
            wx = temp_f * wind.get(wind_category(r.wind), 1.0)
            spi = season.pix.get(r.sp_id)
            sp_bf, sp_hr, sp_hits, sp_k = ((season.P[n][spi, d] for n in ("bf", "hr", "hits", "k"))
                                           if spi is not None else (0.0, 0.0, 0.0, 0.0))
            ti = season.tix.get(r.opp)
            pen_bf, pen_hr, pen_hits = ((season.PEN[n][ti, d] for n in ("bf", "hr", "hits"))
                                        if ti is not None else (0.0, 0.0, 0.0))
            sp_idx = float(np.clip(_shrink(sp_hr, sp_bf, lg_hr, SP_HR_PRIOR_BF) / lg_hr, 0.65, 1.45))
            pen_idx = (float(np.clip(_shrink(pen_hr, pen_bf, lg_hr, PEN_HR_PRIOR_BF) / lg_hr, 0.65, 1.45))
                       if pen_bf > 0 else 1.0)
            pen_hit = (float(np.clip(_shrink(pen_hits, pen_bf, lg_hit, PEN_HIT_PRIOR_BF) / lg_hit,
                                     0.75, 1.30)) if pen_bf > 0 else 1.0)
            pa_b = season.CB["pa"][b, d]
            exposure = float(np.clip(SLOT_PA[r.slot], 1.5, 5.2))
            rows.append({
                "date": r.date, "di": d, "game_pk": r.game_pk, "batter_id": r.batter_id,
                "batter": season.names.get(b), "team": r.team, "opp": r.opp, "venue": r.venue,
                "slot": r.slot, "is_home": r.is_home, "sp_id": r.sp_id, "sp_hand": hand,
                "bats": side, "y_hr": int(r.y_hr > 0), "y_hits": int(r.y_hits), "y_pa": int(r.y_pa),
                "lg_hr": lg_hr, "lg_hit": lg_hit, "lg_k": lg_k,
                # HR components (ensemble.stacked_game_prob)
                "r_hand": r_hand, "r_all": r_all, "park": park, "weather": wx,
                "sp_index": sp_idx, "pen_index": pen_idx, "exposure": exposure,
                "xhr_rate": float((xhr_w[b] + XHR_PRIOR_PA * lg_hr) / (W["pa"][i] + XHR_PRIOR_PA)),
                # hit components (pitching.project_hits)
                "hitter_hit_rate": _shrink(season.CB["hits"][b, d], pa_b, lg_hit, HIT_PRIOR_PA),
                "sp_hit_rate": _shrink(sp_hits, sp_bf, lg_hit, SP_HIT_PRIOR_BF) if sp_bf > 0 else lg_hit,
                "pen_hit_index": pen_hit,
                "park_hit": pk[3] if vs == "L" else pk[4],
                "xhit_rate": _shrink(xhit[b], pa_b, lg_hit, XHIT_PRIOR_PA),
                "k_rate": _shrink(season.CB["k"][b, d], pa_b, lg_k, BATTER_K_PRIOR_PA),
                "sp_k_rate": _shrink(sp_k, sp_bf, lg_k, SP_K_PRIOR_BF) if sp_bf > 0 else lg_k,
                "prior": prior[i], "K": K, "pa": pa_b,
            })
        if d % 20 == 0:
            log(f"[replay]   {days[d]}: {len(rows):,} starter-games ({time.perf_counter() - t0:.0f}s)")
    out = pd.DataFrame(rows)
    log(f"[replay] {len(out):,} starter-games in {time.perf_counter() - t0:.0f}s")
    return out


# ------------------------------------------------------------ design matrices

def hr_design(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(X in HR_TERMS order, offset) -- the terms of ensemble.stacked_game_prob."""
    lg = np.clip(rows["lg_hr"].values, 1e-4, 0.2)
    r_all = np.maximum(rows["r_all"].values, 1e-6)
    cols = {
        "batter_rate": np.log(r_all / lg),
        "hand": np.log(np.maximum(rows["r_hand"].values, 1e-6) / r_all),
        "park": np.log(np.maximum(rows["park"].values, 1e-3)),
        "weather": np.log(np.maximum(rows["weather"].values, 1e-3)),
        "starter": np.log(np.maximum(rows["sp_index"].values, 1e-3)),
        "bullpen": np.log(np.maximum(rows["pen_index"].values, 1e-3)),
        "exposure": np.log(np.clip(rows["exposure"].values, 1.5, 5.2)),
        "xhr": np.log(np.maximum(rows["xhr_rate"].values, 1e-6) / lg),
    }
    return np.column_stack([cols[t] for t in HR_TERMS]), np.log(lg / (1 - lg))


def hit_design(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(X in HIT_TERMS order, offset) -- the terms of pitching.project_hits' stack."""
    lg = rows["lg_hit"].values
    lg_k = np.maximum(rows["lg_k"].values, 1e-3)
    cols = {
        "hitter_rate": np.log(rows["hitter_hit_rate"].values / lg),
        "starter_hits": np.log(rows["sp_hit_rate"].values / lg),
        "bullpen": np.log(np.maximum(rows["pen_hit_index"].values, 1e-3)),
        "park": np.log(np.maximum(rows["park_hit"].values, 1e-3)),
        "xhit": np.log(np.maximum(rows["xhit_rate"].values, 1e-6) / lg),
        "k_rate": np.log(rows["k_rate"].values / lg_k),
        "starter_k": np.log(rows["sp_k_rate"].values / lg_k),
        "exposure": np.log(np.clip(rows["exposure"].values, 1.5, 5.2)),
    }
    return np.column_stack([cols[t] for t in HIT_TERMS]), np.log(lg)


def fit_logistic(X, y, offset, lam: float = 1.0, w=None) -> np.ndarray:
    Z = np.column_stack([np.ones(len(X)), X])

    def f(b):
        t = offset + Z @ b
        loss = np.logaddexp(0, t) - y * t
        g = Z.T @ ((expit(t) - y) * (1 if w is None else w))
        g[1:] += lam * b[1:]
        return np.sum(loss if w is None else w * loss) + 0.5 * lam * b[1:] @ b[1:], g
    return minimize(f, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B").x


def fit_poisson(X, y, offset, lam: float = 1.0) -> np.ndarray:
    Z = np.column_stack([np.ones(len(X)), X])

    def f(b):
        eta = offset + Z @ b
        mu = np.exp(eta)
        g = Z.T @ (mu - y)
        g[1:] += lam * b[1:]
        return np.sum(mu - y * eta) + 0.5 * lam * b[1:] @ b[1:], g
    return minimize(f, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B").x


def _predict_hr(b, X, offset):
    return np.clip(expit(offset + np.column_stack([np.ones(len(X)), X]) @ b), 1e-4, 0.6)


def fit_top_cal(p: np.ndarray, y: np.ndarray) -> dict:
    """ensemble.HR_TOP_CAL: logistic on [logit(p), logit(p)^2]."""
    z = np.log(p / (1 - p))
    b = fit_logistic(np.column_stack([z, z * z]), y, np.zeros(len(y)), lam=0.0)
    return {"a": float(b[0]), "b": float(b[1]), "c": float(b[2])}


def apply_top_cal(p: np.ndarray, cal: dict) -> np.ndarray:
    from mlb_hr.ensemble import top_calibrate

    return np.array([top_calibrate(v, cal) if np.isfinite(v) else np.nan for v in p])


def fit_hit_top_cal(mu: np.ndarray, y: np.ndarray) -> dict:
    """pitching.HIT_TOP_CAL: Poisson on [log(mu), log(mu)^2]."""
    x = np.log(mu)
    b = fit_poisson(np.column_stack([x, x * x]), y, np.zeros(len(y)), lam=0.0)
    return {"a": float(b[0]), "b": float(b[1]), "c": float(b[2])}


def apply_hit_top_cal(mu: np.ndarray, cal: dict) -> np.ndarray:
    from mlb_hr.pitching import hit_top_calibrate

    return np.array([hit_top_calibrate(v, cal) if np.isfinite(v) else np.nan for v in mu])


def walk_forward_top_cal(di: np.ndarray, p: np.ndarray, y: np.ndarray, min_days: int = 35,
                         every: int = 7, fit=None, apply=None) -> np.ndarray:
    """A top calibration refit weekly on earlier walk-forward predictions."""
    fit, apply = fit or fit_top_cal, apply or apply_top_cal
    ok = ~np.isnan(p)
    out = np.full(len(p), np.nan)
    days = np.sort(np.unique(di[ok]))
    cal, last = None, None
    for d in days:
        idx = ok & (di == d)
        if d - days[0] < min_days:
            out[idx] = p[idx]
            continue
        if cal is None or d - last >= every:
            train = ok & (di < d)
            cal, last = fit(p[train], y[train]), d
        out[idx] = apply(p[idx], cal)
    return out


def _predict_hits(b, X, offset):
    return np.exp(offset + np.column_stack([np.ones(len(X)), X]) @ b)


def walk_forward(di: np.ndarray, X: np.ndarray, off: np.ndarray, y: np.ndarray, fit, predict,
                 warmup_days: int = 21, every: int = 7) -> np.ndarray:
    """Predict each week from a fit on every earlier day (NaN during warm-up)."""
    days = np.sort(np.unique(di))
    out = np.full(len(di), np.nan)
    coef, last = None, None
    for d in days:
        if d - days[0] < warmup_days:
            continue
        train = di < d
        if coef is None or d - last >= every:
            coef, last = fit(X[train], y[train], off[train]), d
        idx = di == d
        out[idx] = predict(coef, X[idx], off[idx])
    return out


def level_series(rows: pd.DataFrame, p: np.ndarray, half_life: float, pseudo: float) -> np.ndarray:
    """snapshot.hr_level, applied walk-forward to a column of predictions.

    Rows without a prediction (NaN, the warm-up) neither receive a level nor
    count toward one, as a day the tracker did not project would not.
    """
    from mlb_hr.snapshot import LEVEL_CLAMP

    y = rows["y_hr"].values
    ordinal = np.array([date.fromisoformat(d).toordinal() for d in rows["date"]])
    ok = ~np.isnan(p)
    by_day = pd.DataFrame({"o": ordinal[ok], "y": y[ok], "p": p[ok]}).groupby("o")[["y", "p"]].sum()
    out = np.array(p, dtype=float)
    for o in np.unique(ordinal[ok]):
        prev = by_day.index.values < o
        if not prev.any():
            continue
        w = 0.5 ** ((o - by_day.index.values[prev]) / half_life)
        m = (w @ by_day["y"].values[prev] + pseudo) / (w @ by_day["p"].values[prev] + pseudo)
        m = min(max(m, LEVEL_CLAMP[0]), LEVEL_CLAMP[1])
        idx = (ordinal == o) & ok
        odds = p[idx] / (1 - p[idx]) * m
        out[idx] = odds / (1 + odds)
    return out


# -------------------------------------------------------------------- scoring

def _top6_rows(rows: pd.DataFrame, score: np.ndarray, kind: str) -> np.ndarray:
    """The slate's own pick rules (slate.choose_hr_picks / choose_hit_picks)."""
    from mlb_hr.slate import choose_hit_picks, choose_hr_picks

    picked = []
    for _d, idx in rows.groupby("di").indices.items():
        games = {}
        for i in idx:
            r = rows.iloc[i]
            g = games.setdefault(r["game_pk"], {"game_pk": r["game_pk"], "venue": r["venue"],
                                                 "hitters": []})
            g["hitters"].append({"i": i, "batter": f"{r['batter_id']}", "team": r["team"],
                                 "game_pk": r["game_pk"], "prob_hr": score[i],
                                 "hits_proj": {"projected_hits": score[i],
                                               "projected_hits_raw": score[i]}})
        chooser = choose_hr_picks if kind == "hr" else choose_hit_picks
        picked.extend(h["i"] for h in chooser(list(games.values())))
    return np.array(picked, dtype=int)


BANDS = (0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 1.0)


def _bands(p: np.ndarray, y: np.ndarray) -> list:
    """Projected vs actual by percentile band of the projection (top bands matter:
    the picks come from the top 2-3%)."""
    cuts = np.quantile(p, BANDS)
    band = np.clip(np.searchsorted(cuts, p, side="right") - 1, 0, len(BANDS) - 2)
    return [[f"{BANDS[k]:.0%}-{BANDS[k + 1]:.0%}", round(float(p[band == k].mean()), 4),
             round(float(y[band == k].mean()), 4)] for k in range(len(BANDS) - 1)]


def score_hr(rows: pd.DataFrame, p: np.ndarray) -> dict:
    y = rows["y_hr"].values
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    top = _top6_rows(rows, p, "hr")
    return {
        "n": int(len(y)), "log_loss": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
        "auc": float(roc_auc_score(y, p)), "projected": float(p.sum()), "actual": int(y.sum()),
        "top6_rate": float(y[top].mean()), "top6_n": int(len(top)),
        "top6_projected": float(p[top].sum()), "top6_actual": int(y[top].sum()),
        "bands": _bands(p, y),
        "by_slot": {int(s): round(float(y[m].sum() / p[m].sum()), 3)
                    for s, m in ((s, rows["slot"].values == s) for s in range(1, 10))},
        "by_month": {mo: round(float(y[m].sum() / p[m].sum()), 3)
                     for mo, m in ((mo, rows["date"].str[:7].values == mo)
                                   for mo in sorted(rows["date"].str[:7].unique()))},
    }


def score_hits(rows: pd.DataFrame, mu: np.ndarray) -> dict:
    y = rows["y_hits"].values.astype(float)
    exposure = np.clip(rows["exposure"].values, 1.5, 5.2)
    p1 = 1 - (1 - np.clip(mu / exposure, 0, 0.9)) ** exposure
    y1 = (y > 0).astype(float)
    pc = np.clip(p1, 1e-6, 1 - 1e-6)
    dev = 2 * np.mean(np.where(y > 0, y * np.log(np.maximum(y, 1e-9) / mu), 0) - (y - mu))
    top = _top6_rows(rows, mu, "hits")
    return {
        "n": int(len(y)), "poisson_deviance": float(dev),
        "p1_log_loss": float(-np.mean(y1 * np.log(pc) + (1 - y1) * np.log(1 - pc))),
        "p1_auc": float(roc_auc_score(y1, p1)), "projected": float(mu.sum()), "actual": int(y.sum()),
        "top6_one_plus": float(y1[top].mean()), "top6_projected": float(mu[top].sum()),
        "top6_actual": int(y[top].sum()), "bands": _bands(mu, y),
        "by_slot": {int(s): round(float(y[m].sum() / mu[m].sum()), 3)
                    for s, m in ((s, rows["slot"].values == s) for s in range(1, 10))},
    }


def parlay_record(rows: pd.DataFrame, p: np.ndarray, y: np.ndarray, legs: int = 5) -> dict:
    """mlb_hr.parlays' ticket, rebuilt on every day with a prediction.

    Each game's likeliest leg, the best `legs` of those; the ticket cashes when
    every leg does. The replay knows lineups but not injury notes, so it can
    only take a leg the live page would have passed over for one -- a minor gap.
    """
    df = pd.DataFrame({"date": rows["date"].values, "game": rows["game_pk"].values,
                       "p": p, "y": y}).dropna(subset=["p"])
    days = won = 0
    products, wins_on = [], []
    for day, g in df.groupby("date", sort=True):
        best = g.loc[g.groupby("game")["p"].idxmax()]
        top = best.nlargest(legs, "p")
        if len(top) < legs:
            continue
        days += 1
        products.append(float(np.prod(top["p"].values)))
        if bool(top["y"].all()):
            won += 1
            wins_on.append(day)
    return {"legs": legs, "days": days, "won": won,
            "rate": round(won / days, 4) if days else None,
            "predicted": round(float(np.mean(products)), 6) if products else None,
            "expected_wins": round(float(np.sum(products)), 3), "wins_on": wins_on}


def log5_hits(rows: pd.DataFrame) -> np.ndarray:
    """The pre-stack hit projection (log5 + bullpen index), same inputs."""
    from mlb_hr.pitching import odds_ratio_rate, split_exposure

    out = np.empty(len(rows))
    for i, r in enumerate(rows.itertuples(index=False)):
        vs_sp = odds_ratio_rate(r.sp_hit_rate, r.hitter_hit_rate, r.lg_hit)
        vs_pen = min(max(r.hitter_hit_rate * r.pen_hit_index, 0.0), 0.9)
        a, b = split_exposure(float(np.clip(r.exposure, 1.5, 5.2)))
        out[i] = a * vs_sp + b * vs_pen
    return out


def _fmt_hr(label: str, s: dict) -> str:
    return (f"  {label:<34} log loss {s['log_loss']:.5f}  AUC {s['auc']:.4f}  "
            f"projected {s['projected']:7.1f} vs {s['actual']:5d}  top-6 {s['top6_rate']:.1%} "
            f"({s['top6_actual']}/{s['top6_n']}, projected {s['top6_projected']:.1f})")


# ----------------------------------------------------------------------- main

def fit_and_validate(rows: pd.DataFrame, log=print) -> dict:
    """Full-season stack fits plus their walk-forward validation."""
    from mlb_hr.pitching import split_exposure
    from mlb_hr.snapshot import LEVEL_HALF_LIFE_DAYS, LEVEL_PSEUDO_HR

    rows = rows.reset_index(drop=True)
    y_hr = rows["y_hr"].values.astype(float)
    y_hits = rows["y_hits"].values.astype(float)
    di = rows["di"].values
    scored = di - di.min() >= 21

    Xh, oh = hr_design(rows)
    b_hr = fit_logistic(Xh, y_hr, oh)
    wf_hr = walk_forward(di, Xh, oh, y_hr, fit_logistic, _predict_hr)
    wf_lvl = level_series(rows, wf_hr, LEVEL_HALF_LIFE_DAYS, LEVEL_PSEUDO_HR)
    # For reference: the old structure -- the product of every multiplier
    # through 1 - (1 - rate)^PA -- with a single Platt layer refit weekly.
    exposure = np.clip(rows["exposure"].values, 1.5, 5.2)
    sp_share = np.array([split_exposure(e)[0] for e in exposure])
    rate_sp = np.clip(rows["r_hand"] * rows["park"] * rows["weather"] * rows["sp_index"], 5e-4, .15)
    rate_pen = np.clip(rows["r_all"] * rows["park"] * rows["weather"] * rows["pen_index"], 5e-4, .15)
    product = (1 - (1 - rate_sp) ** sp_share * (1 - rate_pen) ** (exposure - sp_share)).values
    wf_platt = walk_forward(di, np.log(product / (1 - product))[:, None], np.zeros(len(rows)), y_hr,
                            lambda X, y, o: fit_logistic(X, y, o, lam=0.0), _predict_hr)

    # The top calibration: fitted on the walk-forward (out-of-sample) stacked
    # predictions, which is how the shipped stack will behave on new days.
    ok = scored & ~np.isnan(wf_lvl)
    top_cal = fit_top_cal(wf_lvl[ok], y_hr[ok])
    wf_top = walk_forward_top_cal(di, np.where(scored, wf_lvl, np.nan), y_hr)

    Xt, ot = hit_design(rows)
    b_hit = fit_poisson(Xt, y_hits, ot)
    wf_hit = walk_forward(di, Xt, ot, y_hits, fit_poisson, _predict_hits)
    ok_hit = scored & ~np.isnan(wf_hit)
    hit_top = fit_hit_top_cal(wf_hit[ok_hit], y_hits[ok_hit])
    wf_hit_top = walk_forward_top_cal(di, np.where(scored, wf_hit, np.nan), y_hits,
                                      fit=fit_hit_top_cal, apply=apply_hit_top_cal)
    old_hits = log5_hits(rows)

    sub = rows[scored].reset_index(drop=True)
    report = {
        "hr": {
            "product_platt": score_hr(sub, wf_platt[scored]),
            "stack": score_hr(sub, wf_hr[scored]),
            "stack_level": score_hr(sub, wf_lvl[scored]),
            "served": score_hr(sub, wf_top[scored]),
        },
        "hits": {
            "log5": score_hits(sub, old_hits[scored]),
            "stack": score_hits(sub, wf_hit[scored]),
            "served": score_hits(sub, wf_hit_top[scored]),
        },
    }
    log(f"[replay] walk-forward, {int(scored.sum()):,} starter-games after a 21-day warm-up:")
    log("  HOME RUNS")
    for key, label in (("product_platt", "multipliers + weekly Platt"),
                       ("stack", "stacked (HR_STACK form)"), ("stack_level", "stacked + tracker level"),
                       ("served", "stacked + level + top cal")):
        log(_fmt_hr(label, report["hr"][key]))
    served = report["hr"]["served"]
    log(f"  served, by percentile band [band, projected, actual]: {served['bands']}")
    log(f"  served, by slot (actual / projected): {served['by_slot']}")
    log(f"  served, by month: {served['by_month']}")
    log("  HITS")
    for key, label in (("log5", "log5 + bullpen (old)"), ("stack", "stacked (HIT_STACK form)"),
                       ("served", "stacked + top cal")):
        s = report["hits"][key]
        log(f"  {label:<34} deviance {s['poisson_deviance']:.5f}  P(1+) log loss {s['p1_log_loss']:.5f}"
            f"  AUC {s['p1_auc']:.4f}  projected {s['projected']:8.0f} vs {s['actual']:6d}  "
            f"top-6 1+ {s['top6_one_plus']:.1%} (hits {s['top6_actual']} vs {s['top6_projected']:.0f})")
    log(f"  hits served, by band [band, projected, actual]: {report['hits']['served']['bands']}")
    log(f"  hits served, by slot (actual / projected): {report['hits']['served']['by_slot']}")

    # The two 5-pick parlays (mlb_hr.parlays), built from the served numbers.
    exposure = np.clip(rows["exposure"].values, 1.5, 5.2)
    one_plus = 1 - (1 - np.clip(wf_hit_top / exposure, 0, 0.9)) ** exposure
    parlays = {
        "hr5": parlay_record(sub, wf_top[scored], y_hr[scored]),
        "hit5": parlay_record(sub, one_plus[scored], (y_hits[scored] > 0).astype(float)),
    }
    for key, rec in parlays.items():
        log(f"  5-pick parlay {key}: cashed {rec['won']} of {rec['days']} days "
            f"(model expected {rec['expected_wins']:.2f} wins, {rec['predicted']:.4%} a day)")
    return {
        "hr_stack": {"coef": dict(zip(["intercept"] + HR_TERMS, map(float, b_hr)))},
        "hr_top_cal": {"coef": top_cal},
        "hit_stack": {"coef": dict(zip(["intercept"] + HIT_TERMS, map(float, b_hit)))},
        "hit_top_cal": {"coef": hit_top},
        "parlays": parlays,
        "walk_forward": report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("pa_path", nargs="?", default=str(DATA_DIR / "season_pa_v2.jsonl"))
    parser.add_argument("--start", default=None, help="first day projected (YYYY-MM-DD)")
    parser.add_argument("--rows", default=str(DATA_DIR / "replay_rows.parquet"))
    parser.add_argument("--reuse-rows", action="store_true",
                        help="refit from the saved rows instead of replaying")
    parser.add_argument("--no-write", action="store_true", help="report only")
    parser.add_argument("--out", default=str(FIT_PATH), help="where to write the fit")
    args = parser.parse_args()

    if args.reuse_rows and Path(args.rows).exists():
        rows = pd.read_parquet(args.rows)
    else:
        from mlb_hr.espn import ESPNData
        espn = ESPNData(str(DATA_DIR / "espn_cache.json"), ttl_hours=1e9).load()
        rows = replay(args.pa_path, args.start, espn)
        rows.to_parquet(args.rows)
    fit = fit_and_validate(rows)
    fit.update({
        "fitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_version": MODEL_VERSION,
        "through": str(rows["date"].max()), "from": str(rows["date"].min()),
        "starter_games": int(len(rows)),
    })
    for key in ("hr_stack", "hr_top_cal", "hit_stack", "hit_top_cal"):
        print(f"[replay] {key}: " + ", ".join(f"{k} {v:+.3f}" for k, v in fit[key]["coef"].items()))
    if not args.no_write:
        Path(args.out).write_text(json.dumps(fit, indent=2), encoding="utf-8")
        print(f"[replay] wrote {args.out}")


if __name__ == "__main__":
    main()
