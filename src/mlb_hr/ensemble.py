"""Comparables-prior ensemble for home-run probability.

WHAT WAS BROKEN
---------------
The previous implementation trained a KNeighborsClassifier on one row per
plate appearance, but every row for a given batter carried the *same*
batter-level aggregate feature vector. Three failures followed:

1. Zero within-batter variance. No feature distinguished a home run from a
   strikeout, so no classifier could learn anything about individual PAs.
2. Distance-zero self-matching. All of a batter's rows are the identical point,
   so a query for that batter retrieved his own training rows at distance 0.
   With weights="distance" those get effectively infinite weight, making the
   prediction a leaked in-sample average of 5 of his own PAs -- quantized to
   {0, .2, .4, .6, .8, 1}. Joe Mack's 0.400 was "2 of my own 5 sampled PAs
   were home runs", not a prediction.
3. Target leakage. hr_total and hr_vs_r were features; they are the numerator
   of the quantity being predicted.

Separately, the empirical-Bayes baseline was never evaluated -- it was scored
against np.random.uniform(0.03, 0.08) -- and that placeholder Brier score then
set the ensemble blend weights.

THE FIX
-------
KNN stays, but gets the job it is actually good at: finding *comparable
players*. The unit of observation becomes one batter, the features become
quality-of-contact and physical attributes (never home-run counts), and the
target becomes a rate rather than a binary event.

    prior(batter)  = pooled HR rate of his k nearest comparables, self excluded
    posterior      = (hr_observed + K * prior) / (pa_observed + K)

So KNN supplies the *prior mean* for the empirical-Bayes shrinkage, replacing
the flat 3.4% league constant. This is load-bearing but bounded:

  - a rookie with 30 PA is pulled toward what similar hitters do (given his
    exit velocity, physique and age) instead of toward the league average;
  - a regular with 600 PA barely moves, because K is estimated from data;
  - the prior can never produce an absurd number, because it is a pooled rate
    over real players rather than an additive vote.

K is estimated by method of moments from the residual variance around the
prior, not hand-tuned. Everything is validated on a held-out *later* slice of
the season, so reported metrics are genuine out-of-sample numbers.
"""
from __future__ import annotations

import json
import pickle
from datetime import date as _date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from mlb_hr.features import (
    BatterProfile,
    build_matrix,
    build_profiles,
    platoon_factors,
    season_bounds,
    FEATURE_NAMES,
)
from mlb_hr.model import HRModel, BatterStats, compute_calibration_metrics
from mlb_hr.pitching import split_exposure

# Fraction of the season (by date) used for fitting; the remainder is held out.
TRAIN_FRACTION = 0.80

# Batters below this PA count are too noisy to inform prior fitting.
RELIABLE_PA = 100

DEFAULT_PA_PER_GAME = 4.1

# Per-game HR probabilities come out overconfident: across the rolling backtest
# (~39,600 starter-games) the top decile projected ~22% and homered under 18%,
# the bottom projected ~3.4% and homered ~5.6%. A logistic recalibration
#     p' = expit(CAL_INTERCEPT + CAL_SLOPE * logit(p))
# fitted on all prior months gave slopes of 0.586-0.632 and intercepts of
# -0.788 to -0.898 for each of June-September (all weeks: -0.787 / 0.644), and
# cut forward log loss 9265 -> 9221 (2026-09-13 backtest). It is monotone, so
# it never reorders hitters or changes a pick;
# it only makes the published percentages mean what they say. Refit it with
# `python -m mlb_hr.calibration_fit` after model changes -- and check it
# against tracked picks once ~100 have settled, since the live projection
# carries park, weather and pitcher terms the backtest scoring does not.
CAL_INTERCEPT = -0.79
CAL_SLOPE = 0.64


def calibrate_game_prob(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return float(1.0 / (1.0 + np.exp(-(CAL_INTERCEPT + CAL_SLOPE * np.log(p / (1 - p))))))


class ComparablesKNN:
    """Nearest-neighbour prior: pooled HR rate of a batter's closest comparables.

    Self-exclusion is mandatory. Without it the model retrieves the batter's own
    row and hands back his observed rate, which is the leak that made the old
    ensemble produce 40% home-run probabilities.
    """

    def __init__(self, n_neighbors: int = 40):
        self.n_neighbors = n_neighbors
        self.nn: Optional[NearestNeighbors] = None
        self.hr = np.empty(0)
        self.pa = np.empty(0)
        self.ids: list[int] = []
        self._index_of: dict[int, int] = {}
        self.league_rate = 0.03

    def fit(self, X: np.ndarray, hr: np.ndarray, pa: np.ndarray, ids: list[int]) -> "ComparablesKNN":
        k = min(self.n_neighbors, max(1, len(X) - 1))
        # +1 neighbour so a self-match can be dropped without losing a comparable.
        self.nn = NearestNeighbors(n_neighbors=min(k + 1, len(X))).fit(X)
        self.hr, self.pa, self.ids = hr, pa, ids
        self._index_of = {bid: i for i, bid in enumerate(ids)}
        self.league_rate = float(hr.sum() / pa.sum()) if pa.sum() else 0.03
        return self

    def prior(self, x: np.ndarray, batter_id: Optional[int] = None) -> float:
        """Pooled HR rate over the k nearest comparables, excluding the batter.

        Pooling (sum of HRs over sum of PAs) rather than averaging per-batter
        rates keeps a 30-PA neighbour from carrying the same weight as a
        600-PA neighbour.
        """
        if self.nn is None or not len(self.ids):
            return self.league_rate

        _, indices = self.nn.kneighbors(x.reshape(1, -1))
        neighbors = [i for i in indices[0] if self.ids[i] != batter_id]
        neighbors = neighbors[: self.n_neighbors]
        if not neighbors:
            return self.league_rate

        hr_sum = float(self.hr[neighbors].sum())
        pa_sum = float(self.pa[neighbors].sum())
        if pa_sum <= 0:
            return self.league_rate
        return hr_sum / pa_sum


class EnsembleHRModel:
    """Empirical-Bayes home-run model with a KNN/SVR comparables prior."""

    def __init__(
        self,
        model: HRModel,
        pa_jsonl_path: str,
        espn=None,
        n_neighbors: int = 40,
        half_life_days: Optional[float] = None,
        serve_variant: str = "core",
        cutoff: Optional[str] = None,
        holdout_end: Optional[str] = None,
    ):
        self.model = model
        self.pa_path = pa_jsonl_path
        self.espn = espn
        self.half_life_days = half_life_days
        # The backtest moves the split through the season: fit before `cutoff`,
        # score [cutoff, holdout_end). Left unset, the split is TRAIN_FRACTION
        # of the season and the holdout runs to the last game.
        self.cutoff = cutoff
        self.holdout_end = holdout_end
        # Raw held-out outcomes and per-variant predictions, kept so windows can
        # be pooled into one score rather than averaged.
        self.holdout: dict = {}
        self.scaler = StandardScaler()
        self.feature_weights: Optional[np.ndarray] = None
        self.knn = ComparablesKNN(n_neighbors=n_neighbors)
        self.svm: Optional[ScaledSVR] = None  # kept as `svm` for API compatibility
        self.rf: Optional[RandomForestRegressor] = None
        # Comparison-only priors: they are scored beside the served model on
        # the Model Lab page but do not feed the blend, which stays KNN + SVR
        # until something beats it on held-out data.
        self.linear: Optional[LinearRegression] = None
        self.logistic: Optional[LogisticRegression] = None
        self.nn: Optional[RateNet] = None
        self.xgb = None
        self.profiles: dict[int, BatterProfile] = {}
        self.platoon: dict[tuple[str, str], float] = {}
        self.league_rate = 0.03
        self.prior_weights = {"knn": 0.5, "svr": 0.25, "rf": 0.25}
        self.core_weights = {"knn": 1.0, "svr": 0.0}
        # Which blend the slate is served from. The core model is the default
        # because the forest variant is unproven on held-out data.
        self.serve_variant = serve_variant
        self.k_shrink = {
            "league": 300.0, "knn": 300.0, "svr": 300.0,
            "rf": 300.0, "core": 300.0, "ensemble": 300.0,
            "linear": 300.0, "logistic": 300.0, "nn": 300.0, "xgb": 300.0,
        }
        self.calibration: dict = {}
        self._prior_cache: dict[tuple[int, str], float] = {}
        self._train()

    # ------------------------------------------------------------------ fitting

    def _train(self) -> None:
        cutoff = self._cutoff_date()
        print(f"[ensemble.py] temporal split at {cutoff} (fit before, score after)")

        # Profiles for serving (full season) and for fitting (pre-cutoff only).
        # Recency decay is anchored to the cutoff for the fitting profiles, so
        # the weighting cannot see into the scoring window.
        self.profiles = build_profiles(
            self.pa_path, half_life_days=self.half_life_days
        )
        train_profiles = build_profiles(
            self.pa_path,
            end_date=cutoff,
            half_life_days=self.half_life_days,
            ref_date=cutoff,
        )
        self.platoon = platoon_factors(self.profiles)

        X_raw, hr, pa, ids = build_matrix(train_profiles, self.espn, min_pa=1)
        if len(X_raw) < 50:
            print("[ensemble.py] too few batters to fit; empirical-Bayes only")
            self._fit_league_only()
            return

        self.league_rate = float(hr.sum() / pa.sum())
        X = self._scale_fit(X_raw, hr, pa)
        print(
            f"[ensemble.py] fitting on {len(X):,} batters "
            f"({int(pa.sum()):,} PA, league HR rate {self.league_rate:.4f})"
        )

        # Everything below is fit on out-of-fold priors. Comparing an
        # in-sample SVR fit against a self-excluded KNN would hand SVR the
        # blend on the strength of its own overfit, which is the same class of
        # mistake as the leak being repaired here.
        self._select_neighbors(X, hr, pa, ids)
        oof = self._out_of_fold_priors(X, hr, pa, ids)
        # Prefer forward selection; fall back to same-window error only when the
        # inner split is too thin to decide.
        if not self._forward_selection(cutoff):
            print("[ensemble.py] inner split too thin; selecting on same-window error")
            self._fit_blend_weights(oof, hr, pa)

        oof["core"] = (
            self.core_weights["knn"] * oof["knn"]
            + self.core_weights["svr"] * oof["svr"]
        )
        oof["ensemble"] = (
            self.prior_weights["knn"] * oof["knn"]
            + self.prior_weights["svr"] * oof["svr"]
            + self.prior_weights["rf"] * oof["rf"]
        )
        oof["league"] = np.full(len(hr), self.league_rate)

        # Shrinkage strength per prior, by method of moments on OOF residuals.
        # A sharper prior leaves less residual signal and so earns a larger K.
        # When the residual falls below binomial noise the estimate is
        # unidentifiable; fall back to the flat prior's K rather than an
        # arbitrary large number, since there is no evidence the informative
        # prior deserves *more* weight than the flat one.
        k_league = _estimate_k(hr, pa, oof["league"]) or 300.0
        self.k_shrink["league"] = k_league
        for name in ("knn", "svr", "rf", "core", "ensemble",
                     "linear", "logistic", "nn", "xgb"):
            self.k_shrink[name] = _estimate_k(hr, pa, oof[name]) or k_league
        print(
            "[ensemble.py] shrinkage K: "
            + ", ".join(f"{k}={v:.0f}" for k, v in self.k_shrink.items())
        )

        # Refit on the full fitting window now that weights and K are set.
        self._fit_svr(X, hr, pa)
        self._fit_rf(X, hr, pa)
        self._fit_comparison_models(X, hr, pa)

        self._validate(cutoff, train_profiles, X, ids)

    def _fit_league_only(self) -> None:
        self.knn.nn = None
        self.svm = None
        self.calibration = {}

    def _cutoff_date(self) -> str:
        """Date splitting the season into fit and held-out windows."""
        if self.cutoff:
            return self.cutoff
        lo, hi = season_bounds(self.pa_path)
        try:
            start = _date.fromisoformat(lo)
            end = _date.fromisoformat(hi)
            span = (end - start).days
            return str(start + timedelta(days=int(span * TRAIN_FRACTION)))
        except Exception:  # noqa: BLE001 - fall back to no holdout
            return hi

    def _scale_fit(self, X_raw: np.ndarray, hr: np.ndarray, pa: np.ndarray) -> np.ndarray:
        """Standardize, then weight each feature by its correlation with HR rate.

        Plain Euclidean distance treats age and barrel rate as equally
        important. Weighting by |correlation| makes "nearest" mean nearest in
        the directions that actually track home-run production.
        """
        X = self.scaler.fit_transform(X_raw)
        rates = np.divide(hr, np.maximum(pa, 1))
        reliable = pa >= RELIABLE_PA
        if reliable.sum() < 30:
            reliable = np.ones(len(pa), dtype=bool)

        weights = np.ones(X.shape[1])
        for j in range(X.shape[1]):
            col = X[reliable, j]
            if np.std(col) < 1e-9:
                weights[j] = 0.0
                continue
            corr = np.corrcoef(col, rates[reliable])[0, 1]
            weights[j] = abs(corr) if np.isfinite(corr) else 0.0

        if weights.sum() <= 0:
            weights = np.ones(X.shape[1])
        weights = weights / weights.sum() * len(weights)
        self.feature_weights = weights

        top = sorted(zip(FEATURE_NAMES, weights), key=lambda t: -t[1])[:5]
        print(
            "[ensemble.py] top comparables features: "
            + ", ".join(f"{n} {w:.2f}" for n, w in top)
        )
        return X * weights

    def _fit_svr(self, X: np.ndarray, hr: np.ndarray, pa: np.ndarray) -> None:
        """Support-vector *regression* on rates.

        The old code used SVC to classify individual PAs from batter-constant
        features, which is unlearnable. Regressing the rate on the same
        comparables features gives a smooth second opinion on the prior.
        """
        self.svm = _new_svr().fit(
            X,
            np.divide(hr, np.maximum(pa, 1)),
            sample_weight=np.sqrt(pa),  # reliability weighting
        )

    def _fit_comparison_models(
        self, X: np.ndarray, hr: np.ndarray, pa: np.ndarray
    ) -> None:
        """Fit the priors that exist to be compared, not to be served.

        Least squares, binomial logistic regression and a small torch network,
        all on the identical feature matrix the KNN and the forest see, so the
        Model Lab comparison isolates the model and nothing else.
        """
        rates = np.divide(hr, np.maximum(pa, 1))
        weights = np.sqrt(pa)
        try:
            self.linear = _new_linear().fit(X, rates, sample_weight=weights)
        except Exception as exc:  # noqa: BLE001 - a comparison prior is optional
            print(f"[ensemble.py] linear fit failed ({exc})")
            self.linear = None
        try:
            self.logistic = _fit_logistic_rate(X, hr, pa)
        except Exception as exc:  # noqa: BLE001
            print(f"[ensemble.py] logistic fit failed ({exc})")
            self.logistic = None
        if _XGB:
            try:
                self.xgb = _new_xgb().fit(X, rates, sample_weight=weights)
            except Exception as exc:  # noqa: BLE001
                print(f"[ensemble.py] xgboost fit failed ({exc})")
                self.xgb = None
        else:
            print("[ensemble.py] xgboost not installed; boosted prior disabled")
        if _TORCH:
            try:
                self.nn = _new_nn().fit(X, rates, weights)
            except Exception as exc:  # noqa: BLE001
                print(f"[ensemble.py] neural fit failed ({exc})")
                self.nn = None
        else:
            print("[ensemble.py] torch not installed; neural prior disabled")
        print(
            "[ensemble.py] comparison priors: linear"
            f"{'' if self.linear is None else ' ok'}, logistic"
            f"{'' if self.logistic is None else ' ok'}, xgboost"
            f"{'' if self.xgb is None else ' ok'}, torch net"
            f"{'' if self.nn is None or self.nn.net is None else ' ok'}"
        )

    def _forward_selection(self, cutoff: str) -> bool:
        """Pick blend weights on an inner temporal split inside the fit window.

        Priors are fitted on the first part of the fitting window and scored
        against what the same hitters did in the remainder. Nothing at or after
        `cutoff` is touched, so the held-out window stays untouched while the
        criterion still measures prediction rather than recall.

        Returns False when the inner windows are too thin to decide, leaving
        the caller to fall back on the same-window criterion.
        """
        lo, _hi = season_bounds(self.pa_path)
        try:
            start, end = _date.fromisoformat(lo), _date.fromisoformat(cutoff)
        except (ValueError, TypeError):
            return False
        inner = str(start + timedelta(days=int((end - start).days * TRAIN_FRACTION)))
        if inner <= lo or inner >= cutoff:
            return False

        fit_profiles = build_profiles(
            self.pa_path, end_date=inner,
            half_life_days=self.half_life_days, ref_date=inner,
        )
        future = build_profiles(self.pa_path, start_date=inner, end_date=cutoff)
        X_raw, hr, pa, ids = build_matrix(fit_profiles, self.espn, min_pa=1)
        if len(X_raw) < 100:
            return False

        # A private scaler and weighting: the outer ones are fitted on the full
        # window and would carry information from the inner scoring slice.
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)
        rates = np.divide(hr, np.maximum(pa, 1))
        reliable = pa >= RELIABLE_PA
        if reliable.sum() < 30:
            return False
        fw = np.ones(X.shape[1])
        for j in range(X.shape[1]):
            col = X[reliable, j]
            if np.std(col) < 1e-9:
                fw[j] = 0.0
                continue
            corr = np.corrcoef(col, rates[reliable])[0, 1]
            fw[j] = abs(corr) if np.isfinite(corr) else 0.0
        fw = fw / fw.sum() * len(fw) if fw.sum() > 0 else np.ones(X.shape[1])
        Xw = X * fw

        knn = ComparablesKNN(n_neighbors=self.knn.n_neighbors).fit(Xw, hr, pa, ids)
        sample_weight = np.sqrt(pa)
        svr = _new_svr().fit(Xw, rates, sample_weight=sample_weight)
        rf = _new_rf().fit(Xw, rates, sample_weight=sample_weight)

        priors = {
            "knn": np.array([knn.prior(Xw[i], ids[i]) for i in range(len(Xw))]),
            "svr": np.clip(svr.predict(Xw), 0.002, 0.15),
            "rf": np.clip(rf.predict(Xw), 0.002, 0.15),
        }

        future_hr = np.array(
            [future[b].hr if b in future else 0.0 for b in ids], dtype=float
        )
        future_pa = np.array(
            [future[b].pa if b in future else 0.0 for b in ids], dtype=float
        )
        scorable = (pa >= RELIABLE_PA) & (future_pa >= 30)
        if scorable.sum() < 50:
            return False

        target = np.zeros(len(ids))
        np.divide(future_hr, np.maximum(future_pa, 1), out=target)
        print(
            f"[ensemble.py] forward selection: fit before {inner}, "
            f"scored on {inner}..{cutoff} ({int(scorable.sum())} batters)"
        )
        self._fit_blend_weights(
            priors, hr, pa, target=target, weights=future_pa, mask=scorable
        )
        return True

    def _fit_rf(self, X: np.ndarray, hr: np.ndarray, pa: np.ndarray) -> None:
        """Random-forest regression on rates.

        A forest is itself a learned similarity metric: two hitters are alike
        if they fall in the same leaves. That is the same job the comparables
        KNN does, except the metric is learned from the target rather than
        hand-weighted by correlation, and it captures interactions between
        exit velocity, launch angle and pull tendency that a weighted Euclidean
        distance treats as independent. Tree splits are invariant to the
        per-feature scaling applied for the KNN, so the same matrix is reused.
        """
        self.rf = _new_rf().fit(
            X,
            np.divide(hr, np.maximum(pa, 1)),
            sample_weight=np.sqrt(pa),
        )

    def _select_neighbors(
        self, X: np.ndarray, hr: np.ndarray, pa: np.ndarray, ids: list[int]
    ) -> None:
        """Choose k by leave-one-out error, then fit the comparables index.

        Self-exclusion makes the KNN prior inherently leave-one-out, so k can be
        tuned honestly using only the fitting window.
        """
        rates = np.divide(hr, np.maximum(pa, 1))
        mask = pa >= RELIABLE_PA
        if mask.sum() < 30:
            self.knn.fit(X, hr, pa, ids)
            return

        best_k, best_mse = self.knn.n_neighbors, float("inf")
        for candidate in (10, 15, 20, 30, 40, 60):
            if candidate >= len(X):
                continue
            trial = ComparablesKNN(n_neighbors=candidate).fit(X, hr, pa, ids)
            prior = np.array([trial.prior(X[i], ids[i]) for i in range(len(X))])
            mse = float(np.average((prior[mask] - rates[mask]) ** 2, weights=pa[mask]))
            if mse < best_mse:
                best_k, best_mse = candidate, mse

        print(f"[ensemble.py] comparables k={best_k} (LOO weighted MSE {best_mse:.3e})")
        self.knn = ComparablesKNN(n_neighbors=best_k).fit(X, hr, pa, ids)

    def _out_of_fold_priors(
        self, X: np.ndarray, hr: np.ndarray, pa: np.ndarray, ids: list[int]
    ) -> dict[str, np.ndarray]:
        """Honest prior estimates for every fitting-window batter.

        KNN is already out-of-fold via self-exclusion. The SVR is not, so it is
        refit across K folds and each batter scored by the fold that did not
        see him.
        """
        knn_prior = np.array([self.knn.prior(X[i], ids[i]) for i in range(len(X))])

        rates = np.divide(hr, np.maximum(pa, 1))
        svr_prior = np.full(len(X), self.league_rate)
        rf_prior = np.full(len(X), self.league_rate)
        # The comparison priors are cross-validated the same way, so their
        # shrinkage constants and held-out scores rest on the same footing as
        # the served model's rather than on an in-sample fit.
        linear_prior = np.full(len(X), self.league_rate)
        logistic_prior = np.full(len(X), self.league_rate)
        nn_prior = np.full(len(X), self.league_rate)
        xgb_prior = np.full(len(X), self.league_rate)
        n_splits = min(5, max(2, len(X) // 50))
        try:
            for train_idx, test_idx in KFold(
                n_splits=n_splits, shuffle=True, random_state=0
            ).split(X):
                weights = np.sqrt(pa[train_idx])
                svr_prior[test_idx] = _new_svr().fit(
                    X[train_idx], rates[train_idx], sample_weight=weights
                ).predict(X[test_idx])
                rf_prior[test_idx] = _new_rf().fit(
                    X[train_idx], rates[train_idx], sample_weight=weights
                ).predict(X[test_idx])
                linear_prior[test_idx] = _new_linear().fit(
                    X[train_idx], rates[train_idx], sample_weight=weights
                ).predict(X[test_idx])
                logit = _fit_logistic_rate(
                    X[train_idx], hr[train_idx], pa[train_idx]
                )
                if logit is not None:
                    logistic_prior[test_idx] = logit.predict_proba(
                        X[test_idx]
                    )[:, 1]
                if _XGB:
                    xgb_prior[test_idx] = _new_xgb().fit(
                        X[train_idx], rates[train_idx], sample_weight=weights
                    ).predict(X[test_idx])
                if _TORCH:
                    net = _new_nn().fit(X[train_idx], rates[train_idx], weights)
                    preds = net.predict(X[test_idx])
                    if not np.isnan(preds).any():
                        nn_prior[test_idx] = preds
        except Exception as exc:  # noqa: BLE001 - fall back to the flat prior
            print(f"[ensemble.py] cross-validation failed ({exc}); KNN prior only")

        return {
            "knn": np.clip(knn_prior, 0.002, 0.15),
            "svr": np.clip(svr_prior, 0.002, 0.15),
            "rf": np.clip(rf_prior, 0.002, 0.15),
            "linear": np.clip(linear_prior, 0.002, 0.15),
            "logistic": np.clip(logistic_prior, 0.002, 0.15),
            "nn": np.clip(nn_prior, 0.002, 0.15),
            "xgb": np.clip(xgb_prior, 0.002, 0.15),
        }

    def _fit_blend_weights(
        self,
        oof: dict[str, np.ndarray],
        hr: np.ndarray,
        pa: np.ndarray,
        target: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
    ) -> None:
        """Choose blend weights by error against a target rate.

        The target must be *future* production, not the same window's observed
        rate. Scoring against observed rates rewards fitting binomial noise: a
        random forest beat the KNN 7.5e-05 to 8.5e-05 against observed rates and
        then lost 3.9e-04 to 3.7e-04 against what those same hitters actually did
        next. The ranking inverts, so the criterion has to look forward.
        """
        rates = np.divide(hr, np.maximum(pa, 1)) if target is None else target
        if mask is None:
            mask = pa >= RELIABLE_PA
        if weights is None:
            weights = pa
        if mask.sum() < 30:
            return

        def mse_of(pred: np.ndarray) -> float:
            return float(np.average(
                (pred[mask] - rates[mask]) ** 2, weights=weights[mask]
            ))

        solo = {name: mse_of(oof[name]) for name in ("knn", "svr", "rf")}

        # Search the weight simplex directly rather than weighting by inverse
        # error. Inverse-MSE hands a strictly worse member a large share -- it
        # gave the SVR 28% despite losing on every metric, dragging the blend
        # below KNN alone. Optimising the actual objective cannot do that.
        best, best_mse = None, float("inf")
        for i in range(21):
            for j in range(21 - i):
                w_knn, w_svr = i / 20.0, j / 20.0
                w_rf = 1.0 - w_knn - w_svr
                blended = (
                    w_knn * oof["knn"] + w_svr * oof["svr"] + w_rf * oof["rf"]
                )
                mse = mse_of(blended)
                if mse < best_mse:
                    best, best_mse = (w_knn, w_svr, w_rf), mse

        # The core model is fitted the same way but with the forest excluded,
        # so the two are a clean A/B rather than one nested inside the other.
        core, core_mse = (1.0, 0.0), float("inf")
        for i in range(21):
            w_knn = i / 20.0
            blended = w_knn * oof["knn"] + (1.0 - w_knn) * oof["svr"]
            mse = mse_of(blended)
            if mse < core_mse:
                core, core_mse = (w_knn, 1.0 - w_knn), mse

        self.prior_weights = {"knn": best[0], "svr": best[1], "rf": best[2]}
        self.core_weights = {"knn": core[0], "svr": core[1]}
        print(
            "[ensemble.py] prior error: "
            + ", ".join(f"{n.upper()} {v:.3e}" for n, v in solo.items())
        )
        print(
            f"[ensemble.py] core blend (no forest): KNN {core[0]:.2f} / "
            f"SVR {core[1]:.2f} -> {core_mse:.3e}"
        )
        print(
            f"[ensemble.py] forest blend: KNN {best[0]:.2f} / SVR {best[1]:.2f} "
            f"/ RF {best[2]:.2f} -> {best_mse:.3e}"
        )

    # ------------------------------------------------------------------- priors

    def _prior_for_row(self, x: np.ndarray, batter_id: Optional[int], which: str) -> float:
        if which == "league":
            return self.league_rate
        if which == "knn":
            return self.knn.prior(x, batter_id)
        if which == "svr":
            if self.svm is None:
                return self.league_rate
            return float(np.clip(self.svm.predict(x.reshape(1, -1))[0], 0.002, 0.15))
        if which == "rf":
            if self.rf is None:
                return self.league_rate
            return float(np.clip(self.rf.predict(x.reshape(1, -1))[0], 0.002, 0.15))
        if which == "linear":
            if self.linear is None:
                return self.league_rate
            return float(np.clip(
                self.linear.predict(x.reshape(1, -1))[0], 0.002, 0.15
            ))
        if which == "logistic":
            rate = _logistic_rate(self.logistic, x)
            if not np.isfinite(rate):
                return self.league_rate
            return float(np.clip(rate, 0.002, 0.15))
        if which == "xgb":
            if self.xgb is None:
                return self.league_rate
            return float(np.clip(
                self.xgb.predict(x.reshape(1, -1))[0], 0.002, 0.15
            ))
        if which == "nn":
            if self.nn is None:
                return self.league_rate
            rate = float(self.nn.predict(x.reshape(1, -1))[0])
            if not np.isfinite(rate):
                return self.league_rate
            return float(np.clip(rate, 0.002, 0.15))
        knn_p = self.knn.prior(x, batter_id)
        svr_p = (
            float(np.clip(self.svm.predict(x.reshape(1, -1))[0], 0.002, 0.15))
            if self.svm is not None else self.league_rate
        )
        if which == "core":
            return (
                self.core_weights["knn"] * knn_p
                + self.core_weights["svr"] * svr_p
            )
        rf_p = (
            float(np.clip(self.rf.predict(x.reshape(1, -1))[0], 0.002, 0.15))
            if self.rf is not None else self.league_rate
        )
        return (
            self.prior_weights["knn"] * knn_p
            + self.prior_weights["svr"] * svr_p
            + self.prior_weights["rf"] * rf_p
        )

    def _features_for(self, profile: BatterProfile) -> np.ndarray:
        bio = self.espn.bio_for(profile.name) if self.espn else None
        x = self.scaler.transform(profile.feature_vector(bio).reshape(1, -1))[0]
        if self.feature_weights is not None:
            x = x * self.feature_weights
        return x

    def prior_for(self, profile: BatterProfile, which: str = "ensemble") -> float:
        cache_key = (profile.batter_id, which)
        if cache_key in self._prior_cache:
            return self._prior_cache[cache_key]
        if self.knn.nn is None:
            value = self.league_rate
        else:
            value = self._prior_for_row(
                self._features_for(profile), profile.batter_id, which
            )
        self._prior_cache[cache_key] = value
        return value

    # ---------------------------------------------------------------- inference

    def per_pa_rate(
        self,
        profile: BatterProfile,
        pitcher_hand: str = "R",
        which: str = "ensemble",
    ) -> float:
        """Empirical-Bayes posterior HR rate per plate appearance."""
        prior = self.prior_for(profile, which)
        # "ALL" means a hand-agnostic rate, used for bullpen exposure where the
        # relievers are not known in advance.
        factor = (
            1.0 if pitcher_hand == "ALL"
            else self.platoon.get((profile.bat_side, pitcher_hand), 1.0)
        )
        prior_split = prior * factor

        hr, pa = profile.observed_weighted(pitcher_hand)
        if pa <= 0:
            hr, pa = profile.observed_weighted("ALL")

        k = self.k_shrink.get(which if which != "svr" else "svr", 300.0)
        posterior = (hr + k * prior_split) / (pa + k)
        return float(np.clip(posterior, 0.0005, 0.15))

    def pr_hr_ensemble(
        self,
        batter: BatterStats,
        pitcher_id: int,
        venue: str,
        pitcher_hand: str = "R",
        bullpen=None,
        league_hr_rate: Optional[float] = None,
        expected_pa: Optional[float] = None,
        weather_factor: float = 1.0,
        tto_factors: Optional[tuple] = None,
        sp_hr_index: float = 1.0,
    ) -> dict[str, float]:
        """Per-game HR probability from each prior variant.

        Returned probabilities are per *game*: a per-PA rate converted through
        the batter's own plate appearances per game. The old code reported a
        per-PA rate under a "HR Probability" label, which understated the daily
        question the slate is actually asking.

        Exposure is split between the starter and the bullpen. A starter turns
        over about 2.4 times through the order, so a regular's last one or two
        trips come against relief -- charging every plate appearance to the
        starter's handedness misattributes roughly 40% of the exposure, and
        bullpens allow home runs at rates well away from league average.

        `sp_hr_index` is the opposing starter's shrunk HR rate over league
        (PitcherProfile.hr_index), applied to the starter share only, the same
        way the bullpen index is applied to the relief share.
        """
        profile = self.profiles.get(batter.batter_id)
        if profile is None:
            base = self.model.pr_hr_today(batter, pitcher_id, venue, pitcher_hand)
            return {"empirical_bayes": base, "ensemble": base}

        # Handedness-split park factor, leveraged by how much this hitter pulls.
        # A short porch is worth more to a pull-heavy hitter than to a spray
        # hitter of the same power, so the park's departure from neutral is
        # scaled by pull tendency relative to league.
        park_side = self.model.park_factor_for(venue, profile.bat_side)
        pull_leverage = float(np.clip(profile.pull_rate / 0.40, 0.7, 1.3))
        park = 1.0 + (park_side - 1.0) * pull_leverage

        if expected_pa is None:
            expected_pa = profile.pa_per_game or DEFAULT_PA_PER_GAME
        expected_pa = float(np.clip(expected_pa, 1.5, 5.2))
        pa_vs_sp, pa_vs_pen = split_exposure(expected_pa)

        # Third time through the order is worse for the pitcher; the starter
        # share of a hitter's night is spread across trips one to three.
        tto_sp, tto_pen = tto_factors or (1.0, 1.0)

        # Relief arms are faced without knowing their handedness in advance, so
        # the batter's overall rate is used and scaled by the pen's HR index.
        pen_index = 1.0
        if bullpen is not None and league_hr_rate:
            pen_index = bullpen.hr_index(league_hr_rate)

        result: dict[str, float] = {}
        for label, which in (
            ("empirical_bayes", "league"),
            ("knn", "knn"),
            ("svm", "svr"),
            ("rf", "rf"),
            ("core", "core"),
            ("forest", "ensemble"),
            ("linear", "linear"),
            ("logistic", "logistic"),
            ("neural", "nn"),
            ("xgboost", "xgb"),
            # The served number, from whichever variant is configured.
            ("ensemble", self.serve_variant),
        ):
            rate_sp = float(np.clip(
                self.per_pa_rate(profile, pitcher_hand, which)
                * park * weather_factor * tto_sp * sp_hr_index,
                0.0005, 0.15,
            ))
            rate_pen = float(np.clip(
                self.per_pa_rate(profile, "ALL", which)
                * park * weather_factor * pen_index * tto_pen,
                0.0005, 0.15,
            ))
            raw = (
                1.0
                - (1.0 - rate_sp) ** pa_vs_sp
                * (1.0 - rate_pen) ** pa_vs_pen
            )
            result[label] = calibrate_game_prob(raw)
            result[f"{label}_uncalibrated"] = float(raw)
            result[f"{label}_per_pa"] = rate_sp
            result[f"{label}_per_pa_pen"] = rate_pen

        result["served_variant"] = self.serve_variant
        result["expected_pa"] = expected_pa
        result["pa_vs_sp"] = round(pa_vs_sp, 1)
        result["pa_vs_pen"] = round(pa_vs_pen, 1)
        result["pen_hr_index"] = round(pen_index, 3)
        result["sp_hr_index"] = round(sp_hr_index, 3)
        result["park_side"] = round(park_side, 3)
        result["pull_rate"] = round(profile.pull_rate, 3)
        result["park_effective"] = round(park, 3)
        result["weather_factor"] = round(weather_factor, 3)
        return result

    # ------------------------------------------------------------- validation

    def _validate(
        self,
        cutoff: str,
        train_profiles: dict[int, BatterProfile],
        X: np.ndarray,
        ids: list[int],
    ) -> None:
        """Score each prior on plate appearances after the cutoff date.

        Predictions use only pre-cutoff data; outcomes are all post-cutoff, so
        these are true walk-forward numbers. Scored on the batter-by-pitcher-hand
        per-PA rate, isolating model quality from the hardcoded park factors.
        """
        events = _load_events(
            self.pa_path, start_date=cutoff, end_date=self.holdout_end
        )
        if not events:
            print("[ensemble.py] no held-out plate appearances; skipping validation")
            return

        variants = (
            "league", "knn", "svr", "rf", "core", "ensemble",
            "linear", "logistic", "nn", "xgb",
        )
        # Precompute each batter's predicted rate vs L and vs R, per variant.
        table: dict[str, dict[tuple[int, str], float]] = {v: {} for v in variants}
        row_of = {bid: i for i, bid in enumerate(ids)}
        # Platoon multipliers from the fitting window only. self.platoon is
        # measured on the full season for serving, which includes the outcomes
        # being scored here.
        platoon = platoon_factors(train_profiles)

        for batter_id, profile in train_profiles.items():
            if batter_id not in row_of:
                continue
            x = X[row_of[batter_id]]
            for variant in variants:
                prior = self._prior_for_row(x, batter_id, variant)
                k = self.k_shrink.get(variant, 300.0)
                for hand in ("L", "R"):
                    factor = platoon.get((profile.bat_side, hand), 1.0)
                    hr, pa = profile.observed_weighted(hand)
                    if pa <= 0:
                        hr, pa = profile.observed_weighted("ALL")
                    rate = (hr + k * prior * factor) / (pa + k)
                    table[variant][(batter_id, hand)] = float(np.clip(rate, 0.0005, 0.15))

        y_list, preds = [], {v: [] for v in variants}
        low_pa_flags = []
        skipped = 0
        for batter_id, hand, is_hr in events:
            if (batter_id, hand) not in table["league"]:
                skipped += 1
                continue
            y_list.append(is_hr)
            low_pa_flags.append(train_profiles[batter_id].pa < RELIABLE_PA)
            for variant in variants:
                preds[variant].append(table[variant][(batter_id, hand)])

        if not y_list:
            print("[ensemble.py] no scorable held-out events; skipping validation")
            return

        y = np.array(y_list, dtype=np.float64)
        self.calibration = {
            "n_test_samples": int(len(y)),
            "cutoff_date": cutoff,
            "holdout_hr_rate": float(y.mean()),
            "unmatched_events": int(skipped),
            "prior_weights": self.prior_weights,
            "shrinkage_k": self.k_shrink,
            "unit": "per plate appearance",
        }
        label_for = {
            "league": "empirical_bayes", "knn": "knn",
            "svr": "svm", "rf": "rf", "core": "core", "ensemble": "forest",
            "linear": "linear", "logistic": "logistic", "nn": "neural",
            "xgb": "xgboost",
        }
        for variant in variants:
            metrics = compute_calibration_metrics(y, np.array(preds[variant]))
            self.calibration[label_for[variant]] = metrics
        self.holdout = {
            "y": y,
            "low_pa": np.array(low_pa_flags, dtype=bool),
            "preds": {
                label_for[v]: np.array(preds[v], dtype=np.float64) for v in variants
            },
        }

        print(
            f"[ensemble.py] held-out: {len(y):,} PA after {cutoff}, "
            f"actual HR rate {y.mean():.4f}"
        )
        for variant in variants:
            m = self.calibration[label_for[variant]]
            print(
                f"  {label_for[variant]:<16} Brier {m['brier']:.5f}  "
                f"LogLoss {m['log_loss']:.5f}  AUC {m['auc']:.4f}  "
                f"Lift {m['top_decile_lift']:.2f}x  CalRMSE {m['calibration_rmse']:.5f}"
            )

        # Stratify by how much the batter's own record was worth. For regulars
        # the observed rate swamps any prior, so all variants converge; the
        # prior only earns its keep on thin samples -- which is precisely the
        # population the old model turned into 18% picks.
        low = np.array(low_pa_flags, dtype=bool)
        if low.sum() >= 200:
            self.calibration["low_pa"] = {
                "threshold_pa": RELIABLE_PA,
                "n_test_samples": int(low.sum()),
                "holdout_hr_rate": float(y[low].mean()),
            }
            print(
                f"[ensemble.py] held-out subset: batters with <{RELIABLE_PA} PA "
                f"before cutoff ({int(low.sum()):,} PA, HR rate {y[low].mean():.4f})"
            )
            for variant in variants:
                m = compute_calibration_metrics(y[low], np.array(preds[variant])[low])
                self.calibration["low_pa"][label_for[variant]] = m
                print(
                    f"  {label_for[variant]:<16} Brier {m['brier']:.5f}  "
                    f"LogLoss {m['log_loss']:.5f}  AUC {m['auc']:.4f}  "
                    f"Lift {m['top_decile_lift']:.2f}x  CalRMSE {m['calibration_rmse']:.5f}"
                )

    # ------------------------------------------------------------- persistence

    def to_json(self, out_path: str) -> None:
        """Export calibration metrics as JSON."""
        with open(out_path, "w") as fh:
            json.dump(self.calibration, fh, indent=2, default=float)

    def save_models(self, dir_path: str) -> None:
        """Persist fitted estimators to disk."""
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        with open(f"{dir_path}/scaler.pkl", "wb") as fh:
            pickle.dump(self.scaler, fh)
        if self.svm is not None:
            with open(f"{dir_path}/svr_model.pkl", "wb") as fh:
                pickle.dump(self.svm, fh)
        if self.knn.nn is not None:
            with open(f"{dir_path}/knn_index.pkl", "wb") as fh:
                pickle.dump(self.knn, fh)


def _estimate_k(hr: np.ndarray, pa: np.ndarray, prior: np.ndarray) -> Optional[float]:
    """Method-of-moments shrinkage strength, or None if unidentifiable.

    Observed rate variance around the prior is part real spread between hitters
    and part binomial noise. Subtracting the binomial component leaves the true
    signal variance, which fixes K without hand-tuning. A prior that explains
    more of the spread leaves less residual signal and therefore earns a larger
    K -- more pull toward itself.

    Returns None when the residual sits at or below binomial noise: the data
    cannot distinguish the prior's error from chance, so the caller should not
    invent a value.
    """
    mask = pa >= RELIABLE_PA
    if mask.sum() < 20:
        return None

    rates = hr[mask] / pa[mask]
    residual_var = float(np.mean((rates - prior[mask]) ** 2))
    binomial_var = float(np.mean(rates * (1 - rates) / pa[mask]))
    signal_var = residual_var - binomial_var

    mu = float(np.mean(prior[mask]))
    if signal_var <= 1e-9 or not 0 < mu < 1:
        return None
    return float(np.clip(mu * (1 - mu) / signal_var - 1, 20.0, 3000.0))


def _load_events(
    pa_jsonl_path: str, start_date: str, end_date: Optional[str] = None
) -> list[tuple[int, str, int]]:
    """(batter_id, pitcher_hand, is_hr) for every PA in [start_date, end_date)."""
    events: list[tuple[int, str, int]] = []
    with open(pa_jsonl_path) as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            game_date = record.get("date") or ""
            if game_date < start_date or (end_date and game_date >= end_date):
                continue
            batter_id = record.get("batter_id")
            if not batter_id:
                continue
            hand = "L" if record.get("pitch_hand") == "L" else "R"
            events.append((batter_id, hand, int(record.get("is_hr", 0) or 0)))
    return events




# ── neural prior ──────────────────────────────────────────────────────────
# Torch is optional: the image installs the CPU build, but a bare checkout
# without it still runs, with the neural prior falling back to the league rate
# rather than failing the whole fit.
try:  # pragma: no cover - import guard
    import torch
    from torch import nn as _nn

    _TORCH = True
except ImportError:  # pragma: no cover - exercised only without torch
    torch = None
    _TORCH = False

# XGBoost is optional on the same terms as torch.
try:  # pragma: no cover - import guard
    from xgboost import XGBRegressor

    _XGB = True
except ImportError:  # pragma: no cover
    XGBRegressor = None
    _XGB = False


class RateNet:
    """A small multilayer perceptron predicting a batter's HR rate.

    The forest already captures interactions between the contact-quality
    features; a net is here to see whether a smooth function of the same
    inputs does better than either it or the weighted-distance KNN. It is kept
    deliberately small -- roughly 600 batters is not a deep-learning sample, so
    two hidden layers with weight decay is the honest ceiling.

    The output passes through a sigmoid scaled to the plausible rate range, so
    the network cannot predict a negative or absurd home-run rate the way an
    unconstrained linear head can.
    """

    MAX_RATE = 0.15

    def __init__(self, epochs: int = 600, hidden: tuple = (32, 16), seed: int = 0):
        self.epochs = epochs
        self.hidden = hidden
        self.seed = seed
        self.net = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray):
        if not _TORCH or len(X) < 40:
            return self
        torch.manual_seed(self.seed)
        layers, prev = [], X.shape[1]
        for width in self.hidden:
            layers += [_nn.Linear(prev, width), _nn.ReLU()]
            prev = width
        layers += [_nn.Linear(prev, 1), _nn.Sigmoid()]
        self.net = _nn.Sequential(*layers)

        xb = torch.tensor(np.asarray(X, dtype=np.float32))
        yb = torch.tensor(
            np.asarray(y, dtype=np.float32).reshape(-1, 1) / self.MAX_RATE
        )
        # Plate appearances are the reliability weight, as everywhere else in
        # the model: a 30-PA line should not pull the fit like a 600-PA one.
        wb = torch.tensor(
            np.asarray(sample_weight, dtype=np.float32).reshape(-1, 1)
        )
        wb = wb / wb.mean()

        opt = torch.optim.Adam(self.net.parameters(), lr=0.01, weight_decay=1e-3)
        self.net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = (wb * (self.net(xb) - yb) ** 2).mean()
            loss.backward()
            opt.step()
        self.net.eval()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.net is None:
            return np.full(len(X), np.nan)
        with torch.no_grad():
            out = self.net(torch.tensor(np.asarray(X, dtype=np.float32)))
        return out.numpy().ravel() * self.MAX_RATE


def _new_nn() -> RateNet:
    return RateNet()


def _new_xgb():
    """Gradient-boosted trees on the comparables features.

    The forest and the booster are both tree ensembles, but they fail in
    opposite directions: the forest averages deep independent trees and
    over-smooths, boosting stacks shallow corrections and will happily chase
    binomial noise. With roughly 600 batters the guard against that is
    structural -- depth 3, a slow learning rate, row and column subsampling and
    an explicit L2 penalty -- rather than early stopping, which would need a
    validation split carved out of an already thin fitting window.
    """
    if not _XGB:
        return None
    return XGBRegressor(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        min_child_weight=5,
        objective="reg:squarederror",
        random_state=0,
        n_jobs=4,
        verbosity=0,
    )


def _new_linear() -> LinearRegression:
    """Ordinary least squares on the comparables features.

    The plainest possible prior above a league constant, and the useful
    baseline for every model that costs more: if the KNN, the forest and the
    net cannot beat a straight line through the same features, their extra
    machinery is not earning anything.
    """
    return LinearRegression()


def _new_logistic() -> LogisticRegression:
    return LogisticRegression(max_iter=2000, C=1.0)


def _fit_logistic_rate(X: np.ndarray, hr: np.ndarray, pa: np.ndarray):
    """Logistic regression on the per-PA home-run outcome.

    The features are constant within a batter, so rather than expanding to one
    row per plate appearance -- 130,000 identical-feature rows -- each batter
    is entered twice: once as home runs, once as the plate appearances that
    were not. Weighted that way the fit is mathematically the same binomial
    logistic regression, at a six-hundredth of the size.

    This is the one prior of the set that models the actual Bernoulli outcome
    rather than regressing a rate, which is why it is worth showing next to the
    regressors.
    """
    successes = np.asarray(hr, dtype=np.float64)
    failures = np.maximum(np.asarray(pa, dtype=np.float64) - successes, 0.0)
    X2 = np.vstack([X, X])
    y2 = np.concatenate([np.ones(len(X)), np.zeros(len(X))])
    w2 = np.concatenate([successes, failures])
    keep = w2 > 0
    if keep.sum() < 20 or len(np.unique(y2[keep])) < 2:
        return None
    return _new_logistic().fit(X2[keep], y2[keep], sample_weight=w2[keep])


def _logistic_rate(model, x: np.ndarray) -> float:
    """Predicted per-PA home-run probability for one batter."""
    if model is None:
        return float("nan")
    return float(model.predict_proba(x.reshape(1, -1))[0, 1])

class ScaledSVR:
    """SVR that standardizes its target before fitting.

    WHY THIS EXISTS
    ---------------
    The old configuration was `SVR(C=1.0, epsilon=0.002)` fitted directly on
    home-run rates, and it was crippled by the scale of that target. Rates have
    a standard deviation near 0.015, so:

      - epsilon=0.002 is an insensitive tube spanning a quarter of the spread
        of the entire target. Differences that matter were priced at zero loss.
      - C=1.0 is not a mild penalty here but an overwhelming one. The objective
        trades ||w||^2/2 against C * sum(errors), and when the errors are of
        order 0.01 the penalty term dominates unless w collapses toward zero.
        The fit degenerates toward a constant.

    A near-constant prediction at the wrong level is exactly what the
    diagnostics found: SVR was the only prior that failed to beat a naive
    baseline on a known synthetic signal, the only one that got *worse* with
    more data, and its calibration error was thirty times everyone else's.

    Standardizing the target fixes the units problem: epsilon and C are then
    expressed in standard deviations, where their conventional values mean what
    they are supposed to mean. The remaining two knobs are chosen by weighted
    cross-validation rather than by hand.
    """

    GRID_C = (1.0, 10.0, 100.0)
    GRID_EPS = (0.05, 0.1, 0.2)

    def __init__(self, C: float = 10.0, epsilon: float = 0.1):
        self.C = C
        self.epsilon = epsilon
        self.svr: Optional[SVR] = None
        self.mu = 0.0
        self.sd = 1.0

    def _select(self, X, z, sample_weight) -> tuple:
        """Weighted 3-fold search over C and epsilon, in standardized units."""
        best, best_err = (self.C, self.epsilon), float("inf")
        folds = list(KFold(n_splits=3, shuffle=True, random_state=0).split(X))
        for c in self.GRID_C:
            for eps in self.GRID_EPS:
                errs, weights = [], []
                for tr, te in folds:
                    w = None if sample_weight is None else sample_weight[tr]
                    model = SVR(kernel="rbf", C=c, epsilon=eps, gamma="scale")
                    model.fit(X[tr], z[tr], sample_weight=w)
                    errs.append((model.predict(X[te]) - z[te]) ** 2)
                    weights.append(
                        np.ones(len(te)) if sample_weight is None
                        else sample_weight[te]
                    )
                err = float(np.average(
                    np.concatenate(errs), weights=np.concatenate(weights)
                ))
                if err < best_err:
                    best, best_err = (c, eps), err
        return best

    def fit(self, X, y, sample_weight=None) -> "ScaledSVR":
        y = np.asarray(y, dtype=np.float64)
        w = None if sample_weight is None else np.asarray(sample_weight, float)
        self.mu = float(np.average(y, weights=w))
        var = float(np.average((y - self.mu) ** 2, weights=w))
        self.sd = float(np.sqrt(var)) or 1.0
        z = (y - self.mu) / self.sd

        if len(X) >= 60:
            self.C, self.epsilon = self._select(X, z, w)
        self.svr = SVR(
            kernel="rbf", C=self.C, epsilon=self.epsilon, gamma="scale"
        ).fit(X, z, sample_weight=w)
        return self

    def predict(self, X) -> np.ndarray:
        if self.svr is None:
            return np.full(len(X), self.mu)
        return self.svr.predict(X) * self.sd + self.mu


def _new_svr() -> ScaledSVR:
    """Shared SVR configuration, so cross-validation folds and the final fit
    are the same estimator."""
    return ScaledSVR()


def _new_rf() -> RandomForestRegressor:
    """Shared random-forest configuration.

    A five-sample leaf beat 15, 30 and 50 on out-of-fold error. That is looser
    than it looks: the target is a batter's season rate rather than a single
    event, so each leaf already averages hundreds of plate appearances, and the
    out-of-fold comparison is what selected it.
    """
    return RandomForestRegressor(
        n_estimators=400,
        min_samples_leaf=5,
        random_state=0,
        n_jobs=-1,
    )
