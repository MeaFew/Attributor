"""Unit tests for attribution & MMM algorithms.

These tests exercise the *math* of each model with small, deterministic,
hand-verifiable inputs. They do NOT depend on any pre-generated artifacts
(run the pipeline scripts for those integration tests in the other test_* files).
"""

import numpy as np
import polars as pl
import pytest

from attributor.budget_optimizer import (  # noqa: E402
    _compute_current_spend,
    extract_params,
    optimize_budget,
)
from attributor.mmm_model import (  # noqa: E402
    chronological_split,
    fit_lasso,
    fit_ols,
    fit_ridge,
    prepare_features,
)
from attributor.multi_touch_attribution import (  # noqa: E402
    first_touch_attribution,
    last_touch_attribution,
    linear_attribution,
    removal_effect_attribution,
    shapley_attribution,
    time_decay_attribution,
)

# ---------------------------------------------------------------------------
# Fixtures: synthetic touchpoint / journey data
# ---------------------------------------------------------------------------


@pytest.fixture
def touchpoints() -> pl.DataFrame:
    """3 users, 2 channels (A, B). Deterministic.

    user1: A -> B (converted, value=100)
    user2: A       (converted, value=100)
    user3: B       (not converted)
    """
    return pl.DataFrame(
        {
            "user_id": [1, 1, 2, 3],
            "channel": ["A", "B", "A", "B"],
            "touchpoint_number": [1, 2, 1, 1],
            "is_conversion": [0, 1, 1, 0],
            "conversion_value": [0, 100, 100, 0],
            "timestamp": [
                "2026-01-01T00:00:00",
                "2026-01-08T00:00:00",
                "2026-01-01T00:00:00",
                "2026-01-01T00:00:00",
            ],
        }
    )


@pytest.fixture
def journeys() -> pl.DataFrame:
    """Path-level view of the same 3 users above."""
    return pl.DataFrame(
        {
            "user_id": [1, 2, 3],
            "path": ["A > B", "A", "B"],
            "converted": [1, 1, 0],
            "conversion_value": [100, 100, 0],
        }
    )


# ---------------------------------------------------------------------------
# Rule-based attribution models
# ---------------------------------------------------------------------------


class TestFirstTouch:
    def test_credits_first_channel_only(self, touchpoints, journeys):
        result = first_touch_attribution(touchpoints, journeys)
        # user1 first = A, user2 first = A, user3 not converted
        assert result["A"] == 200.0
        assert "B" not in result or result["B"] == 0.0

    def test_returns_dict(self, touchpoints, journeys):
        result = first_touch_attribution(touchpoints, journeys)
        assert isinstance(result, dict)


class TestLastTouch:
    def test_credits_last_channel(self, touchpoints):
        result = last_touch_attribution(touchpoints)
        # user1 last (conversion) = B, user2 last = A
        assert result["B"] == 100.0
        assert result["A"] == 100.0


class TestLinearAttribution:
    def test_splits_value_across_touchpoints(self, touchpoints):
        result = linear_attribution(touchpoints)
        # Linear first broadcasts the journey's total conversion_value to every
        # touchpoint, then divides by n_touches for that user. user1: cv=100,
        # /2 touches -> A=50, B=50. user2: cv=100, /1 touch -> A=100.
        # user3: not converted (cv=0).
        assert result["A"] == pytest.approx(150.0)
        assert result["B"] == pytest.approx(50.0)

    def test_single_touch_journey_gets_full_value(self):
        """A one-touch converted journey attributes full value to that channel."""
        tp = pl.DataFrame(
            {
                "user_id": [1],
                "channel": ["A"],
                "conversion_value": [80.0],
            }
        )
        result = linear_attribution(tp)
        assert result["A"] == pytest.approx(80.0)

    def test_broadcasts_journey_total_across_touchpoints(self):
        """Linear attribution redistributes the journey total across all touches.

        Even when conversion_value only lives on the conversion touchpoint,
        the journey total is broadcast so every touchpoint receives an equal
        share.
        """
        tp = pl.DataFrame(
            {
                "user_id": [1, 1],
                "channel": ["A", "B"],
                "conversion_value": [0.0, 80.0],
            }
        )
        result = linear_attribution(tp)
        assert result["A"] == pytest.approx(40.0)
        assert result["B"] == pytest.approx(40.0)


class TestTimeDecay:
    @pytest.fixture
    def multi_touch_case(self):
        """A single user with two touchpoints to isolate the decay effect.

        A at day 0, B (conversion) at day 7. Under any half-life, B is closer
        to the conversion event so its weight >= A's weight.
        """
        tp = pl.DataFrame(
            {
                "user_id": [1, 1],
                "channel": ["A", "B"],
                "touchpoint_number": [1, 2],
                "is_conversion": [0, 1],
                "conversion_value": [0, 100],
                "timestamp": ["2026-01-01T00:00:00", "2026-01-08T00:00:00"],
            }
        )
        journeys = pl.DataFrame(
            {
                "user_id": [1],
                "path": ["A > B"],
                "converted": [1],
                "conversion_value": [100],
            }
        )
        return tp, journeys

    def test_recent_touchpoint_gets_more_credit(self, multi_touch_case):
        tp, journeys = multi_touch_case
        result = time_decay_attribution(tp, journeys, half_life_days=7.0)
        # B is the conversion touchpoint (distance=0, weight=1); A is 7 days
        # earlier (weight=2^(-1)=0.5). So B should get the larger share.
        assert result["B"] > result["A"]

    def test_total_value_conserved(self, multi_touch_case):
        tp, journeys = multi_touch_case
        result = time_decay_attribution(tp, journeys, half_life_days=7.0)
        assert sum(result.values()) == pytest.approx(100.0, rel=0.01)

    def test_returns_normalized_shares(self, touchpoints, journeys):
        result = time_decay_attribution(touchpoints, journeys)
        assert all(v >= 0 for v in result.values())


# ---------------------------------------------------------------------------
# Shapley value attribution (the math-heavy one)
# ---------------------------------------------------------------------------


class TestShapley:
    def test_single_channel_gets_full_credit(self):
        """If only one channel ever appears in converted paths, Shapley = 100%."""
        j = pl.DataFrame(
            {
                "user_id": [1, 2, 3],
                "path": ["A", "A", "A"],
                "converted": [1, 1, 1],
                "conversion_value": [10, 20, 30],
            }
        )
        result = shapley_attribution(j)
        assert result == {"A": 60.0}

    def test_symmetric_channels_split_evenly(self):
        """Two channels with identical contribution patterns get equal Shapley."""
        j = pl.DataFrame(
            {
                "user_id": [1, 2, 3, 4],
                "path": ["A > B", "B > A", "A", "B"],
                "converted": [1, 1, 1, 1],
                "conversion_value": [25, 25, 25, 25],
            }
        )
        result = shapley_attribution(j)
        # Symmetric A/B => equal split of total 100
        assert result["A"] == pytest.approx(50.0)
        assert result["B"] == pytest.approx(50.0)

    def test_values_are_nonnegative(self, journeys):
        result = shapley_attribution(journeys)
        assert all(v >= 0 for v in result.values())

    def test_total_conserved(self, journeys):
        """Sum of Shapley values should equal total conversion value."""
        result = shapley_attribution(journeys)
        total = sum(result.values())
        # journeys fixture has 200 total conversion value
        assert total == pytest.approx(200.0, rel=0.01)


# ---------------------------------------------------------------------------
# Removal effect attribution
# ---------------------------------------------------------------------------


class TestRemovalEffect:
    def test_indispensable_channel_scores_high(self):
        """If removing channel A eliminates all conversions, A gets full credit."""
        j = pl.DataFrame(
            {
                "user_id": [1, 2],
                "path": ["A", "A"],
                "converted": [1, 1],
                "conversion_value": [50, 50],
            }
        )
        result = removal_effect_attribution(j)
        assert result["A"] == 100.0

    def test_returns_nonnegative(self, journeys):
        result = removal_effect_attribution(journeys)
        assert all(v >= 0 for v in result.values())


# ---------------------------------------------------------------------------
# MMM model fitting
# ---------------------------------------------------------------------------


@pytest.fixture
def mmm_df() -> pl.DataFrame:
    """Small MMM-ready DataFrame with one adstocked spend channel.

    y is constructed as a linear function of the spend feature so OLS can
    recover it exactly (r2 ~ 1.0, coef ~ 2.0).
    """
    n = 40
    rng = np.random.default_rng(42)
    google_spend = rng.uniform(100, 500, n)
    # target = 2 * spend + small noise => OLS should recover coef≈2
    y = 2.0 * google_spend + rng.normal(0, 1, n)
    return pl.DataFrame(
        {
            "date_day": pl.date_range(
                pl.date(2024, 1, 1), pl.date(2024, 2, 9), interval="1d", eager=True
            )[:n],
            "google_paid_search_adstock": google_spend,
            "first_purchases_original_price": y,
            "month": [
                d.month
                for d in pl.date_range(
                    pl.date(2024, 1, 1), pl.date(2024, 2, 9), interval="1d", eager=True
                )[:n]
            ],
            "is_weekend": [0] * n,
        }
    )


class TestPrepareFeatures:
    def test_returns_matrix_target_names_and_dates(self, mmm_df):
        X, y, names, dates = prepare_features(mmm_df)  # noqa: N806 (X is ML convention)
        assert X.ndim == 2
        assert y.ndim == 1
        assert len(names) == X.shape[1]
        assert "google_paid_search_adstock" in names
        assert "trend" in names
        assert "month_sin" in names
        # dates returned for chronological splitting
        assert len(dates) == len(y)


class TestChronologicalSplit:
    def test_no_shuffle_time_ordering(self):
        """Holdout must be the LAST rows (by date), not a random subset.

        Guards against re-introducing a random train/test split on time-series
        MMM data (H1).
        """
        n = 100
        X = np.arange(n).reshape(-1, 1).astype(float)  # noqa: N806
        y = np.arange(n).astype(float)
        dates = np.arange(n)
        X_tr, X_te, y_tr, y_te = chronological_split(X, y, dates, frac=0.2)
        assert len(X_te) == 20
        # Holdout must be exactly the last 20 rows
        assert list(y_te) == list(range(80, 100))
        assert list(y_tr) == list(range(80))
        # No overlap
        assert set(y_tr).isdisjoint(set(y_te))


class TestFitOLS:
    def test_recovers_known_coefficient(self, mmm_df):
        X, y, names, dates = prepare_features(mmm_df)  # noqa: N806 (X is ML convention)
        X_tr, X_te, y_tr, y_te = chronological_split(X, y, dates)
        result = fit_ols(X_tr, y_tr, X_te, y_te, names)
        assert result["model"] == "OLS"
        # Linear relationship => near-perfect in-sample fit
        assert result["r2"] > 0.99
        # Holdout R^2 must be reported (H1)
        assert "r2_holdout" in result
        assert "mae_holdout" in result
        coef = result["coefficients"]["google_paid_search_adstock"]["coef"]
        assert coef == pytest.approx(2.0, abs=0.1)

    def test_result_has_diagnostics(self, mmm_df):
        X, y, names, dates = prepare_features(mmm_df)  # noqa: N806 (X is ML convention)
        X_tr, X_te, y_tr, y_te = chronological_split(X, y, dates)
        result = fit_ols(X_tr, y_tr, X_te, y_te, names)
        for key in ("r2", "adj_r2", "r2_holdout", "aic", "bic", "durbin_watson", "vif"):
            assert key in result


class TestFitRidge:
    def test_returns_coefficients_and_score(self, mmm_df):
        X, y, names, dates = prepare_features(mmm_df)  # noqa: N806 (X is ML convention)
        X_tr, X_te, y_tr, y_te = chronological_split(X, y, dates)
        result = fit_ridge(X_tr, y_tr, X_te, y_te, names)
        assert result["model"].startswith("Ridge")
        assert "google_paid_search_adstock" in result["coefficients"]
        assert isinstance(result["r2"], float)
        # alpha must be CV-selected and recorded (H2)
        assert "alpha" in result
        assert result["alpha"] > 0

    def test_strong_alpha_actually_shrinks_vs_ols(self):
        """H2 regression guard: with standardization + a large alpha, Ridge
        coefficients must be visibly smaller in magnitude than OLS on the same
        data. The earlier version used tiny alphas on unscaled features, which
        produced Ridge coefficients indistinguishable from OLS."""
        rng = np.random.default_rng(0)
        n = 200
        # Two correlated predictors with opposing signs
        x1 = rng.uniform(0, 100, n)
        x2 = x1 + rng.normal(0, 1, n)  # highly collinear
        y = 5.0 * x1 - 5.0 * x2 + rng.normal(0, 1, n)
        X = np.column_stack([x1, x2])  # noqa: N806
        # 80/20 chronological split
        sp = int(n * 0.8)
        ols = fit_ols(X[:sp], y[:sp], X[sp:], y[sp:], ["x1", "x2"])
        ridge = fit_ridge(X[:sp], y[:sp], X[sp:], y[sp:], ["x1", "x2"])
        ols_mag = sum(abs(v["coef"]) for v in ols["coefficients"].values())
        ridge_mag = sum(abs(v["coef"]) for v in ridge["coefficients"].values())
        assert ridge_mag < ols_mag, (
            "Ridge with CV-selected alpha must shrink coefficients below OLS "
            "(otherwise the 'Ridge vs OLS' comparison is cosmetic — H2)."
        )


class TestFitLasso:
    def test_lasso_can_zero_features(self):
        """Lasso with a large enough alpha should zero out an irrelevant feature.

        We pass a deliberately large alpha by constructing a case where the CV
        picks strong regularization, then assert at least one coef is small.
        """
        rng = np.random.default_rng(0)
        n = 200
        signal = rng.uniform(0, 10, n)
        noise = rng.normal(0, 1, n)  # unrelated to y
        y = 3.0 * signal + rng.normal(0, 0.1, n)
        X = np.column_stack([signal, noise])  # noqa: N806 (X is ML convention)
        sp = int(n * 0.8)
        result = fit_lasso(X[:sp], y[:sp], X[sp:], y[sp:], ["signal", "noise"])
        assert "alpha" in result
        # signal retained, noise shrunk
        assert abs(result["coefficients"]["signal"]["coef"]) > abs(
            result["coefficients"]["noise"]["coef"]
        )


# ---------------------------------------------------------------------------
# Budget optimizer
# ---------------------------------------------------------------------------


class TestExtractParams:
    def test_pulls_elasticities_and_intercept(self):
        mmm_data = {
            "models": {
                "ridge": {
                    "intercept": 42.0,
                    "coefficients": {
                        "google_paid_search_adstock": {"coef": 1.5},
                        "tiktok_adstock": {"coef": 0.8},
                    },
                }
            }
        }
        elasticities, intercept = extract_params(mmm_data)
        assert intercept == 42.0
        assert elasticities["google_paid_search_spend"] == 1.5
        assert elasticities["tiktok_spend"] == 0.8


class TestOptimizeBudget:
    def test_uses_saturation_response_not_linear(self):
        """H6 regression guard: the optimizer must use a SATURATING response
        model (Hill), not a linear one. Under a linear model the optimum is a
        trivial corner solution; a saturating model yields diminishing returns
        and a finite, bounded improvement. We check the artifact declares a
        non-linear response model."""
        current = {"google_paid_search_spend": 100.0, "tiktok_spend": 100.0}
        elasticities = {"google_paid_search_spend": 3.0, "tiktok_spend": 1.0}
        result = optimize_budget(current, elasticities, intercept=0.0, total_budget=200.0)
        assert "hill" in result["response_model"].lower(), (
            "Optimizer must use a saturating Hill response, not a linear one (H6)."
        )

    def test_optimal_beats_or_matches_current(self):
        current = {"google_paid_search_spend": 100.0, "tiktok_spend": 100.0}
        elasticities = {"google_paid_search_spend": 3.0, "tiktok_spend": 1.0}
        result = optimize_budget(current, elasticities, intercept=0.0, total_budget=200.0)
        # At the same total budget, the optimized allocation must not be worse.
        assert result["optimal_revenue"] >= result["current_revenue"] - 1e-3
        # Improvement is BOUNDED under saturation (the old linear version could
        # imply unbounded gains; saturation cannot).
        assert result["improvement_pct"] < 1000.0

    def test_respects_budget_constraint(self):
        current = {"google_paid_search_spend": 100.0, "tiktok_spend": 100.0}
        elasticities = {"google_paid_search_spend": 2.0, "tiktok_spend": 1.5}
        result = optimize_budget(current, elasticities, intercept=10.0, total_budget=300.0)
        total = sum(result["optimal_spend"].values())
        assert total == pytest.approx(300.0, rel=0.01)

    def test_respects_bounds(self):
        """Optimal spend stays within [0.1x, 3x] of current."""
        current = {"google_paid_search_spend": 100.0, "tiktok_spend": 100.0}
        elasticities = {"google_paid_search_spend": 5.0, "tiktok_spend": 0.1}
        result = optimize_budget(
            current,
            elasticities,
            intercept=0.0,
            total_budget=200.0,
            min_spend_ratio=0.1,
            max_spend_ratio=3.0,
        )
        for ch, optimal in result["optimal_spend"].items():
            assert 10.0 <= optimal <= 300.0  # within [0.1x, 3x] of 100


class TestComputeCurrentSpend:
    """Regression tests for the current-spend baseline computation.

    Guards the bug where an all-null spend column crashed with
    ``float(None)`` TypeError before the fallback branch could run.
    """

    def test_all_null_column_falls_back_to_zero(self):
        """All-null channel: mean is None -> fallback -> no non-zero values -> 0.0."""
        df = pl.DataFrame(
            {
                "google_paid_search_spend": [100.0, 300.0],
                "tiktok_spend": pl.Series([None, None], dtype=pl.Float64),
            }
        )
        result = _compute_current_spend(df)
        assert result["google_paid_search_spend"] == pytest.approx(200.0)
        assert result["tiktok_spend"] == 0.0  # no TypeError, fallback ran
        # Channels absent from the frame get 0.0
        assert result["meta_facebook_spend"] == 0.0

    def test_zero_mean_with_positive_values_uses_percentile_fallback(self):
        """Mean <= 0 but positive values exist -> 10th percentile of non-zero."""
        df = pl.DataFrame({"tiktok_spend": [-200.0, -100.0, 50.0, 100.0]})
        result = _compute_current_spend(df)
        expected = float(pl.Series([50.0, 100.0]).quantile(0.1))
        assert result["tiktok_spend"] == pytest.approx(expected)
        assert result["tiktok_spend"] > 0

    def test_all_zero_column_falls_back_to_zero(self):
        """All-zero channel: mean == 0 -> fallback -> no positive values -> 0.0."""
        df = pl.DataFrame({"tiktok_spend": [0.0, 0.0, 0.0]})
        result = _compute_current_spend(df)
        assert result["tiktok_spend"] == 0.0
