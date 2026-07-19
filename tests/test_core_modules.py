"""Additional unit tests for core modules: MMM fitting, Shapley, SLSQP optimizer.

Extends coverage beyond test_algorithms.py with edge cases, multi-channel
scenarios, and integration-level checks on the optimization pipeline.
"""

import numpy as np
import polars as pl
import pytest

from attributor.budget_optimizer import optimize_budget, scenario_analysis
from attributor.mmm_model import (
    LASSO_ALPHAS,
    RIDGE_ALPHAS,
    _cv_select_alpha,
    _fit_regularized,
    chronological_split,
    fit_lasso,
    fit_ols,
    fit_ridge,
    prepare_features,
)
from attributor.multi_touch_attribution import (
    first_touch_attribution,
    last_touch_attribution,
    linear_attribution,
    removal_effect_attribution,
    shapley_attribution,
    time_decay_attribution,
)

# ---------------------------------------------------------------------------
# MMM: _cv_select_alpha and _fit_regularized
# ---------------------------------------------------------------------------


class TestCvSelectAlpha:
    def test_returns_valid_alpha_from_grid(self):
        """CV-selected alpha must be one of the candidate grid values."""
        rng = np.random.default_rng(42)
        n = 120
        X = rng.uniform(0, 100, (n, 2))
        y = 3.0 * X[:, 0] + rng.normal(0, 5, n)
        alpha = _cv_select_alpha(X, y, RIDGE_ALPHAS, "ridge")
        assert alpha in RIDGE_ALPHAS

    def test_lasso_alpha_from_grid(self):
        rng = np.random.default_rng(7)
        n = 100
        X = rng.uniform(0, 50, (n, 3))
        y = 2.0 * X[:, 0] - X[:, 1] + rng.normal(0, 2, n)
        alpha = _cv_select_alpha(X, y, LASSO_ALPHAS, "lasso")
        assert alpha in LASSO_ALPHAS

    def test_high_noise_prefers_larger_alpha(self):
        """With very noisy data, CV should prefer stronger regularization."""
        rng = np.random.default_rng(0)
        n = 150
        X = rng.uniform(0, 10, (n, 2))
        # Very noisy: signal-to-noise ratio is low
        y = 0.1 * X[:, 0] + rng.normal(0, 100, n)
        alpha = _cv_select_alpha(X, y, RIDGE_ALPHAS, "ridge")
        # Should pick a larger alpha (more regularization)
        assert alpha >= RIDGE_ALPHAS[len(RIDGE_ALPHAS) // 2]


class TestFitRegularized:
    def test_ridge_returns_model_and_scaler(self):
        rng = np.random.default_rng(1)
        X = rng.uniform(0, 100, (80, 3))
        y = X @ np.array([1.0, 2.0, 3.0]) + rng.normal(0, 1, 80)
        model, scaler = _fit_regularized(X, y, ["a", "b", "c"], "ridge", alpha=1.0)
        assert hasattr(model, "coef_")
        assert hasattr(scaler, "scale_")
        assert len(model.coef_) == 3

    def test_lasso_returns_model_and_scaler(self):
        rng = np.random.default_rng(2)
        X = rng.uniform(0, 100, (80, 2))
        y = 5.0 * X[:, 0] + rng.normal(0, 1, 80)
        model, scaler = _fit_regularized(X, y, ["x1", "x2"], "lasso", alpha=0.5)
        assert hasattr(model, "coef_")
        assert len(scaler.mean_) == 2


# ---------------------------------------------------------------------------
# MMM: multi-channel prepare_features
# ---------------------------------------------------------------------------


@pytest.fixture
def multi_channel_mmm_df() -> pl.DataFrame:
    """MMM DataFrame with 3 adstocked spend channels."""
    n = 60
    rng = np.random.default_rng(99)
    dates = pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 2, 29), interval="1d", eager=True)[:n]
    ch1 = rng.uniform(100, 500, n)
    ch2 = rng.uniform(50, 300, n)
    ch3 = rng.uniform(10, 100, n)
    y = 2.0 * ch1 + 1.5 * ch2 + 0.5 * ch3 + rng.normal(0, 5, n)
    return pl.DataFrame(
        {
            "date_day": dates,
            "google_paid_search_adstock": ch1,
            "meta_facebook_adstock": ch2,
            "tiktok_adstock": ch3,
            "first_purchases_original_price": y,
            "month": [d.month for d in dates],
            "is_weekend": [0] * n,
        }
    )


class TestMultiChannelMMM:
    def test_prepare_features_multi_channel(self, multi_channel_mmm_df):
        X, y, names, dates = prepare_features(multi_channel_mmm_df)
        assert "google_paid_search_adstock" in names
        assert "meta_facebook_adstock" in names
        assert "tiktok_adstock" in names
        assert X.shape[1] == len(names)

    def test_ols_multi_channel_recovers_coefficients(self, multi_channel_mmm_df):
        X, y, names, dates = prepare_features(multi_channel_mmm_df)
        X_tr, X_te, y_tr, y_te = chronological_split(X, y, dates)
        result = fit_ols(X_tr, y_tr, X_te, y_te, names)
        assert result["r2"] > 0.95
        assert result["coefficients"]["google_paid_search_adstock"]["coef"] == pytest.approx(
            2.0, abs=0.3
        )

    def test_ridge_multi_channel(self, multi_channel_mmm_df):
        X, y, names, dates = prepare_features(multi_channel_mmm_df)
        X_tr, X_te, y_tr, y_te = chronological_split(X, y, dates)
        result = fit_ridge(X_tr, y_tr, X_te, y_te, names)
        assert result["r2"] > 0.9
        assert "intercept" in result

    def test_lasso_multi_channel(self, multi_channel_mmm_df):
        X, y, names, dates = prepare_features(multi_channel_mmm_df)
        X_tr, X_te, y_tr, y_te = chronological_split(X, y, dates)
        result = fit_lasso(X_tr, y_tr, X_te, y_te, names)
        assert "n_zeroed_coefs" in result
        assert result["r2"] > 0.8


# ---------------------------------------------------------------------------
# Shapley: additional edge cases
# ---------------------------------------------------------------------------


class TestShapleyExtended:
    def test_three_channels_asymmetric(self):
        """Three channels with different contribution patterns."""
        j = pl.DataFrame(
            {
                "user_id": [1, 2, 3, 4, 5, 6],
                "path": ["A > B > C", "A", "B", "C", "A > B", "A > C"],
                "converted": [1, 1, 1, 1, 1, 1],
                "conversion_value": [30, 20, 15, 10, 15, 10],
            }
        )
        result = shapley_attribution(j)
        # All channels should get positive credit
        assert all(v >= 0 for v in result.values())
        # Total should be conserved
        assert sum(result.values()) == pytest.approx(100.0, rel=0.01)
        # A appears in most paths → should get the most credit
        assert result["A"] >= result["C"]

    def test_empty_journeys_returns_empty(self):
        """No converted users → empty Shapley."""
        j = pl.DataFrame(
            {
                "user_id": [1, 2],
                "path": ["A", "B"],
                "converted": [0, 0],
                "conversion_value": [0, 0],
            }
        )
        result = shapley_attribution(j)
        assert result == {}

    def test_single_user_single_channel(self):
        j = pl.DataFrame(
            {
                "user_id": [1],
                "path": ["X"],
                "converted": [1],
                "conversion_value": [42.0],
            }
        )
        result = shapley_attribution(j)
        assert result == {"X": 42.0}

    def test_four_channels_total_conserved(self):
        """With 4 channels, Shapley values must still sum to total conversion."""
        j = pl.DataFrame(
            {
                "user_id": list(range(1, 9)),
                "path": [
                    "A > B > C > D",
                    "A > C",
                    "B > D",
                    "A",
                    "C > D",
                    "B > C",
                    "A > D",
                    "D",
                ],
                "converted": [1] * 8,
                "conversion_value": [10.0] * 8,
            }
        )
        result = shapley_attribution(j)
        assert sum(result.values()) == pytest.approx(80.0, rel=0.01)
        assert len(result) == 4


# ---------------------------------------------------------------------------
# SLSQP optimizer: extended scenarios
# ---------------------------------------------------------------------------


class TestOptimizeBudgetExtended:
    def test_scenario_analysis_returns_four_scenarios(self):
        current = {"google_paid_search_spend": 200.0, "tiktok_spend": 100.0}
        elasticities = {"google_paid_search_spend": 2.5, "tiktok_spend": 1.0}
        scenarios = scenario_analysis(current, elasticities, intercept=50.0)
        assert set(scenarios.keys()) == {
            "reallocate",
            "increase_10pct",
            "increase_20pct",
            "decrease_10pct",
        }
        # Each scenario should have the standard keys
        for name, result in scenarios.items():
            assert "optimal_spend" in result
            assert "current_revenue" in result
            assert "optimal_revenue" in result
            assert "converged" in result

    def test_budget_increase_yields_more_revenue(self):
        """More budget should yield more (or equal) revenue under saturation."""
        current = {"google_paid_search_spend": 150.0, "meta_facebook_spend": 100.0}
        elasticities = {"google_paid_search_spend": 2.0, "meta_facebook_spend": 1.5}
        r_same = optimize_budget(current, elasticities, total_budget=250.0)
        r_more = optimize_budget(current, elasticities, total_budget=300.0)
        assert r_more["optimal_revenue"] >= r_same["optimal_revenue"] - 1e-3

    def test_zero_elasticity_channel_gets_min_spend(self):
        """A channel with zero elasticity should receive minimal allocation."""
        current = {"google_paid_search_spend": 100.0, "tiktok_spend": 100.0}
        elasticities = {"google_paid_search_spend": 5.0, "tiktok_spend": 0.0}
        result = optimize_budget(current, elasticities, total_budget=200.0)
        # tiktok should get pushed toward its lower bound (10% of current = 10)
        assert result["optimal_spend"]["tiktok_spend"] < 50.0

    def test_convergence_flag(self):
        """Optimizer should converge on a well-behaved problem."""
        current = {"a_spend": 100.0, "b_spend": 100.0}
        elasticities = {"a_spend": 2.0, "b_spend": 1.0}
        result = optimize_budget(current, elasticities, total_budget=200.0)
        assert result["converged"] is True

    def test_warnings_list_present(self):
        current = {"a_spend": 100.0, "b_spend": 100.0}
        elasticities = {"a_spend": 1.0, "b_spend": 1.0}
        result = optimize_budget(current, elasticities, total_budget=200.0)
        assert "warnings" in result
        assert isinstance(result["warnings"], list)

    def test_hill_saturation_diminishing_returns(self):
        """Doubling spend should NOT double revenue (saturation property)."""
        current = {"ch_spend": 100.0}
        elasticities = {"ch_spend": 3.0}
        r1 = optimize_budget(current, elasticities, total_budget=100.0)
        r2 = optimize_budget(current, elasticities, total_budget=200.0)
        # Revenue at 2x budget should be less than 2x revenue at 1x budget
        assert r2["optimal_revenue"] < 2.0 * r1["optimal_revenue"] + 1e-3

    def test_single_channel_optimization(self):
        """Single channel: all budget goes to that channel."""
        current = {"only_spend": 500.0}
        elasticities = {"only_spend": 2.0}
        result = optimize_budget(current, elasticities, total_budget=500.0)
        assert result["optimal_spend"]["only_spend"] == pytest.approx(500.0, rel=0.01)

    def test_many_channels(self):
        """Optimizer handles 5+ channels without error."""
        channels = [f"ch{i}_spend" for i in range(6)]
        current = {ch: 100.0 for ch in channels}
        elasticities = {ch: float(i + 1) * 0.5 for i, ch in enumerate(channels)}
        result = optimize_budget(current, elasticities, total_budget=600.0)
        total = sum(result["optimal_spend"].values())
        assert total == pytest.approx(600.0, rel=0.01)
        assert result["converged"] is True


# ---------------------------------------------------------------------------
# Removal effect: additional coverage
# ---------------------------------------------------------------------------


class TestRemovalEffectExtended:
    def test_two_channels_different_importance(self):
        """Channel appearing in more converting paths gets more credit."""
        j = pl.DataFrame(
            {
                "user_id": [1, 2, 3, 4, 5],
                "path": ["A > B", "A", "A > B", "B", "A"],
                "converted": [1, 1, 1, 0, 1],
                "conversion_value": [25, 25, 25, 0, 25],
            }
        )
        result = removal_effect_attribution(j)
        # A appears in all converting paths; removing A kills all conversions
        assert result["A"] > result["B"]

    def test_no_conversions_returns_empty(self):
        j = pl.DataFrame(
            {
                "user_id": [1, 2],
                "path": ["A", "B"],
                "converted": [0, 0],
                "conversion_value": [0, 0],
            }
        )
        result = removal_effect_attribution(j)
        assert result == {}


# ---------------------------------------------------------------------------
# Attribution models: additional edge cases
# ---------------------------------------------------------------------------


class TestAttributionEdgeCases:
    def test_linear_single_user_multi_touch(self):
        """Single user with 3 touchpoints splits value equally."""
        tp = pl.DataFrame(
            {
                "user_id": [1, 1, 1],
                "channel": ["A", "B", "C"],
                "conversion_value": [0, 0, 90.0],
            }
        )
        result = linear_attribution(tp)
        assert result["A"] == pytest.approx(30.0)
        assert result["B"] == pytest.approx(30.0)
        assert result["C"] == pytest.approx(30.0)

    def test_first_touch_multiple_users(self):
        """First touch credits the first channel per user."""
        tp = pl.DataFrame(
            {
                "user_id": [1, 1, 2, 2],
                "channel": ["X", "Y", "Y", "X"],
                "touchpoint_number": [1, 2, 1, 2],
                "is_conversion": [0, 1, 0, 1],
                "conversion_value": [0, 50, 0, 80],
            }
        )
        journeys = pl.DataFrame(
            {
                "user_id": [1, 2],
                "path": ["X > Y", "Y > X"],
                "converted": [1, 1],
                "conversion_value": [50, 80],
            }
        )
        result = first_touch_attribution(tp, journeys)
        assert result["X"] == pytest.approx(50.0)
        assert result["Y"] == pytest.approx(80.0)

    def test_last_touch_only_conversion_touchpoint(self):
        """Last touch only credits the conversion touchpoint."""
        tp = pl.DataFrame(
            {
                "user_id": [1, 1, 1],
                "channel": ["A", "B", "C"],
                "touchpoint_number": [1, 2, 3],
                "is_conversion": [0, 0, 1],
                "conversion_value": [0, 0, 120],
            }
        )
        result = last_touch_attribution(tp)
        assert result["C"] == pytest.approx(120.0)
        assert "A" not in result or result.get("A", 0) == 0
