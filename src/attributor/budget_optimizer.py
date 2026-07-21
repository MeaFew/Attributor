"""Budget allocation optimization based on MMM elasticities."""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import minimize

from attributor.config import (
    CLEANED_PARQUET_PATH,
    HILL_GAMMA,
    MODEL_OUTPUT_DIR,
    SPEND_CHANNELS,
)
from attributor.logging_setup import get_logger, setup_logging

logger = get_logger(__name__)


def load_mmm_results() -> dict:
    """Load MMM results to extract channel elasticities.

    Raises:
        FileNotFoundError: if ``mmm_results.json`` does not exist — run
            ``python -m attributor.mmm_model`` first to generate it.
        json.JSONDecodeError: if the file exists but is corrupted.
    """
    path = MODEL_OUTPUT_DIR / "mmm_results.json"
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.info(f"ERROR: MMM results not found at {path}")
        logger.info("Run 'python -m attributor.mmm_model' first to generate results.")
        raise
    except json.JSONDecodeError as e:
        logger.info(f"ERROR: Failed to parse MMM results: {e}")
        logger.info("The file may be corrupted. Re-run 'python -m attributor.mmm_model'.")
        raise
    return data


def hill_response(
    x: np.ndarray, elasticities: np.ndarray, tau: np.ndarray, gamma: float
) -> np.ndarray:
    """Hill saturation response per channel (diminishing returns).

        revenue_i = elasticities_i * x_i^gamma / (x_i^gamma + tau_i^gamma)

    Shared by ``optimize_budget`` and the Streamlit dashboard's budget
    simulator so the response formula lives in exactly one place.
    """
    # x >= 0 enforced by bounds; guard tau=0 channels.
    safe_tau = np.where(tau > 0, tau, 1.0)
    denom = x**gamma + safe_tau**gamma
    denom = np.where(denom <= 0, 1e-9, denom)
    return elasticities * (x**gamma) / denom


def extract_params(mmm_data: dict) -> tuple[dict[str, float], float]:
    """Extract per-channel elasticity and Ridge intercept from MMM results."""
    ridge = mmm_data["models"]["ridge"]
    coefs = ridge["coefficients"]
    intercept = ridge.get("intercept", 0.0)

    elasticities = {}
    for ch in SPEND_CHANNELS:
        adstock_key = ch.replace("_spend", "_adstock")
        if adstock_key in coefs:
            elasticities[ch] = coefs[adstock_key]["coef"]
    return elasticities, intercept


def optimize_budget(
    current_spend: dict[str, float],
    elasticities: dict[str, float],
    intercept: float = 0.0,
    total_budget: float | None = None,
    min_spend_ratio: float = 0.1,
    max_spend_ratio: float = 3.0,
) -> dict:
    """Optimize budget allocation using scipy.optimize.

    Response model (per channel) is a HILL / saturation function — NOT linear:

        revenue_i = coef_i * x_i^gamma / (x_i^gamma + tau_i^gamma)

    where ``coef_i`` is the Ridge elasticity, ``gamma`` the Hill slope, and
    ``tau_i`` a per-channel half-saturation point set to the channel's current
    average spend (so the model agrees with the linear elasticity near the
    observed operating point but exhibits diminishing returns far from it).
    Total revenue = sum_i(revenue_i) + intercept.

    This fixes the earlier "linear response" issue: under a purely linear
    objective the optimum is a trivial corner solution that just moves all
    budget to the highest-elasticity channel and extrapolates far outside the
    spend range the model was trained on. A saturating response makes the
    optimal allocation non-trivial and bounds the implied revenue gain.

    Constraint: sum(spend_i) = total_budget (if provided).
    Bounds: min_spend_ratio * current <= spend <= max_spend_ratio * current.
    """
    channels = list(current_spend.keys())
    current = np.array([current_spend[c] for c in channels])
    elastic = np.array([elasticities.get(c, 0.0) for c in channels])

    # Per-channel half-saturation point: anchor at the current average spend so
    # the Hill curve matches the linear elasticity near the observed operating
    # point and bends over as spend grows well beyond it.
    tau = np.where(current > 0, current, 1.0)
    gamma = HILL_GAMMA  # Hill slope; >1 gives an S-curve, mild saturation near tau.

    # If no total_budget, keep total constant
    if total_budget is None:
        total_budget = current.sum()

    def total_revenue(x: np.ndarray) -> float:
        return float(np.sum(hill_response(x, elastic, tau, gamma)) + intercept)

    # Objective: maximize total revenue (minimize negative)
    def objective(x):
        return -total_revenue(x)

    # Constraint: sum(x) = total_budget
    def budget_constraint(x):
        return np.sum(x) - total_budget

    # Bounds: each channel can vary between min and max ratio of current
    bounds = []
    for i, c in enumerate(channels):
        min_spend = max(0.0, current[i] * min_spend_ratio)
        # Methodological choice: a channel with current=0 gets an upper bound of
        # total_budget (not 0) so the optimizer CAN activate an unused channel —
        # with a 0 upper bound unused channels could never receive any budget.
        max_spend = current[i] * max_spend_ratio if current[i] > 0 else total_budget
        bounds.append((min_spend, max_spend))

    constraints = [{"type": "eq", "fun": budget_constraint}]

    result = minimize(
        objective,
        current,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 1000},
    )

    warnings = []
    if not result.success:
        warnings.append(
            f"SLSQP did not report convergence (status={result.status}, "
            f"msg={result.message}); the returned allocation is the last iterate."
        )

    optimal = result.x
    predicted_revenue_current = total_revenue(current)
    predicted_revenue_optimal = total_revenue(optimal)

    if predicted_revenue_current < 0:
        warnings.append(
            f"Negative predicted baseline revenue (${predicted_revenue_current:,.0f}). "
            "The MMM fit for this brand is weak — optimization results may be unreliable."
        )

    improvement = (
        float(
            (predicted_revenue_optimal - predicted_revenue_current)
            / abs(predicted_revenue_current)
            * 100
        )
        if predicted_revenue_current != 0
        else 0.0
    )

    return {
        "channels": channels,
        "response_model": f"hill_saturation (gamma={HILL_GAMMA}, tau=current_spend)",
        "current_spend": {c: round(float(v), 2) for c, v in zip(channels, current)},
        "optimal_spend": {c: round(float(v), 2) for c, v in zip(channels, optimal)},
        "current_revenue": round(predicted_revenue_current, 2),
        "optimal_revenue": round(predicted_revenue_optimal, 2),
        "improvement_pct": round(improvement, 2),
        "total_budget": round(float(total_budget), 2),
        "converged": bool(result.success),
        "warnings": warnings,
    }


def scenario_analysis(
    current_spend: dict[str, float],
    elasticities: dict[str, float],
    intercept: float = 0.0,
) -> dict:
    """Run multiple budget scenarios."""
    total = sum(current_spend.values())
    scenarios = {}

    # Scenario 1: same budget, reallocate
    scenarios["reallocate"] = optimize_budget(
        current_spend, elasticities, intercept, total_budget=total
    )

    # Scenario 2: +10% budget
    scenarios["increase_10pct"] = optimize_budget(
        current_spend, elasticities, intercept, total_budget=total * 1.1
    )

    # Scenario 3: +20% budget
    scenarios["increase_20pct"] = optimize_budget(
        current_spend, elasticities, intercept, total_budget=total * 1.2
    )

    # Scenario 4: -10% budget
    scenarios["decrease_10pct"] = optimize_budget(
        current_spend, elasticities, intercept, total_budget=total * 0.9
    )

    return scenarios


def _compute_current_spend(df: pl.DataFrame) -> dict[str, float]:
    """Per-channel average daily spend used as the optimizer's baseline.

    A channel whose mean is null (all-null column) or <= 0 (unused for this
    brand) falls back to the 10th percentile of its non-zero values across ALL
    brands; a channel with no positive values at all gets 0.0.
    """
    current_spend = {}
    for ch in SPEND_CHANNELS:
        if ch not in df.columns:
            current_spend[ch] = 0.0
            continue
        # df[ch].mean() is None for an all-null column — check BEFORE float(),
        # otherwise float(None) raises TypeError and the fallback never runs.
        mean_val = df[ch].mean()
        if mean_val is not None and mean_val > 0:
            current_spend[ch] = float(mean_val)
        else:
            # Channel unused for this brand; compute 10th percentile
            # of non-zero spend across ALL brands as fallback.
            # Series.filter needs a boolean Series, not an Expr.
            non_zero = df[ch].filter(df[ch] > 0)
            if len(non_zero) > 0:
                fallback = float(non_zero.quantile(0.1))
                current_spend[ch] = fallback
            else:
                current_spend[ch] = 0.0
    return current_spend


def main() -> None:
    """Run budget optimization."""
    try:
        mmm = load_mmm_results()
    except (FileNotFoundError, json.JSONDecodeError):
        # Reason already logged by load_mmm_results; exit non-zero as before.
        sys.exit(1)
    elasticities, intercept = extract_params(mmm)

    # Use average daily spend from the MMM training data as current spend baseline
    try:
        df = pl.read_parquet(CLEANED_PARQUET_PATH)
        current_spend = _compute_current_spend(df)
    except (OSError, pl.exceptions.PolarsError, pl.exceptions.ArrowError) as e:
        logger.info(f"Warning: Could not load cleaned data: {e}")
        logger.info("Using zero baseline for all channels.")
        current_spend = {ch: 0.0 for ch in SPEND_CHANNELS}

    logger.info("Running budget optimization...")
    scenarios = scenario_analysis(current_spend, elasticities, intercept)

    # Save results
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = MODEL_OUTPUT_DIR / "budget_optimization.json"
    with open(out, "w") as f:
        json.dump(scenarios, f, indent=2)
    logger.info(f"  Saved scenarios to {out}")

    # Print summary
    for name, result in scenarios.items():
        logger.info(f"\nScenario: {name}")
        logger.info(f"  Total budget: ${result['total_budget']:,.0f}")
        logger.info(
            f"  Predicted revenue: ${result['current_revenue']:,.0f} -> ${result['optimal_revenue']:,.0f}"
        )
        logger.info(f"  Improvement: {result['improvement_pct']:.1f}%")
        logger.info("  Top reallocation:")
        changes = {
            c: result["optimal_spend"][c] - result["current_spend"][c] for c in result["channels"]
        }
        for ch, delta in sorted(changes.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
            logger.info(
                f"    {ch}: ${result['current_spend'][ch]:,.0f} -> ${result['optimal_spend'][ch]:,.0f} ({delta:+,.0f})"
            )


if __name__ == "__main__":
    setup_logging()
    main()
