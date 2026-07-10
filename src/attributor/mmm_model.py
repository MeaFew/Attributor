"""Marketing Mix Modeling (MMM) with OLS, Ridge, and Lasso.

Leakage / methodology notes (vs. an earlier version):
- Evaluation now uses a CHRONOLOGICAL train/test split (MMM data is daily time
  series, so random splitting leaks the future). Both in-sample and holdout
  R^2/MAE are reported so the gap is visible.
- Ridge/Lasso now STANDARDIZE the features (StandardScaler) and select the
  regularization strength via time-series CV over a log-spaced alpha grid. An
  earlier version hardcoded tiny alphas (1.0 / 0.1) on unscaled features, which
  produced coefficients essentially identical to OLS — making the "three-model
  comparison" cosmetic. Standardizing + CV-selecting alpha makes the shrinkage
  meaningful and the three models genuinely distinct.
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl
import statsmodels.api as sm
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

matplotlib.use("Agg")
import sys

import matplotlib.pyplot as plt

from attributor.config import (
    CLEANED_PARQUET_PATH,
    HOLDOUT_FRACTION,
    IMAGES_DIR,
    MODEL_OUTPUT_DIR,
    SPEND_CHANNELS,
    TARGET_NEW_REVENUE,
)
from attributor.logging_setup import get_logger, setup_logging

logger = get_logger(__name__)

# HOLDOUT_FRACTION now comes from config (single source of truth). MMM is a
# daily time series, so the split is by date — never a random shuffle.
# Log-spaced regularization grid; alpha is selected by time-series CV.
# Log-spaced regularization grid; alpha is selected by time-series CV.
RIDGE_ALPHAS = np.logspace(-3, 3, 13)
LASSO_ALPHAS = np.logspace(-3, 1, 13)


def prepare_features(
    df: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Build feature matrix for MMM.

    Returns ``(X, y, feature_names, dates)`` — the dates are returned so callers
    can perform a chronological (no-shuffle) train/test split instead of a
    random one.
    """
    df = df.sort("date_day")

    # Base features: adstocked spend
    feature_cols = [c.replace("_spend", "_adstock") for c in SPEND_CHANNELS]
    feature_cols = [c for c in feature_cols if c in df.columns]

    # Drop zero-variance columns (e.g. a channel with no spend for this brand)
    # — they make VIF undefined and add nothing to the fit.
    feature_cols = [c for c in feature_cols if df[c].std() is not None and df[c].std() > 0]

    # Temporal features
    df = df.with_columns(
        pl.col("date_day").cast(pl.Int64).alias("trend"),  # days since epoch as proxy
    )
    # Normalize trend to start at 0
    min_trend = df["trend"].min()
    df = df.with_columns((pl.col("trend") - min_trend).alias("trend"))

    # Seasonality: sine/cosine of month
    df = df.with_columns(
        (2 * np.pi * pl.col("month") / 12).sin().alias("month_sin"),
        (2 * np.pi * pl.col("month") / 12).cos().alias("month_cos"),
        pl.col("is_weekend").cast(pl.Float64).alias("is_weekend"),
    )

    temporal_cols = ["trend", "month_sin", "month_cos", "is_weekend"]
    all_feature_cols = feature_cols + temporal_cols

    # Build matrix (frame is already sorted by date, so X rows are in time order)
    X = df.select(all_feature_cols).to_numpy()
    y = df.select(TARGET_NEW_REVENUE).to_numpy().ravel()
    dates = df["date_day"].to_numpy()

    return X, y, all_feature_cols, dates


def chronological_split(
    X: np.ndarray, y: np.ndarray, dates: np.ndarray, frac: float = HOLDOUT_FRACTION
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Time-respecting split: the last ``frac`` of rows (by date) form the holdout.

    ``X`` must already be sorted by date (``prepare_features`` sorts the frame).
    Returns ``(X_train, X_test, y_train, y_test)``.
    """
    n = len(y)
    split_idx = int(n * (1 - frac))
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]


def _fit_regularized(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    kind: str,
    alpha: float,
) -> tuple[object, StandardScaler]:
    """Fit a standardized Ridge/Lasso and return (model, fitted scaler)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    if kind == "ridge":
        model = Ridge(alpha=alpha)
    else:
        model = Lasso(alpha=alpha, max_iter=50000)
    model.fit(X_scaled, y)
    return model, scaler


def _cv_select_alpha(X: np.ndarray, y: np.ndarray, alphas: np.ndarray, kind: str) -> float:
    """Pick the alpha with the best mean holdout R^2 under TimeSeriesSplit CV."""
    tscv = TimeSeriesSplit(n_splits=4)
    best_alpha, best_score = float(alphas[0]), -np.inf
    for alpha in alphas:
        scores = []
        for tr_idx, va_idx in tscv.split(X):
            model, scaler = _fit_regularized(X[tr_idx], y[tr_idx], [], kind, float(alpha))
            y_pred = model.predict(scaler.transform(X[va_idx]))
            scores.append(r2_score(y[va_idx], y_pred))
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score, best_alpha = mean_score, float(alpha)
    return best_alpha


def fit_ols(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
) -> dict:
    """Fit OLS with statsmodels for diagnostics; report in-sample + holdout."""
    X_const = sm.add_constant(X_train, has_constant="add")
    model = sm.OLS(y_train, X_const).fit()

    # VIF
    vif_data = []
    for i, name in enumerate(["const"] + feature_names):
        if i == 0:
            continue
        try:
            vif = variance_inflation_factor(X_const, i)
            if np.isnan(vif) or np.isinf(vif):
                vif_data.append(
                    {"feature": name, "vif": None, "note": "Cannot compute (zero variance)"}
                )
            else:
                vif_data.append({"feature": name, "vif": round(float(vif), 2)})
        except (ValueError, np.linalg.LinAlgError):
            vif_data.append(
                {"feature": name, "vif": None, "note": "Cannot compute (zero variance)"}
            )

    # Durbin-Watson
    dw = durbin_watson(model.resid)

    # Holdout performance (X_test seen during no fitting)
    X_test_const = sm.add_constant(X_test, has_constant="add")
    y_pred_test = model.predict(X_test_const)

    # Coefficients (skip const for channel-level interpretation)
    coefs = {}
    for name, coef, pval in zip(feature_names, model.params[1:], model.pvalues[1:]):
        coefs[name] = {
            "coef": round(float(coef), 4),
            "pvalue": round(float(pval), 4),
            "significant": bool(pval < 0.05),
        }

    return {
        "model": "OLS",
        "r2": round(float(model.rsquared), 4),
        "adj_r2": round(float(model.rsquared_adj), 4),
        "r2_holdout": round(float(r2_score(y_test, y_pred_test)), 4),
        "mae_holdout": round(float(mean_absolute_error(y_test, y_pred_test)), 2),
        "aic": round(float(model.aic), 2),
        "bic": round(float(model.bic), 2),
        "durbin_watson": round(float(dw), 4),
        "vif": vif_data,
        "coefficients": coefs,
        "residuals": model.resid.tolist(),
    }


def fit_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
) -> dict:
    """Fit Ridge with standardized features and a CV-selected alpha."""
    alpha = _cv_select_alpha(X_train, y_train, RIDGE_ALPHAS, "ridge")
    model, scaler = _fit_regularized(X_train, y_train, feature_names, "ridge", alpha)

    y_pred_train = model.predict(scaler.transform(X_train))
    y_pred_test = model.predict(scaler.transform(X_test))
    r2_train = r2_score(y_train, y_pred_train)
    n, p = X_train.shape

    # Convert scaled coefs back to original scale
    coefs_original = model.coef_ / scaler.scale_
    intercept_original = model.intercept_ - np.sum(model.coef_ * scaler.mean_ / scaler.scale_)

    coefs = {name: {"coef": round(float(c), 4)} for name, c in zip(feature_names, coefs_original)}

    return {
        "model": f"Ridge(alpha={alpha:.4g})",
        "alpha": float(alpha),
        "r2": round(float(r2_train), 4),
        "adj_r2": round(float(1 - (1 - r2_train) * (n - 1) / (n - p - 1)), 4),
        "r2_holdout": round(float(r2_score(y_test, y_pred_test)), 4),
        "mae_holdout": round(float(mean_absolute_error(y_test, y_pred_test)), 2),
        "mae": round(float(mean_absolute_error(y_train, y_pred_train)), 2),
        "coefficients": coefs,
        "intercept": round(float(intercept_original), 2),
    }


def fit_lasso(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
) -> dict:
    """Fit Lasso with standardized features and a CV-selected alpha."""
    alpha = _cv_select_alpha(X_train, y_train, LASSO_ALPHAS, "lasso")
    model, scaler = _fit_regularized(X_train, y_train, feature_names, "lasso", alpha)

    y_pred_train = model.predict(scaler.transform(X_train))
    y_pred_test = model.predict(scaler.transform(X_test))
    r2_train = r2_score(y_train, y_pred_train)
    n, p = X_train.shape

    coefs_original = model.coef_ / scaler.scale_
    intercept_original = model.intercept_ - np.sum(model.coef_ * scaler.mean_ / scaler.scale_)

    coefs = {name: {"coef": round(float(c), 4)} for name, c in zip(feature_names, coefs_original)}
    n_zero = int(np.sum(np.isclose(model.coef_, 0.0)))

    return {
        "model": f"Lasso(alpha={alpha:.4g})",
        "alpha": float(alpha),
        "r2": round(float(r2_train), 4),
        "adj_r2": round(float(1 - (1 - r2_train) * (n - 1) / (n - p - 1)), 4),
        "r2_holdout": round(float(r2_score(y_test, y_pred_test)), 4),
        "mae_holdout": round(float(mean_absolute_error(y_test, y_pred_test)), 2),
        "mae": round(float(mean_absolute_error(y_train, y_pred_train)), 2),
        "coefficients": coefs,
        "intercept": round(float(intercept_original), 2),
        "n_zeroed_coefs": n_zero,
    }


def plot_model_comparison(
    ols_result: dict,
    ridge_result: dict,
    lasso_result: dict,
    feature_names: list[str],
    output_dir: Path,
) -> None:
    """Plot coefficient comparison across models."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Only plot spend channel coefficients
    spend_features = [f.replace("_spend", "_adstock") for f in SPEND_CHANNELS]
    spend_features = [f for f in spend_features if f in feature_names]

    x = np.arange(len(spend_features))
    width = 0.25

    ols_vals = [ols_result["coefficients"].get(f, {}).get("coef", 0) for f in spend_features]
    ridge_vals = [ridge_result["coefficients"].get(f, {}).get("coef", 0) for f in spend_features]
    lasso_vals = [lasso_result["coefficients"].get(f, {}).get("coef", 0) for f in spend_features]

    # Clean labels
    labels = [
        f.replace("_adstock", "").replace("google_", "G_").replace("meta_", "M_")
        for f in spend_features
    ]

    ax.bar(x - width, ols_vals, width, label="OLS", alpha=0.8)
    ax.bar(x, ridge_vals, width, label="Ridge", alpha=0.8)
    ax.bar(x + width, lasso_vals, width, label="Lasso", alpha=0.8)

    ax.set_xlabel("Channel")
    ax.set_ylabel("Coefficient (Revenue per Spend Unit)")
    ax.set_title("MMM Coefficient Comparison Across Models")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    ax.axhline(0, color="gray", linewidth=0.5)
    fig.tight_layout()

    out = output_dir / "mmm_coefficient_comparison.png"
    fig.savefig(out, dpi=150)
    logger.info(f"  Saved coefficient comparison to {out}")
    plt.close(fig)


def plot_residuals(ols_result: dict, output_dir: Path) -> None:
    """Plot residual diagnostics."""
    residuals = np.array(ols_result["residuals"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Residual histogram
    axes[0].hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    axes[0].set_title("Residual Distribution")
    axes[0].set_xlabel("Residual")
    axes[0].set_ylabel("Frequency")
    axes[0].axvline(0, color="red", linestyle="--")

    # Q-Q plot
    sm.qqplot(residuals, line="45", fit=True, ax=axes[1])
    axes[1].set_title("Q-Q Plot")

    fig.tight_layout()
    out = output_dir / "mmm_residual_diagnostics.png"
    fig.savefig(out, dpi=150)
    logger.info(f"  Saved residual diagnostics to {out}")
    plt.close(fig)


def run_mmm(df: pl.DataFrame, brand_id: str | None = None, territory: str | None = None) -> dict:
    """Run MMM pipeline on selected brand/territory or full dataset."""
    if brand_id:
        df = df.filter(pl.col("organisation_id") == brand_id)
    if territory:
        df = df.filter(pl.col("territory_name") == territory)

    if df.height == 0:
        raise ValueError("No data after filtering")

    logger.info(f"Running MMM on {df.height:,} rows (brand={brand_id}, territory={territory})")

    X, y, feature_names, dates = prepare_features(df)

    # Chronological split (MMM is a daily time series — never shuffle).
    X_train, X_test, y_train, y_test = chronological_split(X, y, dates)
    split_date = str(dates[len(y_train)])
    logger.info(
        f"  Chronological split: train={len(y_train)} rows, "
        f"holdout={len(y_test)} rows (holdout from {split_date})"
    )

    ols = fit_ols(X_train, y_train, X_test, y_test, feature_names)
    ridge = fit_ridge(X_train, y_train, X_test, y_test, feature_names)
    lasso = fit_lasso(X_train, y_train, X_test, y_test, feature_names)

    # Summary
    summary = {
        "sample_size": df.height,
        "brand_id": brand_id,
        "territory": territory,
        "date_range": {
            "min": str(df["date_day"].min()),
            "max": str(df["date_day"].max()),
        },
        "split": {
            "method": "chronological",
            "holdout_fraction": HOLDOUT_FRACTION,
            "train_size": int(len(y_train)),
            "test_size": int(len(y_test)),
            "test_start_date": split_date,
            "note": (
                "r2 is in-sample; r2_holdout is on the time-respecting holdout. "
                "MMM is a daily time series so the split is by date (no shuffle)."
            ),
        },
        "models": {
            "ols": ols,
            "ridge": ridge,
            "lasso": lasso,
        },
    }

    # Save JSON
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = MODEL_OUTPUT_DIR / "mmm_results.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  Saved results to {json_path}")

    # Plots
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    plot_model_comparison(ols, ridge, lasso, feature_names, IMAGES_DIR)
    plot_residuals(ols, IMAGES_DIR)

    return summary


def select_best_brand(df: pl.DataFrame) -> tuple[str, str]:
    """Select brand+territory with most complete data."""
    summary = (
        df.group_by(["organisation_id", "territory_name"])
        .agg(
            [
                pl.len().alias("n_rows"),
                pl.col("total_spend").sum().alias("total_spend"),
            ]
        )
        .sort("total_spend", descending=True)
    )
    best = summary.row(0, named=True)
    return best["organisation_id"], best["territory_name"]


def run_cross_brand_elasticity(df: pl.DataFrame) -> pl.DataFrame:
    """Run MMM per brand-territory and aggregate elasticities."""
    logger.info("Running cross-brand MMM elasticity analysis...")
    groups = df.group_by(["organisation_id", "territory_name"]).agg(pl.count())
    results = []

    for row in groups.iter_rows(named=True):
        brand, territory = row["organisation_id"], row["territory_name"]
        sub = df.filter(
            (pl.col("organisation_id") == brand) & (pl.col("territory_name") == territory)
        )
        if sub.height < 100:
            continue
        try:
            X, y, feature_names, _dates = prepare_features(sub)
            X_tr, X_te, y_tr, y_te = chronological_split(X, y, _dates)
            ridge = fit_ridge(X_tr, y_tr, X_te, y_te, feature_names)
            for feat, info in ridge["coefficients"].items():
                if "adstock" in feat:
                    results.append(
                        {
                            "brand": brand,
                            "territory": territory,
                            "channel": feat.replace("_adstock", ""),
                            "elasticity": info["coef"],
                        }
                    )
        except (ValueError, pl.exceptions.PolarsError) as e:
            logger.info(f"  Skip {brand}/{territory}: {e}")
            continue

    result_df = pl.DataFrame(results)
    out = MODEL_OUTPUT_DIR / "cross_brand_elasticities.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    result_df.write_parquet(out)
    logger.info(f"  Saved {len(results)} elasticity records to {out}")
    return result_df


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="Run Marketing Mix Modeling")
    parser.add_argument("--brand", type=str, default=None, help="Brand ID")
    parser.add_argument("--territory", type=str, default=None, help="Territory name")
    parser.add_argument("--cross-brand", action="store_true", help="Run cross-brand elasticity")
    args = parser.parse_args()

    df = pl.read_parquet(CLEANED_PARQUET_PATH)

    if args.cross_brand:
        run_cross_brand_elasticity(df)
    else:
        if not args.brand or not args.territory:
            args.brand, args.territory = select_best_brand(df)
            logger.info(f"Auto-selected brand={args.brand}, territory={args.territory}")
        run_mmm(df, brand_id=args.brand, territory=args.territory)
