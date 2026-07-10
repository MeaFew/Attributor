"""Data preprocessing for the Conjura MMM dataset."""

import argparse

# Allow running from repo root
import sys
from pathlib import Path

import polars as pl

from attributor.config import (
    ADSTOCK_DECAY,
    CLEANED_PARQUET_PATH,
    CLICK_CHANNELS,
    IMPRESSION_CHANNELS,
    ORGANIC_CHANNELS,
    RAW_CSV_PATH,
    SPEND_CHANNELS,
    TARGET_ALL_CUSTOMERS,
    TARGET_ALL_REVENUE,
    TARGET_NEW_CUSTOMERS,
    TARGET_NEW_REVENUE,
)
from attributor.logging_setup import get_logger, setup_logging

logger = get_logger(__name__)


def load_raw_data(path: Path) -> pl.DataFrame:
    """Load raw CSV with proper dtypes."""
    logger.info(f"Loading raw data from {path} ...")
    df = pl.read_csv(path, try_parse_dates=True, null_values=[""])
    # Ensure date_day is Date type
    if df["date_day"].dtype != pl.Date:
        df = df.with_columns(pl.col("date_day").str.to_date("%Y-%m-%d"))

    # Coerce numeric columns that may have been inferred as String due to empty values
    numeric_cols = (
        SPEND_CHANNELS
        + CLICK_CHANNELS
        + IMPRESSION_CHANNELS
        + ORGANIC_CHANNELS
        + [
            TARGET_NEW_CUSTOMERS,
            TARGET_ALL_CUSTOMERS,
            TARGET_NEW_REVENUE,
            TARGET_ALL_REVENUE,
            "first_purchases_units",
            "all_purchases_units",
            "first_purchases_gross_discount",
            "all_purchases_gross_discount",
        ]
    )
    for col_name in numeric_cols:
        if col_name in df.columns and df[col_name].dtype == pl.String:
            df = df.with_columns(
                pl.col(col_name).str.replace_all(",", "").cast(pl.Float64).alias(col_name)
            )

    logger.info(f"  Loaded {df.height:,} rows x {df.width} columns")
    return df


def handle_missing_values(df: pl.DataFrame) -> pl.DataFrame:
    """Fill missing spend/clicks/impressions with 0 (brand doesn't use that channel)."""
    numeric_cols = SPEND_CHANNELS + CLICK_CHANNELS + IMPRESSION_CHANNELS
    df = df.with_columns([pl.col(c).fill_null(0) for c in numeric_cols if c in df.columns])
    # Fill organic channels
    df = df.with_columns([pl.col(c).fill_null(0) for c in ORGANIC_CHANNELS if c in df.columns])
    return df


def create_derived_features(df: pl.DataFrame) -> pl.DataFrame:
    """Create total spend, CTR, CPM, and other derived metrics."""
    available_spend = [c for c in SPEND_CHANNELS if c in df.columns]
    available_clicks = [c for c in CLICK_CHANNELS if c in df.columns]

    # Total spend
    df = df.with_columns(pl.sum_horizontal(available_spend).alias("total_spend"))

    # Total clicks (paid + organic)
    all_click_cols = available_clicks + ORGANIC_CHANNELS
    all_click_cols = [c for c in all_click_cols if c in df.columns]
    df = df.with_columns(pl.sum_horizontal(all_click_cols).alias("total_clicks"))

    # CTR per channel
    for spend_col, click_col, imp_col in zip(SPEND_CHANNELS, CLICK_CHANNELS, IMPRESSION_CHANNELS):
        if click_col not in df.columns or imp_col not in df.columns:
            continue
        channel_name = spend_col.replace("_spend", "")
        df = df.with_columns(
            pl.when(pl.col(imp_col) > 0)
            .then(pl.col(click_col) / pl.col(imp_col))
            .otherwise(0)
            .alias(f"{channel_name}_ctr")
        )

    # CPM per channel (cost per 1000 impressions)
    for spend_col, imp_col in zip(SPEND_CHANNELS, IMPRESSION_CHANNELS):
        if spend_col not in df.columns or imp_col not in df.columns:
            continue
        channel_name = spend_col.replace("_spend", "")
        df = df.with_columns(
            pl.when(pl.col(imp_col) > 0)
            .then(pl.col(spend_col) / pl.col(imp_col) * 1000)
            .otherwise(0)
            .alias(f"{channel_name}_cpm")
        )

    # ROAS (Return on Ad Spend) per channel
    for spend_col in available_spend:
        channel_name = spend_col.replace("_spend", "")
        df = df.with_columns(
            pl.when(pl.col(spend_col) > 0)
            .then(pl.col(TARGET_NEW_REVENUE) / pl.col(spend_col))
            .otherwise(0)
            .alias(f"{channel_name}_roas")
        )

    # Overall ROAS
    df = df.with_columns(
        pl.when(pl.col("total_spend") > 0)
        .then(pl.col(TARGET_NEW_REVENUE) / pl.col("total_spend"))
        .otherwise(0)
        .alias("overall_roas")
    )

    return df


def create_temporal_features(df: pl.DataFrame) -> pl.DataFrame:
    """Create year, month, day_of_week, is_weekend, week_of_year."""
    df = df.with_columns(
        pl.col("date_day").dt.year().alias("year"),
        pl.col("date_day").dt.month().alias("month"),
        pl.col("date_day").dt.day().alias("day"),
        pl.col("date_day").dt.weekday().alias("day_of_week"),
        (pl.col("date_day").dt.weekday() >= 5).cast(pl.Int8).alias("is_weekend"),
        pl.col("date_day").dt.week().alias("week_of_year"),
    )
    return df


def create_lag_features(df: pl.DataFrame) -> pl.DataFrame:
    """Create lagged spend features to capture adstock/carryover effect."""
    # Sort by brand and date before lag
    df = df.sort(["organisation_id", "territory_name", "date_day"])

    lag_periods = [1, 3, 7]
    for spend_col in SPEND_CHANNELS:
        if spend_col not in df.columns:
            continue
        channel = spend_col.replace("_spend", "")
        for lag in lag_periods:
            df = df.with_columns(
                pl.col(spend_col)
                .shift(lag)
                .over(["organisation_id", "territory_name"])
                .alias(f"{channel}_lag_{lag}")
                .fill_null(0)
            )

    # Adstock (exponentially decayed lag) with decay rate from config.
    decay = ADSTOCK_DECAY
    for spend_col in SPEND_CHANNELS:
        if spend_col not in df.columns:
            continue
        channel = spend_col.replace("_spend", "")
        df = df.with_columns(
            (
                pl.col(spend_col)
                + decay * pl.col(f"{channel}_lag_1")
                + decay**2 * pl.col(f"{channel}_lag_3")
                + decay**3 * pl.col(f"{channel}_lag_7")
            ).alias(f"{channel}_adstock")
        )

    return df


def filter_extreme_outliers(df: pl.DataFrame) -> pl.DataFrame:
    """Remove rows where revenue is > 5 std from mean within brand-territory."""
    df = df.with_columns(
        pl.col(TARGET_NEW_REVENUE)
        .mean()
        .over(["organisation_id", "territory_name"])
        .alias("_mean_revenue"),
        pl.col(TARGET_NEW_REVENUE)
        .std()
        .over(["organisation_id", "territory_name"])
        .alias("_std_revenue"),
    )
    before = df.height
    df = df.filter(
        (pl.col(TARGET_NEW_REVENUE) - pl.col("_mean_revenue")).abs() <= 5 * pl.col("_std_revenue")
    )
    after = df.height
    if before != after:
        logger.info(f"  Removed {before - after:,} outlier rows (>5σ)")
    df = df.drop(["_mean_revenue", "_std_revenue"])
    return df


def preprocess(output_path: Path | None = None) -> pl.DataFrame:
    """Run full preprocessing pipeline."""
    df = load_raw_data(RAW_CSV_PATH)
    df = handle_missing_values(df)
    df = create_derived_features(df)
    df = create_temporal_features(df)
    df = create_lag_features(df)
    df = filter_extreme_outliers(df)

    out = output_path or CLEANED_PARQUET_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    logger.info(f"Saved cleaned data to {out} ({df.height:,} rows)")
    return df


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="Preprocess Conjura MMM data")
    parser.add_argument("--output", type=str, default=None, help="Output Parquet path")
    args = parser.parse_args()
    out = Path(args.output) if args.output else None
    preprocess(out)
