"""Generate a synthetic Conjura-style MMM dataset for local validation.

This is a fallback when the real ~31MB release archive cannot be downloaded.
The output matches the schema expected by attributor.preprocess and the
corresponding tests (>=100k rows, 2019-2025 daily dates, spend/clicks/
impressions/revenue columns, and organic channel columns).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

from attributor.config import (
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


def generate_mock_mmm_data(
    n_rows: int = 132_759,
    n_brands: int = 100,
    n_territories: int = 19,
    random_seed: int = 42,
) -> pl.DataFrame:
    """Generate a synthetic daily MMM panel."""
    rng = np.random.default_rng(random_seed)

    # Brands and territories
    brands = [f"brand_{i:04d}" for i in range(n_brands)]
    territories = [f"territory_{i:02d}" for i in range(n_territories)]
    verticals = ["Apparel", "Electronics", "Home", "Beauty", "Sports"]
    subverticals = ["Shoes", "Phones", "Furniture", "Skincare", "Equipment"]
    currencies = ["USD", "EUR", "GBP"]

    # Each brand gets one primary territory; optionally also "All Territories"
    brand_territory = {b: rng.choice(territories) for b in brands}

    records = []
    start_date = np.datetime64("2019-01-01")
    end_date = np.datetime64("2024-06-30")
    date_range = np.arange(start_date, end_date + 1, dtype="datetime64[D]")
    n_days = len(date_range)

    # Generate brand-level base parameters to create realistic correlations
    brand_bases = {}
    for b in brands:
        base_revenue = rng.lognormal(8.0, 1.2)
        trend = rng.uniform(-0.0005, 0.001)
        seasonality = rng.uniform(0.05, 0.3)
        brand_bases[b] = {"base_revenue": base_revenue, "trend": trend, "seasonality": seasonality}

    rows_per_day = n_rows // n_days
    remainder = n_rows % n_days

    for day_idx, dt in enumerate(date_range):
        date_str = str(dt)
        month = dt.astype(object).month
        day_of_week = dt.astype(object).weekday()
        is_weekend = int(day_of_week >= 5)

        n_today = rows_per_day + (1 if day_idx < remainder else 0)
        # Sample brand/territory combinations for today
        chosen_brands = rng.choice(brands, size=n_today)

        for b in chosen_brands:
            t = brand_territory[b]
            params = brand_bases[b]

            # Seasonality and trend
            seasonal_factor = 1.0 + params["seasonality"] * np.sin(2 * np.pi * month / 12)
            trend_factor = 1.0 + params["trend"] * day_idx
            weekend_factor = 1.2 if is_weekend else 1.0

            # Spend per channel (randomized, sometimes zero)
            spend = {}
            for ch in SPEND_CHANNELS:
                if rng.random() < 0.85:
                    spend[ch] = round(rng.lognormal(5.0, 1.5) * seasonal_factor * weekend_factor, 2)
                else:
                    spend[ch] = 0.0

            # Clicks and impressions derived from spend with noise
            clicks = {}
            impressions = {}
            for spend_col, click_col, imp_col in zip(
                SPEND_CHANNELS, CLICK_CHANNELS, IMPRESSION_CHANNELS
            ):
                cpc = rng.uniform(0.5, 5.0)
                cpm = rng.uniform(2.0, 20.0)
                clicks[click_col] = (
                    round(spend[spend_col] / cpc * rng.uniform(0.8, 1.2), 2)
                    if spend[spend_col] > 0
                    else 0.0
                )
                impressions[imp_col] = (
                    round(spend[spend_col] / cpm * 1000 * rng.uniform(0.8, 1.2), 2)
                    if spend[spend_col] > 0
                    else 0.0
                )

            # Organic channels
            organic = {
                ch: round(rng.lognormal(4.0, 1.0) * weekend_factor, 2) for ch in ORGANIC_CHANNELS
            }

            # Revenue is a function of total spend + organic + noise
            total_paid = sum(spend.values())
            total_clicks = sum(clicks.values()) + sum(organic.values())
            roas = rng.uniform(2.0, 8.0)
            base_rev = params["base_revenue"] * seasonal_factor * trend_factor * weekend_factor
            paid_rev = total_paid * roas * rng.uniform(0.8, 1.2)
            organic_rev = total_clicks * rng.uniform(0.5, 2.0)
            revenue_new = max(
                0, round(base_rev + paid_rev + organic_rev + rng.normal(0, base_rev * 0.1), 2)
            )
            revenue_all = round(revenue_new * rng.uniform(1.5, 3.0), 2)

            new_customers = max(1, int(revenue_new / rng.uniform(50, 150)))
            all_customers = max(new_customers, int(revenue_all / rng.uniform(50, 150)))

            record = {
                "mmm_timeseries_id": f"ts_{b}_{t}",
                "organisation_id": b,
                "organisation_vertical": rng.choice(verticals),
                "organisation_subvertical": rng.choice(subverticals),
                "organisation_marketing_sources": "Google,Meta,Tiktok",
                "organisation_primary_territory_name": t,
                "territory_name": t,
                "currency_code": rng.choice(currencies),
                "date_day": date_str,
                TARGET_NEW_CUSTOMERS: new_customers,
                "first_purchases_units": int(new_customers * rng.uniform(1.0, 2.0)),
                TARGET_NEW_REVENUE: revenue_new,
                "first_purchases_gross_discount": round(revenue_new * rng.uniform(0.0, 0.15), 2),
                TARGET_ALL_CUSTOMERS: all_customers,
                "all_purchases_units": int(all_customers * rng.uniform(1.0, 2.0)),
                TARGET_ALL_REVENUE: revenue_all,
                "all_purchases_gross_discount": round(revenue_all * rng.uniform(0.0, 0.15), 2),
            }
            record.update(spend)
            record.update(clicks)
            record.update(impressions)
            record.update(organic)
            records.append(record)

    df = pl.DataFrame(records)
    # Shuffle rows so brand/time ordering is not perfectly sequential
    df = df.sample(fraction=1.0, shuffle=True, seed=random_seed)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Conjura MMM CSV")
    parser.add_argument("--output", type=str, default=str(RAW_CSV_PATH), help="Output CSV path")
    parser.add_argument("--n-rows", type=int, default=132_759, help="Number of rows")
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating synthetic MMM data ({args.n_rows:,} rows) ...")
    df = generate_mock_mmm_data(n_rows=args.n_rows)
    df.write_csv(out)
    logger.info(f"Saved to {out} ({df.height:,} rows x {df.width} columns)")


if __name__ == "__main__":
    setup_logging()
    main()
