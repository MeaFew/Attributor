"""Preprocess Criteo Attribution Modeling dataset into user journeys.

Input: raw/criteo_attribution_dataset.tsv.gz (impression-level)
Output: processed/criteo_touchpoints.parquet, processed/criteo_journeys.parquet

The Criteo data contains one row per impression. We aggregate by uid,
sort by timestamp, and build channel sequences for multi-touch attribution.
"""

import argparse
from pathlib import Path

import polars as pl

from attributor.config import (
    CRITEO_JOURNEYS_PATH,
    CRITEO_RAW_PATH,
    CRITEO_TOUCHPOINTS_PATH,
)
from attributor.logging_setup import get_logger, setup_logging

logger = get_logger(__name__)

# Number of top campaigns to keep as individual channels.
# Remaining campaigns are grouped into an "other" bucket.
# Keep this <= 12 so that Shapley enumeration (2^N subsets) stays tractable.
DEFAULT_TOP_N_CAMPAIGNS = 10


def build_channel_mapping(df: pl.DataFrame, top_n: int = DEFAULT_TOP_N_CAMPAIGNS) -> dict[int, str]:
    """Map top campaigns to named channels; group the rest as 'other'."""
    campaign_stats = df.group_by("campaign").agg(
        pl.len().alias("impressions"),
        pl.col("conversion").sum().alias("conversions"),
    )
    # Rank by conversions first, then impressions as tie-breaker
    top_campaigns = (
        campaign_stats.sort(["conversions", "impressions"], descending=True)
        .head(top_n)["campaign"]
        .to_list()
    )

    mapping = {}
    for campaign_id in top_campaigns:
        mapping[campaign_id] = f"campaign_{campaign_id}"
    return mapping


def preprocess_criteo(
    raw_path: Path,
    top_n: int = DEFAULT_TOP_N_CAMPAIGNS,
    output_touchpoints: Path | None = None,
    output_journeys: Path | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Convert Criteo impression data into touchpoints and journeys."""
    logger.info(f"Reading Criteo data from {raw_path}...")
    cols = ["timestamp", "uid", "campaign", "conversion"]
    df = pl.read_csv(raw_path, separator="\t", columns=cols)
    logger.info(f"  Loaded {df.height:,} impressions, {df['uid'].n_unique():,} users")

    logger.info(f"\nBuilding channel mapping (top {top_n} campaigns)...")
    channel_mapping = build_channel_mapping(df, top_n=top_n)
    top_campaign_ids = set(channel_mapping.keys())
    df = df.with_columns(
        pl.when(pl.col("campaign").is_in(top_campaign_ids))
        .then(pl.format("campaign_{}", pl.col("campaign")))
        .otherwise(pl.lit("other"))
        .alias("channel")
    )

    logger.info("\nChannel distribution:")
    logger.info(
        df.group_by("channel")
        .agg(
            pl.len().alias("impressions"),
            pl.col("conversion").sum().alias("conversions"),
        )
        .sort("impressions", descending=True)
    )

    logger.info("\nAggregating into user journeys (this may take ~1 minute)...")
    # Sort by user and timestamp to build ordered paths
    df = df.sort(["uid", "timestamp"])

    # --- Journeys: polars-native group-by aggregation ---
    journeys_df = (
        df.group_by("uid", maintain_order=True)
        .agg(
            pl.col("channel").str.concat(" > ").alias("path"),
            pl.len().alias("path_length"),
            pl.col("conversion").max().alias("converted"),
        )
        .with_columns(
            pl.col("uid").cast(pl.Utf8).alias("user_id"),
            pl.col("conversion").cast(pl.Float64).alias("conversion_value"),
        )
        .select(["user_id", "path", "path_length", "converted", "conversion_value"])
    )

    # --- Touchpoints: polars-native cumcount and conversion flag ---
    touchpoints_df = (
        df.with_columns(
            pl.col("uid").cast(pl.Utf8).alias("user_id"),
            pl.col("uid").cum_count().over("uid").alias("touchpoint_number"),
        )
        .with_columns(
            pl.col("touchpoint_number").max().over("uid").alias("_max_tp"),
        )
        .with_columns(
            ((pl.col("conversion") == 1) & (pl.col("touchpoint_number") == pl.col("_max_tp")))
            .cast(pl.Int64)
            .alias("is_conversion"),
        )
        .with_columns(
            pl.col("is_conversion").cast(pl.Float64).alias("conversion_value"),
        )
        .select(
            [
                "user_id",
                "timestamp",
                "channel",
                "touchpoint_number",
                "is_conversion",
                "conversion_value",
            ]
        )
    )

    tp_out = output_touchpoints or CRITEO_TOUCHPOINTS_PATH
    j_out = output_journeys or CRITEO_JOURNEYS_PATH
    tp_out.parent.mkdir(parents=True, exist_ok=True)

    touchpoints_df.write_parquet(tp_out)
    journeys_df.write_parquet(j_out)

    logger.info(f"\n  Touchpoints: {touchpoints_df.height:,} rows -> {tp_out}")
    logger.info(
        f"  Journeys: {journeys_df.height:,} rows "
        f"({journeys_df['converted'].sum():,} converters) -> {j_out}"
    )

    logger.info("\nPath length distribution:")
    logger.info(
        journeys_df.group_by("path_length")
        .agg(pl.len().alias("count"))
        .sort("path_length")
        .head(15)
    )

    return touchpoints_df, journeys_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(CRITEO_RAW_PATH))
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N_CAMPAIGNS)
    parser.add_argument("--touchpoints", type=str, default=None)
    parser.add_argument("--journeys", type=str, default=None)
    args = parser.parse_args()

    preprocess_criteo(
        raw_path=Path(args.input),
        top_n=args.top_n,
        output_touchpoints=Path(args.touchpoints) if args.touchpoints else None,
        output_journeys=Path(args.journeys) if args.journeys else None,
    )


if __name__ == "__main__":
    setup_logging()
    main()
