"""Multi-touch attribution models comparison."""

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import polars as pl

from attributor.config import (
    ATTRIBUTION_HALF_LIFE_DAYS,
    CRITEO_JOURNEYS_PATH,
    CRITEO_TOUCHPOINTS_PATH,
    MODEL_OUTPUT_DIR,
    SIMULATED_JOURNEYS_PATH,
    SIMULATED_TOUCHPOINTS_PATH,
)
from attributor.logging_setup import get_logger, setup_logging

logger = get_logger(__name__)


def load_data(
    touchpoints_path: Path | None = None,
    journeys_path: Path | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load touchpoint and journey data.

    Defaults to Criteo real data if available; falls back to simulated data.
    """
    tp_path = touchpoints_path or CRITEO_TOUCHPOINTS_PATH
    j_path = journeys_path or CRITEO_JOURNEYS_PATH

    if not tp_path.exists() or not j_path.exists():
        logger.info(
            f"Criteo data not found at {tp_path} / {j_path}; falling back to simulated data."
        )
        tp_path = SIMULATED_TOUCHPOINTS_PATH
        j_path = SIMULATED_JOURNEYS_PATH

    tp = pl.read_parquet(tp_path)
    j = pl.read_parquet(j_path)
    return tp, j


# ---------------------------------------------------------------------------
# Rule-based attribution models
# ---------------------------------------------------------------------------


def _channel_share(df: pl.DataFrame, value_col: str = "attributed") -> dict[str, float]:
    """Aggregate per-channel attribution and return a {channel: value} dict.

    The 4-5 line `group_by("channel").agg(sum).sort(descending) -> dict`
    epilogue was repeated in every attribution model; centralize it here.
    """
    result = (
        df.group_by("channel")
        .agg(pl.sum(value_col).alias("attributed"))
        .sort("attributed", descending=True)
    )
    return {row["channel"]: row["attributed"] for row in result.iter_rows(named=True)}


def first_touch_attribution(tp: pl.DataFrame, journeys: pl.DataFrame) -> dict[str, float]:
    """Attribute 100% to the first touchpoint."""
    # Get conversion value per user from journeys
    conv_values = journeys.select(["user_id", "conversion_value"])
    first = tp.filter(pl.col("touchpoint_number") == 1).drop("conversion_value")
    first = first.join(conv_values, on="user_id")
    return _channel_share(first, "conversion_value")


def last_touch_attribution(tp: pl.DataFrame) -> dict[str, float]:
    """Attribute 100% to the last touchpoint (conversion touchpoint)."""
    last = tp.filter(pl.col("is_conversion") == 1)
    return _channel_share(last, "conversion_value")


def linear_attribution(tp: pl.DataFrame) -> dict[str, float]:
    """Attribute the journey's total conversion value equally across all touchpoints."""
    # Broadcast the journey's total conversion value to every touchpoint
    journey_value = tp.group_by("user_id").agg(pl.sum("conversion_value").alias("journey_value"))
    tp = tp.drop("conversion_value").join(journey_value, on="user_id")

    # Count touchpoints per user
    counts = tp.group_by("user_id").agg(pl.len().alias("n_touches"))
    tp = tp.join(counts, on="user_id")
    # Equal split: journey_value / n_touches. Write to a dedicated "attributed"
    # column (explicit alias) instead of overwriting "journey_value" in place —
    # the in-place overwrite relied on Polars keeping the expression's source
    # column name, which is fragile across refactors/Polars versions.
    tp = tp.with_columns((pl.col("journey_value") / pl.col("n_touches")).alias("attributed"))

    return _channel_share(tp, "attributed")


def time_decay_attribution(
    tp: pl.DataFrame, journeys: pl.DataFrame, half_life_days: float = ATTRIBUTION_HALF_LIFE_DAYS
) -> dict[str, float]:
    """Attribute more weight to touchpoints closer to conversion.

    Each touchpoint receives a share of the journey's total conversion_value,
    proportional to its time-decay weight relative to all touchpoints in the
    same journey.
    """
    # Drop per-touchpoint conversion_value (only set on conversion touchpoint)
    tp = tp.drop("conversion_value")

    # Get conversion timestamp per user
    conv = tp.filter(pl.col("is_conversion") == 1).select(["user_id", "timestamp"])
    tp = tp.join(conv, on="user_id", suffix="_conv")

    # Compute days to conversion.
    # Simulated data uses ISO datetime strings; Criteo data uses integer seconds.
    if tp["timestamp"].dtype == pl.Utf8:
        tp = tp.with_columns(
            (
                pl.col("timestamp_conv").str.to_datetime(strict=False)
                - pl.col("timestamp").str.to_datetime(strict=False)
            )
            .dt.total_days()
            .alias("days_to_conv")
        )
    else:
        # Assume numeric timestamp (e.g., seconds); convert to days
        tp = tp.with_columns(
            ((pl.col("timestamp_conv") - pl.col("timestamp")) / 86_400.0).alias("days_to_conv")
        )

    # Decay weight: 2^(-days/half_life)
    tp = tp.with_columns((2.0 ** (-pl.col("days_to_conv") / half_life_days)).alias("weight"))

    # Normalize weights per user
    weights = tp.group_by("user_id").agg(pl.sum("weight").alias("total_weight"))
    tp = tp.join(weights, on="user_id")

    # Join total conversion_value from the full journey table
    conv_values = journeys.select(["user_id", "conversion_value"])
    tp = tp.join(conv_values, on="user_id")

    # Attribute: weight/total_weight * user's total conversion_value. Write to a
    # dedicated "attributed" column via explicit alias rather than overwriting
    # "weight" in place (see linear_attribution for the rationale).
    tp = tp.with_columns(
        (pl.col("weight") / pl.col("total_weight") * pl.col("conversion_value")).alias("attributed")
    )

    return _channel_share(tp, "attributed")


# ---------------------------------------------------------------------------
# Shapley Value attribution
# ---------------------------------------------------------------------------


def shapley_attribution(journeys: pl.DataFrame) -> dict[str, float]:
    """Compute Shapley Value for each channel based on all subset combinations.

    Uses the standard definition: v(S) = total conversion value of paths whose
    channel set is a SUBSET of S. Larger S means more paths are included.

    Implementation note: the per-subset v(S) is computed with a subset-zeta
    transform (sum-over-subsets DP) in O(2^n * n) instead of the naive
    O(2^n * m) double loop, and factorials are precomputed once.
    """
    # Build exact conversion value per channel set
    v_exact = defaultdict(float)
    for row in journeys.filter(pl.col("converted") == 1).iter_rows(named=True):
        channels = frozenset(row["path"].split(" > "))
        v_exact[channels] += row["conversion_value"]

    all_channels = set()
    for cs in v_exact:
        all_channels.update(cs)
    all_channels = sorted(all_channels)
    n = len(all_channels)

    if n == 0:
        return {}

    # Map channels to bit indices so subsets become integer bitmasks.
    ch_to_bit = {ch: i for i, ch in enumerate(all_channels)}

    # Seed v[S] with the exact values, then propagate each value to all
    # supersets via the sum-over-subsets (zeta) transform.
    size = 1 << n
    v = [0.0] * size
    for cs, val in v_exact.items():
        mask = 0
        for ch in cs:
            mask |= 1 << ch_to_bit[ch]
        v[mask] += val
    for i in range(n):
        bit = 1 << i
        for mask in range(size):
            if mask & bit:
                v[mask] += v[mask ^ bit]

    # Precompute factorials once (was recomputed inside the hot loop before).
    fact = [math.factorial(k) for k in range(n + 1)]

    # Compute Shapley values: for each channel c, sum the weighted marginal
    # contribution of adding c to every subset S not containing c.
    shapley = {ch: 0.0 for ch in all_channels}
    for ch in all_channels:
        c_bit = 1 << ch_to_bit[ch]
        for mask in range(size):
            if mask & c_bit:
                continue
            s = bin(mask).count("1")
            weight = fact[s] * fact[n - s - 1] / fact[n]
            marginal = v[mask | c_bit] - v[mask]
            shapley[ch] += marginal * weight

    # Check for negative Shapley values and warn
    negative_channels = [(k, val) for k, val in shapley.items() if val < 0]
    if negative_channels:
        names = ", ".join(f"{k} ({val:.2f})" for k, val in negative_channels)
        logger.info(
            f"  Warning: {len(negative_channels)} channel(s) had negative Shapley values "
            f"and were clipped to 0: {names}"
        )
    # Ensure non-negative
    shapley = {k: max(0.0, val) for k, val in shapley.items()}

    # Normalize to total conversion value
    total_conv = sum(v_exact.values())
    total_shapley = sum(shapley.values())
    if total_shapley > 0:
        shapley = {k: round(val / total_shapley * total_conv, 2) for k, val in shapley.items()}

    return dict(sorted(shapley.items(), key=lambda x: x[1], reverse=True))


# ---------------------------------------------------------------------------
# Removal Effect attribution
# ---------------------------------------------------------------------------


def removal_effect_attribution(journeys: pl.DataFrame) -> dict[str, float]:
    """Channel removal effect analysis.

    Computes attribution by measuring the drop in overall conversion rate when
    a channel is removed from all user journeys. Channels whose removal causes
    the largest conversion drop get more credit.

    This is a removal effect analysis, NOT a full Markov chain model
    (no transition probability matrix). The name reflects the underlying
    intuition: measure each channel's contribution by observing what happens
    when it is removed.

    Implementation: a one-hot presence column is built once per channel, then a
    single pass computes (converted, total) counts per channel — avoiding the
    previous O(N * channels) pattern of one full regex scan per channel.
    """
    total_users = journeys.height
    total_conv = journeys.filter(pl.col("converted") == 1).height
    baseline_rate = total_conv / total_users if total_users > 0 else 0

    if baseline_rate == 0:
        return {}

    all_channels = set()
    for row in journeys.iter_rows(named=True):
        all_channels.update(row["path"].split(" > "))
    all_channels = sorted(all_channels)

    # Build one-hot presence columns in a single with_columns call, then
    # compute per-channel (users-without, conversions-without) via aggregation.
    # `present_<ch>` is 1 when the channel appears in the journey path.
    present_cols = []
    exprs = []
    for ch in all_channels:
        col = f"_present_{ch}"
        present_cols.append(col)
        exprs.append(pl.col("path").str.contains(rf"\b{re.escape(ch)}\b").cast(pl.Int8).alias(col))
    enriched = journeys.with_columns(exprs)

    removal_effects = {}
    for ch, col in zip(all_channels, present_cols):
        without = enriched.filter(pl.col(col) == 0)
        n_without = without.height
        conv_without = without.filter(pl.col("converted") == 1).height
        rate_without = conv_without / n_without if n_without > 0 else 0
        effect = (baseline_rate - rate_without) / baseline_rate
        removal_effects[ch] = max(0, effect)

    # Attribute conversions proportional to removal effect
    total_effect = sum(removal_effects.values())
    if total_effect > 0:
        total_conv_value = journeys.filter(pl.col("converted") == 1)["conversion_value"].sum()
        result = {
            ch: round(effect / total_effect * total_conv_value, 2)
            for ch, effect in removal_effects.items()
        }
    else:
        result = {ch: 0.0 for ch in removal_effects}

    return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))


# ---------------------------------------------------------------------------
# Run all models and compare
# ---------------------------------------------------------------------------


def run_all_models(
    touchpoints_path: Path | None = None,
    journeys_path: Path | None = None,
) -> dict:
    """Run all attribution models and return comparison."""
    tp, journeys = load_data(touchpoints_path, journeys_path)

    logger.info("Running attribution models...")

    results = {
        "first_touch": first_touch_attribution(tp, journeys),
        "last_touch": last_touch_attribution(tp),
        "linear": linear_attribution(tp),
        "time_decay": time_decay_attribution(tp, journeys),
        "shapley": shapley_attribution(journeys),
        "removal_effect": removal_effect_attribution(journeys),
    }

    # Normalize to percentages
    for model_name, values in results.items():
        total = sum(values.values())
        if total > 0:
            results[model_name] = {k: round(v / total * 100, 2) for k, v in values.items()}

    # Save JSON
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = MODEL_OUTPUT_DIR / "attribution_comparison.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"  Saved comparison to {out}")

    # Print summary
    logger.info("\nAttribution model comparison (% of total conversions):")
    all_channels = set()
    for v in results.values():
        all_channels.update(v.keys())

    header = f"{'Channel':<20}" + "".join(f"{m:<12}" for m in results)
    logger.info(header)
    logger.info("-" * len(header))
    for ch in sorted(all_channels):
        row = f"{ch:<20}" + "".join(f"{results[m].get(ch, 0):>10.1f}%  " for m in results)
        logger.info(row)

    return results


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--touchpoints", type=str, default=None)
    parser.add_argument("--journeys", type=str, default=None)
    args = parser.parse_args()

    tp_path = Path(args.touchpoints) if args.touchpoints else None
    j_path = Path(args.journeys) if args.journeys else None
    run_all_models(touchpoints_path=tp_path, journeys_path=j_path)
