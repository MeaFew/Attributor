"""Generate simulated user touchpoint data for multi-touch attribution."""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

from attributor.config import (
    SIMULATED_JOURNEYS_PATH,
    SIMULATED_TOUCHPOINTS_PATH,
    SIMULATION_PARAMS,
)
from attributor.logging_setup import get_logger, setup_logging

logger = get_logger(__name__)


def generate_touchpoints(
    n_users: int = 50_000,
    max_touchpoints: int = 8,
    conversion_rate: float = 0.035,
    channels: list[str] | None = None,
    channel_weights: dict[str, float] | None = None,
    date_range_days: int = 365,
    random_seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Generate simulated user journey data."""
    rng = np.random.default_rng(random_seed)

    channels = channels or SIMULATION_PARAMS["channels"]
    weights = channel_weights or SIMULATION_PARAMS["channel_weights"]
    weights_list = np.array([weights.get(c, 0.01) for c in channels])
    weights_list = weights_list / weights_list.sum()

    # Generate user-level data
    n_converters = int(n_users * conversion_rate)
    n_non_converters = n_users - n_converters

    touchpoint_records = []
    journey_records = []

    user_id = 0
    base_date = np.datetime64("2023-01-01")

    for is_converter in [True, False]:
        n = n_converters if is_converter else n_non_converters
        for _ in range(n):
            uid = f"u{user_id:06d}"
            user_id += 1

            # Number of touchpoints (converters tend to have more)
            if is_converter:
                n_touches = rng.integers(2, max_touchpoints + 1)
            else:
                n_touches = rng.integers(1, max_touchpoints)

            # Generate touchpoint sequence
            touch_channels = rng.choice(channels, size=n_touches, p=weights_list)
            touch_dates = base_date + rng.integers(0, date_range_days, size=n_touches)
            touch_dates = np.sort(touch_dates)

            # Conversion value (only for converters)
            conv_value = round(rng.lognormal(4.5, 0.8), 2) if is_converter else 0.0

            path = []
            for i, (ch, dt) in enumerate(zip(touch_channels, touch_dates)):
                is_conv = 1 if (is_converter and i == n_touches - 1) else 0
                touchpoint_records.append(
                    {
                        "user_id": uid,
                        "timestamp": str(dt),
                        "channel": ch,
                        "touchpoint_number": i + 1,
                        "is_conversion": is_conv,
                        "conversion_value": conv_value if is_conv else 0.0,
                    }
                )
                path.append(ch)

            journey_records.append(
                {
                    "user_id": uid,
                    "path": " > ".join(path),
                    "path_length": len(path),
                    "converted": 1 if is_converter else 0,
                    "conversion_value": conv_value,
                }
            )

    touchpoints_df = pl.DataFrame(touchpoint_records)
    journeys_df = pl.DataFrame(journey_records)

    return touchpoints_df, journeys_df


def main(output_touchpoints: Path | None = None, output_journeys: Path | None = None) -> None:
    """Run touchpoint generation."""
    # Read parameters from shared config
    params = SIMULATION_PARAMS
    logger.info("Generating simulated user touchpoint data...")
    touchpoints, journeys = generate_touchpoints(
        n_users=params["n_users"],
        max_touchpoints=params["max_touchpoints_per_user"],
        conversion_rate=params["conversion_rate"],
        channels=params["channels"],
        channel_weights=params["channel_weights"],
        date_range_days=params["date_range_days"],
    )

    tp_out = output_touchpoints or SIMULATED_TOUCHPOINTS_PATH
    j_out = output_journeys or SIMULATED_JOURNEYS_PATH
    tp_out.parent.mkdir(parents=True, exist_ok=True)

    touchpoints.write_parquet(tp_out)
    journeys.write_parquet(j_out)

    logger.info(f"  Touchpoints: {touchpoints.height:,} rows -> {tp_out}")
    logger.info(
        f"  Journeys: {journeys.height:,} rows ({journeys['converted'].sum():,} converters) -> {j_out}"
    )

    # Summary stats
    logger.info("\nChannel distribution in touchpoints:")
    logger.info(
        touchpoints.group_by("channel").agg(pl.len().alias("count")).sort("count", descending=True)
    )

    logger.info("\nPath length distribution:")
    logger.info(journeys.group_by("path_length").agg(pl.len().alias("count")).sort("path_length"))


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--touchpoints", type=str, default=None)
    parser.add_argument("--journeys", type=str, default=None)
    args = parser.parse_args()

    tp = Path(args.touchpoints) if args.touchpoints else None
    j = Path(args.journeys) if args.journeys else None
    main(tp, j)
