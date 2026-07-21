"""Unit tests for data preprocessing.

The cleaned parquet is a generated artifact (gitignored) — these tests run
locally after preprocess. In CI without the raw data they skip gracefully
rather than failing (matching the pattern in the other repos).
"""

import polars as pl
import pytest

from attributor.config import CLEANED_PARQUET_PATH
from attributor.preprocess import filter_extreme_outliers


# ---------------------------------------------------------------------------
# Pure unit tests (no artifacts needed — run everywhere)
# ---------------------------------------------------------------------------
class TestFilterExtremeOutliers:
    """Regression tests for the 5-sigma outlier filter.

    Guards the bug where rows in groups with null std (single-row groups)
    were silently dropped by the filter.
    """

    def test_single_row_groups_are_kept(self):
        """A group with one row has std == null — the row must be KEPT."""
        df = pl.DataFrame(
            {
                "organisation_id": ["a", "b", "b", "b"],
                "territory_name": ["t1", "t2", "t2", "t2"],
                "first_purchases_original_price": [100.0, 100.0, 102.0, 98.0],
            }
        )
        out = filter_extreme_outliers(df)
        assert out.height == 4
        assert (
            out.filter(
                (pl.col("organisation_id") == "a") & (pl.col("territory_name") == "t1")
            ).height
            == 1
        )

    def test_constant_groups_are_kept(self):
        """A group with constant revenue has std == 0 — rows must be KEPT."""
        df = pl.DataFrame(
            {
                "organisation_id": ["c"] * 3,
                "territory_name": ["t3"] * 3,
                "first_purchases_original_price": [50.0, 50.0, 50.0],
            }
        )
        out = filter_extreme_outliers(df)
        assert out.height == 3

    def test_true_outliers_are_removed(self):
        """A >5-sigma outlier is still removed (filter keeps its purpose).

        Needs >= 30 rows: a single extreme point's max z-score within its
        group is bounded by ~(n-1)^1.5 / n, which only exceeds 5 for n >= 30.
        """
        df = pl.DataFrame(
            {
                "organisation_id": ["b"] * 30,
                "territory_name": ["t2"] * 30,
                "first_purchases_original_price": [100.0] * 29 + [1e7],
            }
        )
        out = filter_extreme_outliers(df)
        assert out.height == 29
        assert out["first_purchases_original_price"].max() == 100.0


def test_cleaned_data_exists():
    """Cleaned data must exist (local; skips in CI where raw data is gitignored)."""
    if not CLEANED_PARQUET_PATH.exists():
        pytest.skip(
            f"Cleaned data not found at {CLEANED_PARQUET_PATH} (run python -m attributor.preprocess first)"
        )
    assert CLEANED_PARQUET_PATH.exists()


def test_cleaned_data_schema():
    if not CLEANED_PARQUET_PATH.exists():
        pytest.skip(f"Cleaned data not found at {CLEANED_PARQUET_PATH}")
    df = pl.read_parquet(CLEANED_PARQUET_PATH)
    assert df.height > 100_000
    assert "total_spend" in df.columns
    assert "year" in df.columns
    assert "month" in df.columns
    assert "google_paid_search_adstock" in df.columns


def test_no_null_total_spend():
    if not CLEANED_PARQUET_PATH.exists():
        pytest.skip(f"Cleaned data not found at {CLEANED_PARQUET_PATH}")
    df = pl.read_parquet(CLEANED_PARQUET_PATH)
    assert df["total_spend"].null_count() == 0


def test_date_range():
    if not CLEANED_PARQUET_PATH.exists():
        pytest.skip(f"Cleaned data not found at {CLEANED_PARQUET_PATH}")
    df = pl.read_parquet(CLEANED_PARQUET_PATH)
    min_date = df["date_day"].min()
    max_date = df["date_day"].max()
    assert min_date.year >= 2019
    assert max_date.year <= 2025
