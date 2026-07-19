"""Unit tests for data preprocessing.

The cleaned parquet is a generated artifact (gitignored) — these tests run
locally after preprocess. In CI without the raw data they skip gracefully
rather than failing (matching the pattern in the other repos).
"""

import sys
from pathlib import Path

import polars as pl
import pytest

from attributor.config import CLEANED_PARQUET_PATH


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
