"""Unit tests for attribution models.

The attribution artifacts are generated files (gitignored) — these tests run
locally after the pipeline. In CI without the artifacts they skip gracefully
rather than failing (same pattern as tests/test_preprocess.py).
"""

import json

import pytest

from attributor.config import MODEL_OUTPUT_DIR

ATTRIBUTION_RESULTS_PATH = MODEL_OUTPUT_DIR / "attribution_comparison.json"


def test_attribution_results_exist():
    if not ATTRIBUTION_RESULTS_PATH.exists():
        pytest.skip(
            f"Attribution results not found at {ATTRIBUTION_RESULTS_PATH} "
            "(run python -m attributor.multi_touch_attribution first)"
        )
    assert ATTRIBUTION_RESULTS_PATH.exists()


def test_attribution_models_present():
    if not ATTRIBUTION_RESULTS_PATH.exists():
        pytest.skip(f"Attribution results not found at {ATTRIBUTION_RESULTS_PATH}")
    with open(ATTRIBUTION_RESULTS_PATH) as f:
        data = json.load(f)
    expected_models = [
        "first_touch",
        "last_touch",
        "linear",
        "time_decay",
        "shapley",
        "removal_effect",
    ]
    for model in expected_models:
        assert model in data, f"Model {model} not found in attribution results"


def test_attribution_sums_to_100():
    if not ATTRIBUTION_RESULTS_PATH.exists():
        pytest.skip(f"Attribution results not found at {ATTRIBUTION_RESULTS_PATH}")
    with open(ATTRIBUTION_RESULTS_PATH) as f:
        data = json.load(f)
    for model, values in data.items():
        total = sum(values.values())
        assert 95 <= total <= 105, f"Model {model} sums to {total:.1f}% (expected ~100%)"
