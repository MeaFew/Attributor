"""Unit tests for attribution models."""

import json
import sys
from pathlib import Path

from attributor.config import MODEL_OUTPUT_DIR


def test_attribution_results_exist():
    path = MODEL_OUTPUT_DIR / "attribution_comparison.json"
    assert path.exists(), (
        "Attribution results not found. Run python -m attributor.multi_touch_attribution first."
    )


def test_attribution_models_present():
    with open(MODEL_OUTPUT_DIR / "attribution_comparison.json") as f:
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
    with open(MODEL_OUTPUT_DIR / "attribution_comparison.json") as f:
        data = json.load(f)
    for model, values in data.items():
        total = sum(values.values())
        assert 95 <= total <= 105, f"Model {model} sums to {total:.1f}% (expected ~100%)"
