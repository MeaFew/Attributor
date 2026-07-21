"""Unit tests for MMM model.

The model artifacts are generated files (gitignored) — these tests run locally
after the pipeline. In CI without the artifacts they skip gracefully rather
than failing (same pattern as tests/test_preprocess.py).
"""

import json

import pytest

from attributor.config import MODEL_OUTPUT_DIR

MMM_RESULTS_PATH = MODEL_OUTPUT_DIR / "mmm_results.json"
BUDGET_RESULTS_PATH = MODEL_OUTPUT_DIR / "budget_optimization.json"


def test_mmm_results_exist():
    if not MMM_RESULTS_PATH.exists():
        pytest.skip(
            f"MMM results not found at {MMM_RESULTS_PATH} (run python -m attributor.mmm_model first)"
        )
    assert MMM_RESULTS_PATH.exists()


def test_mmm_results_structure():
    if not MMM_RESULTS_PATH.exists():
        pytest.skip(f"MMM results not found at {MMM_RESULTS_PATH}")
    with open(MMM_RESULTS_PATH) as f:
        data = json.load(f)
    assert "models" in data
    assert "ols" in data["models"]
    assert "ridge" in data["models"]
    assert "lasso" in data["models"]
    assert "coefficients" in data["models"]["ols"]


def test_budget_results_exist():
    if not BUDGET_RESULTS_PATH.exists():
        pytest.skip(
            f"Budget results not found at {BUDGET_RESULTS_PATH} (run python -m attributor.budget_optimizer first)"
        )
    assert BUDGET_RESULTS_PATH.exists()
