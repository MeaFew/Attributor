"""Unit tests for MMM model."""

import json
import sys
from pathlib import Path

from attributor.config import MODEL_OUTPUT_DIR


def test_mmm_results_exist():
    path = MODEL_OUTPUT_DIR / "mmm_results.json"
    assert path.exists(), "MMM results not found. Run python -m attributor.mmm_model first."


def test_mmm_results_structure():
    with open(MODEL_OUTPUT_DIR / "mmm_results.json") as f:
        data = json.load(f)
    assert "models" in data
    assert "ols" in data["models"]
    assert "ridge" in data["models"]
    assert "lasso" in data["models"]
    assert "coefficients" in data["models"]["ols"]


def test_budget_results_exist():
    path = MODEL_OUTPUT_DIR / "budget_optimization.json"
    assert path.exists(), "Budget results not found. Run python -m attributor.budget_optimizer first."
