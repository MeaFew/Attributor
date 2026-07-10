"""Cross-reference audit: README claims vs. actual pipeline outputs.

Run after `make all` to verify that key metrics declared in README.md
match the actual values produced by the pipeline.

Usage: python -m attributor.audit_consistency

Add project-specific checks in the `main()` function.
"""

import json
import logging
import re
import sys
from pathlib import Path

from attributor.logging_setup import get_logger, setup_logging

logger = get_logger(__name__)

# Defensive: avoid UnicodeEncodeError on Windows consoles using GBK code page.
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def check(condition: bool, msg: str) -> bool:
    """Assert-like check that prints pass/fail."""
    if condition:
        logger.info(f"  PASS: {msg}")
    else:
        logger.info(f"  FAIL: {msg}")
    return condition


def main():
    root = Path(__file__).resolve().parents[2]
    readme = root / "README.md"
    passed = 0
    failed = 0

    # --- Check 1: R^2 in README matches mmm_results.json ---
    mmm_path = root / "data" / "processed" / "models" / "mmm_results.json"
    if mmm_path.exists():
        with open(mmm_path) as f:
            mmm = json.load(f)
        r2_actual = round(mmm["models"]["ridge"]["r2"], 3)

        # Extract R^2 from README benchmark table Ridge row
        readme_text = readme.read_text(encoding="utf-8")
        # Match the Ridge row in the benchmark/result table: | **Ridge** | 0.569 | ...
        r2_match = re.search(r"\|\s*\*\*?Ridge\*\*?\s*\|\s*(\d+\.\d+)", readme_text)
        if r2_match:
            r2_readme = float(r2_match.group(1))
            r2_readme_r = round(r2_readme, 3)
            ok = check(
                abs(r2_readme_r - r2_actual) < 0.01,
                f"R^2 (Ridge): README={r2_readme_r:.3f}, actual (mmm_results.json)={r2_actual:.3f}",
            )
            if ok:
                passed += 1
            else:
                failed += 1
        else:
            failed += 1
            check(False, "R^2 (Ridge): could not extract from README.md")
    else:
        logger.info(f"  SKIP: mmm_results.json not found at {mmm_path}")
        logger.info("         Run 'python scripts/mmm_model.py' first.")

    # --- Check 2: Attribution percentages sum to ~100% per model ---
    attr_path = root / "data" / "processed" / "models" / "attribution_comparison.json"
    if attr_path.exists():
        with open(attr_path) as f:
            attr = json.load(f)
        for model_name, values in attr.items():
            total = sum(values.values())
            ok = check(
                95 <= total <= 105,
                f"Attribution sum ({model_name}): {total:.1f}% (expected ~100%)",
            )
            if ok:
                passed += 1
            else:
                failed += 1
    else:
        logger.info(f"  SKIP: attribution_comparison.json not found at {attr_path}")
        logger.info("         Run 'python scripts/multi_touch_attribution.py' first.")

    # --- Check 3: Exactly 6 attribution models in attribution_comparison.json ---
    # The pipeline ships 6 models (First/Last/Linear/TimeDecay/Shapley/Removal).
    # A precise count catches a dropped model; the old loose `>= 5` did not.
    EXPECTED_ATTR_MODELS = 6
    if attr_path.exists():
        with open(attr_path) as f:
            attr = json.load(f)
        n_models = len(attr)
        ok = check(
            n_models == EXPECTED_ATTR_MODELS,
            f"Attribution model count: {n_models} (expected == {EXPECTED_ATTR_MODELS})",
        )
        if ok:
            passed += 1
        else:
            failed += 1

    # --- Summary ---
    total = passed + failed
    if total == 0:
        logger.info("No checks configured. Add project-specific checks to main().")
        return

    logger.info(f"\n{'=' * 40}")
    logger.info(f"Results: {passed}/{total} passed, {failed} failed")
    if failed > 0:
        logger.info("ACTION: Update README.md or pipeline to resolve mismatches.")
        sys.exit(1)


if __name__ == "__main__":
    setup_logging()
    main()
