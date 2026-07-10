.PHONY: setup preprocess eda mmm attribution attribution-simulated optimize dashboard test typecheck verify clean

# ============================================================
# Marketing Attribution & Budget Optimization
# ============================================================

setup:
	pip install -r requirements.lock
	pip install -e ".[dev]"
	pre-commit install

preprocess:
	python -m attributor.preprocess

eda:
	jupyter notebook notebooks/01_eda.ipynb

mmm:
	python -m attributor.mmm_model

attribution:
	python -m attributor.preprocess_criteo
	python -m attributor.multi_touch_attribution

attribution-simulated:
	python -m attributor.generate_touchpoints
	python -m attributor.multi_touch_attribution --touchpoints data/processed/simulated_touchpoints.parquet --journeys data/processed/simulated_journeys.parquet

optimize:
	python -m attributor.budget_optimizer

dashboard:
	streamlit run dashboard/app.py

test:
	pytest tests/ -v --cov=attributor --cov-report=term-missing --cov-fail-under=20

typecheck:
	mypy src/attributor

lint:
	ruff check src/ tests/ dashboard/

format:
	ruff format src/ tests/ dashboard/

format-check:
	ruff format --check src/ tests/ dashboard/

audit:
	python -m attributor.audit_consistency

verify: lint format-check typecheck test audit

all: preprocess mmm attribution optimize

clean:
	rm -rf data/processed/*.parquet
	rm -rf data/processed/*.duckdb
	rm -rf reports/images/*.png
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
