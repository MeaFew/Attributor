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
	python -c "import shutil, pathlib; [p.unlink() for p in pathlib.Path('data/processed').glob('*.parquet')]; [p.unlink() for p in pathlib.Path('data/processed').glob('*.duckdb')]; [p.unlink() for p in pathlib.Path('reports/images').glob('*.png') if p.exists()]; [shutil.rmtree(d, ignore_errors=True) for d in pathlib.Path('.').rglob('__pycache__')]; [shutil.rmtree(d, ignore_errors=True) for d in pathlib.Path('.').rglob('.pytest_cache')]; [shutil.rmtree(d, ignore_errors=True) for d in pathlib.Path('.').rglob('.ruff_cache')]"
