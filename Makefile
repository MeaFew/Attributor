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

# Attribution data prep mirrors run_all.py: use the real Criteo journeys when
# the raw TSV is present (config.CRITEO_RAW_PATH), otherwise generate synthetic
# touchpoints. multi_touch_attribution.load_data() itself also falls back from
# Criteo to simulated parquet, so the downstream step is robust either way.
attribution:
	@if [ -f data/raw/criteo_attribution_dataset.tsv.gz ]; then \
		python -m attributor.preprocess_criteo; \
	else \
		echo "Criteo raw data not found (data/raw/criteo_attribution_dataset.tsv.gz); falling back to synthetic touchpoints."; \
		python -m attributor.generate_touchpoints; \
	fi
	python -m attributor.multi_touch_attribution

attribution-simulated:
	python -m attributor.generate_touchpoints
	python -m attributor.multi_touch_attribution --touchpoints data/processed/simulated_touchpoints.parquet --journeys data/processed/simulated_journeys.parquet

optimize:
	python -m attributor.budget_optimizer

dashboard:
	streamlit run dashboard/app.py

test:
	pytest tests/ -v --cov=attributor --cov-report=term-missing --cov-fail-under=40

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
