# Convenience targets. Everything here is a thin wrapper over a documented CLI;
# nothing is only reachable through make.
PYTHON ?= python
SEEDS  ?= 30
WORKERS ?= $(shell $(PYTHON) -c "import os;print(os.cpu_count() or 1)")
RESULTS ?= results

.PHONY: help install data test test-all lint format baseline experiments benchmark \
        calibrate optimize sensitivity sumo report dashboard all clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## install runtime and dev dependencies
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e ".[dev,dashboard]"

data:  ## regenerate the synthetic data layer
	$(PYTHON) scripts/generate_data.py
	$(PYTHON) -m src.calibration.calibrate --generate --seeds 3 --maxiter 1 --popsize 2 || true

test:  ## fast test suite
	$(PYTHON) -m pytest tests -q -m "not slow"

test-all:  ## every test, including slow and SUMO
	$(PYTHON) -m pytest tests -q --cov=src --cov-report=term-missing

lint:  ## static checks
	ruff check src tests scripts

format:  ## auto-format
	ruff format src tests scripts

baseline:  ## baseline scenario only
	$(PYTHON) -m src.experiments.runner --scenario baseline --seeds $(SEEDS) --workers $(WORKERS) --out $(RESULTS)

experiments:  ## every scenario x $(SEEDS) seeds
	$(PYTHON) -m src.experiments.runner --all --seeds $(SEEDS) --workers $(WORKERS) --out $(RESULTS)

benchmark:  ## parallel scaling benchmark
	$(PYTHON) -m src.experiments.runner --benchmark --seeds 16 --out $(RESULTS)

sensitivity:  ## cross-modal elasticities and demand sweep
	$(PYTHON) -m src.experiments.sensitivity --seeds 10 --workers $(WORKERS) --out $(RESULTS)/sensitivity

calibrate:  ## fit behavioural parameters and score the hold-outs
	$(PYTHON) -m src.calibration.calibrate --seeds 3 --workers $(WORKERS) --out $(RESULTS)/calibration

optimize:  ## search for the best curb allocation
	$(PYTHON) -m src.optimization.optimizer --workers $(WORKERS) --out $(RESULTS)/optimization

sumo:  ## one replication through the SUMO backend
	$(PYTHON) -m src.sumo.backend --horizon 60 --out $(RESULTS)/sumo

report:  ## build the static HTML report from existing results
	$(PYTHON) -m src.viz.report --results $(RESULTS)

dashboard:  ## interactive dashboard
	streamlit run src/viz/dashboard.py

all: data experiments benchmark sensitivity calibrate optimize report  ## full pipeline

clean:  ## remove generated results and caches
	rm -rf $(RESULTS)/* .pytest_cache .ruff_cache .coverage htmlcov
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	touch $(RESULTS)/.gitkeep
